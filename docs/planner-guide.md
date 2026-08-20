# HSEM Planner — Technical Guide

This guide explains how the HSEM (Home Assistant Solar Energy Management) planner works.
It covers inputs, outputs, the cost function, safety modes, and worked examples for five
common scenarios a real installation will encounter.

> **See also:** `docs/planner-spec.md` — the normative specification that governs
> all planner invariants and implementation rules.

---

## Table of contents

1. [Overview](#overview)
2. [Planning inputs](#planning-inputs)
3. [Planning outputs](#planning-outputs)
4. [EV planned load integration](#ev-planned-load-integration)
5. [Cost function](#cost-function)
6. [Candidate generation and selection](#candidate-generation-and-selection)
7. [Safety modes](#safety-modes)
8. [Data quality diagnostics](#data-quality-diagnostics)
9. [Diagnostic accuracy and daily accounting](#diagnostic-accuracy-and-daily-accounting)
10. [Scenario examples](#scenario-examples)
   - [Winter price arbitrage](#scenario-1-winter-price-arbitrage)
   - [Summer day — high PV surplus](#scenario-2-summer-day--high-pv-surplus)
   - [Cheap night price — grid charge opportunity](#scenario-3-cheap-night-price--grid-charge-opportunity)
   - [High PV day — export now, refill later](#scenario-4-high-pv-day--export-now-refill-later)
   - [Flat price day — self-consume, do not arbitrage](#scenario-5-flat-price-day--self-consume-do-not-arbitrage)
   - [EV charging — MILP co-optimisation](#scenario-6-ev-charging--milp-co-optimisation)
11. [Reading the plan explanation](#reading-the-plan-explanation)
12. [Known limitations](#known-limitations)

---

## Overview

The HSEM planner is a forward-looking, cost-minimising battery scheduler.
Every time the coordinator runs (typically every minute) the planner:

1. Reads the current battery state, electricity prices, and PV forecast.
2. Generates a time grid of **slots** covering the planning horizon (24, 48, or 72 hours).
3. Populates each slot with expected house load, PV production, and prices.
4. Evaluates several candidate strategies (charge from grid, discharge only, solar only, etc.).
5. Scores every candidate with the cost function.
6. Writes the lowest-cost valid plan to the `HourlyRecommendation` objects consumed by the coordinator.

The planner is **pure Python with no Home Assistant imports**. It runs synchronously,
produces deterministic output for identical input, and is fully testable with plain pytest.

---

## Planning inputs

All inputs are collected in the `PlannerInput` dataclass
(`custom_components/hsem/models/planner_inputs.py`).

### Temporal context

| Field | Type | Description |
|---|---|---|
| `now_iso` | `str` | ISO-8601 timezone-aware timestamp of the planning moment (e.g. `"2024-06-15T14:00:00+02:00"`) |
| `interval_minutes` | `int` | Slot width in minutes — `15` or `60` |
| `interval_length_hours` | `int` | Planning horizon length — `24`, `48`, or `72` hours |

The total number of slots generated is `(interval_length_hours * 60) // interval_minutes`.

| Horizon | 15-min slots | 60-min slots |
|---|---|---|
| 24 h | 96 | 24 |
| 48 h | 192 | 48 |
| 72 h | 288 | 72 |

### Battery hardware

| Field | Type | Description |
|---|---|---|
| `battery_soc_pct` | `float` | Current SoC percentage (0–100) |
| `battery_rated_capacity_kwh` | `float` | Nameplate capacity in kWh |
| `battery_end_of_discharge_soc_pct` | `float` | Minimum allowed SoC floor (%) |
| `battery_max_soc_pct` | `float` | Maximum allowed SoC ceiling (%, default 100) |
| `battery_max_charge_power_w` | `float` | Maximum charge power in Watts |
| `battery_max_discharge_power_w` | `float \| None` | Maximum discharge power in Watts (`None` = unlimited). Derived from the rated capacity via `get_max_discharge_power()`, which covers S0/S1 single- and two-stack capacities (5–30 kWh); unknown capacities log a warning and fall back to 2500 W |
| `battery_conversion_loss_pct` | `float` | Round-trip conversion loss (%) |

The planner converts power limits to per-slot energy limits internally:

```text
max_charge_per_slot_kwh = battery_max_charge_power_w / 1000 * (interval_minutes / 60)
```

### Battery economics

| Field | Type | Description |
|---|---|---|
| `battery_purchase_price` | `float` | Purchase price of the battery (local currency) |
| `battery_expected_cycles` | `int` | Expected total lifetime cycles |
| `battery_cycle_cost_per_kwh` | `float` | Explicit depreciation cost per kWh cycled |
| `battery_capacity_loss_pct` | `float` | Expected capacity loss at end-of-life (%), default 30 |
| `battery_charge_efficiency_pct` | `float` | Charge-side efficiency (%), e.g. 98 |
| `battery_discharge_efficiency_pct` | `float` | Discharge-side efficiency (%), e.g. 98 |

When `battery_cycle_cost_per_kwh` is `0.0`, the planner auto-derives cycle cost from
purchase price, rated capacity, expected cycles, capacity loss at EOL, and round-trip
conversion loss:

```text
depreciation      = (purchase_price × capacity_loss_pct / 100)
                   / (2 × usable_capacity_kwh × expected_cycles)
threshold         = depreciation
```

Conversion (in)efficiency losses are priced per-slot by the MILP objective
and the cost function's ``conversion_loss_cost`` term, both of which use the
actual import price of each slot rather than a fixed add-on.  The 2× factor
in the depreciation term accounts for one full cycle (charge + discharge).  The
``capacity_loss_pct`` (default 30 %) accounts for the fraction of the battery's
value that is consumed over its lifetime — typically 20 % physical capacity
loss at 6000 LiFePO4 cycles plus ~10 % margin for calendar ageing.

`excess_export_price_threshold` remains a derived diagnostic/context value.
The MILP objective and scorer use the canonical cycle-wear cost; a positive
`battery_cycle_cost_per_kwh` is added to its auto-derived depreciation cost.

#### Dynamic discharge floor

The planner computes a **dynamic discharge floor** — a bridge-to-refill
minimum SoC that prevents the battery from being discharged below the
level needed to reach the next solar refill window:

```text
bridge_reserve_pct = (next_refill_kwh − expected_charge_kwh)
                    / usable_capacity_kwh × 100
effective_floor   = max(configured_min_soc_pct,
                        bridge_reserve_pct × safety_margin)
```

where `safety_margin` is a **self-correcting multiplier** that starts at
1.50 and gradually decays toward 1.05 as the tracker observes successful
refills.  The floor is clamped to the hardware minimum SoC.

This prevents the planner from discharging the battery late in the
evening when the next day's solar forecast is insufficient to refill it —
the battery retains enough energy to cover the gap.  Without this guard,
the planner would discharge to the configured floor every night, forcing
morning grid imports when solar is scarce.

### Consumption prediction

HSEM predicts house load for each slot.  Two modes are available (toggled via
``hsem_ml_consumption_enabled``):

- **Legacy (default):** Weighted average across four rolling windows (1d, 3d,
  7d, 14d) with IQR outlier detection.  Requires HSEM custom sensor entities.
- **ML (optional):** Ridge regression on recorder history with day-of-week,
  day-of-year seasonality, and optional outdoor temperature. No custom sensors
  are needed.

ML calendar features use Home Assistant-local date, day-of-week, day-of-year,
and wall slot. Recorder ordering, elapsed age, adjacency, cache expiry, and
slot lookup use canonical UTC instants, preserving both autumn DST folds and
the 92/96/100-slot civil-day shape. Accumulator deltas are labelled with the
current slot they measure, never the preceding slot, and recorder gaps are not
collapsed into one oversized observation.

ML requires the configured history span (14 days by default) and fails closed
to legacy averages when it cannot train. Processed caches are keyed by the Home
Assistant instance, meter entities, net/gross mode, cadence, and history
requirement; changing context replaces the predictor. Net mode uses only
physically aligned import/export slots and never treats a missing export sample
as zero. Optional temperature is trained only with sufficient history; the
latest nearby observation is held across inference because it is not a weather
forecast. An untrained model never publishes a zero-load horizon.

Regardless of mode, the planner receives a per-hour ``HourlyConsumptionAverage``:

| Field | Type | Description |
|---|---|---|
| `consumption_averages` | `list[HourlyConsumptionAverage]` | Per-hour (0–23) historical averages |
| `weight_1d` | `int` | Weight for the 1-day average (integer %) |
| `weight_3d` | `int` | Weight for the 3-day average (integer %) |
| `weight_7d` | `int` | Weight for the 7-day average (integer %) |
| `weight_14d` | `int` | Weight for the 14-day average (integer %) |

Weights must sum to 100. Default split: 1d=25 %, 3d=30 %, 7d=30 %, 14d=15 %.

Each `HourlyConsumptionAverage` carries:

- `hour` — 0-based clock-hour (0–23)
- `avg_1d`, `avg_3d`, `avg_7d`, `avg_14d` — average kWh for that hour over each window

The planner applies a median-ratio outlier detection algorithm that flags anomalous
windows and redistributes their weight to stable windows before combining the averages.

In addition, the optional **weekday/weekend profiling** feature (`WeekdayProfile`,
#612) maintains separate 24-slot EWMA profiles for Mon–Fri and Sat–Sun.  When
active, the appropriate profile is merged into the consumption prediction to
better capture the distinct usage patterns between workdays and weekends.

### Price data

| Field | Type | Description |
|---|---|---|
| `price_points` | `list[PricePoint]` | Hourly import/export prices |

Each `PricePoint` carries:

- `hour` — 0-based clock-hour
- `import_price` — cost to buy 1 kWh from the grid (local currency/kWh)
- `export_price` — revenue from selling 1 kWh to the grid (local currency/kWh)

Prices sourced from Energi Data Service (EDS) are normalised through the
`eds_share` pipeline before reaching the planner so the engine always receives
the full hourly rate, regardless of the EDS update interval (15 min or 60 min).
See [Price interval semantics](planner-spec.md#price-interval-semantics) in the spec.

### PV forecast

| Field | Type | Description |
|---|---|---|
| `solcast_slots` | `list[SolcastSlot]` | Forecast PV production per hour |

Each `SolcastSlot` carries:

- `hour` — 0-based clock-hour
- `pv_estimate` — expected PV energy (kWh) for that hour

For multi-day horizons, a **confidence decay** factor is applied to PV estimates
for future days to account for forecast uncertainty:

| Day offset | Decay | Meaning |
|---|---|---|
| 0 (today) | 1.00 | No decay |
| 1 (tomorrow) | 0.90 | 10 % conservative discount |
| 2 (day after) | 0.80 | 20 % conservative discount |

Prices are **not** decayed because spot-market prices are typically firm by mid-day.

#### Solar forecast auto-correction

Raw Solcast PV estimates are corrected in two stages before entering the
planner (issue #602):

**Per-hour accuracy factors:** The `SolarForecastCorrector` maintains a
rolling history of the most recent eligible `(raw forecast, actual)` pairs
for each hour. The mean actual/raw ratio is clamped to **[0.3, 1.5]**.

**Short-term residual correction:** The mean actual/corrected ratio from the
most recent eligible closed slots is clamped to the same bounds and decays
linearly toward 1.0 over eight future slots:

```text
corrected_pv = raw_pv × hour_factor × residual_factor
residual_factor = 1.0 + (mean_recent_ratio − 1.0)
                × max(0, 1 − slots_ahead / 8)
```

Lead distance is measured from the current planning instant on the UTC
timeline, not from local midnight. The correction therefore remains active
after early morning and treats both DST folds as separate physical slots.

The raw Solcast data is never mutated; corrections are only applied at
consumption time when the planner reads PV estimates. Learning accepts only the
last pre-slot baseline with complete trusted PV and load coverage. Current-slot
live injection, a post-start replan, missing telemetry, or a stale sample gap
cannot teach the corrector. Upgrading to v7.1.2 intentionally resets old
unversioned learned factors and residuals; the confidence value is retained.

### Excess export and grid controls

| Field | Default | Description |
|---|---|---|
| `excess_export_enabled` | `False` | Permit the MILP to schedule battery → grid export when full-horizon economics and constraints justify it; it does not force export by itself |
| `excess_export_discharge_buffer_pct` | `10.0` | Conditional safety SoC buffer retained through the demand window following an intentional battery-export slot; one contiguous PV-surplus run shares one checkpoint |
| `excess_export_price_threshold` | Auto-calculated | Legacy/diagnostic depreciation threshold computed by `calculate_recommended_threshold()`. It is not a hard MILP export trigger; cycle wear and actual prices are already part of the objective. |
| `export_min_price` | `0.0` | Below this export price the inverter throttles export to zero |
| `battery_export_min_price` | `0.0` | Per-slot hard floor for intentional battery-to-grid export (issue #752). When > 0 and a slot's raw `export_price` is strictly below this value, the MILP fixes `bx[t] = 0`, so `primary_battery_export_kwh` is zero while normal local discharge and `pv_export_kwh` remain available. `force_batteries_discharge` is never labelled there. Reaching the threshold does NOT auto-trigger export; the optimizer still decides. Set to 0 to disable. |

### Seasonal configuration

| Field | Default | Description |
|---|---|---|
| `months_winter` | `[1,2,3,4,10,11,12]` | Months classified as winter. All 12 months may be winter (TOU year-round, issue #725); the summer set is then empty. |
| `house_power_includes_ev` | `True` | Whether the house consumption sensor already includes EV charger power |

### Main fuse / tariff protection

| Field | Default | Description |
|---|---|---|
| `main_fuse_amps` | `0` (disabled) | Main fuse/breaker rating in amps (e.g., 25, 35). When set, the MILP optimizer respects this limit as a soft constraint on total grid import power per slot. Set to 0 to disable. |

The MILP uses a **soft** (penalty-based) constraint so the solver never becomes
infeasible — if house base load alone exceeds the fuse rating, the plan is still
returned with the violation flagged.  The formula for converting amps to kWh/slot is:

```text
max_grid_import_per_slot_kwh = main_fuse_amps × 230 V × 3 phases / 1000 × (interval_minutes / 60)
```

This assumes balanced three-phase load at 230 V phase-to-neutral.  When the
constraint is active, the MILP will throttle battery and EV charging to stay
within the fuse limit whenever possible.

### Grid export power cap (issue #726)

| Field | Default | Description |
|---|---|---|
| `max_grid_export_power_kw` | `0` (disabled) | Maximum grid export power in kW — the DNO/inverter export cap for export-limited connections. When set, the MILP hard-bounds per-slot grid export to `max_grid_export_power_kw × slot_hours`. Set to 0 to disable. |

Unlike the fuse, the export cap is a **hard** bound (`ge[t] ≤ cap × slot_hours`)
because it is physically enforced by the inverter/DNO — exceeding it is never
required for feasibility. Battery export and PV export obey
`ge[t] = discharge_efficiency × bx[t] + pv_export[t]` and compete for the
same cap, so the optimal plan front-loads battery
export into low-PV slots and tapers it as PV ramps; PV that cannot be exported
at the cap is curtailed.  Without the cap the planner overstates export revenue
and can schedule forced battery discharge that displaces PV export.

The source split is not chosen by that economic competition:
`bx[t] = min(ed[t], ge[t]/discharge_efficiency)` is enforced by a binary
source branch. `pv_export[t]` is capped by available forecast PV plus only PV
revealed when PowMr SBU removes a dedicated load already present in the site
measurement. Active flexible EV demand is excluded from battery-eligible local
sinks, so PV serves it before residual PV can be classified as export.
An SBU slot may therefore both avoid residual import and reveal PV export. The
PowMr discharge still pays its full terminal-inventory value, conversion loss,
and wear, so Utility wins when the incremental meter value does not cover those
costs.

The corresponding slot diagnostics are
`primary_battery_export_kwh = discharge_efficiency × bx[t]` and
`pv_export_kwh = pv_export[t]`. They are non-negative; the raw solution sums
within solver tolerance and the public 0.001 kWh fields are reconciled exactly.

### EV planned load — primary EV

All fields are prefixed `ev_planned_load_`.

| Field | Default | Description |
|---|---|---|
| `ev_planned_load_enabled` | `False` | Enable EV planned load integration for the primary EV |
| `ev_planned_load_connected` | `False` | Whether a vehicle is currently plugged in |
| `ev_planned_load_smart_charging_enabled` | `True` | Whether smart EV charging scheduling is permitted |
| `ev_planned_load_current_soc_pct` | `0.0` | Current EV battery SoC (%) |
| `ev_planned_load_target_soc_pct` | `80.0` | Target SoC the EV must reach by the deadline (%) |
| `ev_planned_load_battery_capacity_kwh` | `0.0` | EV battery nameplate capacity (kWh) |
| `ev_planned_load_charger_power_kw` | `0.0` | Charger AC output power (kW) |
| `ev_planned_load_charger_efficiency_pct` | `100.0` | Charger efficiency (%) — energy delivered to EV / AC draw |
| `ev_planned_load_deadline` | `None` | Timezone-aware datetime by which charging must be complete |
| `ev_planned_load_base_load_includes_ev` | Auto (derived) | Automatically derived from the `hsem_house_power_includes_ev_charger_power` setting in the EV charger config step. When that is `True`, this is `True` (EV load already in the house consumption data). |

### EV planned load — second EV

All fields are prefixed `ev_second_planned_load_`. The schema is identical to the
primary EV fields above:

| Field | Default | Description |
|---|---|---|
| `ev_second_planned_load_enabled` | `False` | Enable EV planned load integration for the second EV |
| `ev_second_planned_load_connected` | `False` | Whether a second vehicle is currently plugged in |
| `ev_second_planned_load_smart_charging_enabled` | `True` | Smart charging permission |
| `ev_second_planned_load_current_soc_pct` | `0.0` | Current second EV battery SoC (%) |
| `ev_second_planned_load_target_soc_pct` | `80.0` | Target SoC (%) |
| `ev_second_planned_load_battery_capacity_kwh` | `0.0` | Second EV battery nameplate capacity (kWh) |
| `ev_second_planned_load_charger_power_kw` | `0.0` | Charger AC output power (kW) |
| `ev_second_planned_load_charger_efficiency_pct` | `100.0` | Charger efficiency (%) |
| `ev_second_planned_load_deadline` | `None` | Timezone-aware charging deadline |
| `ev_second_planned_load_base_load_includes_ev` | Auto (derived) | Automatically derived from the global `hsem_house_power_includes_ev_charger_power` setting — same value as the primary EV. |

---

## Planning outputs

All outputs are collected in the `PlannerOutput` dataclass
(`custom_components/hsem/models/planner_outputs.py`).

### Per-slot decisions (`slots`)

Each `PlannedSlot` in the output list covers one time interval and carries:

| Field | Unit | Description |
|---|---|---|
| `start` / `end` | datetime | Slot boundaries (timezone-aware) |
| `price.import_price` | currency/kWh | Import price for this slot |
| `price.export_price` | currency/kWh | Export price for this slot |
| `solcast_pv_estimate` | kWh | Forecast PV production |
| `avg_house_consumption` | kWh | Predicted house load (weighted average) |
| `estimated_net_consumption` | kWh | `avg_house_consumption + ev_planned_load_kwh − solcast_pv_estimate` (negative = PV surplus) |
| `batteries_charged` | kWh | Energy scheduled to be stored (after losses) |
| `batteries_discharged` | kWh | Energy drawn from battery |
| `grid_import_kwh` | kWh | Grid import this slot |
| `grid_export_kwh` | kWh | Grid export this slot |
| `estimated_battery_soc` | % | Estimated SoC at end of slot |
| `estimated_battery_capacity` | kWh | Usable remaining capacity at end of slot |
| `ev_planned_load_kwh` | kWh | **Extra** EV AC load added to net consumption (zero when `base_load_includes_ev = True`) |
| `ev_accounted_load_kwh` | kWh | EV AC load already included in the house consumption sensor (non-zero when `base_load_includes_ev = True`) |
| `ev_total_planned_load_kwh` | kWh | Total planned EV AC load: `ev_planned_load_kwh + ev_accounted_load_kwh`. Non-zero whenever EV charging is planned, regardless of `base_load_includes_ev` |
| `estimated_cost` | currency | Net grid cost this slot (positive = import, negative = export) |
| `recommendation` | string | The action chosen for this slot (see below) |
| `primary_battery_hold` | boolean | `true` when an idle MILP slot explicitly holds the Huawei battery while preserving its solved grid/PV flow |

#### Recommendation values

| Value | Meaning |
|---|---|
| `batteries_charge_grid` | Charge battery from grid as selected by the dynamic MILP |
| `batteries_charge_solar` | Battery is charging from PV surplus |
| `batteries_discharge_mode` | Battery discharges to cover house load during high-price window |
| `force_batteries_discharge` | Forced discharge (excess export to grid) |
| `force_export` | Negative import price — all available energy exported to earn money |
| `ev_smart_charging` | EV charging load is allocated to this slot (planner or runtime resolver) |
| `batteries_wait_mode` | Battery idle by default; when **Wait mode behaviour** is set to *Self-consumption with reserve*, normal household self-consumption is allowed using energy above the planner's required reserve |
| `time_passed` | Slot is in the past — no recommendation applied |
| `missing_input_entities` | Required HA entities were unavailable when this slot was scheduled |

---

#### How recommendations are assigned — priority layers

Recommendations are set in three consecutive layers. Each later layer can
override an earlier one only within defined priority rules.

---

##### Layer 1 — Planner engine (optimisation and slot population)

The dynamic MILP is the sole authority for actively optimised primary-battery
actions. It jointly chooses charge, local discharge, hold, and optional export
across the actionable horizon from prices, forecasts, losses, wear, SoC, fuse,
phase, and inverter limits. No user-defined daily windows pre-author battery
actions. Recommendation labels are derived from the accepted solution's energy
flows; a failed solve uses the passive fallback instead of restoring a
heuristic active-battery plan.

**Intentional battery export** — MILP only:

The retired `apply_excess_export` scheduling pass does not author executable
labels. With `excess_export_enabled = True`, the MILP may allocate explicit
battery-origin export when actual horizon economics and all constraints permit
it. Internally `bx[t]` is battery-side DC export; the output exposes
`primary_battery_export_kwh = discharge_efficiency × bx[t]`. A slot is
`force_batteries_discharge` only when that field exceeds solver tolerance.
Aggregate grid export alone may be PV and is not evidence of battery export.
The binary `export_source_mode[t]` enforces the causal split
`bx[t] = min(ed[t], grid_export_kwh[t]/discharge_efficiency)`; any concurrent
battery discharge and grid export is attributed to the battery before a
non-battery/PV remainder. The separate binary `primary_action_mode[t]`
enforces exact charge-or-discharge eligibility, so a slot cannot contain both
primary actions. `grid_flow_mode[t]` independently selects import or export
with finite physical per-slot bounds; both meter directions may be idle, but
they cannot be positive together.

The conditional export buffer uses one reserve checkpoint for every slot in a
contiguous PV-surplus run. That checkpoint is after the run's following demand
window—immediately before the next distinct PV-surplus run, or horizon end for
the final run. Planned PV or cheap grid charging before the checkpoint may
restore the reserve. If the full SoC trajectory cannot retain it, the planner
may suppress the run's battery export instead of merely moving it to the
highest-price slot. This grouping changes only checkpoint preprocessing; it
does not add solver variables or constraints or change ordinary
self-consumption, direct PV export, export caps/floors, or PowMr control.

**Seasonal optimisation fill** (`apply_optimization_strategy`) — for all remaining `None` slots:

This heuristic runs only for non-MILP candidates. A validated MILP result is
already a complete energy allocation; its idle slots receive a label-only
`batteries_wait_mode` plus `primary_battery_hold = true`, without changing
battery charge/discharge or grid import/export.

| Priority | Condition | Recommendation |
|---|---|---|
| 1 | Export price > import price AND export price ≥ `export_min_price` | `force_export` |
| 2 | Actual PV surplus (`estimated_net_consumption_kwh < 0`) and battery not full | `batteries_charge_solar` |
| 3 | Future `force_batteries_discharge` slot exists AND battery > required | `batteries_wait_mode` |
| 4 | Winter month | `batteries_wait_mode` |
| 5 | Summer month, actual PV surplus | `batteries_charge_solar` |
| 5 | Summer month, no PV surplus (zero or positive net consumption) | `batteries_discharge_mode` |

> **Note:** `BatteriesChargeSolar` is only assigned when there is a genuine PV
> surplus (negative net consumption).  A small positive house load with zero PV
> must not be mislabeled as solar charging — that would cause the applier to
> write `MaximizeSelfConsumption` instead of `TimeOfUse` + charge TOU
> (issue #720).

> **Wait mode behaviour:** the `batteries_wait_mode` recommendation normally keeps the
> battery idle.  When `hsem_batteries_wait_mode_behavior` is set to
> `self_consumption_with_reserve`, the applier switches the inverter to
> `MaximizeSelfConsumption` and caps discharge power so only surplus energy above
> the planner's required reserve is used.  This reduces unnecessary grid import
> while still preserving capacity for future energy commitments in the accepted plan.
> An explicit MILP `primary_battery_hold` takes precedence over that fallback:
> it uses Time of Use, a 0 W discharge cap, and exports incidental surplus PV.

**Discharge concentration** (`concentrate_discharge_on_expensive_slots`) runs after the
seasonal fill but before candidate generation. It re-evaluates all discharge-mode
slots and clears the cheapest ones that exceed the battery's capacity, turning them
into `batteries_wait_mode` (grid-import) so the battery is reserved for the most
expensive slots.

Slots are grouped by **calendar day** and each day receives its own independent
battery budget (`usable_kwh`). This correctly accounts for solar recharging between
dynamically planned discharge groups on different days — day N+1's discharge slots
do not compete with day N's for the same capacity pool.

---

##### Layer 2 — EV planned load labelling (engine, post-simulation)

After the winning candidate is selected and the final SoC simulation is complete,
slots with **`ev_total_planned_load_kwh > 0`** are re-labelled.
`ev_total_planned_load_kwh` is used — not `ev_planned_load_kwh` — so that EV-scheduled
slots are correctly labelled even when `base_load_includes_ev = True`, where
`ev_planned_load_kwh` is `0.0` but EV charging is still planned.

| Current recommendation | Has EV load? | Result |
|---|---|---|
| `batteries_charge_solar` | Yes (`ev_total > 0`) | → `ev_smart_charging` |
| `batteries_wait_mode` | Yes (`ev_total > 0`) | → `ev_smart_charging` |
| `batteries_discharge_mode` | Yes (`ev_total > 0`) | → `ev_smart_charging` (EV label wins) |
| `batteries_charge_grid` | Yes | Kept — grid charge takes priority |
| `force_batteries_discharge` | Yes | Kept — forced export takes priority |
| `force_export` | Yes | Kept |
| `time_passed` | Yes | Kept |

---

##### Layer 3 — Runtime resolver (`resolve_current_recommendation`)

Applied to the **current slot only** at hardware-write time. Overrides the planner
output with live sensor readings that were unknown at planning time.

| Priority | Condition | Result |
|---|---|---|
| 1 (highest) | Current slot actionable and available live import price < 0 | → `force_export` |
| 2 | Current recommendation = `batteries_charge_grid` | Kept — grid charge never overridden |
| 3 | Any EV (primary or second) is actively charging right now | → `ev_smart_charging` |
| — | None of the above | Accepted planner recommendation kept unchanged |

> **Note:** Priorities 1 and 3 interact. A published, actionable negative
> import price always wins — even when an EV is charging. However, a grid-charge
> slot (priority 2) is never overridden by an actively charging EV (priority 3).
> Display relabelling and window hysteresis preserve an explicit
> `primary_battery_hold`; neither may introduce battery energy absent from the
> solved MILP allocation.

---

##### Summary: full priority stack (highest → lowest)

```
1. actionable available import_price < 0 → force_export    [runtime resolver]
2. batteries_charge_grid active   → batteries_charge_grid  [runtime resolver guard]
3. EV actively charging (live)    → ev_smart_charging      [runtime resolver]
   ──────────────────────────────────────────────────────── resolver boundary ──
4. accepted MILP flow labels      [dynamic battery optimisation]
5. ev_smart_charging              [EV load labelling]
6. passive solar/wait labels      [solver-failure fallback]
7. time_passed / missing_input_entities
```

### Derived charge and discharge windows (`charge_windows`, `discharge_windows`)

These output-only windows group consecutive dynamically planned slots with the same charge or discharge recommendation:

- `ChargeWindow` — `start`, `end`, `total_energy_kwh`, `avg_import_price`, `recommendation`
- `DischargeWindow` — `start`, `end`, `total_energy_kwh`, `avg_export_price`, `recommendation`

### Plan metadata

| Field | Description |
|---|---|
| `plan_cost` | Total estimated grid cost for the selected plan (local currency) |
| `missing_inputs` | List of diagnostic labels for absent input data |
| `warnings` | Human-readable warning messages about data quality or configuration |
| `data_quality` | Structured `DataQuality` report (see below) |
| `explanation` | `PlanExplanation` with strategy summary, score, and rejected alternatives |
| `time_series_index` | `TimeSeriesIndex` — shared slot grid used internally |
| `ev_charging_plan` | `EVChargingPlan` for the primary EV (`None` when disabled) |
| `ev_second_charging_plan` | `EVChargingPlan` for the second EV (`None` when disabled) |

### Plan explanation (`explanation`)

The `PlanExplanation` object is designed to be surfaced directly as a HA sensor attribute:

| Field | Description |
|---|---|
| `selected_strategy` | Short identifier (e.g. `"charge_grid_discharge_peak"`) |
| `summary` | One-sentence reason for the selected plan |
| `score` | Savings vs. doing nothing (positive = saves money) |
| `estimated_total_cost` | Net grid cost for the horizon |
| `price_spread` | Max − min import price (larger = more arbitrage potential) |
| `peak_import_price` / `off_peak_import_price` | Price extremes |
| `forecast_pv_kwh` | Total PV production for the horizon |
| `forecast_net_consumption_kwh` | Total load − PV (negative = net solar surplus) |
| `battery_soc_pct` / `battery_soc_at_end_pct` | Starting and ending SoC |
| `terminal_cost_to_go_source` / `terminal_cost_to_go_boundary` | Primary terminal model source and published-price boundary |
| `terminal_cost_to_go_tier_count` / `terminal_cost_to_go_total_quantity_kwh` | Number and total battery-side quantity of bounded post-boundary tiers |
| `terminal_cost_to_go_highest_value_per_kwh` / `terminal_cost_to_go_lowest_value_per_kwh` | Marginal-value range of the retained tiers |
| `terminal_cost_to_go_initial_valued_quantity_kwh` / `terminal_cost_to_go_final_valued_quantity_kwh` | Initial and selected-final inventory covered by tiers |
| `terminal_cost_to_go_initial_value` / `terminal_cost_to_go_final_value` | Piecewise inventory values used to derive the selector-only primary terminal term |
| `constraints` | Active flags (e.g. `"winter_month"`, `"excess_export_enabled"`) |
| `rejected_plans` | Alternatives with name, reason, and estimated cost |

---

## EV planned load integration

### Overview

When one or both EV planned load features are enabled, the planner allocates
EV charging demand into slots **before** the final net consumption is computed.
This ensures the home battery planner sees the true net demand and does not
misinterpret EV-consumed solar as available for battery charging.

The EV planner is a separate, pure-Python module (`planner/ev_planner.py`).
It runs once per planning cycle and writes three per-slot load fields to each
`PlannedSlot`.

### No circular dependency

EV plans are built from raw inputs only (EV SoC, target SoC, capacity, charger
power, deadline, and the per-slot net surplus). They are computed independently
of the home battery planner output. The one-pass design prevents circular dependency.

### Three-field EV load model

Three fields capture EV load intent precisely:

| Field | Meaning |
|---|---|
| `ev_planned_load_kwh` | Extra EV AC load **added to net consumption** — only the portion not already in `avg_house_consumption`. Zero when `base_load_includes_ev = True`. |
| `ev_accounted_load_kwh` | EV AC load **already included** in the house consumption sensor. Non-zero when `base_load_includes_ev = True`. |
| `ev_total_planned_load_kwh` | Total planned EV AC load: `ev_planned_load_kwh + ev_accounted_load_kwh`. Always non-zero when EV charging is planned. |

### Net load formula with EV

```text
effective_net_load_kwh
    = avg_house_consumption
    + ev_planned_load_kwh          ← extra load only (zero when base includes EV)
    − solcast_pv_estimate
```

Only `ev_planned_load_kwh` is added.  When `base_load_includes_ev = True`,
`ev_planned_load_kwh` is `0.0`; the EV load is already captured in
`avg_house_consumption` and must not be added a second time.

### Slot selection strategy

The EV planner selects slots in two passes, using **net surplus after house
consumption** as the priority signal:

1. **Net-surplus slots first** — slots where `−estimated_net_consumption > 0`
   (i.e. solar production exceeds house demand) are prioritised.  The energy
   is free for the EV because the house has already consumed its share of
   solar and the remainder would otherwise be exported.
2. **Cheapest grid-import slots next** — among remaining slots, the lowest
   import price comes first.

Allocation stops when the total energy needed to reach `target_soc_pct` is
satisfied, or the deadline is reached.

**Why net surplus, not raw PV?**  The house sits between the PV inverter and the
EV charger at the AC bus.  It always consumes solar first.  The EV charger only
sees what is left over after house demand is satisfied.  Using raw PV would
over-estimate the free energy available and schedule more EV load on
"solar" slots than is physically available.

### Engine execution order

The engine processes EV load in three steps:

1. **Base net consumption** — `populate_net_consumption(slots)` is called first,
   populating `estimated_net_consumption = house − pv` (without EV).  PV
   confidence decay (day+1 at 90 %, day+2 at 80 %) is applied before this
   step so the surplus signal is already conservatively adjusted.

2. **EV planning** — net surplus is derived from step 1:
   ```text
   slot_net_surplus = max(−estimated_net_consumption, 0.0)
   ```
   The EV planner selects slots using this signal and builds charging plans
   for both EVs.  Per-slot loads are accumulated additively (primary + second).

3. **Final net consumption** — `populate_net_consumption(slots)` runs a second
   time to incorporate `ev_planned_load_kwh` into the final
   `estimated_net_consumption` values.

### Partial current-slot scaling

The currently active slot is scaled by its remaining duration to avoid
over-counting energy in the partially elapsed slot:

```text
slot_remaining_hours = remaining_minutes_in_slot(now, slot_end) / 60.0
max_charge_this_slot = charger_power_kw × slot_remaining_hours × (efficiency / 100)
```

### Double-count prevention (base_load_includes_ev)

When the house consumption sensor already includes EV charger power
(e.g. the CT clamp is upstream of the EVSE), set `base_load_includes_ev = True`
for that EV.

- `ev_planned_load_kwh` is **not** added to net consumption for that EV.
- The load is captured in `ev_accounted_load_kwh` instead.
- `ev_total_planned_load_kwh` is still set and non-zero, so diagnostics,
  logs, and the `ev_smart_charging` label all reflect the planned EV activity.

This prevents double-counting while keeping full observability.

### EV plan states

| State | Meaning |
|---|---|
| `not_connected` | No vehicle plugged in |
| `smart_charging_disabled` | Feature disabled or smart charging turned off |
| `fully_charged` | EV has already reached target SoC — no load allocated |
| `charging` | Slots have been allocated; charging is active or planned |
| `waiting` | EV is connected but no candidate slots exist before the deadline |
| `unavailable` | Required config values (capacity or charger power) are zero/missing |

### HA sensor entities

Two sensor entities expose the EV charging plan as attributes:

| Entity | Purpose |
|---|---|
| `sensor.hsem_ev_optimal_charging_plan` | Primary EV plan state and slot details |
| `sensor.hsem_ev_second_optimal_charging_plan` | Second EV plan state and slot details |

Both sensors share the same attribute schema:

```json
{
  "battery_capacity_kwh": 60.0,
  "charge_power_kw": 11.0,
  "current_soc": 32.0,
  "target_soc": 80.0,
  "ev_connected": true,
  "total_kwh_needed": 28.8,
  "deadline": "2026-05-15T07:00:00+02:00",
  "charging_slots": [
    {
      "start": "2026-05-14T10:00:00+02:00",
      "end":   "2026-05-14T11:00:00+02:00",
      "estimated_charged_kwh": 8.5,
      "solar_surplus_kwh": 9.2,
      "import_needed_kwh": 0.0,
      "import_price": 1.25,
      "estimated_cost": 0.0
    }
  ],
  "planned_load_by_slot": {
    "2026-05-14T10:00:00+02:00": 8.5
  },
  "current_slot_planned_load_kwh": 8.5,
  "data_quality": {}
}
```

### Net surplus model (and historical notes)

**Current approach (PR #406):** The engine runs `populate_net_consumption` once
before EV planning to derive the per-slot net surplus:

```text
slot_net_surplus = max(−estimated_net_consumption, 0.0)
                 = max(pv_estimate − avg_house_consumption, 0.0)
```

This is the correct physical model: the house uses solar first; only what
remains is available to the EV charger at no extra grid cost.
Using `estimated_net_consumption` as the starting point also ensures that
PV confidence decay (day+1 at 90 %, day+2 at 80 %) is automatically applied
before EV slot selection.

After EV injection, `populate_net_consumption` runs a second time to produce
final slot values that incorporate `ev_planned_load_kwh`.

**Historical note (PR #397 fix):** Before PR #397 the surplus was computed from
`slot.estimated_net_consumption` which was `0.0` at EV planning time (net
consumption had not been populated yet). Every slot appeared to have zero surplus,
so the EV was always scheduled as grid-import.

The PR #397 workaround derived surplus directly from raw base fields:
```text
surplus = max(slot.solcast_pv_estimate − slot.avg_house_consumption, 0.0)
```
This was correct but did not yet apply PV confidence decay.  PR #406 replaced it
with the pre-populated `estimated_net_consumption` approach, which is both
conceptually cleaner and more accurate.

---

### Session-aware EV demand

When an EV is **actively charging** (session in progress, current draw
detected), the next 2 hours are treated as **certain demand** in the MILP.
The number of slots covered is derived from the configured slot interval
(8 slots at 15-minute, 4 at 30-minute, 2 at 60-minute).  The live charger
power is used as a fixed lower bound on EV load for those slots, preventing
the MILP from re-allocating demand away from a charging session that is
already underway:

```text
For slots t in [now, now + 2h]:
    ev_c_lower_bound[t] = min(session_charge_kw × slot_hours,
                               ev_max_charge_per_slot)
```

This keeps the MILP's plan consistent with the physical state of the EV
charger and avoids oscillation between charging and idle states within a
single session.  Slots beyond the 2-hour window are optimised freely by
the MILP.

---

## Cost function

The cost function returns two aggregates. `total_cost` contains auditable
money terms only. `score` adds selector-only penalties, terminal inventory
value, and the primary-action structural tiebreak. **Lower score is
better**.

### Formula

```text
total_cost
  = grid_import_cost
  − export_revenue
  + conversion_loss_cost
  + cycle_cost

score
  = total_cost
  + soc_penalty
  + grid_limit_penalty
  + terminal_soc_value
  + primary_action_tiebreak
```

### Grid import cost

```text
grid_import_cost = Σ (grid_import_kwh[slot] × import_price[slot])
```

The cost function prices actual grid energy drawn, not stored energy.
If the battery stores `x` kWh and charge efficiency is `e`, the grid
import is `x / e`. This means conversion losses are implicitly included
in the import cost before the explicit conversion-loss term.

### Export revenue

```text
export_revenue = Σ (grid_export_kwh[slot] × export_price[slot])
```

Revenue is subtracted from total cost (it reduces the net expense).

For a solved MILP slot:

```text
primary_battery_export_kwh + pv_export_kwh = grid_export_kwh
```

Both source fields are explicit and non-negative. The raw solver values sum
within solver tolerance and the public 0.001 kWh fields sum exactly. They are
used for diagnostics and destination-aware scoring;
source is not inferred from net export or forecast PV.
The raw source split also satisfies:

```text
battery_export_dc =
    min(batteries_discharged_kwh, grid_export_kwh / discharge_efficiency)
```

The `export_source_mode` binary enforces this equality. The
`primary_action_mode` binary independently guarantees that primary charge
and discharge are not both positive. The `grid_flow_mode` binary guarantees
the same for grid import and export:

```text
grid_import_kwh <= M_import * grid_flow_mode
grid_export_kwh <= M_export * (1 - grid_flow_mode)
```

`M_import` and `M_export` are finite physical per-slot bounds derived from
reachable fixed/flexible load and charge, and from PV plus delivered battery
discharge respectively.
Diagnostics expose the exact integral block names/count and
`grid_import_export_overlap_max_kwh`; the latter is the maximum raw
`min(grid_import, grid_export)` and must remain within solver tolerance.

**Export price clamping:** When ``export_min_price > 0``, the applier
blocks all grid export for slots where ``export_price < export_min_price``
by setting the inverter to ``GRID_EXPORT_LIMIT_WATT``.  To keep the
planner consistent with this physical behaviour, both the MILP and the
cost function treat ``export_price`` as 0 for any slot where
``export_price < export_min_price`` — no revenue is counted for exports
that can never happen.  See *Excess export and grid controls* for the
configuration fields.

### Conversion loss cost

Charge-side loss is priced at the sanitised import price. Discharge-side loss
uses the explicit destination split:

```text
charge_loss_cost =
    batteries_charged_kwh × (1 − charge_eff) × max(import_price, 0)
battery_export_dc = primary_battery_export_kwh / discharge_eff
local_discharge_dc = batteries_discharged_kwh − battery_export_dc
discharge_loss_cost =
    local_discharge_dc × (1 − discharge_eff) × max(import_price, 0)
    + battery_export_dc × (1 − discharge_eff) × max(export_price, 0)
```

### Battery cycle cost

Battery depreciation per kWh cycled through the physical cells:

```text
throughput_kwh[slot] = max(batteries_charged[slot], batteries_discharged[slot])
cycle_cost = Σ (throughput_kwh[slot] × cycle_cost_per_kwh)
```

Auto-derived cycle cost (when not explicitly configured):

```text
cycle_cost_per_kwh = purchase_price / (rated_capacity_kwh × expected_cycles)
```

The price threshold used by the profitability guard adds round-trip conversion
loss on top of depreciation:

```text
price_threshold = cycle_cost_per_kwh + conversion_loss
conversion_loss = 1 / (charge_eff × discharge_eff) − 1
```

**Depreciation example:** A 10 kWh battery bought for 30 000 DKK with 6 000 expected
cycles costs `30000 / (10 × 6000) = 0.50 DKK/kWh` of throughput.
**With 98 % efficiency:** conversion loss adds ~0.042 DKK/kWh, giving a combined
threshold of ~0.542 DKK/kWh.

### SoC penalties

Quadratic guard penalties discourage plans that violate SoC bounds:

```text
# Below the floor
if estimated_battery_soc[slot] < min_soc_pct:
    violation = min_soc_pct − estimated_battery_soc[slot]
    soc_penalty += soc_low_penalty_weight × violation²

# Above the ceiling
if estimated_battery_soc[slot] > max_soc_pct:
    violation = estimated_battery_soc[slot] − max_soc_pct
    soc_penalty += soc_high_penalty_weight × violation²
```

These penalties are a soft guard — the SoC simulation already hard-clamps SoC
at the hardware limits, so violations are rare in practice.

### Grid limit penalty

When a grid power limit is configured, slots that exceed it incur a proportional penalty:

```text
slot_power_kw = grid_import_or_export_kwh / slot_duration_hours
if slot_power_kw > grid_limit_kw:
    excess_kwh = (slot_power_kw − grid_limit_kw) × slot_duration_hours
    grid_limit_penalty += excess_kwh × grid_limit_penalty_per_kwh
```

### Terminal inventory accounting

Plans can look artificially cheap when they use inventory needed just beyond
the published-price boundary. Primary storage therefore uses a bounded
post-boundary cost-to-go; secondary storage retains its uniform terminal term:

```text
primary_terminal_soc_value =
    V_primary(initial_battery_kwh)
    − V_primary(final_battery_kwh)

secondary_terminal_soc_value =
    secondary_storage_replacement_price_per_kwh
    × (
        Σ secondary_storage_discharged_kwh
        − Σ secondary_storage_charged_kwh
      )

terminal_soc_value =
    primary_terminal_soc_value + secondary_terminal_soc_value
```

`V_primary(E)` is piecewise linear. Each tier represents only the battery-side
energy that could serve one exactly aligned, non-actionable house-load slot
strictly beyond the contiguous published-price prefix. Tier quantity is capped
by residual house load after PV and accounted EV load, discharge efficiency,
per-slot discharge power, and usable battery capacity. Its Unagi price is
reduced by MAE plus operator margin; conversion loss and cycle wear are then
removed from its marginal value. Duplicate points use the lower prediction,
and invalid inputs fail closed.

Inventory above the combined tier quantity has no synthetic value. When no
valid tier exists, diagnostics report `hardware_floor_only`: primary terminal
value is zero and the existing effective hardware discharge floor is the only
reserve. As official prices arrive, the boundary rolls forward; newly
published demand leaves the tiers and is evaluated at its real price instead
of moving a synthetic hold forward.

Both primary and secondary accounting remain path-independent. Equal discharge
and recharge restore the same final inventory and cancel exactly, so real slot
prices, efficiencies, wear, headroom, and power limits decide whether a cycle
is economic. `resolve_secondary_terminal_price()` remains a separate,
uniform mean-of-window valuation for the dedicated PowMr load.

Unagi remains valuation-only: it never becomes a slot import/export price,
extends price actionability, authorises a storage action, or creates realised
within-window savings.

**Primary-action structural tiebreak (issues #638/#655).** A tiny weighted
selector-only term resolves true economic ties without subsidising
charge/discharge or export/refill cycles:

```text
epsilon = 0.00001 currency / DC kWh
battery_export_dc[t] = primary_battery_export_kwh[t] / discharge_efficiency
local_discharge_dc[t] =
    batteries_discharged_kwh[t] - battery_export_dc[t]
primary_action_tiebreak =
    epsilon × Σ(
        batteries_charged_kwh[t] + batteries_discharged_kwh[t]
    )
    − 1.5 × epsilon × Σ local_discharge_dc[t]
```

Charge and battery-origin export each add `epsilon` per DC kWh; local
discharge subtracts `0.5*epsilon`. Charge/local-discharge and export/refill
cycles therefore remain disfavoured. On a true lossless/economic tie, local
discharge wins. A 97%-efficient discharge at a flat tariff is not an economic
tie and is deliberately no longer forced.

This is a structural weighted tiebreak, not a mathematically lexicographic
objective. It can affect economics below epsilon, and the 0.5% MIP gap does not
promise proof of an epsilon-sized distinction. Same-slot high-price PV export
(#694) and later cheap-surplus refill (#592) remain explicit solver choices:
charging reduces `pv_export_kwh` in its actual slot and a later refill uses
real later PV/headroom.

---

## Candidate generation and selection

The planner evaluates multiple independent strategies before committing to a plan.

### Candidate strategies

| Name | Description |
|---|---|
| `no_action` | Fully simulated and scored diagnostic comparator; never eligible to win |
| `passive` | Solar absorption and normal self-consumption, with no optimized grid charge or intentional battery export; sole executable solver-failure fallback |
| `milp` | Added only after an optimal solve or a fully validated time-limit incumbent; sole candidate that may introduce optimized grid charge or battery export |

The retired baseline/grid-charge/solar-only/discharge/aggressive heuristic
matrix is not generated. Every candidate is independently simulated and scored.
MILP-populated energy and export-source fields are preserved rather than
re-derived from recommendation labels.

### Selection

After scoring, the selector picks the valid, eligible executable candidate with
the lowest `score`. `no_action` remains diagnostic even if its score is
lower.

The invariant **must always hold**:

```text
output.plan_cost == selected_candidate.cost
output.slots == selected_candidate.slots
```

No post-selection mutation is permitted. If a candidate needs adjusting, it must
be re-simulated and re-scored before it can become the output.

---

## Safety modes

HSEM uses a layered safety system to prevent hardware writes when inputs are
unsafe or the system is in a degraded state.

### Degraded mode levels

| Mode | Hardware writes | Trigger |
|---|---|---|
| `Normal` | Allowed | All required live control/telemetry inputs present |
| `Degraded` | Allowed (with warnings) | A non-critical live entity is missing |
| `Error` | **Blocked** | Critical data missing (battery SoC, house load, working mode) |
| `ReadOnly` | **Blocked** | `is_read_only = True` in config or `PlannerInput` |
| `DryRun` | **Blocked** | Dry-run mode active |

### Critical vs. non-critical missing data

Critical keywords in `missing_inputs` block hardware writes:

- `battery` — battery SoC or capacity unavailable
- `house_consumption` — house load sensor unavailable
- `working_mode` — inverter working-mode select unavailable

Forecast price/PV gaps are not live missing-entity labels and do not by
themselves change degraded mode. They are surfaced in `data_quality` and by
per-slot availability. Missing prices close the contiguous actionable prefix;
automatic storage is held beyond it rather than optimized against placeholder
zeros.

### Safety gate behaviour

The write-verify applier (`WriteVerifyApplier`) enforces these gates
before any Huawei Solar service call:

1. Checks `is_read_only` — skip writes if `True`.
2. Checks degraded mode — skip writes in `Error` mode.
3. Verifies the inverter is not unloading.
4. After writing, reads back the entity state to confirm the change applied.

---

## Data quality diagnostics

The `DataQuality` object on `PlannerOutput` reports completeness of the planning inputs.

### Fields

| Field | Type | Description |
|---|---|---|
| `today_price_missing_hours` | `list[int]` | Hours (0–23) with no price data today |
| `today_pv_missing_hours` | `list[int]` | Hours (0–23) with no PV forecast today |
| `tomorrow_price_missing_hours` | `list[int]` | Hours with no price data for tomorrow |
| `tomorrow_pv_missing_hours` | `list[int]` | Hours with no PV forecast for tomorrow |
| `day2_price_missing_hours` | `list[int]` | Hours with no price data for day +2 (72-h horizon only) |
| `day2_pv_missing_hours` | `list[int]` | Hours with no PV forecast for day +2 |
| `horizon_has_tomorrow` | `bool` | `True` when horizon extends beyond 24 h |
| `horizon_days` | `int` | Distinct local calendar dates covered; normally 1/2/3 for 24/48/72 h, or one extra across spring-forward |
| `is_complete` | `bool` | `True` when no missing data was detected |

### Home Assistant attribute serialisation

`data_quality.as_dict()` returns a JSON-safe dictionary that can be attached
directly to a sensor's `extra_state_attributes`:

```json
{
  "is_complete": true,
  "horizon_has_tomorrow": true,
  "horizon_days": 2,
  "tomorrow_price_missing_hours": [],
  "tomorrow_pv_missing_hours": [],
  "day2_price_missing_hours": [],
  "day2_pv_missing_hours": [],
  "today_price_missing_hours": [],
  "today_pv_missing_hours": []
}
```

## Diagnostic accuracy and daily accounting

These trackers are diagnostic and do not alter the selected plan, but their
time and energy accounting follows the same physical-slot rules as the planner.

### Frozen forecast accuracy

For each future slot HSEM retains the last raw PV, corrected PV, load, predicted
end SoC, and action observed before the slot starts. The baseline freezes at
the UTC slot boundary. The previous finite PV/load power sample is integrated
over its half-open interval and split by physical slot overlap. A missing
endpoint advances sample state but invalidates that interval; a gap longer than
twice the effective coordinator cadence is also rejected. Only a frozen slot
with complete trusted actual coverage contributes to accuracy or correction
learning.

### Daily plan versus actual

Cumulative import, export, and PV meter deltas are assumed uniform between
their two sample instants. Each delta is split on the UTC timeline at planner
price-slot boundaries and Home Assistant-local midnight, including DST
transitions. Energy is counted even when a price is unavailable; money is
added only under an authoritative finite price. An uncovered leading segment
uses its prior known price rather than applying the newly sampled price to the
whole interval.

Actual meters are sampled before day rollover. A cross-midnight delta is
therefore apportioned to both local dates before the old date is saved. Meter
and SoC baselines survive rollover, and when a daily meter resets, its first
positive reading is retained as new-day energy since midnight. Planned slot
energy is captured once after the slot starts on the selected local date.
If the configured cadence changes while Home Assistant is running, physical
coverage already captured by the old layout remains authoritative and only the
uncovered fraction of an overlapping new slot is added.

### Savings

Savings consumes the daily tracker's explicit per-cycle, per-date measured
import-cost and export-revenue deltas; it does not re-difference a cumulative
daily total. Cheap-charge savings uses measured positive Huawei battery charge
power from the preceding valid interval, not the full planned charge energy on
every coordinator poll. The interval is rejected after missing telemetry or a
gap beyond twice the effective cadence, then split by overlap with actionable
charge slots and local dates. Each eligible share is valued against that local
day's published mean import price.
The plan snapshot that governed elapsed time masks a later overlapping replan;
the current plan can value only a physical tail that the preceding snapshot did
not cover.

Home Assistant's local date also anchors today, 7-day, and 30-day savings
rollups. Automatic mode records earned savings; non-automatic operation records
the same opportunity as missed savings.

---

## Scenario examples

All examples use the following base configuration:

- Battery: 10 kWh rated, 10 % end-of-discharge floor → 9 kWh usable
- Charge efficiency: 90 % (10 % conversion loss)
- Max charge power: 5 kW (5 kWh/h)
- Horizon: 24 h, 1-hour slots
- Prices and PV in local currency (DKK) and kWh

---

### Scenario 1: Winter price arbitrage

**Conditions:**
- Month: January (winter month)
- PV forecast: 0 kWh across all hours
- House load: approximately 2 kWh/h
- Import prices: 0.50 overnight, 1.50 midday, and 3.00 during 16:00–21:00
- Battery at start: 50 % SoC

**What the planner does:**

```
Cheapest overnight slots: batteries_charge_grid, limited by power and SoC
Mid-price slots:          batteries_wait_mode
16:00–21:00 peak:         batteries_discharge_mode to cover local load
After the peak:           batteries_wait_mode near the configured floor
```

**Why this plan wins:**

The full-horizon MILP sees that avoided peak imports exceed overnight energy,
conversion-loss, and cycle-wear costs. It dynamically chooses only the charge
needed before the expensive slots and discharges where the avoided import is
most valuable. If that spread disappears, it does not create the round trip.

**Explanation excerpt:**

```json
{
  "selected_strategy": "milp",
  "summary": "Charge from low-cost grid energy and discharge into the evening peak.",
  "constraints": ["winter_month", "grid_charge_price_spread_met"],
  "forecast_pv_kwh": 0.0
}
```

---

### Scenario 2: Summer day — high PV surplus

**Conditions:**
- Month: July (summer month)
- PV forecast: 0→2→6→8→6→4→1→0 kWh (ramps from 06:00 to 14:00, falls off by 19:00)
- House load: 0.5 kWh/h (typical summer light load)
- Import prices: moderate, 2.00 DKK/kWh peak (09–11), 0.80 DKK/kWh off-peak
- Battery at start: 20 % SoC (1.8 kWh above floor)
- Excess export disabled

**What the planner does:**

```
Hours 00–06:  batteries_wait_mode  (night, no PV, load from grid)
Hours 06–09:  batteries_charge_solar
              → PV arrives, surplus charges battery
              → net_consumption = 0.5 kWh − PV (surplus) → battery fills
Hours 09–14:  batteries_charge_solar / batteries_wait_mode
              → PV covers load; surplus continues charging battery
              → battery reaches max_soc around 11:00
Hours 14–19:  batteries_discharge_mode (PV falling, prices still moderate)
              → battery discharges to cover load, reduces grid import
Hours 19–24:  batteries_wait_mode (battery near floor, no PV)
```

**Why this plan wins:**

The planner identifies the large solar surplus and assigns `batteries_charge_solar`
slots in the morning. This avoids peak-price grid imports in the morning hours
and accumulates free solar energy. The battery then covers evening load when PV
has stopped. The `no_action` candidate wastes PV surplus by exporting it at the
low export price instead of storing it for later use.

**Explanation excerpt:**

```json
{
  "selected_strategy": "milp",
  "summary": "High PV day: solar surplus stored for evening discharge.",
  "constraints": ["summer_month"],
  "forecast_pv_kwh": 27.0,
  "forecast_net_consumption_kwh": -15.0,
  "battery_soc_at_end_pct": 12.0
}
```

---

### Scenario 3: Cheap night price — grid charge opportunity

**Conditions:**
- Month: March (winter month)
- PV forecast: small midday peak (2–3 kWh/h, 10:00–14:00)
- House load: ~1.5 kWh/h
- Import prices:
  - 00:00–06:00: 0.25 DKK/kWh (very cheap night tariff)
  - 06:00–09:00: 2.50 DKK/kWh
  - 09:00–16:00: 1.80 DKK/kWh
  - 16:00–21:00: 3.20 DKK/kWh (peak)
  - 21:00–24:00: 1.20 DKK/kWh
- Export price: 0.10 DKK/kWh (low, net-metering not attractive)
- Battery at start: 15 % SoC (0.45 kWh above floor)

**What the planner does:**

```
Hours 00–06:  batteries_charge_grid
              → cheap night rate: 0.25 DKK/kWh import
              → charge 5 kWh/h × 5h = 25 kWh capacity requested,
                capped at usable range → battery fills to max_soc (90 %)
Hours 06–10:  batteries_wait_mode (prices rise, battery full)
Hours 10–14:  batteries_charge_solar (PV surplus topping up)
Hours 14–22:  batteries_discharge_mode
              → discharges during expensive slots (1.80–3.20 DKK/kWh)
              → avoids 8h × 1.5 kWh = 12 kWh at avg 2.5 DKK/kWh = 30 DKK import
              → charge cost: ≈ 9 kWh × 0.25 DKK + cycle cost ≈ 2.25 + 4.50 = 6.75 DKK
              → net saving ≈ 23 DKK
Hours 22–24:  batteries_wait_mode
```

**Why this plan wins:**

The price spread of 2.95 DKK/kWh (peak 3.20 − night 0.25) far exceeds
conversion and cycle wear. The MILP allocates charge only up to available
capacity and power, then discharges when the avoided import is worth more. The
passive fallback cannot introduce optimized grid charge; `no_action` remains
an auditable diagnostic comparator.

**Explanation excerpt:**

```json
{
  "selected_strategy": "milp",
  "summary": "Cheap night rate (0.25 DKK/kWh) enables grid pre-charge; discharges during peak (3.20 DKK/kWh).",
  "score": 23.25,
  "price_spread": 2.95,
  "constraints": ["winter_month", "grid_charge_price_spread_met"],
  "battery_soc_at_end_pct": 10.0
}
```

---

### Scenario 4: High PV day — export now, refill later

Suppose an actionable horizon has a high export-price window followed by a
larger, lower-value PV surplus, the battery has usable headroom, and excess
export is enabled.

The MILP evaluates the whole path at once:

- Charging during the high-price window reduces that slot's
  `pv_export_kwh`, so the foregone export is explicit (#694).
- Battery-origin export is `primary_battery_export_kwh`, not aggregate
  `grid_export_kwh`. Only a material battery-origin amount produces
  `force_batteries_discharge`.
- A later refill consumes real later PV and battery headroom. Equal discharge
  and recharge cancel in terminal inventory value, so export revenue, future
  foregone PV export, efficiencies, cycle wear, and power limits decide whether
  the cycle wins (#592).
- A price threshold alone does not trigger export. With excess export disabled,
  or below `battery_export_min_price`, `primary_battery_export_kwh` is zero
  while direct `pv_export_kwh` remains possible.

For every solved slot, diagnostics must satisfy:

```text
primary_battery_export_kwh / discharge_efficiency
    = min(batteries_discharged_kwh,
          grid_export_kwh / discharge_efficiency)
primary_battery_export_kwh + pv_export_kwh = grid_export_kwh
```

exactly at the public 0.001 kWh precision. This makes a simultaneous local
battery discharge and PV export distinguishable from an intentional
battery-to-grid discharge.

---

### Scenario 5: Flat price day — self-consume, do not arbitrage

**Conditions:**
- Month: April (winter/spring boundary, configured as winter)
- PV forecast: modest (1–2 kWh/h, 09:00–15:00)
- House load: 1.0 kWh/h
- Import price: 1.20 DKK/kWh flat all 24 hours
- Export price: 0.10 DKK/kWh flat
- Battery at start: 50 % SoC
- Excess export disabled

With a flat import price of 1.20 DKK/kWh there is no grid-charge arbitrage:
charging from grid and later discharging adds conversion loss and cycle wear.
The MILP therefore does not create a grid round-trip merely to move energy
between equal-price slots.

An already charged battery may still cover local house load when doing so is
economically tied with grid import. The `primary_action_tiebreak` makes that
true tie resolve toward local discharge while keeping a charge/local-discharge
cycle slightly worse. With real conversion loss, a flat tariff is not a true
tie and discharge is deliberately not forced. Because excess export is
disabled, `bx = 0`; the export branch cannot receive the local-discharge
tiebreak.
PV surplus may charge the battery when storing it for later local use beats
exporting at 0.10 DKK/kWh. The passive fallback has the same safety envelope,
while `no_action` remains diagnostic only.

---

### Scenario 6: EV charging — MILP co-optimisation

**Conditions:**
- EV plugged in at 08:00, target SoC 80 %, deadline 07:00 next morning
- EV battery: 60 kWh, current SoC 32 % → 28.8 kWh needed
- Charger: 11 kW AC (efficiency 100 %)
- House load: 0.5 kWh/h
- PV forecast: 0→2→8→10→8→4→1→0 kWh/h (peak midday)
- Import price: 0.80 DKK/kWh off-peak, 2.00 DKK/kWh peak (09–13, 17–21)
- `base_load_includes_ev = False` (CT clamp is downstream of EVSE)

**What the MILP does:**

The MILP co-optimises EV charging, house battery, and grid import/export
simultaneously across all future slots. For pre-deadline slots (`t ≤ D`),
each `ev_c[t]` receives a strong negative coefficient (benefit) equal to
`ev_penalty_cost = max(p_imp) × max(energy_needed, 1.0) × 10`, forcing the LP to
charge the EV.

The LP naturally prefers PV surplus (free) over grid import (costs
`p_imp[t]`), so it allocates EV charging to high-PV slots first:

```
Pre-deadline slots (08:00 → 07:00 next day):
  10:00–11:00: PV surplus = 8 − 0.5 = 7.5 kWh → EV charges 7.5 kWh (free)
  11:00–12:00: PV surplus = 10 − 0.5 = 9.5 kWh → EV charges 9.5 kWh (free)
  09:00–10:00: PV surplus = 2 − 0.5 = 1.5 kWh → EV charges 1.5 kWh (free)
  Remaining 10.3 kWh → cheapest import slot (00:00–01:00 at 0.80 DKK/kWh)

Post-deadline slots (after 07:00):
  ev_c[t] = 0 — hard constraint, no charging allowed
  (unless charge_past_target=True, then surplus-PV-only with tiny benefit)
```

**Cost comparison (EV charging cost only):**

| Strategy | EV cost (DKK) |
|---|---|
| MILP (solar-first + cheapest import) | 0.80 × 10.3 + 0 × 18.5 = 8.24 DKK |
| Dumb (charge immediately from grid at 2.00 DKK/kWh) | 2.00 × 28.8 = 57.60 DKK |

---

## Reading the plan explanation

The `PlanExplanation` object is exposed as a HA sensor attribute on the
`hsem_working_mode` sensor. In the Home Assistant developer tools (States) you
can inspect it directly:

```
Entity: sensor.hsem_working_mode
Attributes:
  explanation:
    selected_strategy: charge_grid_discharge_peak
    summary: "Pre-charge for evening discharge: 0.25 DKK night vs 3.20 DKK peak"
    score: 6.10
    estimated_total_cost: 6.75
    price_spread: 2.95
    peak_import_price: 3.20
    off_peak_import_price: 0.25
    forecast_pv_kwh: 4.5
    forecast_net_consumption_kwh: 16.5
    battery_soc_pct: 15.0
    battery_soc_at_end_pct: 10.0
    constraints: [winter_month, grid_charge_price_spread_met]
    rejected_plans:
      - name: no_action
        reason: "Diagnostic comparator has a higher selector score."
        estimated_cost: 30.00
```

### Understanding the two `score` fields

`PlanExplanation.score` is the legacy human-readable savings comparison:

```text
PlanExplanation.score = idle_cost - estimated_total_cost
```

Positive means the selected plan is cheaper than the idle comparator inside
the actionable window; negative means it costs more there.

`PlanCostBreakdown.score` is the candidate selector objective. Lower is
better:

```text
score = total_cost
      + synthetic penalties
      + terminal_soc_value
      + primary_action_tiebreak
```

`PlanCostBreakdown.total_cost` is the auditable within-horizon money outcome.
Its `score` may be above or below it because terminal inventory and the
structural tiebreak may have either sign. Compare candidate scores only with
other candidates from the same planner run; do not interpret their sign as
profit, loss, or savings versus `no_action`.

### Understanding charge-only explanations

The human-readable `PlanExplanation` describes actions that actually appear
in the selected slots. Grid charge without any scheduled discharge is labelled
`opportunistic_charge`, and its summary says that no discharge window is
scheduled. If it costs more than the idle comparator inside the current
actionable window, the explanation reports that within-window difference; it
does not claim that discharge savings will occur in a window containing no
discharge. Unagi terminal value is retained-inventory context, not realised
revenue.

### Understanding `constraints`

Common constraint tags and their meaning:

| Tag | Meaning |
|---|---|
| `winter_month` | Current month is in `months_winter`; winter scheduling strategy active |
| `summer_month` | Not in winter months; summer scheduling strategy active |
| `no_price_spread` | Max − min import price is near zero; no grid-charge arbitrage |
| `grid_charge_price_spread_met` | Price spread exceeds conversion-loss and cycle-wear costs |
| `excess_export_enabled` | Excess export feature is active in config |
| `export_price_above_threshold` | Export price exceeds `excess_export_price_threshold` |

---

## Known limitations

### Consumption prediction: legacy mode is averaged, not model-based

In legacy mode, the planner predicts house load from a weighted average of
1, 3, 7, and 14-day historical consumption per clock-hour.  This works well
for regular households but may under- or over-predict when:

- An EV charges on an irregular schedule.
- Seasonal load shifts (e.g. heating vs. cooling) haven't had time to appear
  in the lookback window.
- Spike days (e.g. a party) pull the average up permanently.

The IQR median-ratio outlier detection algorithm flags anomalous windows, and
a **peer-median clamp** (issue #592) bounds each window to at most
`max(3× the median of the other three windows, 0.15 kWh/h)` — so a single
stale window (e.g. a 14d average still holding pre-change EV-charging
nights) cannot dominate the blend when the other three windows agree.
Only the upward side is clamped: a genuine consumption drop flows through
immediately.  The ML mode (ridge regression with day-of-week, seasonality,
and outdoor temperature) addresses several of these limitations.  See
`docs/consumption-prediction.md`.

### Price authority may cover only a prefix of the horizon

In practice:

- Today's prices are firm (EDS publishes by ~13:00).
- Tomorrow's prices arrive around 13:00 CET and are typically available before the evening planning run.
- Day +2 prices (72-hour horizon) may be unavailable or estimated.

Missing price data is surfaced in `data_quality`. The numeric display fallback
remains `0.0`, but availability distinguishes it from a genuinely published
zero or negative price. The planner optimizes only the contiguous published
prefix and enforces a price-neutral primary Hold plus secondary Utility beyond
the first gap. Price and Solcast PV publication/withdrawal events wake a
debounced refresh; an event that arrives during that refresh guarantees one
coalesced follow-up cycle.

Optional Unagi valuation is a separate horizon-end channel. After MAE and
operator-margin haircut, exactly aligned post-boundary demand becomes bounded
primary terminal tiers; published overlap is ignored. It never populates slot
prices, extends the actionable prefix, changes a physical bound, or authorises
battery export. An empty tier set falls back to
`hardware_floor_only`. Beyond the boundary primary charge/discharge and
`primary_battery_export_kwh` remain zero; natural PV flow may still appear as
`pv_export_kwh`.

### No intra-day re-planning of past slots

Slots marked `time_passed` are frozen. If the morning plan assumed 5 kWh of PV
that didn't materialise (cloudy day), the afternoon plan starts fresh from the
current SoC but does not retroactively account for the morning shortfall.

### Grid export throttle is a binary threshold

The `export_min_price` threshold turns grid export on or off below a price level.
There is no proportional throttle or ramp — the switch is instantaneous.

### Single-zone tariff model

The planner applies a single import price and export price per slot. It does not
model time-of-use (TOU) tariffs with multiple simultaneous price components (e.g.
capacity tariffs, network fees, or spot + fixed-premium structures). These can be
factored in manually by adjusting the import price values fed to the planner.
