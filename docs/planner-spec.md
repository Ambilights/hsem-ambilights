# HSEM Planner Specification

This document defines how the HSEM planner should work.

Use it as the reference for reviewing planner code, cost planning, and optimization changes.

## Goals

The planner must:

- minimize expected total cost within the configured horizon
- respect battery and inverter constraints
- keep energy accounting physically consistent
- avoid hardware writes when inputs are unsafe
- explain why a plan was selected
- produce deterministic output for the same input

## Core concepts

### Slot

A slot is one time interval in the planning horizon.

Each slot must have:

- start time
- end time
- duration in hours
- expected house load in kWh
- expected PV production in kWh
- import price per kWh
- export price per kWh
- recommendation
- planned battery charge in kWh
- planned battery discharge in kWh
- expected SoC before and after the slot

Power values in kW must be converted to energy using:

```text
energy_kwh = power_kw * duration_hours
```

### Slot timeline and DST identity

The horizon is a physical timeline. It starts at local midnight, advances in
UTC by the configured interval, and converts each boundary back to the Home
Assistant IANA timezone. A 24-hour horizon always contains 96 fifteen-minute
slots: on a spring transition it skips the nonexistent local hour, while on an
autumn transition it contains both folds of the repeated hour.

Slot identity is `(day_offset, slot_in_day)`, where `slot_in_day` is elapsed
physical time since that local calendar day's midnight. It is not derived from
the wall-clock hour. Consequently a complete 15-minute civil day has 92, 96,
or 100 ordinals on spring-transition, ordinary, and autumn-transition days.
Prices, PV forecasts, recommendations, and planner slots must preserve that
identity; timestamps carrying `+02:00` and `+01:00` in the repeated hour are
different slots.

All aware datetime containment, ordering, duration, `hours_ahead`, and
current/past/future decisions use UTC instants. `PlannerInput` carries both
`now_iso` and `timezone_name`; the planner rehydrates the IANA timezone before
building future slots because a fixed numeric offset cannot encode a DST rule.

## Recommendation priority rules

### Three-layer model

Recommendations are assigned and potentially overridden in three layers.
Every layer must respect the rules below.

#### Layer 1 — Planner engine (pre-simulation)

The dynamic MILP is the sole authority for actively optimised primary-battery
decisions. Across the actionable horizon it jointly chooses grid and PV charge,
local-load discharge, hold, and optional battery export from prices, load and
PV forecasts, conversion losses, cycle wear, SoC limits, and physical power
constraints. No user-defined daily charge or discharge windows constrain or
pre-author these decisions.

Recommendation labels are derived from the accepted solution's explicit energy
flows. A failed or invalid solve falls back to the passive candidate; it must
not restore a heuristic active-battery plan.

**Intentional battery export:**

1. The MILP is the only active-optimisation authority for primary-battery
   export. With excess export disabled, it may discharge only into house load;
   it cannot schedule battery-to-grid energy.
2. The MILP represents battery-origin export explicitly as battery-side DC
   `bx[t]`. Its AC slot field is
   `primary_battery_export_kwh = discharge_efficiency × bx[t]`; the separate
   `pv_export_kwh` field carries non-primary-battery AC export (normally
   direct PV). They obey
   `primary_battery_export_kwh + pv_export_kwh = grid_export_kwh`.
   The raw solution satisfies this within solver tolerance; public slot fields
   are reconciled exactly at 0.001 kWh precision. Aggregate grid export alone
   is never used to infer battery export.
   Source attribution is causal rather than chosen by the objective:
   `bx[t] = min(ed[t], grid_export_kwh[t] / discharge_efficiency)`. A per-slot
   binary `export_source_mode` enforces that identity together with
   `grid_export_kwh = discharge_efficiency × bx[t] + pv_export_kwh`. Thus any
   concurrent battery discharge and grid export is attributed to the battery
   first; PV supplies local and flexible loads before its residual is labelled
   export. The non-battery export block also has a finite physical upper bound:
   forecast PV surplus plus any PV exposed when PowMr SBU removes a dedicated
   load already included in the site measurement. No objective tiebreak may
   change source attribution.
   A separate binary `grid_flow_mode[t]` makes meter direction exact:
   `gi[t] <= M_import[t] * grid_flow_mode[t]` and
   `ge[t] <= M_export[t] * (1-grid_flow_mode[t])`. The finite bounds are
   derived from the largest physically reachable site load/charge and
   PV-plus-battery export in that slot. Import and export may both be zero,
   but can never both be positive or form a hidden wash flow.
3. With excess export enabled, material `bx[t]` activates an export-mode
   binary. The battery SoC at the end of the following demand window must then
   retain
   `hsem_batteries_excess_export_discharge_buffer`. The reserve is conditional:
   normal self-consumption may use it, and a planned cheap grid charge before
   the checkpoint may restore it. Every slot in one contiguous forecast
   PV-surplus run shares the checkpoint derived from the run's final slot:
   immediately before the next distinct PV-surplus run, or horizon end. The
   common checkpoint prevents the final PV-positive slot from receiving a
   different reserve test solely because it ends the run.
4. `force_export` is separate PV routing: it holds the primary battery at zero
   charge and zero discharge while exporting available PV. It is never evidence
   of battery discharge.

**Seasonal fill** (remaining `None` slots):

1. Export price > import price AND export price ≥ `export_min_price` → `force_export`
2. Actual PV surplus (`estimated_net_consumption_kwh < 0`) and battery not full → `batteries_charge_solar`
3. Insufficient projected energy for the reserve plus executable future
   `force_batteries_discharge` targets → `batteries_wait_mode`
4. With `hsem_seasonal_fill_mode = forecast` (default), use the forward
   refill-headroom decision below.
5. With `hsem_seasonal_fill_mode = months`, or when the forward Solcast
   window is unusable, use the legacy rule: winter month →
   `batteries_wait_mode`; otherwise actual PV surplus →
   `batteries_charge_solar`, and positive/zero net load →
   `batteries_discharge_mode`.

The pre-solar reserve is also battery-side: it sums actionable non-EV house
load divided by discharge efficiency, caps each slot by discharge power, adds
the configured buffer, and caps the result by usable capacity. The scan ends at
the first PV surplus that storage is allowed to absorb or at the price-authority
boundary. PV in a primary hold, `force_export`, or
`force_batteries_discharge` slot is not a refill boundary.

The forecast calculation is cumulative. It starts from the live usable energy
and advances a projected capacity chronologically through existing and newly
assigned recommendations, clamped to `[0, usable_capacity]` after every slot.
It accounts for charge/discharge conversion efficiency, per-slot power, EV
discharge suppression, primary-battery holds, and the executable stored or
discharged energy on each preassigned slot. A preassigned solar-charge slot
projects all PV the simulator can absorb, not merely its original target. A
preassigned `force_batteries_discharge` slot projects its explicit target; when
that slot is reached, a legacy zero target uses only executable energy above
reserve. A missing future target is reserved conservatively as one
power-limited usable-capacity draw. `force_export` projects no battery movement.

For each idle slot, the forward window contains slots strictly after it. Its
potential refill is cumulative battery-side energy that could actually pass the
charge conversion and per-slot charge-power limit, not gross forecast PV:

```text
slot_refill_kwh
    = min(max(-estimated_net_consumption_kwh, 0) × charge_efficiency,
          max_charge_per_slot)

minimum_after_load_kwh
    = required_capacity
      + future_forced_discharge_targets_net_of_intervening_refill
```

The forward suffix resets at a price-authority boundary, a primary-storage
hold, `force_export`, or `force_batteries_discharge`; forbidden energy and PV
sent deliberately to the grid cannot be promised as an earlier refill. An
ordinary actionable `batteries_wait_mode` label is transparent to the suffix:
under self-consumption-with-reserve it may still absorb PV, whereas
`primary_battery_hold` explicitly forbids that refill. Future forced-discharge
commitments are their cumulative, power-limited declared battery targets
(or the conservative missing-target bound), reduced by executable intervening
refill. The suffix passes and chronological projection keep the full
calculation O(n).

The forecast-mode decision is:

1. Current slot has storable PV surplus → `batteries_charge_solar`, limited by
   efficiency, charge power, and projected battery headroom.
2. Otherwise, the projected battery can serve the slot's battery-side load and
   remain above `minimum_after_load_kwh`, while its bounded forward refill
   provides positive headroom → `batteries_discharge_mode`.
3. Otherwise → `batteries_wait_mode`.

The forward forecast is usable only when at least one slot in that bounded
window has `solcast_pv_estimate_kwh > 0`.  Missing or all-zero forward Solcast
therefore invokes the legacy month rule instead of being interpreted as a real
zero-PV forecast.  An invalid configured mode also falls back to `months` and
emits a planner warning.  Every forecast decision logs its net consumption,
refill forecast, required/current capacity, headroom, and recommendation with
the `[disch] seasonal_fill` prefix.

> **Note:** `BatteriesChargeSolar` is only assigned when there is a genuine PV
> surplus (negative net consumption).  A small positive house load with zero PV
> must not be mislabeled as solar charging — that would cause the applier to
> write `MaximizeSelfConsumption` instead of `TimeOfUse` + charge TOU
> (issue #720).

**Seasonal-fill invariants:**

- Existing non-`None` recommendations are never changed.
- Seasonal fill and discharge concentration apply only to heuristic
  candidates. A validated MILP candidate is already a complete energy
  allocation and must not be changed by either heuristic after the solve.
- Explicit future forced-battery-discharge targets reserve their cumulative
  executable energy, net of intervening refill; one small future target must
  not hold every prior load slot.
- `force_export` neither changes battery SoC nor contributes refill. PV after a
  forced battery discharge, primary-storage hold, or authority boundary is not
  counted as earlier refill headroom.
- Projected capacity never becomes negative or exceeds usable capacity.
- A sunny winter forecast and an identical sunny summer forecast produce the
  same idle-slot decision; likewise for identical usable low-refill forecasts.
- `months` mode and unusable forward Solcast preserve the legacy calendar
  behaviour.

### Layer 2 — EV planned load labelling (post-simulation)

After the final SoC simulation, slots with `ev_total_planned_load_kwh > 0` are relabelled.
`ev_total_planned_load_kwh` is used (not `ev_planned_load_kwh`) so that EV-scheduled
slots are correctly labelled even when `base_load_includes_ev = True`, where
`ev_planned_load_kwh` is `0.0` but EV charging is still planned.

`base_load_includes_ev` is automatically derived from the
`hsem_house_power_includes_ev_charger_power` setting in the EV charger config step.
There is no separate user input for it.

- `batteries_charge_solar` → `ev_smart_charging`
- `batteries_wait_mode` → `ev_smart_charging`
- All other recommendations: **kept unchanged** (must not be overridden by EV label)

The following must never be overridden by the EV label:
`batteries_charge_grid`, `force_batteries_discharge`, `force_export`,
`time_passed`, `missing_input_entities`.

`batteries_discharge_mode` is **not** in this protected set — it is intentionally
overrideable. When EV load is allocated to a local-discharge slot, the
`ev_smart_charging` label wins so dashboards correctly reflect the active EV
session. This relabelling does not change the accepted MILP energy flows.

#### Layer 3 — Runtime resolver (current slot only, at hardware-write time)

Applied to the current slot immediately before hardware writes, using live sensor data:

1. `import_price < 0` → `force_export` (overrides everything)
2. `batteries_charge_grid` → kept (must never be overridden by EV relabelling)
3. Any EV actively charging → `ev_smart_charging`
4. Otherwise → keep the accepted planner recommendation

Runtime resolution is copy-on-write. The working-mode sensor detaches the
current recommendation and, when its effective label changes, substitutes that
copy into a sensor-local `CoordinatorData` view. It must never mutate the
planner-owned accepted slot or recommendation list. An in-flight hardware
worker therefore retains the exact snapshot it accepted, and each later live
refresh resolves again from the canonical planner output so a transient EV
override clears when its live condition clears.

### Invariants for tests

- Active grid charge, local discharge, and intentional battery export must come
  from the accepted MILP flows, never from user-defined daily time windows.
- With no negative-price or live-EV override, the runtime resolver must retain
  the accepted planner recommendation.
- A slot assigned `batteries_charge_grid` by the planner must never be relabelled by
  the EV load labelling pass (layer 2).
- A slot assigned `batteries_discharge_mode` **may** be relabelled `ev_smart_charging`
  by the EV load labelling pass when `ev_total_planned_load_kwh > 0`.
- A slot with `ev_planned_load_kwh > 0` and recommendation `batteries_charge_solar`
  must be relabelled `ev_smart_charging` after layer 2.
- A slot with `ev_planned_load_kwh > 0` and recommendation `batteries_wait_mode`
  must be relabelled `ev_smart_charging` after layer 2.
- The runtime resolver must set `force_export` when `import_price < 0`, regardless
  of the planner recommendation.
- The runtime resolver must NOT override `batteries_charge_grid` even when an EV
  is actively charging.
- The runtime resolver must NOT override `batteries_charge_grid` even when
  `import_price < 0` is False and EV is charging.
- Priority 1 (negative price → `force_export`) always beats priority 3 (EV charging).

## Energy balance per slot

For every slot:

```text
net_load_kwh = house_load_kwh + ev_planned_load_kwh - pv_kwh
```

`ev_planned_load_kwh` is the **extra** EV AC load to add to net consumption — the
portion not already captured in `house_load_kwh`.  See the EV load semantics section
for the three-field breakdown.

When EV integration is disabled, `ev_planned_load_kwh` is `0.0` for every slot
and the formula is identical to the non-EV case.

Positive `net_load_kwh` means the house (plus any extra EV load) needs energy.

Negative `net_load_kwh` means there is net surplus (solar minus house and EV load).

### EV charger energy source

The EV charger is an **AC appliance** that draws directly from the grid or from
PV surplus.  **It never draws from the house battery.**  This means:

- The battery's net demand is computed from `house_load - pv` only.
- `ev_planned_load_kwh` is added to `grid_import_kwh` — not to the battery
  discharge calculation.
- When PV surplus is available the EV consumes from it first (reducing
  `grid_export_kwh`); any residual EV demand that cannot be met by PV is
  imported from the grid.
- `batteries_discharged` is therefore independent of `ev_planned_load_kwh`.

Battery and grid flows must satisfy:

```text
house_load_kwh
= pv_used_for_house_kwh
+ battery_discharge_to_house_kwh
+ grid_import_for_house_kwh

grid_import_kwh
= grid_import_for_house_kwh
+ grid_import_for_battery_kwh
+ ev_grid_import_kwh
```

PV production must satisfy:

```text
pv_kwh
= pv_used_for_house_kwh
+ pv_used_for_ev_kwh
+ pv_used_for_battery_kwh
+ pv_export_kwh
+ pv_curtailed_kwh
```

Battery charge must satisfy:

```text
battery_charge_stored_kwh
= pv_used_for_battery_kwh * charge_efficiency
+ grid_import_for_battery_kwh * charge_efficiency
```

Grid import for charging:

```text
grid_import_for_battery_kwh = battery_charge_stored_kwh / charge_efficiency
```

Battery discharge must satisfy:

```text
usable_battery_discharge_kwh
= battery_energy_removed_kwh * discharge_efficiency
```

For MILP output the grid-export sources satisfy:

```text
primary_battery_export_kwh = battery_export_dc_kwh * discharge_efficiency
grid_export_kwh = primary_battery_export_kwh + pv_export_kwh
```

`battery_energy_removed_kwh - battery_export_dc_kwh` is the battery-side
discharge serving local load. Both export-source fields are non-negative. The
raw equality holds within solver tolerance and public values are exact at
0.001 kWh precision.

Battery energy to remove in order to deliver a target house load:

```text
battery_energy_removed_kwh = house_load_kwh / discharge_efficiency
```

## Secondary stationary storage (PowMr dedicated-load topology)

The optional secondary-storage model represents a topology that is materially
different from the Huawei battery:

- all PV is connected to Huawei; PowMr has no solar input
- PowMr can charge from the site AC bus
- PowMr output supplies one dedicated AC load only (the NAS)
- PowMr cannot backfeed the site bus or export to the grid
- PowMr charging and its dedicated load are connected to one configured grid
  phase (phase 3 by default)
- `SBU priority` transfers that dedicated load from utility to the PowMr battery
- utility bypass takes over at the configured hard reserve (20 % by default)

For slot $t$, let $L_t$ be dedicated-load AC energy, $h_t$ the slot duration,
$c_t$ stored charge energy, $d_t$ battery energy removed, $z^c_t$ the charge-mode
binary, and $z^s_t$ the SBU-mode binary. The three physical modes are utility
($z^c_t=z^s_t=0$), charge, and SBU, with:

$$
z^c_t + z^s_t \le 1
$$

The PowMr current entity supports integer 10 A steps. With nominal voltage $V$,
configured current step $\Delta I$, and non-negative integer $q_t$:

$$
c_t = \frac{V \times \Delta I \times h_t}{1000} q_t
$$

For the current slot, $h_t$ is the physical time remaining from `now` to the
slot end; an ended slot has zero duration. Future slots retain their nominal
full duration. The same effective duration is used for dedicated load,
standby overhead, minimum/maximum charge bounds, integer current-step energy,
secondary SoC and grid flows, and conversion of solved charge energy back to a
current command.

The charge-mode constraints bind $c_t$ to the configured minimum and maximum
current. With discharge efficiency $\eta_d$ and inverter overhead $P_o$:

$$
d_t = z^s_t \left(\frac{L_t}{\eta_d} + \frac{P_o h_t}{1000}\right)
$$

This equality is the dedicated-load-node invariant: secondary discharge serves
exactly the dedicated load plus its DC-side overhead and never directly
backfeeds. SBU still changes the aggregate site balance: if PV already serves
part of that dedicated load, moving it to battery may both avoid residual grid
import and reveal PV for export. That export remains non-battery/PV-origin; the
secondary discharge pays its full terminal-inventory, conversion-loss, and wear
terms.

Let $E_0$ be energy above the reserve and $E_{usable}$ the energy between the
minimum and maximum SoC. The secondary state equation is hard-constrained:

$$
E_t = E_0 + \sum_{k=0}^{t}(c_k-d_k), \qquad
0 \le E_t \le E_{usable}
$$

The site-bus adjustment depends on the meaning of the house-consumption history.
When history already includes the dedicated PowMr/NAS utility load (the default),
let $\widehat{L}_t=\min(L_t, house\_load_t)$ be the portion demonstrably present
in that forecast:

$$
\Delta site_t = \frac{c_t}{\eta_c} - z^s_t \widehat{L}_t
$$

When history excludes it:

$$
\Delta site_t = \frac{c_t}{\eta_c} + (1-z^s_t)L_t
$$

Mixed historical Utility/SBU operation is necessarily an approximation because
the same house-history series alternately includes and excludes the NAS. The
clamp prevents an incomplete history sample from modeling PowMr backfeed; the
explicit load sensor remains the source of truth for battery draw.

Secondary charge participates in the same site grid balance and main-fuse limit
as Huawei and EV charging. With phase-aware charging enabled, its entire site
delta is assigned to the configured physical phase instead of being treated as a
balanced three-phase load. By default, cross-battery transfer is forbidden in
both directions. With $M^c_t$ and $M^d_t$ denoting the primary charge and
discharge bounds, respectively:

$$
ec_t + M^c_t z^s_t \le M^c_t
$$

$$
ed_t + M^d_t z^c_t \le M^d_t
$$

The first row prevents PowMr SBU from funding Huawei charging; the second
prevents Huawei discharge from funding PowMr charging. The advanced transfer
option must be explicitly enabled to relax both guards.

The live primary house/PV frame remains the established full-slot projection,
including energy already elapsed in the current slot. When that forecast
includes the dedicated load, SBU removes only its remaining-duration portion;
the elapsed portion remains in the site balance. Phase-fuse rows therefore
also stay in that full-slot frame: reconstruction removes the actual partial
PowMr delta from the balanced $G_t/3$ share, converts it to its full-slot power
equivalent, and assigns that equivalent only to the configured phase. Thus a
60 A command always represents $V\times60$ W at the fuse even late in a slot,
while the secondary battery receives only the remaining-duration energy.

Huawei recommendation labels are reconciled after the PowMr site-bus adjustment
without changing either battery's solved energy. With cross-battery transfer
enabled, SBU may remove the remaining site import from a primary charge slot;
that charge is then labelled solar and runs in Maximize Self Consumption rather
than forced grid-charge mode. With transfer disabled, the hard mutual-exclusion
row makes that combination impossible. If SBU makes a positive
primary-discharge slot export, it is labelled forced battery discharge only
when the final export price clears both the economic and user-configured
battery-export floors; otherwise it remains self-consumption discharge. EV,
session, and explicit battery-hold intents are not rewritten by this pass.

Flow-based recommendation reconciliation follows the shared three-decimal
publication contract. Final grid import or export of exactly 0.001 kWh or less
is numerical/publication residue: it keeps solar charge or normal
self-consumption respectively. Only a flow greater than 0.001 kWh enables
forced grid-charge or forced battery-export hardware modes. This boundary is
identical for the primary MILP write-out and the post-SBU adjustment. The
secondary writer rejects material SBU and primary charge in the same slot when
cross-battery transfer is disabled; it never repairs the solved energy by
post-hoc mutation.

For no-export slots and slots below either export-price floor, the primary
discharge cap is coupled to the SBU binary before solving. When the house
forecast includes the dedicated load, Huawei AC discharge plus the included
load removed by SBU may not exceed the original site-load cap. This prevents a
nominally valid primary self-consumption allocation from becoming unexecutable
export when PowMr changes from utility to SBU.

The MILP and authoritative candidate scorer both include secondary conversion
loss, cycle wear, time discount, and a horizon-tail value for stored energy.
The secondary terminal term is uniform and undiscounted across every actionable
slot:

```text
secondary_terminal_soc_value
    = R_secondary * sum(
          secondary_storage_discharged_kwh
          - secondary_storage_charged_kwh
      )
    = R_secondary * (initial_secondary_kwh - final_secondary_kwh)
```

Equal secondary discharge and refill therefore cancel exactly regardless of
their slot positions or prices. There is no per-slot secondary charge premium
or discharge premium. Actual grid import/export, conversion loss, cycle wear,
headroom, and power constraints decide whether the cycle is worthwhile.
Non-MILP candidates receive a physically valid utility-bypass plan so candidate
comparison never leaves the dedicated load unaccounted.

Each successful MILP solve with valid secondary storage emits exactly one
aggregate debug record with the prefix ``[milp] secondary_result``.  Disabled
secondary storage emits no result record.  The line contains solved ternary-mode
counts, stored charge and discharge energy, isolated grid-import saving/cost,
secondary conversion loss, cycle wear, terminal credit, net diagnostic value,
start/end SoC, and a conservative reason slug.  It is deliberately one line per
solve rather than one line per slot so verbose logging remains usable on
15-minute, multi-day horizons.

The conversion-loss, cycle-cost, and terminal-value fields use the same shared
secondary cost accumulator as :func:`score_plan`; diagnostics must never
introduce a parallel cost formula.  ``terminal_credit`` is the sign-inverted
``secondary_terminal_soc_value`` so a credit is positive in the logged net:

```text
net = sbu_saving - charge_cost - cycle_cost - conversion_loss
      + terminal_credit
```

Summary generation is read-only and must not change any recommendation, mode,
charge-current command, energy flow, or SoC trajectory.

### Secondary-storage control safety

Planning and control are separate opt-ins. `hsem_secondary_storage_enabled`
enables MILP shadow planning; `hsem_secondary_storage_control_enabled` permits the
PowMr adapter. Global `hsem_read_only`, degraded mode, and missing live SoC/load
each independently prevent writes. The adapter applies ordered transitions and
verifies each write before continuing:

- SBU: set charger to `Only Solar`, then output to `SBU priority`
- charge start/increase: set output to `Utility first`, set the 10 A-step
  current, then set charger to `Solar and Utility`
- charge decrease: first verify charger `Only Solar`, set output to
  `Utility first`, set the lower 10 A-step current, then set charger to
  `Solar and Utility`
- utility: first verify charger `Only Solar`, then set output to
  `Utility first`

At or below minimum SoC, a stale SBU plan is forced to utility. At maximum SoC,
a stale charge plan is forced to utility, stopping grid charging with
`Only Solar` before switching the output to `Utility first`.
An invalid live SoC/load sample blocks the PowMr adapter for that cycle.
A failed or unverified preceding Huawei write blocks PowMr-enabling Charge and
SBU transitions, but must not block the independent fail-closed utility plan:
charger `Only Solar`, then output `Utility first`. The adapter revalidates
that the restricted plan contains only those two safe select targets before
writing. If the charger stop fails or cannot be verified, the output write is
blocked rather than adding the dedicated load to L3 while charging may remain
armed.

An already-active utility charge is disarmed with a verified `Only Solar`
selection before a downward current change. If a current write in any charge
transition fails or cannot be verified, the adapter does not reach the final
`Solar and Utility` enable and attempts `Only Solar` unless that stop was
already verified. `Only Solar` removes PowMr grid charging; on the target
installation, which has no PowMr PV input, it stops PowMr charging completely.

### Secondary-storage invariants for tests

- secondary SoC never crosses the configured 20 % reserve or maximum SoC
- current-slot secondary load, standby, charge/discharge energy, SoC movement,
  and current-step energy use only the time remaining; future slots use their
  nominal duration
- SBU discharge equals dedicated load divided by discharge efficiency plus
  inverter overhead
- secondary discharge is tied to the dedicated load and never directly
  backfeeds; it may free Huawei PV that is independently eligible for export
- SBU may both avoid genuine residual grid import and reveal PV export through
  the aggregate site balance
- charge current is always a supported 10 A increment
- PowMr charge and utility/SBU load transitions affect only its configured phase
- a current-slot PowMr command is checked against its full power on that phase,
  not diluted by the fraction of the slot remaining
- Huawei discharge is zero during PowMr charging unless transfer is enabled
- Huawei charge is zero during PowMr SBU unless transfer is enabled
- enabling cross-battery transfer relaxes both battery-to-battery guards but
  does not change export-source attribution or inventory valuation
- post-SBU Huawei charge/discharge labels match the final executable site flow
  and respect the configured battery-export price floor
- no-export and below-floor slots couple Huawei discharge to SBU load removal,
  so Huawei cannot create grid export without post-hoc energy mutation;
  independently eligible PV export remains valid
- disabled secondary storage is numerically identical to the upstream planner
- missing required PowMr telemetry produces no secondary plan and blocks control
- global read-only and the feature control switch each independently block writes
- failed or unverified Huawei writes continue to block PowMr Charge/SBU enables
  but cannot suppress the charger-first utility/`Only Solar` stop
- an unconfirmed `Only Solar` stop blocks the subsequent utility-output write
- a downward live PowMr current retarget verifies `Only Solar` before changing
  the current and never re-enables utility charging after an unconfirmed number
  write
- serialized PlannerOutput slots contain all secondary-storage load, charge,
  discharge, grid-import, capacity, SoC, current, and mode fields
- successful enabled MILP solves emit one ``secondary_result`` line each
- disabled or invalid secondary storage emits no ``secondary_result`` line
- logged secondary conversion, cycle, and terminal terms equal the authoritative
  candidate scorer's terms for the same solved slots
- equal secondary charge and discharge contribute exactly zero net
  ``secondary_terminal_soc_value`` regardless of slot prices or positions
- building or logging the summary leaves the solved slot list unchanged

## Battery efficiency

HSEM tracks charge-side and discharge-side efficiency independently.

### Parameters

| Parameter | Field | Default | Description |
|---|---|---|---|
| Charge efficiency | `battery_charge_efficiency_pct` | 97 % | Fraction of input energy stored. |
| Discharge efficiency | `battery_discharge_efficiency_pct` | 97 % | Fraction of stored energy delivered to house. |

### Semantics

```text
battery_stored = grid_or_pv_input × (charge_efficiency_pct / 100)
house_delivered = battery_removed × (discharge_efficiency_pct / 100)
grid_import_for_battery = battery_stored / (charge_efficiency_pct / 100)
battery_to_remove = house_load / (discharge_efficiency_pct / 100)
```

Round-trip yield:

```text
roundtrip_yield = (charge_efficiency_pct / 100) × (discharge_efficiency_pct / 100)
roundtrip_loss  = 1 − roundtrip_yield
```

Example (90 % / 90 %): yield = 0.81, loss = 19 %.

### Conversion loss pricing (issue #641)

Each side of the round-trip is priced independently at its own slot's price:

- **Charge-side loss**: Priced at the sanitised import price of the charge
  slot (`max(import_price, 0)`).  The lost energy was purchased at that
  price.
- **Discharge-side loss**: Priced from the MILP's explicit export-source split.
  Battery-side `bx[t]` is export-destined; `ed[t] - bx[t]` serves local
  load. Export-destined loss uses sanitised export price (foregone revenue);
  local-use loss uses sanitised import price (avoided import).

```text
local_discharge_dc = ed[t] - bx[t]
battery_export_dc = bx[t]
discharge_loss_cost[t] =
    local_discharge_dc * (1 - dis_eff) * max(import_price, 0)
    + battery_export_dc * (1 - dis_eff) * max(export_price, 0)
```

The LP objective, `cost_function.py::score_plan()`, and diagnostics use this
same split. A slot can export PV while the battery serves local load, so
`grid_export_kwh > 0` is not sufficient evidence that all discharge is
export-destined.

### Invariants for tests

- Charging 10 kWh at 90 % efficiency must draw 10 / 0.9 approx 11.11 kWh from the grid.
- Charging 10 kWh at 100 % efficiency must draw exactly 10 kWh from the grid.
- Discharging 10 kWh battery energy at 90 % efficiency must deliver 9 kWh to the house.
- The round-trip cost term (conversion_loss_cost) must use
  1 - charge_eff * discharge_eff when explicit efficiencies are set.
- When both efficiencies are 100 %, the legacy conversion_loss_pct field drives
  the conversion_loss_cost term (backwards compatibility).
- Discharge-side conversion loss MUST be destination-aware: `bx[t]` uses the
  export price and `ed[t]-bx[t]` uses the import price (issue #641).
- Battery-origin export has a lower loss cost than import-only pricing when
  export price is lower than import price.
- A slot with PV export and local battery discharge keeps import-price
  valuation for the local discharge; net-export status cannot change it.

## Live data injection (current slot)

Before scoring, the engine replaces the current (partially elapsed) slot's
forecast PV and consumption with live measurements
(`engine_population.py::_inject_live_data_into_current_slot`).  Live Watts
are converted to a projected full-slot kWh by multiplying by the slot's full
duration.

When `house_power_includes_ev = True`, the live house reading may contain EV
charging power that the battery must not serve (issue #592).  Two layers
protect against this:

1. **Known EV power subtraction** — when `ev_session_charge_kw` (and/or the
   second charger's) is available, it is subtracted from the live reading
   (floored at 0) before injection.
2. **Spike cap** — if the remaining live reading still exceeds
   `max(3 × forecast, 0.05 kWh)`, it is capped at the forecast (or at the
   0.05 kWh floor when the forecast is ~0, where the ratio test would be
   degenerate).  A spike of that magnitude is unambiguous unmetered load
   (e.g. a boolean-only EV status sensor); normal house load does not
   triple between slots.

The sub-window averages (`avg_house_consumption_1d/3d/7d/14d_kwh`) of the
current slot are **deliberately left unchanged** (issue #592).  The EV
discharge-cap fallback in `applier.async_apply_battery_settings` picks the
*minimum* of those windows to recover a clean house baseline when the live
reading is unreliable; overwriting them with the live-injected value (which
can still include unmeasured EV load when no EV power sensor is configured)
would destroy that fallback and let polluted history inflate the hardware
discharge cap.

The configured 1d/3d/7d/14d weights remain authoritative after outlier
redistribution and the existing safety caps. Agreement between overlapping
windows is not a reliability multiplier: with less than 14 days of history,
the 7d and 14d sensors can contain the same source days, so equality is not
independent corroboration and must not inflate their combined effective share.

## Historical learning and diagnostic accounting

These pipelines do not change the physical energy balance directly, but the
PV corrector changes planner input. Their time and eligibility semantics are
therefore part of the planner contract.

### ML recorder history

Recorder timestamps are normalized to Home Assistant's configured IANA
timezone at the read boundary. Local date, day-of-week, day-of-year, and
wall-clock slot are calendar features. Ordering, elapsed age, adjacency,
cache expiry, and slot dictionary identity use UTC instants.

For a cumulative energy meter, HSEM keeps the last reading of each physical
slot. The difference between the previous and current slot-end readings is
energy used in the **current** slot and is labelled with its current local
timestamp. Only adjacent UTC slot identities may form a delta. Recorder gaps,
counter resets, non-positive deltas, and values above the sanity ceiling are
discarded rather than moved into another slot.

A 15-minute local civil day may supply 92, 96, or 100 observations. Both
autumn folds remain separate physical records even though they intentionally
share one wall-clock model feature; a nonexistent spring slot is not
fabricated. Today's actual replacement begins one physical slot before local
midnight, includes completed slots only, filters by HA-local date, and keys
them by canonical UTC start.
Sequential training resets its lag across non-adjacent physical samples.
Inference chains the real recommendation instants in physical UTC order, so
spring skips nonexistent wall slots and autumn preserves both physical folds.


The configured minimum history span defaults to 14 physical days. Processed
history is cached for one physical hour and keyed by Home Assistant instance,
import entity, optional export entity, net/gross mode, cadence, and required
span. Temperature history is independently source-keyed. A predictor is
replaced whenever its effective source/configuration context changes, so a
sample-count retrain gate cannot preserve coefficients from the old context.
Within an unchanged context, retraining compares valid sample fingerprints and
reruns after four unseen or revised fingerprints. Revised readings therefore
trigger retraining even when the rolling sample count remains constant.


Net mode is fail-closed: an export entity is required, and training/today
actuals use only physical slot keys present in both import and export sources.
An absent export observation is never zero. If the aligned training result no
longer spans the required history, ML population fails and legacy averages are
used.

Temperature is enabled only with sufficient recorder history. Training matches
temperature by physical time; because this source is not a future weather
forecast, inference holds the newest nearby observation across the horizon.
Insufficient temperature history fits a model without that feature. Failure to
fit any trained model returns control to legacy population; an untrained model
must never publish a zero-load horizon.

### Frozen forecast accuracy and solar learning

Before current-slot live injection or planning, the coordinator snapshots raw,
available PV only for future recommendation slots. After a successful plan it
may register corrected PV, load, predicted end SoC, and action only while the
slot remains physically in the future. A future replan may update these values;
the last pre-slot values freeze when the UTC start instant arrives. Current or
past planner values can never become or rewrite an eligible baseline.

Instantaneous PV and load follow prior-sample semantics: the previous finite
power pair represents `[previous_timestamp, now)`. Both the previous and
current endpoint samples for both channels must be finite. Missing telemetry
advances all sample state but rejects the interval, so recovery cannot bridge
the gap. A non-positive interval or one longer than twice the effective
coordinator cadence is also rejected.

Accepted intervals are split by overlap with frozen slots on the UTC timeline.
Each record tracks trusted coverage seconds. A production record is accuracy-
eligible only when it has a frozen raw baseline and covers the full physical
slot within tolerance. Missing time is unknown, never zero PV or load.
Prediction eligibility additionally requires frozen SoC and action fields.

Only eligible finalised records train the PV corrector. The per-hour factor
uses raw PV versus actual PV; the recent residual uses corrected PV versus
actual PV. Residual `slots_ahead` is the UTC physical distance from the
current planning instant, not an index since midnight, so correction remains
active throughout the day and across both DST folds.

Prediction accuracy is emitted only on the coordinator cycle in which an
eligible slot transitions to finalised and only when actual Huawei SoC is
finite. No restored forecast record may be paired with a post-restart live SoC
sample; only a slot newly finalised from the current live process enters the
prediction tracker. Its warm-up/deduplication identity is the unique UTC slot
start, not the number of coordinator polls.

Restored legacy forecast records may deserialize, but records missing the
raw/frozen schema are permanently ineligible for summary metrics and learning.
Solar-corrector state is versioned: pre-v3 factors, history, and residuals
learned under the old live-rewritten baseline are cold-reset on upgrade while
the internal confidence value is retained. Valid v3 state restores the exact
bounded per-hour and residual buffers plus a UTC processed-through watermark.
The restore is atomic; malformed, non-finite, or future-dated watermark state
cold-resets rather than entering planning. A prediction-accuracy scalar
restored by Home Assistant is startup-only; after the first live coordinator
snapshot the sensor reports a fresh metric or unavailable.

### Daily plan-versus-actual and savings

All daily labels and period rollups use the Home Assistant-local date. All
interval ordering, duration, overlap, price boundaries, local-midnight
boundaries, and DST folds use UTC instants.

Cumulative import, export, and PV meter deltas are distributed uniformly
between consecutive sample instants. The interval is split at every overlapping
planner price boundary and local midnight. Energy is recorded even when a
price channel is unavailable; money is added only for a finite authoritative
price. A leading segment not covered by the new planner output uses the prior
sampled price when authoritative rather than pricing the entire delta at the
new value.

Actual meters are accumulated before rollover. A cross-midnight interval is
therefore apportioned by local date before the old day is persisted. Cumulative
meter timestamp/value baselines and the SoC baseline survive rollover. If a
daily meter decreases, its current non-negative reading is treated as energy
since local midnight and retained for the new day. Plan values are recorded
once after their slot starts, after rollover has selected the destination day.
On a live cadence change, already-recorded physical coverage remains
authoritative; only the uncovered fraction of an overlapping replacement slot
may be added.

Savings consumes the daily tracker's explicit per-cycle, per-date measured
import-cost and export-revenue deltas once. It must not re-difference the
cumulative daily totals. Charge savings integrates the **previous** positive,
finite Huawei battery charge-power sample over the interval to the current
finite endpoint. Equal/reversed timestamps, missing telemetry, and gaps beyond
twice the effective cadence add no energy while still advancing sample state.

Measured charge energy is divided by UTC overlap across planner slots and
local dates. Only actionable slots whose recommendation is a charge mode and
whose import price is finite/available contribute; their share is valued
against that local delivery day's published mean import price. Planned charge
The snapshot that governed elapsed time masks a later overlapping replan even
when the prior slot is not actionable; the current snapshot fills only
uncovered physical time.
energy is never added repeatedly per coordinator poll. Automatic operation
records the result as actual savings, other modes as missed savings, and the
injected HA-local date anchors today/7-day/30-day rollups.

### Invariants for historical and accounting tests

- ML wall-calendar features are local, while identity, age, and adjacency are
  physical UTC.
- A cumulative-meter delta is labelled with the current slot, not one slot
  early, and never spans a missing physical slot.
- Both autumn folds remain distinct; spring gaps do not create observations.
- A cache or predictor from another HA instance, entity pair, net mode,
  cadence, history span, temperature context, or sequential mode is not reused.
- Net ML history never substitutes zero for a missing export sample.
- An untrained ML predictor falls back to legacy consumption.
- A baseline observed at or after slot start is ineligible, and a post-start
  replan cannot mutate a frozen baseline.
- Forecast actual energy is assigned from the prior sample by UTC overlap;
  missing endpoints and stale gaps cannot become zero or stale energy.
- Only fully covered records affect accuracy, solar correction, or prediction
  diagnostics.
- Solar residual lead is relative to the current physical slot, not midnight.
- Prediction diagnostics emit once at finalisation with finite actual SoC.
- Cross-boundary meter deltas receive proportional prices and local dates.
- Midnight rollover retains the first new-day energy and all meter baselines.
- Savings uses per-cycle measured deltas and measured prior-sample charge
  energy exactly once.

## SoC simulation

SoC must be simulated forward through the full horizon.

For each slot:

```text
soc_after_kwh
= soc_before_kwh
+ battery_charge_stored_kwh
- battery_energy_removed_kwh
```

The simulator must enforce:

- `soc_after_kwh >= min_soc_kwh`
- `soc_after_kwh <= max_soc_kwh`
- charge power limit
- discharge power limit
- grid import limit
- the MILP export limit when configured; a non-MILP simulation reports
  natural PV export and leaves physical curtailment to the inverter/DNO

The simulator must read the slot recommendation.

If a slot recommends forced battery discharge or discharge-only behaviour,
that energy flow must appear in:

- `batteries_discharged`
- SoC change
- import/export calculation
- plan cost

No recommendation may be energetically invisible.

`force_export` is intentionally different: the primary battery is held, so
both battery charge and discharge are zero. Only PV remaining after site load
may appear as grid export.

### MILP-pre-populated mode (issue #637)

When `milp_prepopulated=True` is passed to `simulate_soc()`, the
simulation uses the slot's **existing** `batteries_discharged_kwh`,
`grid_import_kwh`, and `grid_export_kwh` values verbatim — it does **not**
re-derive them from the recommendation label and net demand.

This mode is used for MILP-sourced candidates.  `solve_milp()` populates
these fields in a **single merged write-out pass** (issue #659) that:

1. Consumes an exact charge-or-discharge solution: the binary
   `primary_action_mode` prevents simultaneous material `ec` and `ed`. The
   existing chronological headroom resolver remains as a defensive guard for
   solver tolerance or legacy direct-helper inputs; a validated production
   solution must not depend on that guard to remove a wash cycle.
2. Writes `batteries_charged_kwh` and `batteries_discharged_kwh` from the
   **resolved** ec/ed (not the raw solver arrays).
3. Derives `grid_import_kwh` and `grid_export_kwh` from the slot's energy
   balance equation using the **same resolved** ec/ed values — they are
   **not** read directly from the raw solver `gi[t]`/`ge[t]` arrays, because
   the raw arrays assume the original (potentially now-invalid) ec/ed
   combination.

Recommendation source/mode classification uses those final published flows.
A charge-source shortfall or final export of exactly 0.001 kWh or less is
rounding residue and cannot enable Time-of-Use grid charge or Fully Fed battery
export; a value greater than 0.001 kWh is material and must select the matching
executable mode.

All four energy-flow fields are consistent with each other and with the
recommendation label for every slot. The resolved values are the source
of truth; candidate selection, seasonal fill, discharge concentration, and
SoC simulation must never silently overwrite them. An idle MILP slot is
completed with a label-only ``batteries_wait_mode`` recommendation and an
explicit primary-battery hold intent; its charge, discharge, import, and
export fields remain byte-for-byte unchanged. The runtime applier executes
that hold in Time-of-Use mode with a 0 W discharge cap and preserves incidental
PV export, even when the configured fallback wait behaviour permits
self-consumption or an EV display relabel is active.

For an explicit `force_batteries_discharge` slot, the runtime converts the
selected battery energy into a stable upper bound:
`floor(batteries_discharged_kwh / slot_hours * 1000)`, limited by the Huawei
hardware maximum. The bound is latched for the selected plan and is not tapered
from Huawei's integer SoC updates; their resolution is too coarse for sub-slot
energy accounting. A changed plan or hardware limit recomputes it, a newly
accepted plan already at its endpoint is rejected, and an expired slot commands
0 W on the next applier callback. Accurate adaptive tapering would require
integrated primary-battery energy.

While a forced-export slot or a materially partial normal-discharge slot is
active, a separate lightweight coordinator timer samples only live house and
PV power every 10 seconds. It compares residual AC demand
(`max(house_power - solar_power, 0)`) with the slot's planned AC battery
delivery (`batteries_discharged_kwh * discharge_efficiency / slot_hours`) plus
any material grid-import share already solved for a partial normal-discharge
slot. A rounded import residue of 0.001 kWh or less is treated as numerical
noise: it neither throttles MSC nor enables the partial-slot monitor. Any
larger planned import remains authoritative. When residual demand exceeds the
solved battery-plus-grid supply by more than `max(150 W, 10% of that supply)`
continuously for 30 seconds, HSEM requests one corrective planner run for that
slot. The trigger is disabled with less than 60 seconds remaining, when required
live state is unavailable/degraded, and while an EV is charging if the house
meter includes EV power. This monitor never changes the v13 latched hardware
cap directly.

The live house-power sensor and the solved Huawei-plus-grid supply are compared
on the same site-bus boundary. Before current-slot injection, the coordinator
removes the measured current PowMr site delta — including Utility/SBU load
routing and any live AC charging draw — to recover the exogenous base load on
the configured history topology. The MILP can then apply its prospective
Utility/SBU/charge delta exactly once. The monitor therefore must not add PowMr
discharge or load again. This prevents double-counting secondary supply or live
charger power and preserves detection of genuinely under-modelled house load.

The corrective planner run receives current live house/PV inputs and bypasses
both candidate hysteresis and current-window hysteresis for that one run; stale
plan preference must not restore the recommendation that caused the correction.
The planner remains authoritative: it may keep forced export if profitable
battery export remains, or choose normal self-consumption when live house demand
uses the available discharge. Only an optimal MILP or a fully validated
time-limit incumbent may replace the active plan. A passive/no-action fallback
is logged but cannot replace the previous validated plan; the attempt is then
closed for that slot to avoid repeated solver timeouts. The
normal-self-consumption transition requires a real solved battery-discharge
allocation. When the solved slot deliberately retains a material grid-import
share, Huawei MSC is bounded to `batteries_discharged_kwh / slot_hours` (and
the physical maximum), so it cannot consume energy the MILP reserved for
later. If live demand then remains materially above the solved battery-plus-grid
supply for 30 seconds, the guarded corrective path asks the MILP to re-evaluate
the split rather than bypassing its reserve. When modeled grid import is zero
or only a rounding residue, MSC retains the normal hardware maximum and follows
live house demand without exporting. The
once-per-slot attempt is otherwise consumed only after the complete coordinator
snapshot is successfully published. A busy or failed update leaves the request
pending for the next 10-second monitor tick. Read-only and degraded-mode
hardware-write gates remain unchanged.

Secondary-storage state listeners do not route every PowMr sample through the
full coordinator. Raw battery net power is sampled by accepted planner runs for
the current-slot topology correction, but is not itself a high-rate planning
trigger. Secondary
SoC and the smoothed dedicated-load input use lightweight event callbacks and
request one coalesced update only after changing by at least 1 percentage point
or 25 W respectively; output/charger priority and charge-current control state
changes use the same short debounce but remain immediately material. The
comparison baseline advances only after a newly accepted plan has been
successfully published. Reused plans and rejected corrective candidates must
not swallow cumulative secondary-storage drift. Failed cycles therefore retry
on a later material state event. If the coordinator is already running when a
debounce expires, the material event waits for the current cycle's lock and is
processed once instead of being dropped. All listener and debounce handles are
cancelled during coordinator teardown.

Control-entity callbacks are ignored whenever secondary hardware control is
disabled, including stale listeners registered before an options change. SoC
and dedicated-load telemetry continue to drive planning while secondary
optimisation remains enabled.

For non-MILP candidates (`milp_prepopulated=False`, the default),
the simulation continues to derive discharge and grid flows greedily
from the recommendation label and net demand — unchanged behaviour.

### Fallback wait-mode fidelity

For non-MILP candidates, SoC simulation must model the configured
``batteries_wait_mode`` hardware behaviour rather than assuming every wait
slot is idle.  This keeps passive fallback scoring aligned with the mode the
runtime applier will actually select when a solver result is unavailable.

The simulator therefore:

- leaves wait slots idle when ``hsem_batteries_wait_mode_behavior = strict``;
- with ``self_consumption_with_reserve``, lets the battery serve only the
  house-load deficit while its usable energy remains above
  ``required_capacity``;
- never exports battery energy from a wait slot;
- keeps the existing EV anti-roundtrip guard, so EV load remains grid/PV-fed;
- respects charge/discharge efficiency and the per-slot discharge limit.

This rule affects candidate simulation and scoring only.  It does not change
the recommendation label or bypass any hardware-write safety gate.

### MILP solve time limit and graceful fallback

``hsem_milp_solver_timeout_seconds`` configures the HiGHS wall-clock budget
(default 15 seconds, constrained to 1-60 seconds).  The same configured
budget is used whether phase-aware fuse constraints, export reserve, EV, or
secondary storage are active.

Solver outcomes follow three explicit tiers:

1. **Optimal MILP** — HiGHS proves optimality and the solution is accepted.
2. **Validated time-limit incumbent** — HiGHS reaches the configured time
   limit after finding an integer-feasible solution.  HSEM validates the full
   returned decision vector before accepting it as the normal ``milp``
   candidate.
3. **Explicit passive fallback** — no incumbent exists, validation fails, or
   another solver failure prevents a safe MILP candidate.  The passive
   candidate remains available and the failure reason is surfaced.

A time-limited incumbent must pass all of these checks against the final model:

- one-dimensional, exact-length, finite decision vector;
- future-slot count and indices align with the planning horizon;
- every per-slot decision-variable block has exactly the active slot count;
- the central `MilpColumnLayout` declares nine core per-slot blocks, appends
  active EV and optional fuse blocks, then the named `primary_battery_export`,
  `pv_export`, `export_source_mode`, `primary_action_mode`, and
  `grid_flow_mode` blocks, followed by optional export-reserve, bounded
  primary-terminal-inventory, and secondary-storage blocks;
- every variable-bound producer assigns one complete block by its declared
  `MilpColumnLayout` name; `MilpBoundsBuilder` places that block
  at its declared offset, independent of producer call order;
- unknown, duplicate, overlapping, wrong-width, or invalid bound assignments
  are rejected, and finalization rejects every declared column that remains
  unassigned;
- the objective length, equality-matrix width, inequality-matrix width, and
  finalized bounds length all equal that layout's final `column_count`;
- variable bounds are satisfied;
- all equality and inequality rows are satisfied within solver tolerance;
- every integer/binary variable is integral within tolerance.

Any failure rejects the incumbent closed; partially decoded plans are never
used.  The candidate name intentionally remains exactly ``milp`` for both
optimal and accepted-incumbent results because that name is a control-flow
key: it preserves pre-populated primary/PowMr flows and prevents the
secondary-storage utility bypass from overwriting the solved PowMr schedule.
Offsets and bounds slices are obtained by block name from the layout declaration
rather than recomputed from a hard-coded base width, positional formula, or
append order. A correctly named bounds assignment is resolved to its declared
slice at write time, so changes to the layout declaration or the order of
independent producers cannot redirect it through positional drift. Construction
fails before HiGHS is called if a consumer supplies an
unknown, duplicate, overlapping, wrong-width, invalid, or incomplete assignment.

This is a structural construction invariant only. It does not alter any
variable's lower or upper bound, add a model column or row, or change planner
economics.

`export_source_mode`, `primary_action_mode`, and `grid_flow_mode` are
per-slot binary blocks in every solve. They make the export-source split,
primary charge/discharge direction, and meter import/export direction exact.
Secondary-storage and conditional export-reserve modes may add further integer
blocks. Positive primary terminal tiers add a non-integral, non-per-slot
`primary_terminal_inventory` block; an empty `hardware_floor_only` model
adds none. The base primary model is already mixed-integer.

Diagnostics publish the ordered block metadata under
`model_variable_blocks` and the common width under `model_column_count`,
`model_objective_column_count`, `model_equality_column_count`,
`model_inequality_column_count`, and `model_bounds_count`. The block
offset/width cursor and every count must agree.
`model_integral_blocks` lists every wholly integral named block and
`model_integrality_count` gives the total integral-column count; all three
per-slot primary modes must appear there.
`grid_import_export_overlap_max_kwh` reports the largest raw
`min(gi[t], ge[t])` and must remain within solver tolerance.

The plan explanation sensor surfaces the most recent solve outcome through:

- ``solver_status`` and ``solver_optimal``;
- ``solver_time_limit_seconds`` and ``solver_elapsed_seconds``;
- ``solver_mip_gap`` and ``solver_message``;
- ``incumbent_used`` and ``incumbent_validation``;
- ``fallback_reason``.

A non-MILP winner adds ``milp_fallback`` to ``constraints``; an accepted
time-limit incumbent adds ``milp_time_limit_incumbent``.

**Invariants:**

- A time-limit result without a complete feasible incumbent never becomes a
  ``milp`` candidate.
- An accepted incumbent satisfies the same complete model used by HiGHS,
  including SoC, fuse, phase, export-reserve, EV, and PowMr constraints.
- Every declared variable block receives exactly one valid, width-matched named
  bounds assignment; unknown, duplicate, overlapping, invalid, or missing
  assignments fail before HiGHS is called.
- Optimal and accepted-incumbent candidates both retain the name ``milp``.
- A fallback retains the failed MILP diagnostics even though the winning
  candidate is passive.
- No solver-status field changes energy allocation or hardware behaviour.

## MILP soft constraints (penalty approach)

The MILP optimizer (`milp_optimizer.py`) uses **soft constraints** with penalty
variables to prevent infeasibility when the initial SoC is outside bounds
(e.g., overcharged battery).

### Penalty variables

- `s_max_pen[t]` — kWh by which SoC exceeds `usable_kwh` in slot `t`
- `s_min_pen[t]` — kWh by which SoC drops below 0 in slot `t`

### Soft SOC bounds

```text
Upper: soc[t] - s_max_pen[t] <= usable_kwh
Lower: -soc[t] - s_min_pen[t] <= 0
```

### Penalty cost

```text
p_soc = max(p_imp) * 100
```

The penalty cost is added to the objective:
`p_soc * (s_max_pen[t] + s_min_pen[t])`.  It is high enough that the solver
never uses penalties unless forced by an out-of-bounds initial SoC.

### Invariants

- The MILP is **never** infeasible due to initial SoC boundary violations.
- When `current_kwh` is within `[0, usable_kwh]`, all penalty values are zero.
- When `current_kwh > usable_kwh`, `s_max_pen[0]` absorbs the excess and
  decreases over time as the solver discharges.
- Violations are logged at WARNING level.
- The diagnostics dict (returned alongside the slot list) captures penalty
  values for the engine to surface.

### Battery discharge upper bounds (hard)

Before the destination-specific discharge bounds below, every active slot uses
a binary `primary_action_mode[t]`:

```text
ec[t] <= max_charge_per_slot * primary_action_mode[t]
ed[t] <= max_discharge_per_slot * (1 - primary_action_mode[t])
```

Both flows may be zero, but they can never both be positive. This exact
charge-or-discharge choice prevents a physically meaningless continuous
half-charge/half-discharge vertex and the plan-versus-writeback divergence it
would create.

In addition to the soft SoC penalties, the MILP applies **hard per-slot upper
bounds** on the discharge variable `ed[t]` (implemented as variable bounds in
`_build_constraints`):

1. **EV discharge guard (issue #592)** — when EV co-optimisation is **not**
   active and a slot has `ev_accounted_load_kwh > 0` (EV load already included
   in the house consumption sensor):

   ```text
   ed[t] <= max(0, base_load[t] - ev_accounted_load_kwh[t]) / discharge_eff
   ```

   The battery may only serve the non-EV portion of demand; the EV load is
   served by grid import or PV.  When co-optimisation **is** active the legacy
   bound is skipped, but `base_load` is rebuilt without flexible EV load and
   the explicit local-sink/source row also excludes that EV load. The battery
   therefore still cannot serve it or have that demand counted as local
   battery discharge.

   **Exactness note**: although `base_load` is net of PV and
   `ev_accounted_load_kwh` is the gross EV load, the formula is exact —
   there is no PV double-counting.  With `H` = gross house consumption
   (incl. EV) and `P` = PV production, `base_load = max(H − P, 0)` and the
   non-EV unmet demand is `max(H − ev − P, 0)`.  When `base_load > 0`:
   `base_load − ev = H − P − ev`, identical.  When `base_load = 0`
   (PV surplus): `H − P ≤ 0`, so both sides are 0.  Hence
   `max(base_load − ev, 0) == max(H − ev − P, 0)` in all cases, and the
   battery is never blocked from serving genuine non-EV house load on
   partially PV-covered EV slots.

2. **No-export cap (issue #592)** — when `excess_export_enabled = False`
   (`no_export=True`):

   ```text
   bx[t] = 0
   ```

   The battery-origin export block is fixed to zero. The battery may still
   discharge into house load through `ed[t] - bx[t]`; PV export through
   `pv_export[t]` is unaffected.
   Note this also suppresses battery-driven grid arbitrage — intentional:
   "excess export disabled" means the battery never feeds the grid.

3. **Battery export minimum price floor (issue #752)** — when
   `battery_export_min_price > 0` and the slot's RAW export price is
   strictly below this floor (`p_exp[t] < battery_export_min_price`), the
   battery can serve house load on that slot but cannot intentionally
   export to the grid:

   ```text
   bx[t] = 0  (only on blocked slots)
   ```

   This is the per-slot, soft-switch companion to the global `no_export`
   cap: instead of blocking battery export everywhere, the floor blocks
   it only on slots whose raw export price is below the user's explicit
   guard.  The mask is evaluated on the RAW `p_exp` (before the
   `export_min_price` and export-≤-import clamps) so the user's explicit
   price signal is honoured even when the recommended threshold or
   inverter physical floor are lower.  Above the floor the optimiser is
   free to decide whether exporting is worthwhile — reaching the
   threshold does NOT auto-trigger export.  The guard applies only to
   intentional battery-to-grid export (`ForceBatteriesDischarge`); it
   does NOT restrict normal battery self-consumption, battery discharge
   for house load, direct PV export, or PV charging of the battery.  The
   scorer uses the solved `primary_battery_export_kwh` attribution directly;
   there is no non-MILP battery-export scheduling path.

   Historical correction for `v6.2.2-powmr.23`: that release described the
   private `apply_excess_export()` helper as deciding force-discharge slots.
   It did not author a selected executable plan: candidate generation cleared
   its labels from the diagnostic and passive candidates, while the MILP solved
   battery allocation independently. Because replacement valuation inspected
   those pre-candidate labels, the helper could still change MILP terminal
   valuation indirectly. That was a heuristic side channel, not an
   authoritative export guard. The scheduling call is retired; intentional
   battery export is now exclusively MILP-owned. `ForceExport` is also excluded
   from the discharge-recommendation set so its PV-only label cannot affect
   replacement valuation as if battery energy moved.

4. **Conditional excess-export reserve** — when excess export is enabled and
   `excess_export_discharge_buffer_pct > 0`, each slot receives a binary
   `z_export[t]`. Material battery-origin export forces that binary on:

   ```text
   bx[t] <= max_discharge_per_slot * z_export[t]
   ```

   A forecast PV-surplus run is a maximal contiguous sequence whose
   `pv_avail[t]` is materially positive. For a slot outside such a run, let
   `checkpoint[t]` be the final demand slot before the next run, or horizon
   end when no later run exists. For every slot in a run `[a, b]`, use the
   checkpoint derived from its final slot for the whole run:

   ```text
   checkpoint[t] = checkpoint[b]  for every t in [a, b]
   ```

   That common checkpoint is the end of the demand window following the run:
   immediately before the next distinct PV-surplus run, or horizon end for the
   final run. When the binary is on, primary SoC at that checkpoint must retain
   the configured percentage of usable capacity:

   ```text
   SoC[checkpoint[t]] >= buffer_kwh - usable_kwh * (1 - z_export[t])
   buffer_kwh = usable_kwh * buffer_pct / 100
   ```

   This protects forecast demand plus an error buffer without creating a hard
   minimum SoC for ordinary self-consumption. Because the condition checks the
   solved SoC trajectory at the common checkpoint, future grid or PV charging
   may legitimately restore the buffer before it is measured. Conversely, if
   subsequent demand consumes the available energy and no economical refill
   restores the buffer, grouping may suppress every battery-export slot in the
   run; it does not promise to move an export to the run's highest-price slot.

   Run grouping changes only the precomputed checkpoint array. It adds no MILP
   variable, binary, bound, or constraint row. `no_export`, export-price floors,
   the shared grid-export cap, hardware and dynamic SoC floors, ordinary local
   self-consumption, direct PV export, and secondary-storage behaviour retain
   their existing scope.

   **Checkpoint invariants:**

   - every slot in one contiguous PV-surplus run has the same checkpoint;
   - that checkpoint follows the run's subsequent demand window, not merely
     the run itself;
   - every slot in the final PV-surplus run uses horizon end;
   - a common checkpoint may move, spread, or suppress intentional export
     according to the same full-horizon economics and physical limits.

When either export block applies, `bx[t]` is zero. The local-discharge
quantity `ed[t]-bx[t]` remains available within house load. Post-processing
labels `ForceBatteriesDischarge` only when
`primary_battery_export_kwh = discharge_eff * bx[t]` exceeds the numerical
action tolerance; otherwise a discharge slot is `BatteriesDischargeMode`.

For every solved slot:

```text
bx[t] = min(ed[t], grid_export_kwh / discharge_eff)
primary_battery_export_kwh = discharge_eff * bx[t]
pv_export_kwh = pv_export[t]
primary_battery_export_kwh + pv_export_kwh = grid_export_kwh
0 <= pv_export[t] <= pv_avail[t] + PowMr_SBU_revealed_PV[t]
```

The binary `export_source_mode[t]` makes the first identity exact: if aggregate
export is smaller than delivered battery discharge, all export is
battery-origin; otherwise all battery discharge is export-origin and only the
remainder may be non-battery/PV export. The local-discharge row permits
`ed[t]-bx[t]` only against fixed residual house demand and eligible PowMr
transfer/dedicated-load sinks; active flexible EV demand is deliberately
excluded. This prevents simultaneous PV, forced EV load, and battery discharge
from disguising battery-caused export as PV export.

The raw solution satisfies the source equalities within solver tolerance.
Public fields are rounded/reconciled so the source sum is exact at 0.001 kWh
precision, and both sources are non-negative. Passive fallback slots set
`primary_battery_export_kwh = 0`.

### EV co-optimisation (MILP)

When one or more `EVConfig` objects are passed to `solve_milp()`, the LP
expands to co-optimise EV charging alongside the battery.  EV loads are no
longer pre-computed by `ev_planner.py` and treated as fixed inputs; instead
the MILP decides **when and how much each EV charges**.

**EV variables** (per active EV):
- `ev_c[t]` — DC-side energy delivered to the EV battery in slot `t` (kWh).
  Bounded by `[0, ev.max_charge_per_slot]`.
- `ev_pen` — single slack variable absorbing unmet deadline target (kWh).

**EV constraints**:
- SOC dynamics (cumulative, no discharge):
  `ev_soc[t] = ev_initial + Σ_{k≤t} ev_c[k]`
- SOC upper bound per slot: `ev_soc[t] ≤ ev_capacity`
- Deadline soft goal: `ev_soc[D] + ev_pen ≥ ev_target` where `D` is the
  LP-slot index of the effective deadline.
- **Post-deadline zero-charge**: For EVs with a deadline and `charge_past_target=False`,
  `ev_c[t] = 0` for all `t > D`. This prevents charging after the deadline.
- **Target-cap constraint** (issue #636): For EVs with a deadline and
  `charge_past_target=False`, a hard upper bound caps cumulative pre-deadline
  charge at the economic shortfall:
  `Σ_{k≤D} ev_c[k] ≤ target_kwh − initial_soc_kwh`.
  Without this, the benefit coefficient on `ev_c[t]` would drive charging all the
  way to `capacity_kwh` regardless of the actual shortfall.
- **Surplus-only for charge-past-target**: When `charge_past_target=True`,
  `ev_c[t]/η_charger ≤ max(0, pv[t] − base_load[t])` — charging only from PV surplus.
- No discharge: `ev_c[t] ≥ 0` (via bounds).

**Energy balance** includes EV AC load:
```text
gi + pv + ed·η_dis = base_load + ec/η_chg + ge + Σ ev_c/eff
```
where `base_load` is recomputed **without** pre-computed EV planned loads
(only house consumption minus PV).

**Objective** includes a high-cost deadline penalty:
```text
ev_penalty_cost = max(p_imp) * max(energy_needed, 1.0) * 10
```
ensuring the MILP always prefers meeting the target when physically possible.

**Pre-deadline slots** (`t ≤ D`): Each `ev_c[t]` receives a negative objective
coefficient of `-ev_penalty_cost`, creating a direct benefit that forces the LP
to charge the EV. The LP will use PV surplus first (free), then grid import
(costs `p_imp[t]`) when PV alone is insufficient.

This pre-deadline benefit is **mutually exclusive** with `charge_past_target`.
The LP construction guards the pre-deadline benefit block with
`and not ev.charge_past_target` (mirroring the post-deadline zero-charge and
target-cap constraints), so an EV in charge-past-target mode never receives the
large penalty-driven benefit. The LP enforces this exclusion directly — it does
not rely on caller discipline in `engine_core.py` to prevent both conditions
from being true simultaneously.

**Post-deadline slots** (`t > D`):
- When `charge_past_target=False`: `ev_c[t]` is hard-constrained to zero —
  no charging allowed after the deadline.
- When `charge_past_target=True`: `ev_c[t]` receives a tiny benefit of
  `-0.0001/η_charger` per kWh AC, but is constrained to PV surplus only
  (`ev_c[t]/η_charger ≤ pv[t] − base_load[t]`). The house battery charges
  first (benefit ~`p_imp`), then export at good prices (benefit `p_exp`),
  and only when both are saturated does the EV get the remaining surplus.

**Output**: the MILP writes EV decisions to `ev_planned_load_kwh`,
`ev_accounted_load_kwh`, and `ev_total_planned_load_kwh` on the output slots.
`estimated_net_consumption_kwh` and `estimated_cost_currency` are recomputed
to reflect the new EV loads.

#### Invariants

- When `ev_configs=None`, behaviour is identical to the pre-#530 code
  (backward compatible).
- EV charge per slot never exceeds `ev.max_charge_per_slot`.
- Cumulative EV SoC never exceeds `ev.capacity_kwh`.
- For EVs with a deadline and `charge_past_target=False`, cumulative
  pre-deadline charge `Σ_{k≤D} ev_c[k]` never exceeds `target_kwh − initial_soc_kwh`.
- When `ev.deadline_slot` is provided and the target is reachable, the
  deadline penalty `ev_pen` is zero.
- When the target is unreachable within the available slots, `ev_pen > 0`
  absorbs the shortfall — the MILP never becomes infeasible due to EV
  constraints.
- EV diagnostics (total DC kWh delivered, deadline penalty, deadline met)
  are included in the diagnostics dict under the `"ev"` key.

### MILP decision priority

The MILP solves a single global cost-minimization across all future slots
simultaneously.  It has no hard-coded priority order — the cost coefficients
in the objective function create a natural decision hierarchy.  Below is
how that plays out per slot, from cheapest to most expensive action.

**Objective** (minimise):

```text
Σ_t [ p_imp[t]·gi[t] − p_exp[t]·ge[t] + α·m[t]
      + (charge_loss·p_imp[t])·ec[t]
      + discharge_loss·(p_imp[t]·(ed[t]−bx[t]) + p_exp_pos[t]·bx[t])
      + p_soc·(s_max_pen[t] + s_min_pen[t]) ]
- V_primary(E_final)  # V_primary(E_initial) is a constant
+ ε·Σ_t(ec[t] + ed[t]) − 1.5ε·Σ_t(ed[t] − bx[t])
+ Σ_t δ_t·[(1−η_secondary_charge)·p_imp[t]·c_secondary[t]
           + (1−η_secondary_discharge)·p_imp[t]·d_secondary[t]
           + α_secondary·m_secondary[t]]
+ R_secondary·Σ_t(d_secondary[t] − c_secondary[t])
+ Σ_ev [ ev_penalty·ev_pen + tiebreaker·Σ_t ev_c[t] ]
```

The aggregate grid terms already contain the secondary branch's actual import
cost or avoided import. Secondary conversion loss and wear are discounted with
the other within-horizon money terms. Its terminal inventory coefficient is
uniform and undiscounted, matching the authoritative scorer.

#### 1. Serve house load from PV (free)

PV surplus `pv[t]` has **zero objective cost**.  Curtailment `curt[t]` also
has zero cost.  The LP always uses available PV to cover house load first.

#### 2. Use remaining PV surplus

| Priority | Action | Cost coefficient | When taken |
|---|---|---|---|
| 2a | Charge house battery | `charge_loss × p_imp[t]` | Battery below `usable_kwh`, future savings justify the minor conversion loss |
| 2b | Charge EV (pre-deadline, below target) | `-ev_penalty_cost` (benefit) + `p_imp[t]` (via grid) or `0` (via surplus) | EV below target, `t ≤ D` — the **deadline benefit** forces charging; PV used first, grid import when PV insufficient |
| 2c | Charge EV (post-deadline, past target) | **−0.0001 / charger_eff** (benefit) | `t > D`, `charge_past_target=True`. Surplus-only constraint: `ev_c/eff ≤ pv − base_load`. House battery fills first, then export, then EV gets remainder |
| 2d | Export to grid | **−p_exp[t]** (revenue) | Battery full, EV doesn't want surplus, export price > 0 |
| 2e | Curtail PV | `0` (free) | Battery full, EV doesn't want surplus, `p_exp ≤ 0` (export costs money or is blocked) |

#### 3. Cover house-load deficit

| Priority | Action | Cost coefficient | When taken |
|---|---|---|---|
| 3a | Discharge battery | `discharge_loss × p_imp[t] + cycle_cost` | Battery has energy, discharging is cheaper than grid import |
| 3b | Import from grid | `p_imp[t]` | Battery empty or discharge not worthwhile (cycle cost > import price spread) |

#### 4. EV deadline charging (hard penalty)

When the EV is **below target SoC** with a deadline approaching:

- Penalty: `max(p_imp) × max(energy_needed, 1.0) × 10` per kWh shortfall
- Constraint: `initial_soc + Σ ev_c + penalty ≥ target`
- **Pre-deadline benefit**: Each slot `t ≤ D` gets coefficient `-ev_penalty_cost`
  on `ev_c[t]`, so the LP always prefers charging over paying the penalty.
- This penalty dominates everything — the LP will import at high prices
  to meet the deadline when physically possible.

#### 5. Post-deadline behaviour

After the deadline slot `D`:

- **Normal mode** (`charge_past_target=False`): Hard constraint `ev_c[t] = 0`
  for all `t > D`. The EV receives zero energy allocation — charging is
  forbidden regardless of PV surplus or grid prices.
- **Charge-past-target mode** (`charge_past_target=True`): The EV may still
  charge, but only from genuine PV surplus that would otherwise be curtailed
  or exported at near-zero prices:
  - Surplus-only constraint: `ev_c[t]/η_charger ≤ max(0, pv[t] − base_load[t])`
  - Benefit: `-future_value_per_kwh/η_charger` per kWh AC (issue #630), where
    `future_value_per_kwh` is the avoided cost of importing the same energy
    later (`confidence_factor × mean(import_price)` over the next 24h — see
    `ev_future_charge_value_per_kwh` in `candidate_selector.py`). Falls back
    to a tiny fixed `0.0001/η_charger` tiebreaker when no future price data
    is available.
  - Because the benefit is priced in real currency terms, charge-past-target
    EV charging competes fairly against house battery charging (worth
    ~`p_imp` via avoided future import) and export (`p_exp`) — whichever has
    the higher genuine avoided-cost value wins the surplus for that slot.
  - Grid import is never used for post-deadline EV charging.

#### 6. Bounded terminal cost-to-go and primary-action structural tiebreak

At the end of the contiguous published-price prefix, primary-battery inventory
is valued **inside the MILP objective** by a bounded, piecewise-linear
`TerminalCostToGo`, not by one replacement-price coefficient. Let its tiers
`(q_i, v_i)` be ordered by decreasing positive marginal value, where both the
quantity `q_i` and inventory are battery-side DC kWh:

```text
V_primary(E) = Σ_i v_i × min(q_i, max(E - Σ_{j<i} q_j, 0))

primary_terminal_soc_value
    = V_primary(initial_battery_kwh)
      - V_primary(final_battery_kwh)
```

The MILP represents the final value with one allocation variable `y_i` per
tier:

```text
0 <= y_i <= q_i
Σ_i y_i <= final_battery_kwh
objective contribution = -Σ_i v_i × y_i
```

The omitted `V_primary(initial_battery_kwh)` is constant for candidate
selection and is restored in scorer and diagnostic accounting. Inventory above
`Σ_i q_i` receives no invented salvage value. The model is path-independent:
only final inventory determines `V_primary`, so an export discharge and equal
later refill cancel exactly. Actual import/export prices, conversion
efficiencies, cycle wear, available headroom, and per-slot power limits still
decide whether the cycle is worthwhile. There is no per-slot
`terminal_premium`, asymmetric charge credit, or deferred executable price
side channel.

Battery-origin export changes final inventory through `ed[t]` and therefore
crosses the same bounded tiers as local discharge. It cannot be valued as
though it displaced house import. If the optimizer can export and later refill
economically, equal battery movements restore the same final inventory and the
explicit refill cost/opportunity cost decides the result.

The bounded terminal term is paired with a tiny primary-action structural
tiebreak:

```text
ε = 0.00001 currency / DC kWh
primary_action_tiebreak
    = ε * Σ_t(ec[t] + ed[t])
      - 1.5ε * Σ_t(ed[t] - bx[t])
```

The exact export-source split makes `ed[t]-bx[t]` local battery discharge.
The resulting per-DC-kWh perturbation is `+ε` for charge, `-0.5ε` for local
discharge, and `+ε` for battery-origin export. A charge/local-discharge cycle
therefore adds `0.5ε` and an export/refill cycle adds `2ε` instead of
harvesting a tie benefit. On a true lossless/economic tie, the negative local
coefficient preserves the self-consumption preference required by issues
#638/#655; with real 97% discharge efficiency, a flat tariff is not an
economic tie and discharge is deliberately no longer forced.

This is a weighted structural tiebreak, not a mathematically lexicographic
objective. It can decide economics smaller than ε, and the configured 0.5%
MIP gap does not promise proof of an ε-sized distinction. At ordinary
48-hour/10 kW primary-battery bounds its total perturbation remains below
about 0.01 currency. The term is reported as
`primary_action_tiebreak`, contributes to selector `score`, and is excluded
from auditable `total_cost`.

**Preserved PV timing behaviour (issues #694/#592):** charging from PV reduces
the same slot's available `pv_export[t]`; a later refill consumes that later
slot's PV/headroom. Because equal charge/discharge cancels in the terminal
term, the optimizer compares the actual same-slot foregone export and future
refill opportunity directly. High-price PV can still be exported instead of
charged (#694), and an inevitable cheaper future surplus can still refill the
battery after expensive PV is exported (#592), subject to real capacity and
power constraints. These are solver outcomes, not terminal-premium caps.

Required behavioural regressions make the distinction concrete:

- One post-boundary 1.0 kWh demand tier can value at most 1.0 kWh, never the
  whole 25.5 kWh usable battery.
- When an official price moves the boundary forward, that newly actionable
  demand leaves the terminal tiers and is served according to its real price;
  only still-unpublished aligned demand remains reserved.
- With no valid tier, a full battery may serve every worthwhile actionable
  demand down to its effective hardware floor instead of remaining full for an
  imaginary tail.
- Equal battery-side discharge and refill restore the same final inventory and
  therefore contribute exactly zero primary terminal value, independent of
  slot order or export destination.
- A terminal tier cannot make its non-actionable source slot executable or
  bypass EV, no-export, or conditional export-reserve constraints.

##### Where primary terminal tiers come from

The primary `TerminalCostToGo` is built only from the opt-in Unagi import-price
forecast **strictly outside the published-price authority window**. The
boundary is the end of the contiguous future `price_actionable` prefix. A
forecast point is eligible only when its UTC start exactly matches a future,
non-actionable planner slot whose start is at or after that boundary. Published
overlap, past points, off-cadence points, and points without an aligned planner
load slot contribute nothing.

Each eligible Unagi price is reduced conservatively before valuation:

```text
effective_forecast_price
    = max(forecast_price - max(MAE, 0) - max(operator_margin, 0), 0)

eligible_house_load_ac
    = max(house_load - PV - accounted_EV_load, 0)

q_i = min(eligible_house_load_ac / discharge_efficiency,
          max_discharge_per_slot,
          usable_battery_kwh)

v_i = effective_forecast_price * discharge_efficiency
      - effective_forecast_price * (1 - discharge_efficiency)
      - cycle_wear_per_kwh
```

Only finite, positive `q_i` and `v_i` survive. Duplicate forecasts for one
physical slot collapse to the lower effective price. Tiers are sorted by
decreasing marginal value, with physical start as the deterministic tie-break,
and their combined quantity is capped by usable capacity. Invalid forecast
uncertainty or load/PV/EV inputs fail closed and create no tier.

This boundary rolls with publication. When an official price arrives, that
slot leaves the cost-to-go and is evaluated through its real in-horizon price;
the synthetic hold cannot migrate with the boundary and keep a newly published
peak uneconomically reserved. If no valid post-boundary tier remains, the model
is empty with source `hardware_floor_only`: terminal synthetic value is zero
for the primary battery, and the existing effective hardware discharge floor
is its only terminal reserve.

The forecast remains valuation-only. It never populates `PlannedSlot.price`,
extends `price_actionable`, creates import cost or export revenue, changes a
physical bound, or authorises a storage action beyond the boundary. The
isolation is structural rather than a flag every slot consumer must remember.
A confidence *interval* is deliberately not used: on the reference feed the
80 % lower bound proved flat across a whole day (0.001–0.021 SEK/kWh through an
evening peak whose point estimate was 0.752), carrying no useful shape.

Secondary storage retains its independent uniform terminal coefficient.
`resolve_secondary_terminal_price()` may combine its published and forecast
context with its dedicated-load mean-of-window rule; that scalar remains
undiscounted and never becomes a per-slot secondary premium.

Enabling forecast valuation requires a configured, existing valuation sensor;
a stale sensor option is ignored while the feature is disabled. The accepted
forecast-authority signature includes the enabled state, normalized MAE and
operator margin, and the sorted finite `(start, value)` forecast points. An
attribute publication, withdrawal, or recovery therefore enters the normal
debounced refresh path and must trigger a new planner run rather than reusing a
plan valued from stale forecast data. Parser-rejected points do not enter the
signature.

The terminal model is exposed without pretending its forecast value is money
already earned. Diagnostics publish `terminal_cost_to_go_source`
(`forecast` or `hardware_floor_only` in production;
`legacy_scalar` only for compatible direct callers), boundary, tier count,
total bounded quantity, and the ordered tier details. They also publish
initial/final inventory, initial/final valued quantity, and initial/final
inventory value;
`terminal_inventory_value` is the initial value minus the final value. MILP
and scorer consume the same model and export-source semantics. Published
diagnostics are recomputed from final reconciled slot fields, so they match the
authoritative scorer even when solver writeback has removed a degenerate raw
flow.

`PlanExplanation` carries the compact source, boundary, tier count/quantity,
highest/lowest marginal value, and initial/final valued quantity/value. Full
ordered tier details remain in MILP diagnostics so recorder-backed explanation
attributes stay bounded.

`terminal_soc_credit` remains a legacy diagnostic key carrying the same
signed value as `terminal_inventory_value`; despite its historical name, a
positive value is a penalty for net inventory loss.

#### Key constraint: EV surplus-only for charge-past-target

The constraint `ev_c[t]/charger_eff ≤ max(0, pv[t] − base_load[t])` ensures
past-target EV charging **never** draws from the battery or grid — only
genuine PV surplus that has nowhere else to go.

#### Charge-past-target benefit: avoided future import cost (issue #630)

The charge-past-target EV benefit (`EVConfig.future_value_per_kwh`) prices
one kWh of past-target EV charging at what it would otherwise cost to
import that same energy later:

```
future_value_per_kwh = confidence_factor × mean(import_price[t] for t in next 24h of slots)
```

- **24h lookahead**: always available even on the minimum-configured
  planning horizon (24h), long enough to smooth daily price cycles, short
  enough to avoid relying on degraded/missing day+2 forecasts.
- **`confidence_factor`** (default `0.9`, configurable per EV via
  `hsem_ev_past_target_confidence_factor` /
  `hsem_ev_second_past_target_confidence_factor`): discounts the estimate
  to account for the EV's future need being less certain than the house
  battery's scheduled discharge (depends on driving pattern, whether the EV
  stays plugged in, etc.).
- This remains an EV-specific scalar heuristic. Primary terminal inventory now
  uses the bounded post-boundary `TerminalCostToGo` tiers described above.

Because this benefit is priced in the same currency units as `p_imp` and
`p_exp`, the MILP lets charge-past-target EV charging compete fairly
against house battery charging and export — whichever has the higher
genuine avoided-cost value wins the surplus for that slot. When no future
price data is available (`future_value_per_kwh` is `None`, e.g. missing
forecast), the MILP falls back to a tiny fixed tiebreaker
(`0.0001`/kWh AC) so surplus PV still prefers the EV over being wastefully
curtailed/exported at near-zero or negative prices.

### Grid import power limit (main fuse / tariff protection)

When `main_fuse_amps` is provided and > 0, the MILP adds a **soft**
constraint on total grid import power per slot:

```text
max_grid_import_per_slot_kwh = main_fuse_amps * 230 * phases / 1000 * (interval_minutes / 60)
```

where ``phases`` is the electrical phase count (1 or 3, default 3).
This assumes balanced load at 230 V phase-to-neutral per phase.

**Penalty approach** (soft constraint):
- A penalty variable `gi_pen[t]` is added for each future slot.
- Constraint: `gi[t] - gi_pen[t] ≤ max_grid_import_per_slot_kwh`
- Penalty cost: `P_fuse * gi_pen[t]` where `P_fuse = max(p_imp) * 100`
  (same magnitude as existing SoC penalties).
- The solver only exceeds the fuse limit when physically unavoidable
  (e.g., house base load alone exceeds the fuse rating).

**Diagnostics**:
- `total_fuse_violation_kwh` in the returned diagnostics dict.
- `has_violations` set to `True` when any fuse violation exists.
- Each violating slot is logged at WARNING level with slot timestamp,
  required import, limit, and excess kWh.

**When disabled** (`main_fuse_amps` is `None` or 0): no constraint is
added — behaviour is identical to the pre-#567 code.

#### Optional hard per-phase charging protection

`hsem_phase_aware_charging_enabled` is an explicit opt-in for three-phase
installations. It requires signed Huawei grid-meter readings for phases A, B,
and C (positive import, negative export), the signed Huawei battery
charge/discharge-power sensor, and the writable Huawei grid-charge maximum-power
number. The aggregate soft constraint remains present; the phase model adds
three hard rows per future slot.

The latest meter snapshot is reduced to a zero-sum fixed-load imbalance after
removing the currently observed PowMr site delta. The same imbalance is projected
across the planning horizon. Let $G_t=gi_t-ge_t$ be total signed grid energy,
$D_t$ the PowMr site delta, $p$ its configured phase, and $\Delta_{i,t}$ the
fixed phase-imbalance energy. Planned phase flow is:

$$
F_{i,t}=\frac{G_t}{3}+\Delta_{i,t}
       +\left(\mathbf{1}_{i=p}-\frac{1}{3}\right)D_t
$$

This makes Huawei battery charge/discharge and Huawei PV balanced while placing
all PowMr charge, utility bypass, and SBU load removal on one phase. Each phase
has the hard target:

$$
F_{i,t}\leq\max\left(
  I_{fuse}\times230\times h_t/1000,
  B_{i,t}
\right)
$$

where $B_{i,t}$ is the uncontrollable baseline phase forecast. The `max` keeps
the MILP feasible if house load alone is already above the target, but permits no
controllable battery charging to worsen that baseline. A 16 A setting therefore
targets 16 A on every phase. Thermal fuse tolerance (for example a brief 25 A
overshoot) is not treated as schedulable capacity.

Immediately before hardware writes, a second guard uses the newest phase-meter
snapshot. It removes the currently observed Huawei/PowMr contributions,
constructs the desired utility/SBU baseline, and allocates charge headroom in
this order:

1. Huawei receives balanced three-phase headroom, limited by the least-free
   phase and rounded down to a 100 W command.
2. PowMr receives only the remaining headroom on its configured phase, rounded
   down to a supported 10 A step.

The Huawei grid-charge maximum-power number is written and verified before TOU
forced charging is enabled. If no full PowMr current step fits, PowMr is placed
in utility mode with grid charging disabled. Missing required telemetry enters
degraded mode and blocks all writes for that cycle. The runtime guard can correct
normal telemetry/control lag on the next coordinator cycle, but never budgets a
sustained overload.

#### Invariants

- When `main_fuse_amps` is `None` or 0, the MILP produces identical
  results to the pre-#567 code (backward compatible).
- When house load is within the fuse limit, `gi_pen[t]` is zero for all
  slots.
- When house load alone exceeds the fuse limit, `gi_pen[t] > 0` absorbs
  the excess — the MILP never becomes infeasible due to fuse constraints.
- When battery + EV + house load would exceed the fuse, the MILP
  throttles charging to stay within the limit.
- With phase-aware charging disabled, results remain identical to the aggregate
  fuse model.
- With phase-aware charging enabled, planned controllable charging never raises
  any phase above the configured target (or above an already-unavoidable
  baseline overload).
- Huawei is allocated live charge headroom before PowMr; PowMr is constrained to
  its configured single phase.

### Grid export power limit (DNO/inverter export cap — issue #726)

When `max_grid_export_power_kw` is provided and > 0, the MILP adds a
**hard** per-slot scheduling bound on grid export:

```
ge[t] <= max_grid_export_power_kw * slot_hours
```

- Implemented as a variable bound on `ge[t]`, not a penalty.
- Battery export and PV export compete for the same cap through
  `ge[t] = discharge_eff*bx[t] + pv_export[t]`, so the optimal plan front-loads battery export
  into low-PV slots and tapers it as PV ramps.
- PV that cannot be exported at the cap is absorbed by the free `curt[t]`
  curtailment variable.

The option is a MILP planning constraint, not a separate HSEM hardware command.
A passive solver fallback schedules no intentional battery export and reports
its natural PV flow without inventing a clipped forecast; the inverter/DNO
remains responsible for physical PV curtailment. A passive fallback slot may
therefore report PV export above the configured MILP cap without authorizing
intentional battery export.

**When disabled** (`max_grid_export_power_kw` is `None` or 0): `ge[t]`
remains unbounded above — behaviour is identical to the pre-#726 code.

#### Invariants

- When `max_grid_export_power_kw` is `None` or 0, the MILP produces
  identical results to the pre-#726 code (backward compatible).
- Every MILP candidate slot's `grid_export_kwh` is ≤
  `max_grid_export_power_kw × slot_hours` (within solver tolerance) when the
  cap is active.
- `primary_battery_export_kwh + pv_export_kwh == grid_export_kwh` within
  solver tolerance before publication and exactly at 0.001 kWh precision on
  every published MILP slot.
- The raw source split satisfies
  `bx[t] == min(ed[t], ge[t]/discharge_efficiency)`; a battery discharge that
  coincides with export cannot be relabelled as local use to receive the
  local-discharge side of `primary_action_tiebreak`.
- Primary `ec[t]` and `ed[t]` are never simultaneously positive because the
  always-present `primary_action_mode[t]` is binary.
- Grid `gi[t]` and `ge[t]` are never simultaneously positive because the
  always-present `grid_flow_mode[t]` is binary and uses finite physical
  per-slot import/export bounds.
- The battery never discharges purely to displace PV export at a saturated
  cap (export-destined discharge gains nothing once `ge[t]` is at its
  bound).
- A passive fallback contains no intentional battery-to-grid energy; natural
  PV flow is not fake-clamped in planner diagnostics.

## Cost function

The cost function returns **two distinct aggregates** for every plan
(issue #413):

- `total_cost` — the **money outcome** of the plan within the horizon.
  Pure monetary value.  Auditable; directly comparable to a real electricity bill.
- `score` — the **selector objective**.  Equals `total_cost` plus every
  synthetic penalty, the terminal-inventory opportunity cost, and the
  separately named primary-action structural tiebreak. The candidate selector
  picks the plan with the **lowest score** — not the lowest money cost.

```text
total_cost
= grid_import_cost
- export_revenue
+ battery_cycle_cost
+ conversion_loss_cost
```

```text
score
= total_cost
+ soc_guard_penalty
+ grid_limit_penalty
+ terminal_soc_value
+ primary_action_tiebreak
```

Where:

- `soc_guard_penalty` and `grid_limit_penalty` are **selector-only**
  synthetic terms. They must **never** appear in `total_cost`, because they
  do not represent real money paid or earned.
- `terminal_soc_value` is **selector-only** and is the sum of independent
  primary and secondary final-inventory terms. The primary contribution is
  bounded by valid post-boundary demand tiers; the secondary remains uniform.
  A component is negative (credit) when the plan ends with more *valued*
  stored energy than it started with and positive (penalty) when it ends with
  less.
- `primary_action_tiebreak` is **selector-only** and may have either sign. It
  is the same ε-weighted primary charge/discharge/export expression used by
  the MILP; it resolves structural ties without being real money.

The implementation exposes both numbers on `PlanCostBreakdown` together with
a deprecated `total` alias that equals `score` (kept so older code and tests
that compared plans by `.total` still select the same winner).

### Grid import cost

Grid import cost must use actual grid energy pulled.

If the battery stores `x` kWh from grid and charge efficiency is `e`, grid import is:

```text
grid_import_for_battery_kwh = x / e
```

Do not price stored energy as if it was grid energy.

### Export revenue

Export revenue is:

```text
grid_export_kwh * export_price_per_kwh
```

For MILP candidates, `grid_export_kwh` is source-conserved:

```text
primary_battery_export_kwh = discharge_eff * bx[t]
pv_export_kwh = pv_export[t]
grid_export_kwh = primary_battery_export_kwh + pv_export_kwh
```

The source fields are written to every future slot and serialised into planner
diagnostics. They are non-negative and sum to aggregate export exactly at the
published 0.001 kWh precision. `export_source_balance_max_error_kwh` reports
the maximum published per-slot residual. Production cost and conversion-loss
diagnostics use these fields directly; they do not reconstruct export origin
from recommendation labels, net-export status, or Solcast availability.

`score_plan()` retains a bounded compatibility path for older tests/callers
that construct aggregate-only slots. If the supplied source sum differs from
`grid_export_kwh` by more than 0.002 kWh, it caps the reconstructed primary
share at
`min(grid_export_kwh, batteries_discharged_kwh * discharge_efficiency)` and
assigns the remainder to PV. It never consults recommendation labels or
Solcast. No production MILP or passive candidate may rely on this fallback;
both publish explicit source fields before scoring.

When the export price is negative (curtailment penalty), ``export_revenue``
is negative — exporting costs money rather than earning it.  The
``total_cost`` formula ``import_cost − export_revenue`` correctly handles
this: subtracting a negative adds the cost.

**Export price clamping (``export_min_price``):**  When
``export_min_price > 0``, the inverter physically blocks all export for
slots where ``export_price < export_min_price`` (applier sets
``GRID_EXPORT_LIMIT_WATT``).  To keep the planner model consistent with
this physical behaviour:

- The MILP clamps ``export_price`` to 0 for all slots where
  ``export_price < export_min_price`` *before* solving the LP.
- The cost function (``score_plan``) applies the same clamping via
  ``CostWeights.export_min_price``.
- This clamping only affects the planner's decision-making; the raw slot
  ``export_price`` is preserved for diagnostics.

Negative export prices are **not** clamped — the LP's ``curt[t]``
variable (zero objective cost) naturally handles them: when
``p_exp < 0``, exporting costs money (``−p_exp·ge`` becomes a positive
cost) and the LP prefers curtailment (cost 0) over export (cost > 0).

Invariant: ``export_price < export_min_price`` → planner treats export
revenue as 0 in both optimisation and scoring.

**Battery export minimum price floor (``battery_export_min_price``, issue
#752):** When ``battery_export_min_price > 0`` and a slot's raw
``export_price`` is strictly below this floor, the MILP forbids
intentional battery-to-grid discharge in that slot by fixing ``bx[t] = 0``.
Normal discharge into eligible local load remains available through
``ed[t] - bx[t]``. To keep cost-function scores consistent with the
optimisation assumptions:

- ``CostWeights.battery_export_min_price`` mirrors the floor in
  ``score_plan``.
- When ``export_price < battery_export_min_price``, the solver fixes
  ``bx[t] = 0``, so ``primary_battery_export_kwh = 0``. The scorer
  uses that explicit attribution rather than inferring source from net flow
  or forecast PV.
- ``pv_export_kwh`` still receives full export revenue. The floor never
  restricts PV export.
- Above the floor the optimizer decides freely — reaching the threshold
  does NOT auto-trigger export.

Invariant: ``battery_export_min_price > 0`` AND ``export_price <
battery_export_min_price`` → ``primary_battery_export_kwh == 0`` while
``pv_export_kwh`` remains eligible for normal export revenue.

**Export-≤-import clamp (MILP unbounded-LP fix, issue #635):**
Before solving, the MILP also clamps ``export_price[t]`` to never
exceed ``import_price[t]`` for the same slot:

```text
export_price[t] = min(export_price[t], import_price[t])
```

Without this, slots where ``export_price > import_price`` create an
**unbounded LP** (HiGHS status=3).  ``gi[t]`` and ``ge[t]`` are both
``[0, ∞)`` and linked only through the energy-balance equality, so the
LP can drive both to infinity (import cheap, export expensive) while
the terms cancel.  A single such slot in the horizon causes
``solve_milp()`` to return ``None`` for the entire cycle.

This condition occurs whenever negative import spot prices coincide
with positive export tariffs (DK/DE/NL markets), or when asymmetric
import/export grid fees create an apparent price spread.  The clamp is
economically correct — no rational agent imports and exports
simultaneously for profit — and removes the unbounded direction
without changing any other optimisation behaviour.

This clamp is applied **after** the ``min_export_price`` clamp.

### Battery cycle cost

Cycle cost should count physical battery throughput.

**Single source of truth:** ``resolve_cycle_cost()`` in ``utils/misc.py``.

```text
battery_throughput_kwh = max(battery_charge_stored_kwh, battery_energy_removed_kwh)
cycle_cost_kwh = resolve_cycle_cost(
    purchase_price, usable_kwh, expected_cycles, capacity_loss_pct, user_margin
)
cycle_cost = battery_throughput_kwh * cycle_cost_kwh
```

Formula:

```text
auto = (purchase_price × capacity_loss_pct / 100) / (2 × usable_kwh × expected_cycles)
result = max(auto, user_margin)
```

The ``2×`` factor accounts for one full round-trip (charge + discharge).
``capacity_loss_pct`` accounts for residual value at EOL (LiFePO4 retains ~70 % at EOL,
so ~30 % is lost).

Avoid double-counting the same energy as both charge and discharge unless the cycle-cost definition explicitly expects throughput.

### Past-slot exclusion

The cost function must **skip** any slot whose recommendation is `time_passed`.

Past slots have `estimated_battery_soc = 0.0` as a sentinel value written by
the SoC simulator.  Including them in SoC-guard penalty calculations would
generate a false `soc_low_penalty` of `soc_low_penalty_weight × min_soc_pct²`
**per past slot**, added equally to every candidate plan.  Because the spurious
penalty is identical across all candidates it does not change the winner but
inflates the reported `total` cost and makes the logs misleading.

All other energy-flow fields (`grid_import_kwh`, `batteries_charged`, etc.) are
also zeroed on past slots by the simulator, so skipping them has no effect on
any cost term other than eliminating the bogus SoC penalty.

**Invariant for tests:**
```text
score_plan(slots_with_past).soc_penalty
== score_plan(future_only_slots).soc_penalty
```

### Terminal inventory value and primary-action structural tiebreak

Plans must not look better merely because they empty either battery before the
horizon ends. The scorer therefore uses the same primary bounded cost-to-go and
secondary uniform final-inventory term as the MILP:

```text
primary_terminal_soc_value =
    V_primary(initial_battery_kwh)
    - V_primary(final_battery_kwh)

secondary_terminal_soc_value =
    secondary_storage_replacement_price_per_kwh
    * (
        sum(secondary_storage_discharged_kwh)
        - sum(secondary_storage_charged_kwh)
      )

terminal_soc_value =
    primary_terminal_soc_value + secondary_terminal_soc_value
```

For the primary, `final_battery_kwh` is initial inventory plus actionable
charge minus actionable discharge; non-actionable storage is held. The bounded
`V_primary` is defined in the MILP objective section. The secondary component
remains its scalar replacement price multiplied by
`initial_battery_kwh - final_battery_kwh`. The aggregate contributes to
`score`, never `total_cost`. Equal net movement for either battery restores
the same final inventory and cancels exactly regardless of path.

The scorer also mirrors the MILP's primary-action structural tiebreak:

```text
epsilon = 0.00001
battery_export_dc[t] =
    primary_battery_export_kwh[t] / discharge_efficiency
local_discharge_dc[t] =
    batteries_discharged_kwh[t] - battery_export_dc[t]
primary_action_tiebreak =
    epsilon * sum(
        batteries_charged_kwh[t] + batteries_discharged_kwh[t]
    )
    - 1.5 * epsilon * sum(local_discharge_dc[t])
```

The field reproduces the MILP expression from final reconciled, three-decimal
slot/source fields. It contributes to `score`, never `total_cost`, and is
exposed separately from `terminal_soc_value`. A 1.0 kWh local discharge
contributes `-0.000005`; six-decimal public scoring therefore retains the
intended tie direction.
Production plans must use explicit `primary_battery_export_kwh`. Only the
bounded aggregate-only compatibility path described above may reconstruct a
source share, and it never consults recommendation labels or PV forecast.

Import prices are sanitised the same way as the MILP's own objective
(`imp_price_obj = max(imp_price, 0.0)`) before being used anywhere in
`score_plan` - including the import-cost term itself.  The charge-side
conversion loss term also uses `imp_price_obj`.  For the discharge-side
conversion loss, destination-aware pricing applies (issue #641):
`battery_export_dc` is priced at the sanitised export price, while
`local_discharge_dc` uses `imp_price_obj` — see Battery efficiency /
Conversion loss pricing above. The LP and scorer use the same explicit split.

Production primary terminal accounting is active when
`initial_battery_kwh` and `terminal_cost_to_go` are supplied to
`score_plan`. An empty `hardware_floor_only` model is still explicit and
contributes zero. The legacy scalar `replacement_price_per_kwh` path remains
only for compatible direct callers that do not pass a terminal model.
Secondary terminal accounting is independently active when secondary scoring
is enabled and `secondary_storage_replacement_price_per_kwh` is supplied in
`CostWeights`. The flow-based `primary_action_tiebreak` remains
independently defined.

### Invariants for tests

- `total_cost` must equal
  `import_cost - export_revenue + cycle_cost + conversion_loss_cost`
  exactly.  No synthetic penalty may enter `total_cost`.
- `score` must equal
  `total_cost + soc_penalty + grid_limit_penalty + terminal_soc_value + primary_action_tiebreak`
  exactly.
- When all penalties and selector-only inventory/tiebreak terms are zero,
  `score == total_cost`.
- The candidate selector must pick the candidate with the lowest `score`,
  not the lowest `total_cost`.
- `winner.score == output.plan_cost.score` for every planner run.
- `winner.slots == output.slots` for every planner run.
- Given two otherwise-identical primary plans whose final inventories differ
  inside an active tier, the one ending with more valued stored energy must
  have the lower `terminal_soc_value` and therefore the lower `score`.
- Primary inventory above total tier quantity has no synthetic salvage value;
  an empty `hardware_floor_only` model contributes exactly zero.
- Equal primary battery-side discharge and recharge must contribute exactly
  zero net `terminal_soc_value`, regardless of their slot positions.
- Equal secondary battery-side discharge and recharge must contribute exactly
  zero net `terminal_soc_value`, regardless of their slot positions.
- `primary_action_tiebreak` must equal
  `εΣ(ec+ed)-1.5εΣ(ed-bx)` using the same reconciled source split as the
  scorer and published diagnostics.
- Every MILP slot must satisfy
  `primary_battery_export_kwh + pv_export_kwh == grid_export_kwh` within
  solver tolerance before publication and exactly at 0.001 kWh published
  precision, with both source fields non-negative.
- (issue #752) When `battery_export_min_price > 0` and a slot's raw
  `export_price` is strictly below this floor, the MILP never schedules
  intentional battery-to-grid export on that slot —
  `primary_battery_export_kwh == 0`; `pv_export_kwh` may remain positive.
- (issue #752) A solver failure cannot reintroduce intentional battery
  export through a non-MILP heuristic path.
- (issue #752) With `battery_export_min_price = 0` (default) the
  planner produces identical results to the pre-#752 code (backward
  compatible).

## Price interval semantics

### Background

HSEM supports two price-data granularities depending on the configured EDS
(Energi Data Service) integration:

| `energi_data_service_update_interval` | Meaning |
|---|---|
| 15 | EDS publishes one price record every 15 minutes |
| 60 | EDS publishes one price record per hour |

The planning slot width is controlled separately by
`recommendation_interval_minutes` (also 15 or 60).

Electricity prices are **rates** (currency per kWh), not energy quantities.
Every slot inside the same EDS update interval shares the same price; the
price is **never summed or averaged** across slots.

### The eds_share conversion factor

When EDS and slot widths differ (most common case: EDS 60 min, slots 15 min),
a conversion factor is needed so internal per-slot storage and the planner
engine both see correct values:

```text
eds_share = energi_data_service_update_interval / recommendation_interval_minutes
```

Common configurations:

| EDS interval | Slot width | eds_share | Effect |
|---|---|---|---|
| 60 min | 15 min | 4.0 | price÷4 stored; planner gets price×4 back |
| 15 min | 15 min | 1.0 | no scaling — price stored and used unchanged |
| 60 min | 60 min | 1.0 | no scaling — price stored and used unchanged |

### How the scaling pipeline works

1. **Population** (`hourly_data_populator._async_update_hourly_field`):
   Each raw EDS value is divided by `eds_share` before writing to the
   per-slot `HourlyRecommendation` object.
   This gives each slot its proportional share of the price-rate value so
   slot boundaries align correctly.
   Data-point timestamps are floored to the start of their enclosing *source*
   interval (the EDS update interval for prices, 60 min for Solcast), never
   to the hour: a 15-min price point covers only the slots whose start lies
   inside that 15-min window, so quarter-hourly prices land on distinct
   slots (issue #720).  With hourly price data and 15-min slots the single
   hourly point fans out to all four quarter-hour slots of the hour.

2. **Planner input** (`coordinator_builder.build_planner_input`):
   Recommendation slots are deduplicated on `(day_offset, hour)` for
   consumption averages and Solcast PV (genuinely hour-granular), but
   **price points are emitted per slot** with an explicit `slot_in_day`
   field, so quarter-hourly prices survive as distinct `PricePoint`
   entries (192 for a 48 h horizon at 15-min slots).  Each stored per-slot
   price is multiplied by `eds_share` to recover the original
   hourly-equivalent rate.  The planner's cost function always works with
   full currency/kWh rates, not fractions.

3. **Slot population** (`planner.slot_population.populate_prices`):
   When price points carry `slot_in_day`, slots are keyed by
   `(day_offset, slot_in_day)` so each quarter-hourly price lands on its
   own planner slot; slots the source does not cover fall back to the
   hourly value.  Points without `slot_in_day` (legacy hourly callers)
   use the existing `align_hourly_prices` fan-out unchanged.

The divide and multiply are exact inverses — they cancel perfectly and the
planner always receives the original price rate regardless of configuration.

### What this is NOT

- `eds_share` is **not** a VAT multiplier.
- `eds_share` is **not** a currency conversion.
- `eds_share` is **not** an energy-splitting factor (prices are rates, not energy).

### Invariants for tests

- A 60-min EDS price of `P` must reach the planner as `P` (not `P/4` or `P*4`).
- A 15-min EDS price of `P` must reach the planner as `P`.
- Intermediate per-slot stored values must equal `P / eds_share`.
- Changing `energi_data_service_update_interval` from 60 to 15 with the same
  price input must not change the price seen by the planner engine.
- Negative prices must survive the full pipeline unchanged.
- With 15-min price data and 15-min slots, each quarter-hour price must land
  on exactly its own slot — four distinct prices within an hour must produce
  four distinct slot prices (issue #720).
- With 15-min price data, 15-min slots, and a 48 h horizon, the planner must
  receive 192 distinct price points (not 48 collapsed hourly ones) and the
  MILP must see intra-hour price variation (issue #720 stage 2).

## Candidate plans

Every candidate plan must be fully simulated and scored.

Rejected-plan diagnostics keep the two cost aggregates distinct:
``estimated_cost`` is the auditable monetary ``total_cost``, while ``score``
is the selector objective including synthetic penalties, terminal inventory
value, and the primary-action structural tiebreak.

Required candidates on every run:

- `no_action`: a fully simulated and scored diagnostic comparator that is
  never eligible to win;
- `passive`: solar absorption and normal self-consumption, with no grid charge
  or intentional battery export; this is the sole executable solver-failure
  fallback;
- `milp`: added only after an optimal solve or a fully validated time-limit
  incumbent.

The retired baseline/grid-charge/discharge/export/aggressive heuristic matrix is
not generated. In particular, a failed solver must never restore the
pre-solve heuristic baseline, because that baseline has not passed the MILP's
fuse, phase, export, EV, and secondary-storage constraints. The MILP is
the only candidate allowed to introduce actively optimized grid charge or
intentional battery export.

The selected plan must have the lowest score among valid, eligible executable
candidates within this implemented search space. `no_action` remains diagnostic
even when its score is lower.

The final returned plan must be the same plan that was selected.

Planner warnings have the same ownership rule. Global input-quality and solver
diagnostics remain visible for every run, but recommendation-specific warnings
created while building a candidate are surfaced only if that candidate wins.
The retired `apply_excess_export` pass can no longer add heuristic export
labels or warnings. The selected MILP's explicit flows own every actively
optimised primary-battery decision; intentional battery export remains owned
by the selected MILP candidate.

This invariant must always hold:

```text
output.plan_cost == selected_candidate.cost
output.slots == selected_candidate.slots
```

No post-selection pass may mutate slots unless the plan is re-simulated and re-scored.

### Truthful plan-explanation semantics

Plan explanations describe only energy actions present in the final selected
slots. A plan with grid charge but no scheduled discharge is
`opportunistic_charge`; its summary must state that no discharge window is
scheduled. If that plan costs more than the idle comparator inside the current
actionable window, the rejected `do_nothing` reason may report the measured
charging overhead or retained post-boundary inventory, but must not promise
that discharge savings occur inside a window containing no discharge.

`estimated_total_cost` and the idle comparator use only actionable slot money.
The explanation's legacy `score` is their signed difference (positive means
the selected plan is cheaper), not the candidate selector's lower-is-better
`PlanCostBreakdown.score`. Terminal cost-to-go may explain why retaining energy
changes selection, but forecast value must never be presented as realised
within-window revenue or savings.

### Plan-level hysteresis (anti-flapping, issue #372)

The selector may optionally apply **plan-level hysteresis** to avoid switching
strategies for tiny cost improvements.  When hysteresis is active, the
previously active plan (identified by candidate name) is re-evaluated with
current data.  If its score improvement over the best new candidate is below
both configured thresholds, the previous plan is kept.

Two thresholds are supported, evaluated in order:

1. **Absolute threshold** (currency): the new plan's score must be lower
   (better) by at least this amount.  ``0.0`` disables the check.
2. **Percentage threshold** (relative): the new plan's score must be lower
   by at least this percentage of the previous plan's score.  ``0.0`` disables
   the check.

If the previous plan's candidate is not found in the current candidate set
(e.g. because the underlying strategy no longer applies), hysteresis falls
back to normal selection.

The hysteresis decision is surfaced in
:attr:`PlanExplanation.hysteresis_active`,
:attr:`PlanExplanation.hysteresis_reason`, and
:attr:`PlanExplanation.previous_plan_name`.

The previous winner's name and score are persisted across planner runs by the
coordinator and passed as part of :class:`PlannerInput`.

Hysteresis is enabled by default with a 5 % percentage threshold; setting
``planner_hysteresis_enabled = False`` disables it entirely.

### Window-level hysteresis (anti-flapping, issue #315)

In addition to plan-level hysteresis, HSEM applies **window-level hysteresis**
on the **current time slot** to suppress display-label flapping only when the
two labels produce the same hardware command.

Only ``batteries_charge_solar`` ↔ unrestricted
``batteries_discharge_mode`` is held: both execute maximise-self-consumption
with the normal discharge cap. Grid charge, EV control, Fully Fed modes,
neutral states, non-actionable prices, and explicit MILP holds pass through
immediately. A BDM slot with material battery discharge and grid import is a
partial allocation with a plan-derived cap, so it also bypasses hysteresis.

The hold time is configured by ``planner_window_hysteresis_minutes``
(default: 10).  When set to a positive integer, only a command-equivalent MSC
alias change on the current slot is suppressed until the previous label has
been active for at least this many minutes.

The previous recommendation and its activation timestamp are persisted across
planner runs so elapsed time is measured from the moment the mode was accepted.
Every accepted in-slot switch starts a fresh full hold period. Elapsed and
expiry arithmetic uses the UTC timeline, so the repeated autumn DST hour cannot
extend a hold by an extra hour.

Window-level hysteresis is applied **after** the planner engine completes but
**before** the current slot recommendation is resolved.  The held
recommendation is written back into the planner output slots so it propagates
to the ``hourly_recommendations`` list and ultimately to hardware writes.

When a transition is held, the coordinator schedules a one-shot callback at
the exact hold expiry and forces a fresh planner run. The expiry does not depend
on a drifting periodic poll, and it waits for an in-progress coordinator cycle
instead of being dropped.

Independently, a one-shot callback at every exact recommendation-slot boundary
queues the new slot through the same durable 250 ms refresh window as price,
PV, and valuation-source events. A quiet boundary still produces exactly one
cycle. Import/export updates arriving before the shared worker starts join that
pending cycle before its snapshot and solver work, avoiding a stale first solve
and duplicate replacement solve in the ordinary boundary burst. If another
event opened the window first, the boundary uses the remaining delay rather
than restarting it. The boundary also advances the authority generation, so a
pre-boundary cycle already in flight cannot publish old-slot intent after the
new slot starts. A genuinely later source event during the shared solve retains
the existing stale-generation rejection and follow-up cycle.

Boundary discovery advances on the UTC timeline while testing
local wall-clock alignment, so DST folds never schedule a callback in the past.
Both callbacks are cancelled during teardown; a degraded or
failed expiry cycle leaves the forced-replan request pending until recovery.

### Invariants for window-level hysteresis tests

- First run (no previous state) always accepts the new recommendation.
- Only command-equivalent ``batteries_charge_solar`` ↔ unrestricted
  ``batteries_discharge_mode`` changes may keep the previous label.
- Every hardware-semantic transition and partial BDM allocation is immediate.
- Changes after the hold time expires switch to the new recommendation.
- Neutral recommendations never trigger hold behaviour.
- Feature disabled (hold minutes = 0) always allows the switch.

## No-action diagnostic

The no-action plan means:

- no forced grid charge
- no intentional battery discharge or battery export
- normal battery self-consumption and PV absorption when physically modelled
- a PV-only `force_export` label may hold the battery and route available PV

It must still account for:

- PV charging battery if that is normal inverter behavior
- PV export
- house load
- battery self-consumption behavior if modeled
- terminal SoC

No-action must not be treated as “zero battery movement” unless the physical
model says no battery movement occurs. It is always simulated and scored for
diagnostics, but it is excluded from winner selection and is not a fallback.

## Safety gates

The planner may compute in read-only or degraded states.

The applier must not write to hardware when:

- read-only mode is enabled
- dry-run mode is enabled
- degraded mode blocks writes
- error mode is active
- required data is missing
- config entry is unloading

Hardware writes are serialized by one worker. Routine coordinator notifications
with the same effective command intent are coalesced while verification is in
flight. The fingerprint contains the quantized inverter export target, phase-
limited Huawei charge command, ordered PowMr write plan and safety guards,
battery/EV cap inputs, entity targets, slot flow, reserve, and mode. A material
safe change never cancels a multi-write transaction. The worker completes the
active sequence, discards snapshots collected against partially changed
hardware, and requests a coordinator refresh. It drains another snapshot only
when a successful listener generation was published after that refresh;
otherwise it stops without retrying stale work, and a later external update
recovers normally. Read-only, Error-mode, and unload gates may still cancel
immediately. Numeric inputs are compared in command-relevant integer units, so
sub-Wh or sub-command floating-point noise cannot restart a write or prevent a
matching apply summary from being published.

The listener resolves live recommendation overrides on a detached sensor-local
snapshot before calculating that fingerprint or queueing work. Neither the
accepted planner output nor a snapshot already owned by the worker may be
mutated by a later coordinator notification.

### EV discharge cap semantics (issue #592)

When an EV is actively charging and the current recommendation is not a
forced-discharge/export mode, the applier caps the inverter's
`maximum_discharging_power` so the battery covers only house load while
100 % of the EV load goes to the grid.  The cap is computed by
`applier.compute_ev_discharge_cap_w()`:

- **History available** → the cap IS the historical house baseline (the
  current slot's weighted average).  The live `net_consumption_w` reading
  is deliberately ignored: downward it ratchets the cap toward zero when
  the CT clamp and the EV sensor disagree (the battery's own capped
  discharge shrinks the CT reading further — a self-poisoning input);
  upward it swings with ordinary house noise and drains the battery into
  what is supposed to be a grid-served EV session.
- **No EV power sensor** → the minimum positive sub-window average
  (1d/3d/7d/14d/weighted).
- **No history** (fresh install) → the live reading.

**SoC guard:** when the battery's remaining usable energy is at or below
the planner's required reserve (`current_required_battery_kwh` — energy
needed until the next solar surplus), the cap is forced to 0 W so the
battery is preserved for its scheduled plans.

## Invariants for tests

Add tests for these invariants:

- Energy balance holds for every slot.
- SoC never leaves configured bounds.
- Forced discharge changes SoC and cost.
- `force_export` holds battery SoC and exports only available PV;
  `force_batteries_discharge` changes SoC and may create export revenue.
- Grid charge prices actual grid import, not stored energy.
- Candidate winner cost equals final output cost.
- Final output slots equal selected candidate slots.
- No post-selection mutation happens without re-score.
- No-action includes normal PV/battery behavior.
- Valid terminal inventory value affects cost; an empty primary
  `hardware_floor_only` model contributes zero.
- Consuming primary valued-tier inventory or uniformly valued secondary
  inventory is not free; primary inventory outside the tiers has no synthetic
  value.
- `no_action` is never selected, even when its diagnostic score is below the
  executable candidates.
- Current partial slot uses remaining duration only.
- Missing price/PV data does not become real zero silently.
- Read-only/degraded/dry-run gates block writes.
- Hysteresis keeps the previous plan when improvement is below absolute threshold.
- Hysteresis keeps the previous plan when improvement is below percentage threshold.
- Hysteresis switches to the new plan when improvement exceeds both thresholds.
- Hysteresis is inactive on the first planner run (no previous plan).
- Hysteresis falls back to normal selection when the previous plan name is not found.
- Hysteresis is inactive when the feature is disabled.
- `PlanExplanation.hysteresis_active` reflects the hysteresis decision.
- `PlanExplanation.hysteresis_reason` describes why hysteresis kept or released the plan.

## Multi-day planning horizon

The planner supports configurable planning horizons: 24, 48, and 72 hours.

The horizon is controlled by `interval_length_hours` in `PlannerInput` (and
`recommendation_interval_length` in `SensorConfig`).  All three values are
accepted without special-casing in the engine.

### Slot count

```text
total_slots = (interval_length_hours * 60) // interval_minutes
```

| Horizon | 15-min slots | 60-min slots |
|---|---|---|
| 24 h | 96 | 24 |
| 48 h | 192 | 48 |
| 72 h | 288 | 72 |

### Confidence decay for future days

Price and PV forecast accuracy degrades for days further in the future.
To avoid over-committing to uncertain future plans, the planner applies a
**confidence decay factor** to PV estimates (not prices) for slots on
day+1 and beyond:

| Day offset | Decay factor | Meaning |
|---|---|---|
| 0 (today) | 1.00 | No decay — current-day forecast |
| 1 (tomorrow) | 0.90 | 10 % conservative discount |
| 2 (day after) | 0.80 | 20 % conservative discount |

Only PV estimates are discounted.  Electricity prices are used as-is because:
- Spot-market prices are typically known for day+1 by mid-day.
- Discounting known prices would distort the cost function.

Decay is applied **after** missing-data diagnostics, so `DataQuality` always
reflects original data gaps, not decayed values.

In addition to the fixed daily decay, the
:class:`~custom_components.hsem.utils.solar_corrector.SolarForecastCorrector`
(introduced in issue #602) applies learned **per-hour accuracy factors** and
an **intra-hour residual correction** to PV estimates before they enter the
planner. For each local clock-hour, the corrector retains the four most recent
eligible physical quarter-slot **actual / raw forecast** ratios. At 15-minute
cadence this is a four-sample window, not four days; the learned mean is clamped
to [0.3, 1.5]. The internal confidence value dampens learned factors toward
neutral below 0.50 and applies them at full strength from 0.50 upward. It is
restored internal state, not a configurable entity. The raw Solcast data is
never mutated; corrections are only applied at slot-population time.

#### Solar correction invariant

The `SolarForecastCorrector` applies two multiplicative corrections to each
raw PV estimate before it enters the planner:

```text
corrected_pv = raw_pv × hour_factor × residual_factor
```

Where:
- `hour_factor ∈ [0.3, 1.5]` — the mean eligible actual/raw ratio for the
  slot's local wall hour, clamped to limit outliers
- `residual_factor` — the mean of recent eligible closed-slot
  actual/corrected ratios, decaying linearly toward 1.0 over the next eight
  physical slots (two hours at 15-minute cadence)

The clamping is symmetric (0.3 lower, 1.5 upper) so the corrector never
amplifies a single outlier beyond these bounds.  Raw Solcast data is never
mutated; both factors are applied only at consumption time.

### Missing future data handling

#### Published price-source precedence

HSEM may be configured with a dedicated pair of ENTSO-E **published-price
backup** sensors in addition to the primary import/export sensors and the
legacy per-channel forecast inputs. The ENTSO-E entities are integration
boundaries: they must already expose final prices in the same local-currency
per-kWh unit and with the same tariff/VAT basis as their corresponding primary
sensors. HSEM never applies VAT, tariffs, foreign-exchange conversion, or
energy-unit scaling to this backup.

Source selection is atomic per slot and follows this order:

1. When both primary channels are finite and available, the primary pair is
   authoritative. Genuine zero and negative prices remain valid.
2. When either primary channel is missing, both channels may switch to ENTSO-E
   for that slot, but only when the import and export backup arrays provide an
   exact, aligned, complete local delivery day at the configured cadence.
   Completeness is defined on the UTC timeline between consecutive local
   midnights, so a 15-minute DST day contains 92, 96, or 100 physical points.
   A partial, duplicated, conflicting, naive, off-cadence, non-finite, or
   wrong-unit backup is rejected as a pair.
3. The existing optional forecast sensors run last and retain their legacy
   independent missing-channel behaviour. They cannot overwrite a primary or
   accepted ENTSO-E value.

Accepted ENTSO-E values are redundant publications of executable day-ahead
prices, not the opt-in valuation forecast described under
`TerminalCostToGo`. They populate `PlannedSlot.price` and can extend the
price-actionable prefix. Source labels (`primary`, `entsoe`, or `forecast`)
are carried with both channels through `HourlyRecommendation`, `PricePoint`,
`PlannedSlot`, the plan-reuse signature, and diagnostics. A source-only
change therefore cannot leave stale provenance on a reused plan.

This backup is deliberately scoped to the publication-gap case where the
primary live entities remain healthy while a future array is missing. It does
not override the current-slot live-price outage gate: if a primary scalar
entity is unavailable, automatic control still enters the strict price-outage
hold described below.

### Load-forecast readiness

Historical-average states preserve their availability provenance: `unknown`,
`unavailable`, unparseable, and non-finite values remain missing instead of
becoming numeric zero. A genuine finite `0.0` remains a valid measurement.

Before candidate generation, the coordinator validates the populated future
load profile. Missing or non-finite source averages and malformed future-slot
values fail closed. A complete identically-zero profile remains valid while
finite live house demand is at most 50 W. Above 50 W it contradicts live
telemetry and reports
`zero_forecast_with_live_demand`.

When the profile is not ready, the coordinator does not run or reuse an
optimised plan. In automatic mode it publishes a current-slot
`batteries_wait_mode` with `primary_battery_hold=True`, zero primary storage
flows, and secondary Utility/zero current. An explicit user force mode remains
higher authority. The coordinator retries at the one-minute pending-data
interval.

The accepted-plan load signature contains each future slot's start plus its
weighted, 1-day, 3-day, 7-day, and 14-day finite load values. Recovery or a
material load correction therefore forces one fresh solve even within the same
slot. Only an accepted published plan advances the reuse baseline, preventing a
rejected or skipped cycle from suppressing that recovery solve.

`DataQuality.load_forecast_ready` and
`load_forecast_reason` expose the gate; `DataQuality.is_complete` is
false while availability is false. This gate does not
change published-price authority, Unagi forecast parsing or haircuts, primary
terminal-tier construction, or terminal cost-to-go scoring; those semantics
resume unchanged once the load forecast is ready.

For every day in the horizon the engine detects and surfaces missing price
and PV data explicitly.  Day-labelled `missing_inputs` entries are emitted
with the format:

```text
tomorrow_price_missing_hours:HH,HH,...
tomorrow_pv_missing_hours:HH,HH,...
day2_price_missing_hours:HH,HH,...
day2_pv_missing_hours:HH,HH,...
```

These labels are data-quality diagnostics, not live missing-entity labels, and
do not by themselves change degraded mode. Price incompleteness is handled by
the explicit authority boundary below, while missing critical battery/house
telemetry continues to block writes through `DegradedMode.Error`.

Numeric missing-price fields remain `0.0` for display/backward compatibility,
but availability is explicit at every boundary (`HourlyRecommendation`,
`PricePoint`, and `PlannedSlot`). A published `0.0` or negative price is valid
and actionable; an unpublished or non-finite (`NaN`/infinite) value is not.

The engine retains the full display horizon and derives a **contiguous
price-actionable future prefix**. The prefix closes at the first slot where
either import or export price is unavailable and cannot reopen later in the
horizon. `DataQuality.price_actionable_slots` and
`DataQuality.price_actionable_until` surface the boundary.

Beyond that boundary:

- Huawei primary storage is held: charge and discharge are both zero. Grid
  import and PV export/curtailment remain physical site-flow outputs, not
  price-driven storage actions. Therefore `bx[t] = 0`,
  `primary_battery_export_kwh = 0`, and any `grid_export_kwh` is carried
  by `pv_export_kwh`.
- PowMr charge and SBU modes are forbidden (Utility only).
- Optional EV arbitrage/deadline allocation is forbidden. A fixed live EV
  session remains a mandatory physical load, and unmet target energy is
  surfaced rather than priced as free.
- Primary terminal tiers may read aligned Unagi points outside the prefix, at
  or after this boundary instant, as valuation-only context. Secondary storage
  retains its separate terminal-price rule. Neither mechanism makes a
  non-actionable slot executable.

An opt-in forecast-derived terminal value does not relax this boundary. It can
value final inventory for actionable decisions, but it cannot make a
non-actionable slot charge, discharge, export battery energy, or change
inverter mode.

Heuristic fallback candidates enforce the same price-neutral hold contract, so a
solver timeout cannot reintroduce actions selected from placeholder zeros.
Window hysteresis cannot restore an older price-driven label onto a
non-actionable current slot. Auto-Full requires both an actionable current slot
and an available live import price. If the live export price is unavailable,
the applier retains the last verified inverter export limit instead of treating
the display placeholder zero as authority for a curtailment write.

For the current slot, populated forecast-channel authority is intersected with
the live price entities: stale attributes cannot remain actionable after the
HA entity becomes unavailable, and live data cannot promote a missing forecast
channel. In automatic mode, a current outage immediately publishes
`batteries_wait_mode` with `primary_battery_hold=True`, zero primary storage
flows, and secondary Utility/zero current, even if another missing input skips
the planner. An explicit user force mode remains higher authority. Primary and
dedicated forecast price entity changes, plus both Solcast PV sources, are
debounced through the same durable queue as the exact recommendation-slot
boundary. Boundary-adjacent source events are absorbed before the queued cycle
starts; an event arriving after that cycle starts is replayed. The worker
consumes pending events only after acquiring the coordinator lock, so updates
that accumulated while another cycle held the lock do not force an unnecessary
second solve. The accepted forecast-authority signature includes each future
slot's finite PV value and availability, so PV publication, withdrawal, or
correction cannot silently reuse a plan built from stale solar data.

Every exact slot boundary and every registered price, PV, or valuation-source
event also increments a monotonic forecast-authority generation synchronously,
before debounce coalescing. An update cycle captures that generation before
collecting its snapshot. If it changes while the planner is solving, or after
asynchronous trackers run but before coordinator publication, the entire stale
cycle is discarded: it publishes no coordinator data, advances no accepted-plan
signature, and produces no hardware intent. The durable debounce worker then
runs one fresh coalesced cycle; an event arriving during that refresh marks one
further pass pending. If the superseded pass raises after that newer event, the
worker logs the failure and still runs the pending fresh pass. Without newer
authority pending, the exception remains visible to the task owner.

This freshness guard changes only which snapshot may publish. It does not alter
the actionable-price boundary, Unagi valuation inputs, forecast haircuts,
terminal tiers, or terminal cost-to-go economics.

### DataQuality fields

`DataQuality.horizon_days` counts the distinct local calendar dates touched by
the physical-time horizon. Ordinary midnight-anchored 24/48/72-hour horizons
cover 1/2/3 dates; across a spring-forward transition the same duration can
touch one extra local date.
`DataQuality.day2_price_missing_hours` and `DataQuality.day2_pv_missing_hours`
carry the day+2 gap lists for 72-hour horizon runs.

`DataQuality.load_forecast_ready` is false when consumption
provenance or the populated future profile cannot safely support a solve.
`DataQuality.load_forecast_reason` carries the machine-readable cause
and is `None` when available. Consumption availability participates in
`DataQuality.is_complete` alongside price and PV completeness.

### Discharge concentration across days

``concentrate_discharge_on_expensive_slots`` clears the cheapest
discharge slots when the battery cannot cover all of them.  This
pre-processing step runs before the SoC simulation and ensures the
battery is reserved for the most expensive slots.

The function groups discharge slots by **calendar day** and gives each
day its own independent ``usable_kwh`` budget.  This correctly accounts
for the fact that the battery can be recharged by solar (or cheap grid
hours) between groups of discharge slots on different days. Without per-day
budgets, slots on day N+1 would compete with slots on day N for the same
capacity pool even when the battery is recharged between them.

Within each day the estimate is conservative: it assumes the battery
starts at full capacity and there is no incoming charge between
discharge slots on the same day.

### Invariants for multi-day horizon tests

- A 24-hour horizon produces exactly `(24 * 60) // interval_minutes` slots.
- A 48-hour horizon produces exactly `(48 * 60) // interval_minutes` slots.
- A 72-hour horizon produces exactly `(72 * 60) // interval_minutes` slots.
- All slots have a non-``None`` recommendation regardless of horizon.
- Day+1 PV estimates are ≤ day+0 estimates for the same hour when both have
  the same raw input (confidence decay applied).
- Day+2 PV estimates are ≤ day+1 estimates for the same raw input.
- On ordinary dates, `DataQuality.horizon_days` equals 1 / 2 / 3 for 24 h /
  48 h / 72 h. A spring-forward physical horizon can touch one extra local
  date and therefore report 2 / 3 / 4 instead.
- Missing day+2 price data surfaces in `day2_price_missing_hours`.
- Missing day+2 PV data surfaces in `day2_pv_missing_hours`.
- `DataQuality.is_complete` is ``False`` when future-day data is missing or
  `load_forecast_ready` is false.
- PV estimate after solar correction is always within `[0.3 × raw_pv, 1.5 × raw_pv]`
  for each hour (clamping enforced).
- The residual correction decays to ≤0.05× the initial deviation after 4 slots.

### Dynamic discharge floor

The dynamic discharge floor computes a per-cycle minimum SoC that bridges the
gap between the last discharge slot and the next solar refill window:

```text
effective_floor_pct = max(configured_min_soc_pct, bridge_reserve_pct)
bridge_reserve_pct  = (next_refill_need_kwh / usable_capacity_kwh) × 100
                    × safety_margin
```

Where `safety_margin` is a self-learning multiplier that starts at **1.50**
and decays toward **1.05** as successful solar refills are observed.  The
floor is never lower than the hardware-configured minimum SoC.

#### Dynamic floor invariant

```text
effective_floor_pct ≥ configured_min_soc_pct    (always)
effective_floor_pct ≤ 1.50 × bridge_reserve_raw  (after learning period)
```

### Session EV invariant

When an active charging session is detected (`session_charge_kw > 0`), the
MILP treats the next 2 hours as **fixed EV demand** with enforced lower
bounds.  The number of slots covered is derived from the configured slot
interval: `round(2 / slot_hours)`, which yields 8 slots at 15-minute
resolution, 4 slots at 30-minute resolution, and 2 slots at 60-minute
resolution.

```text
For t = 1 … SESSION_SLOTS (first 2 hours of future slots):
    ev_c[t] ≥ min(session_charge_kw × slot_duration_hours, ev_max_charge_per_slot)
```

These bound constraints prevent the MILP from re-allocating demand away from
a live charging session.  Slots beyond the 2-hour window are unconstrained
and optimised freely.  When `session_charge_kw == 0` (no active session),
no bounds are applied and the entire EV demand is MILP-determined.

## EV planned load integration

`base_load_includes_ev` is automatically derived from the
`hsem_house_power_includes_ev_charger_power` setting in the EV charger config step.
When the house consumption sensor includes EV charger power, `base_load_includes_ev`
is `True` (EV load is already in the base consumption averages). Otherwise it is `False`.
There is no separate user-facing configuration for this field.

### EV load field semantics

Three per-slot fields capture EV load intent precisely:

| Field | Meaning |
|---|---|
| `ev_planned_load_kwh` | Extra EV AC load **added to net consumption** — only the portion not already in `avg_house_consumption`. Zero when `base_load_includes_ev = True`. |
| `ev_accounted_load_kwh` | EV AC load **already included** in the house consumption sensor. Non-zero when `base_load_includes_ev = True`. Must not be added to net consumption again. |
| `ev_total_planned_load_kwh` | Total planned EV AC load regardless of accounting mode: `ev_planned_load_kwh + ev_accounted_load_kwh`. Always non-zero when any EV charging is planned. |
| `ev_charger_calculated_power` | Target AC power (W) for the primary EV charger during this slot. Computed from the EV planner's per-slot energy target: `round((ac_load_kwh / slot_duration_hours) × 1000)`. For the **current** (partially elapsed) slot, `slot_duration_hours` is the remaining time (minimum 1 s), because the EV planner already scales `ac_load_kwh` to the remaining minutes. For future slots the full slot width is used. Zero when no charging is planned. |
| `ev_second_charger_calculated_power` | Same as above, for the second EV. |

When `base_load_includes_ev = False`:
```text
ev_planned_load_kwh      = summed EV AC load (primary + second)
ev_accounted_load_kwh    = 0
ev_total_planned_load_kwh = summed EV AC load
```

When `base_load_includes_ev = True`:
```text
ev_planned_load_kwh      = 0
ev_accounted_load_kwh    = summed EV AC load (primary + second)
ev_total_planned_load_kwh = summed EV AC load
```

Multiple EVs are always **summed**, never overwritten:
```text
ev_total_planned_load_kwh = primary_ev_ac_load + second_ev_ac_load
```

### Net load formula with EV

```text
effective_net_load_kwh
    = avg_house_consumption
    + ev_planned_load_kwh
    − solcast_pv_estimate
```

Only `ev_planned_load_kwh` (the extra, non-accounted portion) is added.
Using `ev_total_planned_load_kwh` when `base_load_includes_ev = True` would
double-count the EV load.

### Design invariants

The EV planner (`planner/ev_planner.py`) MUST satisfy these invariants:

1. **One-pass, no circularity**: EV plans are built entirely from raw inputs
   (EV SoC, target SoC, capacity, charger power, deadline, and the net
   surplus signal). They must never depend on the home battery planner output.

2. **Net surplus as starting point**: The surplus signal passed to the EV
   planner must represent **net surplus after house consumption**, not raw PV.
   The house always uses solar first; only the leftover is available to the EV
   at no extra grid cost.

   The engine computes base net consumption first, then derives:
   ```text
   slot_net_surplus = max(−estimated_net_consumption, 0.0)
                    = max(pv_estimate − avg_house_consumption, 0.0)
   ```

   `populate_net_consumption` is called **before** EV planning so that
   `estimated_net_consumption` already reflects PV confidence decay
   (day+1 at 90 %, day+2 at 80 %) and any other pre-EV transforms.

3. **`ev_planned_load_kwh` injected before final `populate_net_consumption`**:
   After the EV planner writes per-slot loads, `populate_net_consumption` is
   called a **second time** to incorporate `ev_planned_load_kwh` into the
   final `estimated_net_consumption` values. The final values include both
   house load and any extra EV load.

4. **Additive aggregation**: `apply_ev_planned_load_to_slots` must **add** to
   the existing slot total, never overwrite it (`+=` not `=`). This ensures
   primary and second EV loads are summed when they share a slot.

5. **No double-counting**: When `base_load_includes_ev = True` for an EV, its
   planned load must NOT be added to `ev_planned_load_kwh`. It is captured in
   `ev_accounted_load_kwh` instead.

6. **Partial current slot**: The currently active slot must be scaled by
   remaining slot duration, not the full slot width.

7. **Deadline enforcement**: Slots with `slot_start >= effective_deadline`
   must receive zero EV load (see invariant 8 for the definition of
   `effective_deadline`).

8. **One-midnight-crossing horizon cap** (issue #413): The EV charging
   window may extend into tomorrow but must NEVER reach into the day after
   tomorrow, regardless of the planner's overall slot horizon (which may be
   48 h or 72 h).

   Define:

   ```text
   horizon_cap         = midnight_at_start_of(now.date() + 2 days)
                         in now's timezone
   effective_deadline  = min(user_deadline, horizon_cap) if user_deadline
                         is not None else horizon_cap
   ```

   The EV planner must use `effective_deadline` as the upper bound when
   filtering candidate slots and when clamping per-slot allocation duration.
   This guarantees a single-midnight EV window even when the user-configured
   deadline is missing (`None`) or set to a future instant beyond
   end-of-tomorrow.

   `plan.deadline` (the value surfaced on the EV charging-plan sensor) keeps
   the **user-configured** deadline so dashboards display what the user
   asked for.  When the cap actually changes the deadline, the
   `effective_deadline` and `deadline_clamped` fields are surfaced on
   `plan.data_quality` for debuggability.

9. **Guard states**: The EV planner must return a valid `EVChargingPlan` with
   an appropriate `state` string in all edge cases (disabled, not connected,
   smart charging off, fully charged, no slots before deadline, invalid config).

10. **Disabled EV is zero-cost**: When `ev_planned_load_enabled = False`, all
    three EV load fields must be `0.0` and the home battery planner output
    must be identical to the non-EV case.

11. **Charge past target SoC (MILP only)**: When `allow_charge_past_target_soc`
    is enabled and the EV has reached its target SoC but is below 100 %, the
    EV can receive surplus PV that would otherwise be exported at low/negative
    prices — or, when its avoided-future-import valuation exceeds the export
    price, surplus PV that would otherwise be exported at any price
    (issue #630).  This is handled exclusively by the MILP:

    - The EV is included with `charge_past_target=True`: `target_kwh = capacity_kwh`,
      `deadline_slot = None` (no grid import pressure), a surplus-only constraint
      (`ev_c/eff ≤ pv − base_load`), and a benefit equal to
      `future_value_per_kwh` (avoided cost of importing the same energy
      later, `confidence_factor × mean(import_price)` over the next 24h),
      falling back to a tiny fixed tiebreaker (0.0001/kWh AC) when no future
      price data is available.
    - `future_value_per_kwh` and the per-EV `confidence_factor` are computed
      in `_build_ev_configs_for_milp` (`engine_core.py`) from
      `ev_future_charge_value_per_kwh` (`candidate_selector.py`).
    - The EV planner's Pass 3 has been removed — the MILP is the single
      authority for all EV charging decisions, including charge-past-target.
    - When the MILP fails (scipy unavailable, solver crash), charge-past-target
      is simply unavailable for that cycle.  The next successful MILP solve
      will pick it up.

    The MILP's decisions are authoritative for all EV charging.

12. **EV charger power fields**: `ev_charger_calculated_power` (primary EV)
    and `ev_second_charger_calculated_power` (second EV) are each computed
    **per-EV** from that EV's own charging plan (`EVChargingPlan.charging_slots`)
    by `_compute_ev_charger_power()` (for non-MILP candidates) or directly by
    the MILP's EV power computation (for MILP candidates).

    The per-EV power fields are set **before** candidate selection and
    correctly adjusted by the main-fuse throttling block (per-field loop).

    After candidate selection, a per-EV minimum-power floor check runs:
    each EV's power field is compared against **its own**
    `charger_min_power_w`.  If the power fell below that EV's own minimum
    (due to fuse throttling), only that EV's power field is zeroed, and
    its energy contribution is reverse-engineered from the power value and
    subtracted from the combined slot energy totals.

    **Important**: per-EV power fields MUST NOT be recomputed from the
    combined `ev_planned_load_kwh + ev_accounted_load_kwh` totals, because
    those fields are the sum across both EVs.  Deriving a per-EV power from
    a combined total would corrupt the per-EV output with the sum of both
    EVs' loads.

    The fields are purely planner outputs — the applier must read them to
    throttle the go-e charger; the planner does not control hardware directly.

13. **Slot-stable EV charger power (issue #738)**: Once the current slot has
    started, its per-EV `ev_charger_calculated_power` values must remain
    constant for the remainder of the slot. Replanning inside the same slot
    (triggered, for example, by the EV charging state toggling) must not
    recompute the current slot's charger power from freshly injected live
    PV/consumption data, because that makes the charger command oscillate.

    The coordinator freezes the values computed at slot start and restores
    them to the current slot on every subsequent replan. Explicit runtime
    overrides — force-charge-now and auto-full-EV on negative price — are
    still allowed to replace the frozen value for as long as they are active;
    when the override ends, the frozen slot-start value is restored.

### Invariants for tests

- When `ev_planned_load_enabled = False`, all `ev_planned_load_kwh == 0.0`.
- When EV is at or above target SoC (`current_soc >= target_soc`),
  all EV load fields are `0.0` (early return `"fully_charged"`).
  Charge-past-target is handled exclusively by the MILP.
- When `base_load_includes_ev = True`:
  - `ev_planned_load_kwh == 0.0` for all slots.
  - `ev_accounted_load_kwh > 0` for charging slots.
  - `ev_total_planned_load_kwh == ev_accounted_load_kwh`.
  - Net consumption is not affected by the EV (no double-count).
- `ev_total_planned_load_kwh == ev_planned_load_kwh + ev_accounted_load_kwh` for every slot.
- Net surplus slots are allocated before grid-import slots.
- `sum(ev_total_planned_load_kwh over all slots)` equals `total_kwh_needed` (±charger rounding).
- Deadline: no EV load on slots with `slot_start >= effective_deadline`.
- One-midnight-crossing cap: when `user_deadline is None` and the planner
  horizon extends beyond 24 h, no EV load is scheduled on slots whose
  `slot_start >= midnight_at_start_of(now.date() + 2 days)`.
- Deadline-clamp diagnostic: when the user-configured deadline is later
  than the horizon cap, `plan.data_quality["deadline_clamped"] is True`
  and `plan.data_quality["effective_deadline"]` holds the ISO-format clamp.
- Partial slot: current slot load ≤ `charger_power_kw × remaining_minutes / 60`.
- When EV consumes all net surplus, home battery `batteries_charged == 0.0` in that slot.
- `winner.cost == final_output.cost` still holds when EV load is active (no post-selection mutation).
- Both `ev_charging_plan` and `ev_second_charging_plan` on `PlannerOutput` are `None` when disabled.
- Enabling only the second EV does not affect primary EV fields and vice versa.
- Two EVs charging in the same slot: `ev_total_planned_load_kwh == primary_ac + second_ac`.
- One EV with zero load does not clear the other EV's load.
- `ev_smart_charging` label is applied when `ev_total_planned_load_kwh > 0`, even when
  `ev_planned_load_kwh == 0` (i.e. `base_load_includes_ev = True`).

## Documentation expectations

Every planner change should update:

- this spec if semantics change
- plan explanation output
- tests for at least one hand-calculated scenario

Every test fixture should state:

- slot duration
- input units
- expected SoC trajectory
- expected import/export
- expected total cost
