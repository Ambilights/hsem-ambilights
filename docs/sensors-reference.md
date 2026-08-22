# HSEM Sensors Reference

Comprehensive reference for all entities exposed by the HSEM integration, including
attributes, states, and dashboard examples.

---

## Entity overview

HSEM exposes these entity types:

| Type | Count | Description |
|---|---|---|
| **Sensor** | ~37 | Read-only state, plan, diagnostic, financial, and EV entities |
| **Select** | 2 | Force working mode override and Solcast likelihood selector |
| **Switch** | ~12 | Toggle entities for EV settings, features, and ML options |
| **Time** | 2 | Primary and second-EV charge deadlines |
| **Number** | 4 | Charge/discharge efficiency and EV target SoC controls |

---

## Electricity-price input sensors

These are external Home Assistant entities selected in HSEM's **Electricity
Prices** step; HSEM does not create them.

| Config key | Role |
|---|---|
| `hsem_import_electricity_price_sensor` | Required primary import price |
| `hsem_export_electricity_price_sensor` | Required primary export price |
| `hsem_import_electricity_price_forecast_sensor` | Optional legacy import forecast/gap-fill source |
| `hsem_export_electricity_price_forecast_sensor` | Optional legacy export forecast/gap-fill source |
| `hsem_import_electricity_price_entsoe_sensor` | Optional ENTSO-E published-price import backup |
| `hsem_export_electricity_price_entsoe_sensor` | Optional ENTSO-E published-price export backup |

The ENTSO-E entities must be configured together and must expose aligned,
timezone-aware, finite `{time|start, price}` records in a non-empty `prices`,
`prices_today`, or `prices_tomorrow` list. Their cadence must match
`hsem_electricity_price_update_interval`, including exact interval boundaries,
and their trimmed units must match the corresponding primary sensors exactly.

All four primary/backup entities must already use the final price basis. HSEM
does not convert currency or add VAT, tariffs, markup, or grid fees. Configure
those adjustments on the source sensors. The ENTSO-E selections do not create
new HSEM sensor entities; the selected source is reflected in the planner's
normal import/export price values and provenance attributes. See
[ENTSO-E Price Backup](entsoe-price-backup.md) for setup and validation.

---

## Working mode sensor

The primary HSEM sensor. Exposes the active battery recommendation and carries
all planner output as attributes.

**Entity:** `sensor.hsem_working_mode`

| State | Meaning |
|---|---|
| `batteries_charge_grid` | Battery charging from grid when selected by dynamic optimisation |
| `batteries_charge_solar` | Battery charging from PV surplus |
| `batteries_discharge_mode` | Battery discharging to cover house load |
| `force_batteries_discharge` | Forced discharge to grid (excess export) |
| `force_export` | Negative import price — all energy exported |
| `ev_smart_charging` | EV charging load allocated |
| `batteries_wait_mode` | Battery idle; may allow self-consumption above the planner reserve depending on **Wait mode behaviour** |
| `time_passed` | Slot is in the past |
| `missing_input_entities` | Required HA entities unavailable |

### Standard attributes

| Attribute | Type | Description |
|---|---|---|
| `batteries_current_capacity` | float (kWh) | Current usable battery capacity above discharge floor |
| `batteries_usable_capacity` | float (kWh) | Usable battery capacity (rated − floor) |
| `batteries_recommended_min_price_threshold` | float | Discharge threshold from `calculate_recommended_threshold()` |
| `batteries_capacity_loss_pct` | float | Configured capacity loss percentage |
| `import_electricity_price_state` | float | Live import spot price (currency/kWh) |
| `export_electricity_price_state` | float | Live export spot price (currency/kWh) |
| `export_electricity_min_price` | float | Minimum export price for export gating |
| `electricity_price_update_interval` | int | Price refresh interval (minutes) |
| `house_consumption_power_state` | float (W) | Instantaneous house load |
| `house_power_includes_ev_charger_power` | bool | Whether EV load is baked into house power |
| `net_consumption` | float (W) | Net load (house − solar) |
| `net_consumption_with_ev` | float (W) | Net load including EV |
| `solar_production_power_state` | float (W) | Instantaneous solar production |
| `months_winter` / `months_summer` | list[int] | Configured winter/summer month ranges |
| `batteries_enable_excess_export` | bool | Excess export gating enabled |
| `batteries_excess_export_discharge_buffer` | float | Discharge buffer for excess export |
| `batteries_wait_mode_behavior` | string | Wait mode behaviour: `strict` or `self_consumption_with_reserve` |
| `house_consumption_energy_weight_1d` | float | 1-day consumption prediction weight |
| `house_consumption_energy_weight_3d` | float | 3-day weight |
| `house_consumption_energy_weight_7d` | float | 7-day weight |
| `house_consumption_energy_weight_14d` | float | 14-day weight |
| `last_updated` | string (ISO-8601) | Last coordinator cycle timestamp |
| `status` | string | `ok`, `read_only`, `wait`, or `error` |
| `degraded_mode` | string | `ok`, `degraded`, or `error` |
| `hardware_writes_blocked` | bool | Safety gate preventing hardware writes |
| `apply_status` | string | Last apply result: `ok`, `unverified`, `failed`, `skipped` |
| `apply_failed_entities` | list[string] | Entities that failed the last hardware write |
| `data_quality` | dict | Structured price/PV and load-forecast completeness report |
| `entsoe_price_backup_status` | dict | Paired backup status: `configured`, `matched_slots`, and `rejection_reason` |
| `force_working_mode_state` | string | Active override mode or `auto` |

### Plan output attributes

| Attribute | Type | Description |
|---|---|---|
| `hourly_recommendation` | dict \| null | The recommendation slot active **right now** |
| `hourly_recommendations` | list[dict] | Full list of planner slots for the horizon |

### `hourly_recommendations` slot structure

Each entry in the `hourly_recommendations` list is a dictionary with these keys:

| Key | Type | Description |
|---|---|---|
| `start` | string (ISO-8601) | Slot start timestamp |
| `end` | string (ISO-8601) | Slot end timestamp |
| `recommendation` | string \| null | Working-mode value (see state table above) |
| `import_price` | float | Spot import price (local currency/kWh) |
| `export_price` | float | Spot export price (local currency/kWh) |
| `import_price_source` | string \| null | Import provenance: `primary`, `entsoe`, or `forecast` |
| `export_price_source` | string \| null | Export provenance: `primary`, `entsoe`, or `forecast` |
| `avg_house_consumption_kwh` | float | Weighted spike-aware consumption estimate (kWh) |
| `avg_house_consumption_1d_kwh` | float | 1-day window contribution (kWh) |
| `avg_house_consumption_3d_kwh` | float | 3-day window contribution (kWh) |
| `avg_house_consumption_7d_kwh` | float | 7-day window contribution (kWh) |
| `avg_house_consumption_14d_kwh` | float | 14-day window contribution (kWh) |
| `solcast_pv_estimate_kwh` | float | Forecast PV production for the slot (kWh) |
| `estimated_net_consumption_kwh` | float | avg_consumption + ev_planned_load − pv_estimate (kWh) |
| `ev_planned_load_kwh` | float | Extra EV AC load added to net consumption (kWh, ≥ 0) |
| `ev_accounted_load_kwh` | float | EV AC load already in house consumption (kWh, ≥ 0) |
| `ev_total_planned_load_kwh` | float | Total EV AC load (planned + accounted, kWh, ≥ 0) |
| `ev_charger_calculated_power` | float | Primary EV charger target AC power (W) |
| `ev_second_charger_calculated_power` | float | Second EV charger target AC power (W) |
| `estimated_cost_currency` | float | Estimated grid cost for the slot (local currency) |
| `batteries_charged_kwh` | float | Energy scheduled to charge into battery (kWh) |
| `batteries_discharged_kwh` | float | Energy drawn from battery by SoC simulation (kWh) |
| `estimated_battery_capacity_kwh` | float | Remaining usable battery energy at slot end (kWh) |
| `estimated_battery_soc_pct` | float | Simulated absolute SoC at slot end (0–100 %) |
| `grid_import_kwh` | float | Energy imported from grid (kWh) |
| `grid_export_kwh` | float | Energy exported to grid (kWh) |
| `primary_battery_hold` | bool | Explicit MILP zero-charge/zero-discharge intent; the Huawei battery is held in TOU with a 0 W discharge cap while incidental PV export is preserved |
| `is_ev_surplus_only_slot` | bool | Slot restricted to EV surplus-only charging |
| `secondary_storage_load_kwh` | float | Dedicated PowMr load energy (kWh) |
| `secondary_storage_charged_kwh` | float | Battery-side PowMr charge energy (kWh) |
| `secondary_storage_discharged_kwh` | float | Battery energy removed by PowMr (kWh) |
| `secondary_storage_grid_import_kwh` | float | PowMr branch utility import (kWh) |
| `secondary_storage_estimated_capacity_kwh` | float | PowMr usable energy above reserve at slot end |
| `secondary_storage_estimated_soc_pct` | float | Absolute PowMr SoC at slot end (%) |
| `secondary_storage_charge_current_a` | float | Physical 10 A-step charge target |
| `secondary_storage_mode` | string \| null | `utility`, `charge`, or `sbu` |

### Extended attributes (when enabled)

When the `switch.hsem_extended_attributes` switch is on, additional entity-ID
attributes are exposed. These reference the raw HA entity IDs for troubleshooting:

| Attribute | Description |
|---|---|
| `import_electricity_price_sensor_entity` | Import price sensor entity ID |
| `export_electricity_price_sensor_entity` | Export price sensor entity ID |
| `ev_charger_power_entity` | Primary EV charger power entity ID |
| `ev_charger_status_entity` | Primary EV charger status entity ID |
| `ev_soc_entity` | Primary EV SoC entity ID |
| `ev_connected_entity` | Primary EV connected entity ID |
| `ev_second_charger_power_entity` | Second EV charger power entity ID |
| `ev_second_charger_status_entity` | Second EV charger status entity ID |
| `ev_second_soc_entity` | Second EV SoC entity ID |
| `ev_second_connected_entity` | Second EV connected entity ID |
| `force_working_mode_entity` | Force working mode entity ID |
| `house_consumption_power_entity` | House consumption power entity ID |
| `solar_production_power_entity` | Solar production power entity ID |
| `solcast_pv_forecast_forecast_today_entity` | Solcast today forecast entity ID |
| `solcast_pv_forecast_forecast_tomorrow_entity` | Solcast tomorrow forecast entity ID |
| (plus all `huawei_solar_*` entity IDs) | Huawei battery and inverter entity IDs |
| `recommendation_interval_minutes` | Slot width in minutes |
| `recommendation_interval_length` | Number of slots in the horizon |
| `unique_id` | Integration unique ID |
| `update_interval` | Polling interval in minutes |
| `read_only` | Whether read-only mode is active |

### EV attributes

| Attribute | Type | Description |
|---|---|---|
| `ev_charger_power_state` | float (W) | Primary EV charger instantaneous power |
| `ev_charger_status_state` | bool | Primary EV currently charging |
| `ev_soc_state` | float (%) | Primary EV current SoC |
| `ev_soc_target_state` | float (%) | Primary EV target SoC |
| `ev_connected_state` | bool | Primary EV plugged in |
| `ev_allow_charge_past_target_soc` | bool | Allow charging past target |
| `ev_past_target_confidence_factor` | float | Confidence factor (0.0–1.0) applied to the avoided-future-import valuation of past-target charging |
| `ev_charger_max_discharge_power_state` | float (W) | Max discharge power cap |
| `ev_charger_force_max_discharge_power` | bool | Force max discharge power flag |
| `ev_second_enabled` | bool | Second EV integration enabled |
| `ev_second_charger_power_state` | float (W) | Second EV charger power |
| `ev_second_charger_status_state` | bool | Second EV charging |
| `ev_second_soc_state` | float (%) | Second EV SoC |
| `ev_second_soc_target_state` | float (%) | Second EV target SoC |
| `ev_second_connected_state` | bool | Second EV plugged in |
| `ev_second_allow_charge_past_target_soc` | bool | Second EV past-target flag |
| `ev_second_past_target_confidence_factor` | float | Second EV past-target confidence factor (0.0–1.0) |
| `ev_second_charger_max_discharge_power_state` | float (W) | Second EV max discharge cap |
| `ev_second_charger_force_max_discharge_power` | bool | Second EV force max discharge flag |

### Huawei battery attributes

| Attribute | Type | Description |
|---|---|---|
| `huawei_solar_batteries_charging_cutoff_capacity_state` | float (%) | Inverter charging cutoff SoC |
| `huawei_solar_batteries_grid_charge_cutoff_soc_state` | float (%) | Grid charge cutoff SoC |
| `huawei_solar_batteries_maximum_charging_power_state` | float (W) | Maximum charge power |
| `huawei_solar_batteries_maximum_discharging_power_state` | float (W) | Maximum discharge power |
| `huawei_solar_batteries_rated_capacity_max_state` | float (Wh) | Rated battery capacity |
| `huawei_solar_batteries_rated_capacity_min_state` | float (kWh) | Discharge floor capacity |
| `huawei_solar_batteries_state_of_capacity_state` | float (%) | Battery SoC |
| `huawei_solar_batteries_tou_charging_and_discharging_periods_periods` | list | Parsed TOU periods |
| `huawei_solar_batteries_tou_charging_and_discharging_periods_state` | string | Raw TOU entity state |
| `huawei_solar_batteries_working_mode_state` | string | Inverter working mode |
| `huawei_solar_inverter_active_power_control_state_state` | string | APC mode |
| `huawei_solar_batteries_excess_pv_energy_use_in_tou_state` | string | Excess PV in TOU setting |

---

## Plan explanation sensor

Displays the planner's strategy rationale and per-candidate cost breakdown.

**Entity:** `sensor.hsem_plan_explanation`

| Key attribute | Description |
|---|---|
| **State** | Winning candidate name: `"milp"`, `"passive"`, `"no_action"` |
| `selected_strategy` | Human-readable description (e.g. `"charge_grid_discharge_peak"`) |
| `winner_name` | Winning candidate name (same as state) |
| `summary` | One-sentence human-readable reason |
| `score` | Estimated savings vs doing nothing (currency) |
| `estimated_total_cost` | Net grid cost for the horizon |
| `price_spread` | Max minus min import price (arbitrage potential) |
| `peak_import_price` / `off_peak_import_price` | Price extremes |
| `forecast_pv_kwh` | Total PV forecast for the horizon |
| `forecast_net_consumption_kwh` | Total load minus PV |
| `battery_soc_pct` / `battery_soc_at_end_pct` | Starting and ending SoC |
| `secondary_terminal_price_source` | PowMr terminal inventory valuation source: `configured`, `published`, `forecast`, or `none` |
| `secondary_terminal_price_per_kwh` | Resolved PowMr terminal inventory value in local currency/kWh |
| `constraints` | Active flags (`winter_month`, `excess_export_enabled`, etc.) |
| `rejected_plans` | Alternatives with name, reason, and full cost breakdown |
| `hysteresis_active` | Whether plan-level hysteresis was applied |
| `hysteresis_reason` | Explanation of hysteresis decision |
| `data_quality_complete` | `True` only when price/PV inputs are complete and `load_forecast_ready` is true |

---

## Financial sensors

Cumulative monetary sensors that track grid import cost and export revenue.
All three use the `total` state class because signed prices can make their
values decrease as well as increase.

**Entities:**
- `sensor.hsem_export_income` — Cumulative export revenue
- `sensor.hsem_import_cost` — Cumulative import cost
- `sensor.hsem_net_grid_balance` — Export income minus import cost

### `sensor.hsem_export_income`

| Property | Value |
|---|---|
| **Type** | `sensor` |
| **State class** | `total` |
| **State** | Cumulative export revenue (local currency) |
| **Device class** | `monetary` |

### `sensor.hsem_import_cost`

| Property | Value |
|---|---|
| **Type** | `sensor` |
| **State class** | `total` |
| **State** | Cumulative import cost (local currency) |
| **Device class** | `monetary` |

### `sensor.hsem_net_grid_balance`

| Property | Value |
|---|---|
| **Type** | `sensor` |
| **State class** | `total` |
| **State** | Net grid balance (`export_income − import_cost`, local currency) |
| **Device class** | `monetary` |

**Template example:**

```jinja2
{{ states('sensor.hsem_net_grid_balance') | float | round(2) }}
```

---

## Prediction accuracy sensor

Tracks solar, load, end-of-slot battery SoC, and action accuracy from frozen
pre-slot plans.

**Entity:** `sensor.hsem_prediction_accuracy_sensor`

| State | Meaning |
|---|---|
| `soc_mae_7d` | 7-day battery SoC MAE (percentage points) |

| Attribute | Unit | Description |
|---|---|---|
| `soc_mae_7d` | pp | 7-day SoC Mean Absolute Error |
| `soc_mae_30d` | pp | 30-day SoC Mean Absolute Error |
| `solar_mape` | % | Solar forecast MAPE |
| `load_mae_kwh` | kWh | Load Mean Absolute Error |
| `action_mix` | dict | Distribution of planner actions over the window |
| `records_count` | count | Unique physical slots in the rolling scorecard |

Only a slot that has a frozen baseline and complete trusted actual coverage is
eligible. It is added on the exact coordinator cycle when the slot finalises
and only when live Huawei SoC is finite. Restored or already-finalised records
do not replay. The restored sensor scalar is startup-only: after the first live
coordinator snapshot, the entity reports a fresh metric or unavailable.

---

## Forecast accuracy sensor

Reports corrected PV and load forecast error from fully covered physical slots.

**Entity:** `sensor.hsem_forecast_accuracy_sensor`

| Property | Value |
|---|---|
| **State** | Eligible-slot PV MAE (kWh) |
| **Attributes** | `mae_*`, `bias_*`, `rmse_*`, `mape_*`, eligible `latest_*`, and internal restore data |

The baseline is the last raw/corrected PV and load plan observed before slot
start. Actual PV/load power uses the preceding finite sample interval split by
UTC overlap. Missing endpoints, stale gaps, incomplete coverage, and legacy
live-rewritten records are excluded rather than counted as zero actuals.

---

## Solar confidence sensor

Diagnostic view of the learned per-hour PV correction.

**Entity:** `sensor.hsem_solar_confidence_sensor`

| Attribute | Description |
|---|---|
| **State** | Mean learned hour factor (ratio), unavailable before learning |
| `hour_factors` | JSON-encoded map of local hour to correction factor |
| `confidence` | Internal correction-strength value (default 0.50) |
| `residual_count` | Eligible recent closed slots in the residual buffer |

**Template example:**

```jinja2
{{ states('sensor.hsem_solar_confidence_sensor') | float | round(3) }}
Confidence: {{ state_attr('sensor.hsem_solar_confidence_sensor', 'confidence') }}
```

---

## EV charging plan sensors

Diagnostic sensors displaying the EV charging plan details.

**Entities:**
- `sensor.hsem_ev_optimal_charging_plan` — Primary EV
- `sensor.hsem_ev_second_optimal_charging_plan` — Second EV

| State | Meaning |
|---|---|
| `not_connected` | EV is not plugged in |
| `smart_charging_disabled` | Smart charging turned off |
| `fully_charged` | Already at or above target SoC |
| `charging` | EV scheduled to charge in current slot |
| `waiting` | Connected but no active charging slot |
| `unavailable` | Not configured or capacity/power is zero |

**Key attributes:**

| Attribute | Description |
|---|---|
| `battery_capacity_kwh` | EV battery nameplate capacity |
| `charge_power_kw` | Charger AC output power |
| `current_soc` / `target_soc` | EV SoC values |
| `ev_connected` | Whether vehicle is plugged in |
| `total_kwh_needed` | Energy needed to reach target |
| `deadline` | ISO-8601 charging deadline |
| `charging_slots` | List of allocated charging slots with details |
| `planned_load_by_slot` | Dict of slot → kWh load |
| `data_quality` | Diagnostic warnings |

---

## Secondary storage plan sensor

Diagnostic shadow-plan sensor for the optional non-exporting PowMr battery.

**Entity:** `sensor.hsem_secondary_storage_plan`

| State | Meaning |
|---|---|
| `disabled` | Secondary optimisation is not enabled |
| `utility` | Utility bypass supplies the dedicated NAS load; grid charging is off |
| `charge` | Utility supplies the load and charges PowMr at the planned current |
| `sbu` | PowMr battery supplies only the dedicated load |
| `unavailable` | Required telemetry or a current recommendation is unavailable |

Key attributes:

| Attribute | Description |
|---|---|
| `enabled` | Secondary MILP planning enabled |
| `control_enabled` | Separate PowMr hardware-control gate |
| `read_only` | Global HSEM read-only gate; always overrides control |
| `actual_soc_pct` | Live `sensor.powmr_soc` value |
| `actual_battery_net_power_w` | Live signed PowMr battery power; positive charge, negative discharge |
| `actual_load_power_w` | Live dedicated-load power |
| `actual_output_source_priority` | Current PowMr output mode |
| `actual_charger_source_priority` | Current PowMr charger-source mode |
| `actual_max_charge_current_a` | Current PowMr number-entity setting |
| `target_charge_current_a` | Current slot's 10 A-step charge target |
| `target_soc_at_slot_end_pct` | Planned PowMr SoC at the end of the active slot |
| `planned_soc_at_horizon_end_pct` | Planned PowMr SoC at the end of the horizon |
| `planned_windows` | Consecutive equal-mode slots coalesced into compact windows |

Each `planned_windows` item contains `start`, `end`, `mode`, `charged_kwh`,
`discharged_kwh`, `grid_import_kwh`, `soc_at_end_pct`, and
`charge_current_a`. Keep both control gates off while validating these windows.

---

## Daily plan-vs-actual sensor

Diagnostic sensor tracking Home Assistant-local daily plan-versus-actual
energy and money.

**Entity:** `sensor.daily_plan_vs_actual`

| Property | Value |
|---|---|
| **State** | Today's measured net grid cost: import cost minus export revenue |
| `today` | Nested planned, actual, and difference metrics |
| `yesterday` | Most recent completed local-day record |
| `history` | Recent daily records exposed in state attributes |

Cumulative meter deltas are split by UTC price-slot overlap and local midnight.
Energy remains counted without price authority, while money is omitted. Meter
baselines survive rollover, and a daily reset's first positive reading is
retained as energy since midnight.

---

## EV charging active sensor

Boolean sensor indicating whether any EV is actively drawing power.

**Entity:** `sensor.hsem_ev_charging_sensor`

| State | Meaning |
|---|---|
| `on` | At least one EV is charging |
| `off` | No EV is charging |

---

## Battery SoC sensor

Snapshot of the battery state of charge with optional learned capacity tracking.

**Entity:** `sensor.hsem_battery_soc_sensor`

| State | Unit | Description |
|---|---|---|
| 0–100 | % | Battery SoC percentage |

**Key attributes:**

| Attribute | Unit | Description |
|---|---|---|
| `learned_capacity_kwh` | kWh | Learned usable battery capacity from charge/discharge cycles |
| `capacity_samples` | int | Number of charge/discharge samples contributing to the learned capacity |

---

## Dynamic discharge floor

Controls and reports the effective discharge floor SoC, which the planner uses as a minimum battery SoC when the dynamic floor feature is enabled.

**Entities:**
- `sensor.hsem_effective_discharge_floor` — Current effective floor SoC (%)
- `switch.hsem_dynamic_discharge_floor` — Enable/disable the dynamic floor feature

### `sensor.hsem_effective_discharge_floor`

| Property | Value |
|---|---|
| **Type** | `sensor` |
| **State** | Current effective discharge floor SoC percentage |
| **Unit** | % |

### `switch.hsem_dynamic_discharge_floor`

| Property | Value |
|---|---|
| **Type** | `switch` |
| **State** | `on` (dynamic floor active) or `off` (static floor) |

**Template example:**

```jinja2
{% if is_state('switch.hsem_dynamic_discharge_floor', 'on') %}
  Dynamic floor: {{ states('sensor.hsem_effective_discharge_floor') }}%
{% endif %}
```

---

## Savings tracker sensor

Tracks measured actual versus missed savings on Home Assistant-local dates.

**Entity:** `sensor.hsem_savings_tracker_sensor`

| Property | Value |
|---|---|
| **Type** | `sensor` |
| **State** | `today_actual` savings (local currency) |
| **Daily attributes** | `today_actual`, `today_missed`, `today_baseline` |
| **Period attributes** | `last_7_days_*`, `last_30_days_*`, `total_*` |
| **History attributes** | `daily`, `max_history_days`, `history_total_days` |

Export revenue and import cost come from the daily tracker's explicit measured
per-cycle deltas. Charge savings uses measured positive Huawei charge power
from the preceding valid interval and divides it across overlapping actionable
charge slots; it does not add the full planned charge again on every poll.
Missing telemetry and stale gaps are rejected.

**Template example:**

```jinja2
Actual today: {{ state_attr('sensor.hsem_savings_tracker_sensor', 'today_actual') }}
Missed today: {{ state_attr('sensor.hsem_savings_tracker_sensor', 'today_missed') }}
```

---

## PV curtailment sensor

Detects when the inverter is actively curtailing PV production.

**Entity:** `sensor.hsem_pv_curtailment_sensor`

| Property | Value |
|---|---|
| **Type** | `sensor` |
| **State** | `curtailed` (PV being limited) or `normal` (no curtailment) |
| **Entity category** | `diagnostic` |

**Template example:**

```jinja2
{% if is_state('sensor.hsem_pv_curtailment_sensor', 'curtailed') %}
  PV is being curtailed
{% endif %}
```

---

## Diagnostic sensors

| Entity | Display Name | Purpose | State / Value |
|---|---|---|---|
| `sensor.hsem_applier_status_sensor` | Inverter Apply Status | Hardware write success/failure | `ok`, `unverified`, `failed`, `skipped` |
| `sensor.hsem_battery_soc_sensor` | Battery State of Charge | Battery SoC snapshot | Percentage (0–100) |
| `sensor.daily_plan_vs_actual` | Daily Plan vs Actual | Daily energy plan-vs-actual tracking | Dict with cumulative metrics |
| `sensor.hsem_degraded_mode_sensor` | System Health | Overall system health | `ok`, `degraded`, `error` |
| `sensor.hsem_ev_charging_sensor` | EV Charging Active | Any EV actively charging | `on`, `off` |
| `sensor.hsem_ev_optimal_charging_plan` | EV Optimal Charging Plan | Primary EV plan state | `charging`, `waiting`, etc. |
| `sensor.hsem_ev_second_optimal_charging_plan` | EV Second Optimal Charging Plan | Second EV plan state | `charging`, `waiting`, etc. |
| `sensor.hsem_force_mode_sensor` | Force Working Mode | Override active indicator | `auto` or override mode name |
| `sensor.hsem_solar_confidence_sensor` | Solar Forecast Confidence | Per-hour PV forecast accuracy factors | Mean factor (ratio) |
| `sensor.hsem_hardware_writes_sensor` | Hardware Writes | Writes allowed/blocked by safety gate | `allowed`, `blocked` |
| `sensor.hsem_read_only_sensor` | Read-Only Mode | Read-only mode indicator | `on`, `off` |
| `sensor.hsem_net_consumption_sensor` | Net Consumption | Net load (house minus solar) | Watts (W) |
| `sensor.hsem_last_updated_sensor` | Last Updated | Last coordinator cycle timestamp | ISO-8601 timestamp |
| `sensor.hsem_next_update_sensor` | Next Update | Next scheduled coordinator cycle | ISO-8601 timestamp |
| `sensor.hsem_missing_entities_sensor` | Missing Input Entities | Count of missing input entities | Integer |
| `sensor.hsem_plan_explanation_sensor` | Plan Explanation | Planner strategy and cost breakdown | Winning candidate name |
| `sensor.hsem_prediction_accuracy_sensor` | Prediction Accuracy | Multi-horizon forecast accuracy | `soc_mae_7d` |
| `sensor.hsem_forecast_accuracy_sensor` | Forecast Accuracy | PV and load forecast MAE | kWh |
| `sensor.hsem_recommendation_interval_sensor` | Recommendation Interval | Slot width and horizon info | Minutes |
| `sensor.hsem_update_interval_sensor` | Update Interval | Current polling interval | Minutes |
| `sensor.hsem_working_mode` | Working Mode | Active battery recommendation | Working mode state |
| `sensor.hsem_export_income` | Export Income | Cumulative export revenue | Monetary (total) |
| `sensor.hsem_import_cost` | Import Cost | Cumulative import cost | Monetary (total) |
| `sensor.hsem_net_grid_balance` | Net Grid Balance | Export income minus import cost | Monetary (total) |
| `sensor.hsem_effective_discharge_floor` | Effective Discharge Floor | Current effective floor SoC | Percentage |
| `sensor.hsem_savings_tracker_sensor` | Savings Tracker | Actual vs missed savings (90-day) | Monetary |
| `sensor.hsem_pv_curtailment_sensor` | PV Curtailment | PV curtailment detection | `curtailed` / `normal` |

---

## Select entities

### Force working mode

**Entity:** `select.hsem_force_working_mode`

| Option | Description |
|---|---|
| `auto` | Normal operation — planner controls battery |
| `batteries_charge_grid` | Force grid charge |
| `batteries_charge_solar` | Force solar charge |
| `batteries_discharge_mode` | Force discharge to house |
| `batteries_wait_mode` | Force idle |
| `ev_smart_charging` | Force EV charging |
| `force_batteries_discharge` | Force discharge to grid |
| `force_export` | Force all energy to export |

### Solcast PV forecast likelihood

**Entity:** `select.hsem_solcast_likelihood`

Selects which Solcast likelihood scenario to use for PV forecasts (e.g. `p10`, `p50`, `p90`).
This setting is also configurable in the options flow.

---

## Switch entities

| Entity | Purpose |
|---|---|
| `switch.hsem_read_only` | Block all hardware writes |
| `switch.hsem_extended_attributes` | Enable extended diagnostic attributes |
| `switch.hsem_verbose_logging` | Enable verbose logging |
| `switch.hsem_ev_force_discharge` | Force EV maximum discharge power |
| `switch.hsem_ev_smart_charging` | Enable smart EV charging scheduling |
| `switch.hsem_ev_force_charge_now` | Force immediate EV charging |
| `switch.hsem_ev_second_smart_charging` | Enable smart charging for second EV |
| `switch.hsem_ev_second_force_charge_now` | Force immediate second EV charging |
| `switch.hsem_ml_consumption` | Enable ML-based consumption prediction |
| `switch.hsem_ml_sequential` | Enable sequential (intra-day momentum) ML mode |
| `switch.hsem_dynamic_discharge_floor` | Enable dynamic discharge floor |
| `switch.hsem_ev_auto_full_negative_price` | Auto-Full EV on negative price |

---

## Number entities

| Entity | Purpose | Range |
|---|---|---|
| `number.hsem_battery_charge_efficiency` | Battery charge efficiency | 1–100 % |
| `number.hsem_battery_discharge_efficiency` | Battery discharge efficiency | 1–100 % |
| `number.hsem_ev_target_soc` | Primary EV target SoC | 0–100 % |
| `number.hsem_ev_second_target_soc` | Second EV target SoC | 0–100 % |

## Time entities

| Entity | Purpose |
|---|---|
| `time.hsem_ev_deadline` | Primary EV charge deadline |
| `time.hsem_ev_second_deadline` | Second EV charge deadline |

---

## Config flow additions

### Quick setup (#610)

The config flow includes a `quick_setup` step that auto-detects HA entities
(Huawei inverter, EV charger, Solcast forecasts, price sensors) to reduce
manual configuration.

## Services

### `hsem.create_dashboard`

Logs the path to the bundled HSEM dashboard YAML and provides import
instructions. The dashboard includes cards for working mode, battery SoC,
financial sensors, EV status, and plan explanation. Import the YAML manually
via **Developer Tools → Services** or **Settings → Dashboards**.

> The service does **not** create or modify Home Assistant dashboards
> automatically; it only surfaces the bundled YAML path for manual import.

---

## Internal additions (no new sensors)

These changes are internal to the planner and do not expose new entities:

- **Weekday/weekend profiling (#612):** `WeekdayProfile` module-level
  singleton distinguishes weekday vs weekend consumption patterns for
  more accurate load forecasting.
- **Session EV charging (#615):** `EVConfig.session_charge_kw` field
  allows per-session charge power configuration for EV co-optimisation.

---

## Data quality attribute

The `data_quality` dict on `sensor.hsem_working_mode` is the serialized
planner `DataQuality` report. Example structure:

```json
{
  "is_complete": false,
  "load_forecast_ready": false,
  "load_forecast_reason": "zero_forecast_with_live_demand",
  "horizon_has_tomorrow": true,
  "horizon_days": 2,
  "price_actionable_until": "2030-01-02T00:00:00+01:00",
  "price_actionable_slots": 96,
  "tomorrow_price_missing_hours": [],
  "tomorrow_pv_missing_hours": [],
  "day2_price_missing_hours": [],
  "day2_pv_missing_hours": [],
  "today_price_missing_hours": [],
  "today_pv_missing_hours": []
}
```

`load_forecast_ready` is false when historical-average provenance or a
populated future slot is missing/non-finite. A complete identically-zero
profile is valid while finite live house demand is at most 50 W; above 50 W,
`load_forecast_reason` is `zero_forecast_with_live_demand`.
`load_forecast_reason` is `null` whenever readiness is true.
`is_complete` requires both load readiness and complete price/PV inputs.

The Plan Explanation sensor exposes the aggregate
`data_quality_complete` boolean. Forecast-authority generation mismatches are
written to debug logs only; they do not add an HA attribute or entity. v7.1.6
does not change any Unagi or terminal cost-to-go diagnostic field.
