# HSEM Config Flow Reference

This document describes every step in the HSEM configuration and options flows.

---

## Config flow steps

The config flow is a multi-step wizard. Steps appear in this order:

```
quick_setup → init → prices → months → solcast → huawei_solar
    → battery_economics → power → secondary_storage → ev → [ev_second]
    → ev_planned_load
    → [ev_second_planned_load] → batteries_wait_mode
    → batteries_excess_export → weighted_values
    → energy_and_ml
```

### Step: `quick_setup`

Initial entity auto-detection step. Scans available HA entities and
pre-populates the config flow with discovered sensors and devices.

| Field | Key | Default | Description |
|---|---|---|---|
| Confirm & Continue | — | — | Accept auto-detected entities and skip to final review |
| Advanced Setup | — | — | Proceed through the full step-by-step wizard |

When the user selects "Confirm & Continue", all auto-detected entities
are saved and the config flow jumps directly to `energy_and_ml` for
review and confirmation.  Selecting "Advanced Setup" walks through
every step in order so individual entities can be customised.

### Step: `init`

| Field | Key | Default | Description |
|---|---|---|---|
| Device name | `device_name` | `"Huawei Solar Energy Management"` | Friendly name for the integration |
| Update interval | `hsem_update_interval` | 5 minutes | Coordinator polling interval |
| Read-only mode | `hsem_read_only` | `False` | Block all hardware writes when enabled |
| Verbose logging | `hsem_verbose_logging` | `False` | Enable debug-level planner logging |

### Step: `prices`

Generic electricity price sensor configuration. Provider-agnostic — supports
Energi Data Service, Nordpool, Amber Electric, and any other price source.

| Field | Key | Default | Description |
|---|---|---|---|
| Import price sensor | `hsem_import_electricity_price_sensor` | `sensor.energi_data_service` | HA entity for import price |
| Export price sensor | `hsem_export_electricity_price_sensor` | `sensor.energi_data_service_produktion` | HA entity for export price |
| Import price forecast sensor | `hsem_import_electricity_price_forecast_sensor` | — | Optional dedicated import forecast sensor |
| Export price forecast sensor | `hsem_export_electricity_price_forecast_sensor` | — | Optional dedicated export forecast sensor |
| Export min price | `hsem_export_electricity_min_price` | 0.0 | Below this, inverter throttles export to zero |
| Price update interval | `hsem_electricity_price_update_interval` | 15 minutes | How often the price source publishes (15, 30, or 60) |

### Step: `months`

Seasonal month classification.

| Field | Key | Default | Description |
|---|---|---|---|
| Summer months | `hsem_months_summer` | `[5, 6, 7, 8, 9]` | Months classified as summer (derived as the complement of winter) |
| Winter months | `hsem_months_winter` | `[1, 2, 3, 4, 10, 11, 12]` | Months classified as winter. Selecting **all 12 months** is allowed (issue #725) — this keeps Winter/Spring (TOU) mode active year-round and the summer set is empty. |

### Step: `solcast`

PV forecast sensor configuration.

| Field | Key | Default | Description |
|---|---|---|---|
| Forecast today | `hsem_solcast_pv_forecast_forecast_today` | `sensor.solcast_pv_forecast_forecast_today` | Today's Solcast forecast |
| Forecast tomorrow | `hsem_solcast_pv_forecast_forecast_tomorrow` | `sensor.solcast_pv_forecast_forecast_tomorrow` | Tomorrow's Solcast forecast |
| Forecast likelihood | `hsem_solcast_pv_forecast_forecast_likelihood` | `pv_estimate` | Attribute key for the estimate field |

### Step: `huawei_solar`

Huawei Solar inverter and battery entity configuration (device selectors and entity sensors only).

| Field | Key | Default | Description |
|---|---|---|---|
| Inverter 1 device ID | `hsem_huawei_solar_device_id_inverter_1` | — | Device registry ID for inverter 1 |
| Inverter 2 device ID | `hsem_huawei_solar_device_id_inverter_2` | — | Device registry ID for inverter 2 (optional) |
| Batteries device ID | `hsem_huawei_solar_device_id_batteries` | — | Device registry ID for battery |
| Working mode | `hsem_huawei_solar_batteries_working_mode` | `select.batteries_working_mode` | Battery working mode select |
| End of discharge SoC | `hsem_huawei_solar_batteries_end_of_discharge_soc` | `number.batteries_end_of_discharge_soc` | Min SoC floor entity |
| State of capacity | `hsem_huawei_solar_batteries_state_of_capacity` | `sensor.batteries_state_of_capacity` | SoC sensor |
| Charging cutoff capacity | `hsem_huawei_solar_batteries_charging_cutoff_capacity` | `number.batteries_end_of_charge_soc` | Max SoC during charging |
| Grid charge cutoff SoC | `hsem_huawei_solar_batteries_grid_charge_cutoff_soc` | `number.batteries_grid_charge_cutoff_soc` | Max SoC when charging from grid |
| Grid charge maximum power | `hsem_huawei_solar_batteries_grid_charge_maximum_power` | `number.batteries_grid_charge_maximum_power` | Runtime phase-safe Huawei charge cap |
| Battery charge/discharge power | `hsem_huawei_solar_batteries_charge_discharge_power` | `sensor.batteries_charge_discharge_power` | Signed live battery power; positive charge |
| Max charging power | `hsem_huawei_solar_batteries_maximum_charging_power` | `number.batteries_maximum_charging_power` | Max charge power |
| Max discharging power | `hsem_huawei_solar_batteries_maximum_discharging_power` | `number.batteries_maximum_discharging_power` | Max discharge power |
| Rated capacity | `hsem_huawei_solar_batteries_rated_capacity` | `sensor.batteries_rated_capacity` | Nameplate capacity sensor |
| TOU periods | `hsem_huawei_solar_batteries_tou_charging_and_discharging_periods` | `sensor.batteries_tou_charging_and_discharging_periods` | TOU period schedule |
| Excess PV use | `hsem_huawei_solar_batteries_excess_pv_energy_use_in_tou` | `select.batteries_excess_pv_energy_use_in_tou` | Excess PV mode in TOU |
| Active power control | `hsem_huawei_solar_inverter_active_power_control` | `sensor.inverter_active_power_control` | Export power control mode |
| Phase A/B/C active power | `hsem_huawei_solar_power_meter_phase_[a-c]_active_power` | `sensor.power_meter_phase_[a-c]_active_power` | Signed per-phase grid power used by phase-aware charging |

### Step: `battery_economics`

Battery depreciation and efficiency parameters.

| Field | Key | Default | Description |
|---|---|---|---|
| Purchase price | `hsem_batteries_purchase_price` | 0 | Battery system cost |
| Expected cycles | `hsem_batteries_expected_cycles` | 6000 | Total expected lifetime cycles |
| Cycle cost | `hsem_batteries_cycle_cost` | 0 | Extra per-kWh wear margin |
| Capacity loss at EOL | `hsem_batteries_capacity_loss_pct` | 30 % | Expected capacity loss at end-of-life (%) |
| Charge efficiency | `hsem_batteries_charge_efficiency` | 98 % | Charge-side efficiency |
| Discharge efficiency | `hsem_batteries_discharge_efficiency` | 98 % | Discharge-side efficiency |

### Step: `power`

Power sensor configuration.

| Field | Key | Default | Description |
|---|---|---|---|
| House consumption power | `hsem_house_consumption_power` | `sensor.power_house_load` | House load power sensor |
| Solar production power | `hsem_solar_production_power` | `sensor.power_inverter_input_total` | PV production sensor |
| House includes EV | `hsem_house_power_includes_ev_charger_power` | `True` | Whether house sensor already includes EV charger |
| Main fuse amps | `hsem_main_fuse_amps` | 25 | Main fuse/breaker rating in amps. Set to 0 to disable. The MILP optimizer will respect this limit when scheduling battery and EV charging |
| Main fuse phases | `hsem_main_fuse_phases` | 3 | Electrical phase count (1 or 3). Single-phase installations MUST set this to 1 — setting 3 on a single-phase install makes the fuse constraint 3× too permissive |
| Phase-aware charging | `hsem_phase_aware_charging_enabled` | `False` | Hard per-phase planner limits plus live Huawei-first/PowMr-second charging. Requires the extra Huawei phase and battery entities |
| Max grid export power | `hsem_max_grid_export_power_kw` | 0 | DNO/inverter grid export cap in kW for export-limited connections (issue #726). The MILP planner never schedules export above this limit. Set to 0 to disable |

### Step: `secondary_storage`

Optional topology-aware planning for a non-exporting PowMr battery that powers
only a dedicated AC load. The reference defaults match the PowMr/NAS installation
for which HSEM Ambilights is developed; the feature is disabled and control is
separately disabled by default.

| Field | Key | Default | Description |
|---|---|---|---|
| Enable planning | `hsem_secondary_storage_enabled` | `False` | Add PowMr state and modes to the MILP |
| Enable control | `hsem_secondary_storage_control_enabled` | `False` | Permit the PowMr adapter; global read-only still wins |
| SoC sensor | `hsem_secondary_storage_soc_entity` | `sensor.powmr_soc` | Current battery SoC (%) |
| Battery net power | `hsem_secondary_storage_battery_net_power_entity` | `sensor.powmr_battery_net_power` | Optional diagnostic; positive charge, negative discharge |
| Dedicated load power | `hsem_secondary_storage_load_power_entity` | `sensor.powmr_load_power_internal` | Isolated NAS-side AC load (W) |
| Output priority | `hsem_secondary_storage_output_source_priority_entity` | `select.powmr_output_source_priority` | Supports `Utility first` and `SBU priority` |
| Charger priority | `hsem_secondary_storage_charger_source_priority_entity` | `select.powmr_charger_source_priority` | Uses `Solar and Utility` or `Only Solar` |
| Charge-current number | `hsem_secondary_storage_max_charge_current_entity` | `number.powmr_max_total_charge_current` | PowMr total current control |
| Capacity | `hsem_secondary_storage_capacity_kwh` | 15 kWh | Nameplate battery capacity |
| Minimum / maximum SoC | `hsem_secondary_storage_min_soc_pct`, `hsem_secondary_storage_max_soc_pct` | 20 / 100 % | Hard planning and runtime bounds |
| Nominal voltage | `hsem_secondary_storage_nominal_voltage_v` | 24 V | Converts stored kWh to current target |
| Minimum / maximum current | `hsem_secondary_storage_min_charge_current_a`, `hsem_secondary_storage_max_charge_current_a` | 10 / 60 A | Must be 10 A steps within the verified 10–80 A entity range |
| Grid connection phase | `hsem_secondary_storage_grid_phase` | 3 | Physical phase used by the single-phase PowMr charger and dedicated load |
| Charge / discharge efficiency | `hsem_secondary_storage_charge_efficiency_pct`, `hsem_secondary_storage_discharge_efficiency_pct` | 93 / 93 % | AC↔battery conversion efficiencies |
| SBU overhead | `hsem_secondary_storage_inverter_standby_power_w` | 55 W | Additional DC draw while battery supplies the load |
| Cycle cost | `hsem_secondary_storage_cycle_cost_per_kwh` | 0 | Optional wear cost per throughput kWh |
| History includes load | `hsem_secondary_storage_base_load_includes_dedicated_load` | `True` | House history normally contains the NAS utility load |
| Allow battery transfer | `hsem_secondary_storage_allow_primary_battery_transfer` | `False` | Allow Huawei discharge while PowMr charges |

Safe commissioning order:

1. Enable secondary-storage planning only.
2. Keep `hsem_secondary_storage_control_enabled=False` and global
   `hsem_read_only=True`.
3. Inspect `sensor.hsem_secondary_storage_plan` through several price changes.
4. Before enabling control, disable any automation that also writes the three
   PowMr control entities (for this installation, `automation.powmr_automatik`).
5. Enable the feature control gate only after the shadow plan is verified; lift
   global read-only separately when all HSEM hardware control is intended.

If required PowMr SoC or load telemetry is unavailable, no secondary plan is
produced and degraded-mode safety blocks writes. Mixed Utility/SBU history is an
approximation because the house-consumption series includes the NAS only while it
is supplied from utility.

### Step: `ev`

Primary EV charger configuration.

| Field | Key | Default | Description |
|---|---|---|---|
| EV charger status | `hsem_ev_charger_status` | — | Charger status sensor entity |
| EV charger power | `hsem_ev_charger_power` | — | Charger power sensor entity. **Watts expected** — a sensor reporting kW must have `unit_of_measurement: kW` so HSEM can convert it automatically |
| EV SoC sensor | `hsem_ev_soc` | — | EV battery SoC sensor |
| EV SoC target | `hsem_ev_soc_target` | 80 % | EV target SoC |
| EV connected sensor | `hsem_ev_connected` | — | Binary sensor for EV plugged in |
| Allow charge past target | `hsem_ev_allow_charge_past_target_soc` | `False` | Allow charging beyond target SoC from surplus PV, valued against export by avoided future import cost |
| Past-target confidence factor | `hsem_ev_past_target_confidence_factor` | `0.9` | Discount (0.0–1.0) applied to the avoided-future-import valuation used for past-target charging |
| Auto-Full on negative price | `hsem_ev_auto_full_negative_price` | `False` | Charge EV to 100 % when electricity price is negative |
| Force max discharge power | `hsem_ev_charger_force_max_discharge_power` | `False` | Force maximum discharge power during discharge slots |
| Max discharge power | `hsem_ev_charger_max_discharge_power` | 0 | Maximum discharge power cap (W) |

### Step: `ev_second`

Second EV charger configuration (identical fields to primary EV step; only shown when second EV enabled).

### Step: `ev_planned_load`

Primary EV planned load integration (optional, default disabled).

| Field | Key | Default | Description |
|---|---|---|---|
| Enable | `hsem_ev_planned_load_enabled` | `False` | Master switch |
| Battery capacity | `hsem_ev_planned_load_battery_capacity_kwh` | 0.0 | EV battery nameplate capacity (kWh) |
| Charger power | `hsem_ev_planned_load_charger_power_kw` | 0.0 | Charger AC output (kW) |
| Charger efficiency | `hsem_ev_planned_load_charger_efficiency` | 100 % | Charger efficiency |
| Charger min power | `hsem_ev_planned_load_charger_min_power_w` | 1380 W | Minimum charger power for physical operation |

Target SoC and deadline are configured outside this step:
- **Target SoC**: via the number entity `number.hsem_ev_target_soc`
- **Deadline**: via the HSEM time entity `time.hsem_ev_deadline_time`
- **Smart charging**: via the HSEM switch `switch.hsem_ev_smart_charging`
- **Force charge now**: via the HSEM switch `switch.hsem_ev_force_charge_now`
- **Allow charge past target**: via `hsem_ev_allow_charge_past_target_soc` in the EV charger step
- **Past-target confidence factor**: via `hsem_ev_past_target_confidence_factor` in the EV charger step

### Step: `ev_second_planned_load`

Second EV planned load integration (identical fields; only shown when second EV enabled).

### Step: `batteries_wait_mode`

Battery wait-mode behaviour.

| Field | Key | Default | Description |
|---|---|---|---|
| Wait mode behaviour | `hsem_batteries_wait_mode_behavior` | `strict` | `strict` keeps the battery idle in Wait mode; `self_consumption_with_reserve` allows normal household self-consumption while protecting the planner's required battery reserve |

### Step: `batteries_excess_export`

Excess battery export configuration.

| Field | Key | Default | Description |
|---|---|---|---|
| Enable excess export | `hsem_batteries_enable_excess_export` | `False` | Master switch |
| Discharge buffer | `hsem_batteries_excess_export_discharge_buffer` | 10 % | Safety SoC buffer before forced export |
| Battery export min price | `hsem_batteries_export_min_price` | `0.0` | Per-slot hard floor for intentional battery-to-grid export (issue #752). When > 0, the MILP forbids marking a slot as `force_batteries_discharge` when the export price is strictly below this floor — the battery can still serve house load in those slots. Reaching the threshold does NOT automatically trigger export; the optimizer still decides whether selling is worthwhile. Applies only to intentional battery-to-grid export, not to normal self-consumption, PV export, or PV charging. Set to 0 to disable. |
| Price threshold | — | Auto-calculated | Computed from battery depreciation settings at runtime |


### Step: `weighted_values`

Consumption prediction weight configuration.

| Field | Key | Default | Description |
|---|---|---|---|
| Weight 1-day | `hsem_house_consumption_energy_weight_1d` | 25 % | Weight for 1-day average |
| Weight 3-day | `hsem_house_consumption_energy_weight_3d` | 30 % | Weight for 3-day average |
| Weight 7-day | `hsem_house_consumption_energy_weight_7d` | 30 % | Weight for 7-day average |
| Weight 14-day | `hsem_house_consumption_energy_weight_14d` | 15 % | Weight for 14-day average |

Battery parameters and planner settings in this step duplicate their primary-step
counterparts and are kept for backward compatibility during migration.

### Step: `energy_and_ml`

Energy meter entities and ML consumption prediction (last step, creates entry).

| Field | Key | Default | Description |
|---|---|---|---|
| Grid Import Energy | `hsem_grid_import_energy_entity` | — | Cumulative grid import meter (kWh). Also used as ML data source. |
| Grid Export Energy | `hsem_grid_export_energy_entity` | — | Cumulative grid export meter (kWh). Used for net consumption. |
| PV Energy | `hsem_pv_energy_entity` | — | Cumulative PV production meter (kWh). |
| ML enabled | `hsem_ml_consumption_enabled` | `False` | Enable ridge regression predictor instead of rolling averages. |
| ML history days | `hsem_ml_consumption_history_days` | 14 | Days of recorder history for ML training (7–90). |
| Net consumption | `hsem_ml_consumption_net_consumption` | `False` | Subtract export from import for net house consumption. |
| Sequential prediction | `hsem_ml_consumption_sequential` | `False` | Feed each slot's prediction as lag input to the next (captures intra-day momentum). |
| Temperature sensor | `hsem_ml_consumption_temperature_entity` | — | Outdoor (ambient) temperature in °C for weather-driven predictions. |
