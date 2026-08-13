# Huawei Solar — Available HA Entities

> **Canonical reference for all AI agents and developers.**
> Before using any battery, inverter, or power-meter value in HSEM, look up the correct
> entity ID here. Do **not** guess or invent entity IDs — use only what appears in this file.
>
> This file reflects the actual entities exposed by the `wlcrs/huawei_solar` integration on
> this installation. Update it whenever the integration or hardware changes.

---

## Batteries

### number entities

| Friendly name | Entity ID | Unit | Used by HSEM |
|---|---|---|---|
| End-of-charge SOC | `number.batteries_end_of_charge_soc` | % | ✅ `hsem_huawei_solar_batteries_charging_cutoff_capacity` |
| End-of-discharge SOC | `number.batteries_end_of_discharge_soc` | % | ✅ `hsem_huawei_solar_batteries_end_of_discharge_soc` |
| Grid charge cutoff SOC | `number.batteries_grid_charge_cutoff_soc` | % | ✅ `hsem_huawei_solar_batteries_grid_charge_cutoff_soc` |
| Grid charge maximum power | `number.batteries_grid_charge_maximum_power` | W | ✅ `hsem_huawei_solar_batteries_grid_charge_maximum_power` (phase-aware control) |
| Maximum charging power | `number.batteries_maximum_charging_power` | W | ✅ `hsem_huawei_solar_batteries_maximum_charging_power` |
| Maximum discharging power | `number.batteries_maximum_discharging_power` | W | ✅ `hsem_huawei_solar_batteries_maximum_discharging_power` |
| Peak Shaving SOC | `number.batteries_peak_shaving_soc` | % | — |

### sensor entities

| Friendly name | Entity ID | Unit | Used by HSEM |
|---|---|---|---|
| State of capacity (SoC) | `sensor.batteries_state_of_capacity` | % | ✅ `hsem_huawei_solar_batteries_state_of_capacity` |
| Charge/discharge power | `sensor.batteries_charge_discharge_power` | W | ✅ `hsem_huawei_solar_batteries_charge_discharge_power` (positive charge, negative discharge) |
| Rated capacity | `sensor.batteries_rated_capacity` | Wh | ✅ `hsem_huawei_solar_batteries_rated_capacity` |
| TOU charging and discharging periods | `sensor.batteries_tou_charging_and_discharging_periods` | — | ✅ `hsem_huawei_solar_batteries_tou_charging_and_discharging_periods` |

### select entities

| Friendly name | Entity ID | Used by HSEM |
|---|---|---|
| Working mode | `select.batteries_working_mode` | ✅ `hsem_huawei_solar_batteries_working_mode` |
| Excess PV energy use in TOU | `select.batteries_excess_pv_energy_use_in_tou` | ✅ `hsem_huawei_solar_batteries_excess_pv_energy_use_in_tou` |

### Fully Fed to Grid control behavior

On this installation, Huawei's Fully Fed to Grid mode gives PV priority and lets the battery fill
only the inverter's remaining AC headroom. HSEM combines that mode with
`number.batteries_maximum_discharging_power`: `force_batteries_discharge` divides the planned
battery energy by the full slot duration and holds that power cap for the selected plan. The command
never exceeds either the original planned power or the Huawei hardware maximum. Huawei's integer
SoC samples are too coarse for sub-slot energy pacing (one percent is about 0.3 kWh on a 30 kWh
battery), so SoC changes do not taper a latched cap. A newly accepted stale plan that is already at
its endpoint is blocked, and the first callback at or after slot completion commands 0 W. PV-only
`force_export` uses a 0 W battery cap.

The selected slot's planned discharge energy is authoritative for executing export. Its end capacity
is used only to reject a newly accepted stale plan. The separate `required_capacity_kwh` value is
calculated before candidate selection and is not a constraint on the winning MILP plan, so applying
it as a second hardware floor could contradict that winner.
Accurate within-slot tapering would require integrating a primary-battery power or energy meter;
integer SoC is deliberately not used as a substitute.

`number.batteries_end_of_discharge_soc` exposes only 0–20% here. It remains the inverter's static
hardware floor. The plan-derived cap bounds discharge energy, while the inverter's configured
end-of-discharge SoC remains the absolute hardware backstop.

---

## Inverter

### sensor entities

| Friendly name | Entity ID | Unit | Used by HSEM |
|---|---|---|---|
| Active power control | `sensor.inverter_active_power_control` | — | ✅ `hsem_huawei_solar_inverter_active_power_control` |
| Locking status | `sensor.inverter_locking_status` | — | — |
| Max active power | `sensor.inverter_max_active_power` | W | — |
| Monthly yield | `sensor.inverter_monthly_yield` | kWh | — |
| Off-grid status | `sensor.inverter_off_grid_status` | — | — |
| Off-grid switch | `sensor.inverter_off_grid_switch` | — | — |
| Phase A current | `sensor.inverter_phase_a_current` | A | — |
| Phase A voltage | `sensor.inverter_phase_a_voltage` | V | — |
| Phase B current | `sensor.inverter_phase_b_current` | A | — |
| Phase B voltage | `sensor.inverter_phase_b_voltage` | V | — |
| Phase C current | `sensor.inverter_phase_c_current` | A | — |
| Phase C voltage | `sensor.inverter_phase_c_voltage` | V | — |
| Power factor | `sensor.inverter_power_factor` | — | — |
| PV 1 current | `sensor.inverter_pv_1_current` | A | — |
| PV 1 voltage | `sensor.inverter_pv_1_voltage` | V | — |
| PV 2 current | `sensor.inverter_pv_2_current` | A | — |
| PV 2 voltage | `sensor.inverter_pv_2_voltage` | V | — |
| PV connection status | `sensor.inverter_pv_connection_status` | — | — |
| Rated power | `sensor.inverter_rated_power` | W | — |
| Reactive power | `sensor.inverter_reactive_power` | var | — |
| Shutdown time | `sensor.inverter_shutdown_time` | — | — |
| Startup time | `sensor.inverter_startup_time` | — | — |
| State | `sensor.inverter_inverter_state` | — | — |
| Total DC input energy | `sensor.inverter_total_dc_input_energy` | kWh | — |
| Total yield | `sensor.inverter_total_yield` | kWh | — |
| Yearly yield | `sensor.inverter_yearly_yield` | kWh | — |

### number entities

| Friendly name | Entity ID | Unit | Used by HSEM |
|---|---|---|---|
| MPPT-Scan Interval | `number.inverter_mppt_scan_interval` | min | — |
| Power derating | `number.inverter_power_derating` | W | — |
| Power derating (by percentage) | `number.inverter_power_derating_by_percentage` | % | — |

### switch entities

| Friendly name | Entity ID | Used by HSEM |
|---|---|---|
| MPPT-Scan | `switch.inverter_mppt_scanning` | — |

---

## Power Meter

### sensor entities

| Friendly name | Entity ID | Unit | Used by HSEM |
|---|---|---|---|
| A-B line voltage | `sensor.power_meter_a_b_line_voltage` | V | — |
| Active power | `sensor.power_meter_active_power` | W | — |
| B-C line voltage | `sensor.power_meter_b_c_line_voltage` | V | — |
| C-A line voltage | `sensor.power_meter_c_a_line_voltage` | V | — |
| Consumption | `sensor.power_meter_consumption` | kWh | — |
| Exported | `sensor.power_meter_exported` | kWh | — |
| Frequency | `sensor.power_meter_frequency` | Hz | — |
| Meter status | `sensor.power_meter_meter_status` | — | — |
| Phase A active power | `sensor.power_meter_phase_a_active_power` | W | ✅ `hsem_huawei_solar_power_meter_phase_a_active_power` (phase-aware control) |
| Phase A current | `sensor.power_meter_current` | A | — |
| Phase A voltage | `sensor.power_meter_phase_a_voltage` | V | — |
| Phase B active power | `sensor.power_meter_phase_b_active_power` | W | ✅ `hsem_huawei_solar_power_meter_phase_b_active_power` (phase-aware control) |
| Phase B current | `sensor.power_meter_current_2` | A | — |
| Phase B voltage | `sensor.power_meter_phase_b_voltage` | V | — |
| Phase C active power | `sensor.power_meter_phase_c_active_power` | W | ✅ `hsem_huawei_solar_power_meter_phase_c_active_power` (phase-aware control) |
| Phase C current | `sensor.power_meter_current_3` | A | — |
| Phase C voltage | `sensor.power_meter_phase_c_voltage` | V | — |
| Power factor | `sensor.power_meter_power_factor` | — | — |
| Reactive energy | `sensor.power_meter_reactive_energy` | kvarh | — |
| Reactive power | `sensor.power_meter_reactive_power` | var | — |
