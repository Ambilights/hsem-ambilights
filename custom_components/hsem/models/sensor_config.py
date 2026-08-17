"""Pure-Python dataclass representing the full configuration of HSEMWorkingModeSensor.

This module is the single source of truth for all config-entry values that the
sensor reads during each update cycle.  It carries **no** Home Assistant imports
and can therefore be constructed and inspected in plain unit tests without a
running HA instance.

The :func:`build_sensor_config` factory function (in
``custom_sensors/state_collector.py``) is responsible for reading a
``ConfigEntry`` and returning a populated :class:`SensorConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from custom_components.hsem.const import (
    DEFAULT_CONFIG_VALUES,
    MILP_SOLVER_TIMEOUT_DEFAULT_SECONDS,
    SEASONAL_FILL_MODE_FORECAST,
)
from custom_components.hsem.models.secondary_storage_entity_config import (
    SecondaryStorageEntityConfig,
)


@dataclass
class EVChargerConfig:
    """Configuration for a single EV charger.

    Holds the entity IDs for the charger's status, power, SoC, and
    connection sensors, along with behavioural flags such as whether to
    allow charging past the target SoC and force maximum discharge power.
    """

    status_entity: str | None = None
    power_entity: str | None = None
    soc_entity: str | None = None
    connected_entity: str | None = None
    allow_charge_past_target_soc: bool = False
    #: Confidence multiplier (0.0-1.0) applied to the avoided-future-import
    #: valuation of charge-past-target EV charging in the MILP. See
    #: ``ev_future_charge_value_per_kwh`` in ``planner/candidate_selector.py``.
    past_target_confidence_factor: float = 0.9
    force_max_discharge_power: bool = False
    max_discharge_power: int = 0


@dataclass
class SensorConfig:
    """Complete set of configuration values for :class:`HSEMWorkingModeSensor`.

    All fields are plain Python types; no Home Assistant objects are stored here.
    Fields mirror the config-entry option keys defined in ``const.py`` and
    populated by ``_update_settings()`` in the sensor.

    Attributes:
        read_only: When True the sensor skips all hardware writes.
        verbose_logging: Enable extra debug log output.
        extended_attributes: Expose extra diagnostic attributes on the entity.
        update_interval: Polling interval in minutes.
        recommendation_interval_minutes: Slot width in minutes (15 or 60).
        recommendation_interval_length: Planning horizon in hours.
        electricity_price_update_interval: Price sensor update cadence in minutes (15, 30, or 60).

        huawei_solar_device_id_inverter_1: Device ID for inverter 1.
        huawei_solar_device_id_inverter_2: Device ID for inverter 2 (optional).
        huawei_solar_device_id_batteries: Device ID for the battery pack.
        huawei_solar_batteries_working_mode: Entity ID for working mode select.
        huawei_solar_batteries_end_of_discharge_soc: Entity ID for EoD SoC number.
        huawei_solar_batteries_state_of_capacity: Entity ID for SoC sensor.
        huawei_solar_batteries_charging_cutoff_capacity: Entity ID for charging cutoff SoC (max SoC).
        huawei_solar_batteries_grid_charge_cutoff_soc: Entity ID for grid charge cutoff SoC.
        huawei_solar_batteries_maximum_charging_power: Entity ID for max charge power.
        huawei_solar_batteries_maximum_discharging_power: Entity ID for max discharge power.
        huawei_solar_batteries_tou_charging_and_discharging_periods: Entity ID for TOU periods.
        huawei_solar_batteries_excess_pv_energy_use_in_tou: Entity ID for excess PV use select.
        huawei_solar_inverter_active_power_control: Entity ID for export power control.
        huawei_solar_batteries_rated_capacity: Entity ID for rated battery capacity sensor.

        house_consumption_power: Entity ID for house power meter.
        solar_production_power: Entity ID for solar production meter.
        house_power_includes_ev_charger_power: True if EV draw is already in house meter.

        solcast_pv_forecast_forecast_today: Entity ID for today's Solcast forecast.
        solcast_pv_forecast_forecast_tomorrow: Entity ID for tomorrow's Solcast forecast.
        solcast_pv_forecast_forecast_likelihood: Attribute key for Solcast estimate field.

        import_electricity_price_sensor: Entity ID for the import price sensor.
        export_electricity_price_sensor: Entity ID for the export price sensor.
        import_electricity_price_forecast_sensor: Optional entity ID for a separate import forecast sensor (e.g. Amber Electric).
        export_electricity_price_forecast_sensor: Optional entity ID for a separate export forecast sensor.
        import_electricity_price_entsoe_sensor: Optional entity ID for the
            tariff-, VAT-, and currency-adjusted ENTSO-E import backup sensor.
        export_electricity_price_entsoe_sensor: Optional entity ID for the
            tariff-, VAT-, and currency-adjusted ENTSO-E export backup sensor.
        export_electricity_min_price: Minimum export price to allow grid export.

        ev: First EV charger configuration.
        ev_second_enabled: Whether the second EV charger is active.
        ev_second: Second EV charger configuration.

        batteries_charge_efficiency: Charge efficiency as a percentage (0-100).
            Energy stored in the battery = input_energy × (charge_efficiency / 100).
            Defaults to 97 % (3 % charge-side loss).
        batteries_discharge_efficiency: Discharge efficiency as a percentage (0-100).
            Energy delivered to the house = battery_energy × (discharge_efficiency / 100).
            Defaults to 97 % (3 % discharge-side loss).
        batteries_purchase_price: Battery pack purchase price for depreciation calc.
        batteries_expected_cycles: Expected total cycle life of the battery.
        batteries_cycle_cost: User-configured extra per-kWh cycle-wear cost.
            Added to the auto-derived depreciation cost. 0.0 = no extra margin.


        batteries_enable_excess_export: Enable opportunistic forced-discharge export.
        batteries_excess_export_discharge_buffer: Safety buffer percentage to keep.
        batteries_export_min_price: Per-slot hard floor for intentional
            battery-to-grid export (issue #752).  ``0.0`` disables the floor.
        batteries_wait_mode_behavior: How ``batteries_wait_mode`` is interpreted.
            ``strict`` keeps the battery idle; ``self_consumption_with_reserve``
            allows normal household self-consumption while protecting the
            planner's required battery reserve.

        milp_solver_timeout_seconds: Maximum HiGHS wall-clock budget in seconds.
            Time-limited incumbents are accepted only after complete model
            validation.
        months_winter: List of month integers (1-12) treated as winter.
        months_summer: List of month integers (1-12) treated as summer.
        seasonal_fill_mode: Strategy for otherwise-unassigned planner slots.
            ``forecast`` uses forward PV refill headroom; ``months`` preserves
            the legacy calendar-only behaviour.

        house_consumption_energy_weight_1d: Weight (%) for 1-day consumption average.
        house_consumption_energy_weight_3d: Weight (%) for 3-day consumption average.
        house_consumption_energy_weight_7d: Weight (%) for 7-day consumption average.
        house_consumption_energy_weight_14d: Weight (%) for 14-day consumption average.
    """

    # General
    read_only: bool = False
    verbose_logging: bool = True
    extended_attributes: bool = False
    update_interval: int = 1
    recommendation_interval_minutes: int = 15
    recommendation_interval_length: int = 48
    electricity_price_update_interval: int = 15

    # Huawei Solar device IDs
    huawei_solar_device_id_inverter_1: str | None = None
    huawei_solar_device_id_inverter_2: str | None = None
    huawei_solar_device_id_batteries: str | None = None

    # Huawei Solar entity IDs
    huawei_solar_batteries_working_mode: str | None = None
    huawei_solar_batteries_end_of_discharge_soc: str | None = None
    huawei_solar_batteries_state_of_capacity: str | None = None
    huawei_solar_batteries_charging_cutoff_capacity: str | None = None
    huawei_solar_batteries_grid_charge_cutoff_soc: str | None = None
    huawei_solar_batteries_grid_charge_maximum_power: str | None = None
    huawei_solar_batteries_charge_discharge_power: str | None = None
    huawei_solar_batteries_maximum_charging_power: str | None = None
    huawei_solar_batteries_maximum_discharging_power: str | None = None
    huawei_solar_batteries_tou_charging_and_discharging_periods: str | None = None
    huawei_solar_batteries_excess_pv_energy_use_in_tou: str | None = None
    huawei_solar_batteries_forcible_charge: str | None = None
    huawei_solar_inverter_active_power_control: str | None = None
    huawei_solar_batteries_rated_capacity: str | None = None
    huawei_solar_power_meter_phase_a_active_power: str | None = None
    huawei_solar_power_meter_phase_b_active_power: str | None = None
    huawei_solar_power_meter_phase_c_active_power: str | None = None

    # Power meters
    house_consumption_power: str | None = None
    solar_production_power: str | None = None
    house_power_includes_ev_charger_power: bool = False
    #: Main fuse/breaker rating in amps (0 = disabled).  The MILP optimizer
    #: uses this as a soft constraint on total grid import power.
    main_fuse_amps: int = 0
    #: Electrical phase count (1 or 3).  Single-phase installations MUST
    #: set this to 1 — using 3 on a single-phase install makes the fuse
    #: protection constraint silenty 3× too permissive.
    main_fuse_phases: int = 3
    #: Use signed per-phase Huawei power-meter telemetry for hard planner
    #: constraints and live charge-command limiting.  Disabled by default so
    #: installations without a three-phase meter retain legacy behaviour.
    phase_aware_charging_enabled: bool = False
    #: Maximum grid export power in kW (0 = disabled).  DNO/inverter export
    #: cap for export-limited connections (issue #726).  The MILP optimizer
    #: uses this as a hard bound on per-slot grid export energy.
    max_grid_export_power_kw: float = 0.0

    # Optional dedicated-load secondary storage (PowMr topology).
    secondary_storage: SecondaryStorageEntityConfig = field(
        default_factory=SecondaryStorageEntityConfig
    )

    # Solcast
    solcast_pv_forecast_forecast_today: str | None = None
    solcast_pv_forecast_forecast_tomorrow: str | None = None
    solcast_pv_forecast_forecast_likelihood: str = "pv_estimate"

    # Electricity prices (generic — supports Energi Data Service, Nordpool, Amber Electric, …)
    import_electricity_price_sensor: str | None = None
    export_electricity_price_sensor: str | None = None
    import_electricity_price_forecast_sensor: str | None = None
    export_electricity_price_forecast_sensor: str | None = None
    import_electricity_price_entsoe_sensor: str | None = None
    export_electricity_price_entsoe_sensor: str | None = None
    export_electricity_min_price: float = 0.0

    # Predicted prices for the unpublished horizon. Valuation only — these
    # never become actionable, so they can justify extra charging but never a
    # discharge. See models/price_forecast.py.
    price_forecast_valuation_enabled: bool = False
    price_forecast_valuation_sensor: str | None = None
    price_forecast_valuation_margin: float = 0.0

    # EV chargers
    ev: EVChargerConfig = field(default_factory=EVChargerConfig)
    ev_second_enabled: bool = False
    ev_second: EVChargerConfig = field(default_factory=EVChargerConfig)
    #: Auto-Full on negative electricity price (issue #609)
    ev_auto_full_negative_price: bool = False

    # Battery economics
    batteries_charge_efficiency: float = 98.0
    batteries_discharge_efficiency: float = 98.0
    batteries_purchase_price: float = 0.0
    batteries_expected_cycles: int = 6000
    #: User-configured per-kWh cycle-wear margin. It is added to the canonical
    #: auto-derived depreciation cost used by the MILP objective and scorer.
    #: 0.0 means no extra margin.
    batteries_cycle_cost: float = 0.0

    #: Expected battery capacity loss at end-of-life as a percentage (0-100).
    #: LiFePO4 EOL is typically 20 % (80 % retained).  Default 30 % includes
    #: margin for calendar ageing.
    batteries_capacity_loss_pct: float = field(
        default_factory=lambda: cast(
            float, DEFAULT_CONFIG_VALUES["hsem_batteries_capacity_loss_pct"]
        )
    )

    # Excess export
    batteries_enable_excess_export: bool = False
    batteries_excess_export_discharge_buffer: float = 10.0
    #: Per-slot hard floor for intentional battery-to-grid export (issue #752).
    #: When ``> 0`` and a slot's export price is strictly below this floor the
    #: MILP caps the battery discharge variable so the battery can only cover
    #: house load (no grid export).  ``0.0`` = disabled (default) — fully
    #: backward compatible.
    batteries_export_min_price: float = 0.0

    # Wait mode behaviour
    batteries_wait_mode_behavior: str = "strict"

    # HiGHS MILP solve budget in seconds.
    milp_solver_timeout_seconds: float = MILP_SOLVER_TIMEOUT_DEFAULT_SECONDS

    # EV planned load integration — primary EV (optional, disabled by default)
    ev_planned_load_enabled: bool = False
    ev_planned_load_battery_capacity_kwh: float = 0.0
    ev_planned_load_charger_power_kw: float = 0.0
    ev_planned_load_charger_efficiency_pct: float = 100.0
    ev_planned_load_charger_min_power_w: float = 1380.0
    # EV planned load integration — second EV (optional, disabled by default)
    ev_second_planned_load_enabled: bool = False
    ev_second_planned_load_battery_capacity_kwh: float = 0.0
    ev_second_planned_load_charger_power_kw: float = 0.0
    ev_second_planned_load_charger_efficiency_pct: float = 100.0
    ev_second_planned_load_charger_min_power_w: float = 1380.0

    # Seasonal configuration
    months_winter: list[int] = field(default_factory=list)
    months_summer: list[int] = field(default_factory=list)
    seasonal_fill_mode: str = SEASONAL_FILL_MODE_FORECAST

    # Daily plan-vs-actual tracking — optional cumulative energy meter entities.
    # When not configured, the sensor falls back to Riemann sums from power sensors.
    grid_import_energy_entity: str | None = None
    grid_export_energy_entity: str | None = None
    pv_energy_entity: str | None = None

    # ML consumption prediction — toggle and settings
    ml_consumption_enabled: bool = False
    ml_consumption_energy_entity: str | None = None
    ml_consumption_history_days: int = 14
    ml_consumption_net_consumption: bool = False
    ml_consumption_sequential: bool = False
    ml_consumption_temperature_entity: str | None = None

    # Planner hysteresis — keep the active plan unless a new plan is
    # materially better (anti-flapping, issue #372).
    planner_hysteresis_enabled: bool = True
    planner_hysteresis_absolute: float = 0.0
    planner_hysteresis_percentage: float = 5.0
    # Window-level hysteresis — minimum hold time (minutes) before allowing
    # any recommendation change.  0 disables the feature.
    planner_window_hysteresis_minutes: int = 10

    # Consumption weights
    house_consumption_energy_weight_1d: int = 50
    house_consumption_energy_weight_3d: int = 20
    house_consumption_energy_weight_7d: int = 15
    house_consumption_energy_weight_14d: int = 10

    def weights_sum(self) -> int:
        """Return the sum of all four consumption weights."""
        return (
            self.house_consumption_energy_weight_1d
            + self.house_consumption_energy_weight_3d
            + self.house_consumption_energy_weight_7d
            + self.house_consumption_energy_weight_14d
        )

    def __repr__(self) -> str:
        return (
            f"SensorConfig(read_only={self.read_only}, "
            f"interval_minutes={self.recommendation_interval_minutes}, "
            f"interval_length_hours={self.recommendation_interval_length}, "
            f"weights=[{self.house_consumption_energy_weight_1d}/"
            f"{self.house_consumption_energy_weight_3d}/"
            f"{self.house_consumption_energy_weight_7d}/"
            f"{self.house_consumption_energy_weight_14d}])"
        )
