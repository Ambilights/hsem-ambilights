"""Config-entry reader for HSEMWorkingModeSensor.

Single responsibility: read all config-entry options and convert them into
a typed :class:`~custom_components.hsem.models.sensor_config.SensorConfig`
dataclass.

This module has **no** Home Assistant entity I/O — it only reads
``config_entry.options`` via :func:`get_config_value`.  It can therefore be
called synchronously and tested without a running HA instance.
"""

from __future__ import annotations

from typing import Any, cast

import voluptuous as vol

from custom_components.hsem.const import MILP_SOLVER_TIMEOUT_DEFAULT_SECONDS
from custom_components.hsem.models.secondary_storage_entity_config import (
    SecondaryStorageEntityConfig,
)
from custom_components.hsem.models.sensor_config import (
    EVChargerConfig,
    SensorConfig,
)
from custom_components.hsem.utils.conversion import (
    convert_months_to_int,
    convert_to_boolean,
    convert_to_float,
    convert_to_int,
)
from custom_components.hsem.utils.misc import get_config_value


def build_sensor_config(
    config_entry: Any,
) -> SensorConfig:  # NOSONAR -- HA ConfigEntry; circular import risk
    """Read all config-entry options and return a populated :class:`SensorConfig`.

    This is a pure synchronous function that reads from ``config_entry.options``
    (or ``.data``) via :func:`get_config_value`.  It performs no I/O and can be
    called from tests with a mock config entry.

    Args:
        config_entry: A Home Assistant ``ConfigEntry`` or any object that
            :func:`get_config_value` accepts.

    Returns:
        A fully populated :class:`SensorConfig`.
    """
    cfg = SensorConfig()

    cfg.read_only = convert_to_boolean(get_config_value(config_entry, "hsem_read_only"))
    cfg.verbose_logging = convert_to_boolean(
        get_config_value(config_entry, "hsem_verbose_logging")
    )
    cfg.extended_attributes = convert_to_boolean(
        get_config_value(config_entry, "hsem_extended_attributes")
    )
    cfg.planner_hysteresis_enabled = convert_to_boolean(
        get_config_value(config_entry, "hsem_planner_hysteresis_enabled")
    )
    cfg.planner_hysteresis_absolute = (
        convert_to_float(
            get_config_value(config_entry, "hsem_planner_hysteresis_absolute")
        )
        or 0.0
    )
    cfg.planner_hysteresis_percentage = (
        convert_to_float(
            get_config_value(config_entry, "hsem_planner_hysteresis_percentage")
        )
        or 0.0
    )
    cfg.planner_window_hysteresis_minutes = (
        convert_to_int(
            get_config_value(config_entry, "hsem_planner_window_hysteresis_minutes")
        )
        or 0
    )
    _update_interval = convert_to_int(
        get_config_value(config_entry, "hsem_update_interval")
    )
    cfg.update_interval = _update_interval if _update_interval is not None else 5
    _rec_interval_min = convert_to_int(
        get_config_value(config_entry, "hsem_recommendation_interval_minutes")
    )
    cfg.recommendation_interval_minutes = (
        _rec_interval_min if _rec_interval_min is not None else 15
    )
    _rec_interval_len = convert_to_int(
        get_config_value(config_entry, "hsem_recommendation_interval_length")
    )
    cfg.recommendation_interval_length = (
        _rec_interval_len if _rec_interval_len is not None else 48
    )
    _price_update_interval = convert_to_int(
        get_config_value(config_entry, "hsem_electricity_price_update_interval")
    )
    cfg.electricity_price_update_interval = (
        _price_update_interval if _price_update_interval is not None else 15
    )
    _milp_solver_timeout = convert_to_float(
        get_config_value(config_entry, "hsem_milp_solver_timeout_seconds")
    )
    cfg.milp_solver_timeout_seconds = (
        _milp_solver_timeout
        if _milp_solver_timeout is not None
        else MILP_SOLVER_TIMEOUT_DEFAULT_SECONDS
    )
    # Seasonal months
    months_winter = get_config_value(config_entry, "hsem_months_winter")
    months_summer = get_config_value(config_entry, "hsem_months_summer")
    cfg.months_winter = (
        convert_months_to_int(months_winter) if isinstance(months_winter, list) else []
    )
    cfg.months_summer = (
        convert_months_to_int(months_summer) if isinstance(months_summer, list) else []
    )
    cfg.seasonal_fill_mode = str(
        get_config_value(config_entry, "hsem_seasonal_fill_mode")
    )

    # Huawei Solar device IDs
    cfg.huawei_solar_device_id_inverter_1 = get_config_value(
        config_entry, "hsem_huawei_solar_device_id_inverter_1"
    )
    cfg.huawei_solar_device_id_inverter_2 = get_config_value(
        config_entry, "hsem_huawei_solar_device_id_inverter_2"
    )
    if (
        cfg.huawei_solar_device_id_inverter_2 is not None
        and len(cfg.huawei_solar_device_id_inverter_2) == 0
    ):
        cfg.huawei_solar_device_id_inverter_2 = None

    cfg.huawei_solar_device_id_batteries = get_config_value(
        config_entry, "hsem_huawei_solar_device_id_batteries"
    )

    # Huawei Solar entity IDs
    cfg.huawei_solar_batteries_working_mode = get_config_value(
        config_entry, "hsem_huawei_solar_batteries_working_mode"
    )
    cfg.huawei_solar_batteries_end_of_discharge_soc = get_config_value(
        config_entry, "hsem_huawei_solar_batteries_end_of_discharge_soc"
    )
    cfg.huawei_solar_batteries_state_of_capacity = get_config_value(
        config_entry, "hsem_huawei_solar_batteries_state_of_capacity"
    )
    cfg.huawei_solar_batteries_charging_cutoff_capacity = get_config_value(
        config_entry, "hsem_huawei_solar_batteries_charging_cutoff_capacity"
    )
    cfg.huawei_solar_batteries_grid_charge_cutoff_soc = get_config_value(
        config_entry, "hsem_huawei_solar_batteries_grid_charge_cutoff_soc"
    )
    cfg.huawei_solar_batteries_grid_charge_maximum_power = get_config_value(
        config_entry, "hsem_huawei_solar_batteries_grid_charge_maximum_power"
    )
    cfg.huawei_solar_batteries_charge_discharge_power = get_config_value(
        config_entry, "hsem_huawei_solar_batteries_charge_discharge_power"
    )
    cfg.huawei_solar_batteries_maximum_charging_power = get_config_value(
        config_entry, "hsem_huawei_solar_batteries_maximum_charging_power"
    )
    cfg.huawei_solar_batteries_maximum_discharging_power = get_config_value(
        config_entry, "hsem_huawei_solar_batteries_maximum_discharging_power"
    )
    cfg.huawei_solar_batteries_tou_charging_and_discharging_periods = get_config_value(
        config_entry,
        "hsem_huawei_solar_batteries_tou_charging_and_discharging_periods",
    )
    cfg.huawei_solar_batteries_excess_pv_energy_use_in_tou = get_config_value(
        config_entry, "hsem_huawei_solar_batteries_excess_pv_energy_use_in_tou"
    )
    cfg.huawei_solar_batteries_forcible_charge = get_config_value(
        config_entry, "hsem_huawei_solar_batteries_forcible_charge"
    )
    cfg.huawei_solar_inverter_active_power_control = get_config_value(
        config_entry, "hsem_huawei_solar_inverter_active_power_control"
    )
    cfg.huawei_solar_batteries_rated_capacity = get_config_value(
        config_entry, "hsem_huawei_solar_batteries_rated_capacity"
    )
    cfg.huawei_solar_power_meter_phase_a_active_power = get_config_value(
        config_entry, "hsem_huawei_solar_power_meter_phase_a_active_power"
    )
    cfg.huawei_solar_power_meter_phase_b_active_power = get_config_value(
        config_entry, "hsem_huawei_solar_power_meter_phase_b_active_power"
    )
    cfg.huawei_solar_power_meter_phase_c_active_power = get_config_value(
        config_entry, "hsem_huawei_solar_power_meter_phase_c_active_power"
    )

    # Power meters
    cfg.house_consumption_power = get_config_value(
        config_entry, "hsem_house_consumption_power"
    )
    cfg.solar_production_power = get_config_value(
        config_entry, "hsem_solar_production_power"
    )
    cfg.house_power_includes_ev_charger_power = convert_to_boolean(
        get_config_value(config_entry, "hsem_house_power_includes_ev_charger_power")
    )
    _main_fuse = convert_to_int(get_config_value(config_entry, "hsem_main_fuse_amps"))
    cfg.main_fuse_amps = _main_fuse if _main_fuse is not None else 0
    _main_fuse_phases = convert_to_int(
        get_config_value(config_entry, "hsem_main_fuse_phases")
    )
    cfg.main_fuse_phases = _main_fuse_phases if _main_fuse_phases is not None else 3
    cfg.phase_aware_charging_enabled = convert_to_boolean(
        get_config_value(config_entry, "hsem_phase_aware_charging_enabled")
    )
    _max_export_kw = convert_to_float(
        get_config_value(config_entry, "hsem_max_grid_export_power_kw")
    )
    cfg.max_grid_export_power_kw = _max_export_kw if _max_export_kw is not None else 0.0

    # Optional dedicated-load secondary storage
    secondary = SecondaryStorageEntityConfig()
    secondary.enabled = convert_to_boolean(
        get_config_value(config_entry, "hsem_secondary_storage_enabled")
    )
    secondary.control_enabled = convert_to_boolean(
        get_config_value(config_entry, "hsem_secondary_storage_control_enabled")
    )
    secondary.soc_entity = _optional_entity(
        get_config_value(config_entry, "hsem_secondary_storage_soc_entity")
    )
    secondary.battery_net_power_entity = _optional_entity(
        get_config_value(
            config_entry, "hsem_secondary_storage_battery_net_power_entity"
        )
    )
    secondary.load_power_entity = _optional_entity(
        get_config_value(config_entry, "hsem_secondary_storage_load_power_entity")
    )
    secondary.output_source_priority_entity = _optional_entity(
        get_config_value(
            config_entry, "hsem_secondary_storage_output_source_priority_entity"
        )
    )
    secondary.charger_source_priority_entity = _optional_entity(
        get_config_value(
            config_entry, "hsem_secondary_storage_charger_source_priority_entity"
        )
    )
    secondary.max_charge_current_entity = _optional_entity(
        get_config_value(
            config_entry, "hsem_secondary_storage_max_charge_current_entity"
        )
    )
    secondary.capacity_kwh = _float_config(
        config_entry, "hsem_secondary_storage_capacity_kwh", 15.0
    )
    secondary.min_soc_pct = _float_config(
        config_entry, "hsem_secondary_storage_min_soc_pct", 20.0
    )
    secondary.max_soc_pct = _float_config(
        config_entry, "hsem_secondary_storage_max_soc_pct", 100.0
    )
    secondary.nominal_voltage_v = _float_config(
        config_entry, "hsem_secondary_storage_nominal_voltage_v", 24.0
    )
    secondary.min_charge_current_a = _float_config(
        config_entry, "hsem_secondary_storage_min_charge_current_a", 10.0
    )
    secondary.max_charge_current_a = _float_config(
        config_entry, "hsem_secondary_storage_max_charge_current_a", 60.0
    )
    secondary.charge_efficiency_pct = _float_config(
        config_entry, "hsem_secondary_storage_charge_efficiency_pct", 93.0
    )
    secondary.discharge_efficiency_pct = _float_config(
        config_entry, "hsem_secondary_storage_discharge_efficiency_pct", 93.0
    )
    secondary.inverter_standby_power_w = _float_config(
        config_entry, "hsem_secondary_storage_inverter_standby_power_w", 55.0
    )
    secondary.cycle_cost_per_kwh = _float_config(
        config_entry, "hsem_secondary_storage_cycle_cost_per_kwh", 0.0
    )
    secondary.base_load_includes_dedicated_load = convert_to_boolean(
        get_config_value(
            config_entry,
            "hsem_secondary_storage_base_load_includes_dedicated_load",
        )
    )
    secondary.allow_primary_battery_transfer = convert_to_boolean(
        get_config_value(
            config_entry, "hsem_secondary_storage_allow_primary_battery_transfer"
        )
    )
    _secondary_grid_phase = convert_to_int(
        get_config_value(config_entry, "hsem_secondary_storage_grid_phase")
    )
    secondary.grid_phase = (
        _secondary_grid_phase if _secondary_grid_phase in {1, 2, 3} else 3
    )
    cfg.secondary_storage = secondary

    # Solcast
    cfg.solcast_pv_forecast_forecast_today = get_config_value(
        config_entry, "hsem_solcast_pv_forecast_forecast_today"
    )
    cfg.solcast_pv_forecast_forecast_tomorrow = get_config_value(
        config_entry, "hsem_solcast_pv_forecast_forecast_tomorrow"
    )
    cfg.solcast_pv_forecast_forecast_likelihood = (
        get_config_value(config_entry, "hsem_solcast_pv_forecast_forecast_likelihood")
        or "pv_estimate"
    )

    # Electricity prices (generic — Energi Data Service, Nordpool, Amber Electric, …)
    cfg.import_electricity_price_sensor = get_config_value(
        config_entry, "hsem_import_electricity_price_sensor"
    )
    cfg.export_electricity_price_sensor = get_config_value(
        config_entry, "hsem_export_electricity_price_sensor"
    )
    cfg.import_electricity_price_forecast_sensor = _optional_entity(
        get_config_value(config_entry, "hsem_import_electricity_price_forecast_sensor")
    )
    cfg.export_electricity_price_forecast_sensor = _optional_entity(
        get_config_value(config_entry, "hsem_export_electricity_price_forecast_sensor")
    )
    cfg.export_electricity_min_price = (
        convert_to_float(
            get_config_value(config_entry, "hsem_export_electricity_min_price")
        )
        or 0.0
    )

    # Predicted prices for the unpublished horizon (valuation only)
    cfg.price_forecast_valuation_enabled = bool(
        get_config_value(config_entry, "hsem_price_forecast_valuation_enabled")
    )
    cfg.price_forecast_valuation_sensor = _optional_entity(
        get_config_value(config_entry, "hsem_price_forecast_valuation_sensor")
    )
    cfg.price_forecast_valuation_margin = (
        convert_to_float(
            get_config_value(config_entry, "hsem_price_forecast_valuation_margin")
        )
        or 0.0
    )

    # First EV charger
    ev = EVChargerConfig()
    ev.status_entity = _optional_entity(
        get_config_value(config_entry, "hsem_ev_charger_status")
    )
    ev.power_entity = _optional_entity(
        get_config_value(config_entry, "hsem_ev_charger_power")
    )
    ev.soc_entity = _optional_entity(get_config_value(config_entry, "hsem_ev_soc"))
    ev.connected_entity = _optional_entity(
        get_config_value(config_entry, "hsem_ev_connected")
    )
    ev.allow_charge_past_target_soc = convert_to_boolean(
        get_config_value(config_entry, "hsem_ev_allow_charge_past_target_soc")
    )
    ev.past_target_confidence_factor = (
        convert_to_float(
            get_config_value(config_entry, "hsem_ev_past_target_confidence_factor")
        )
        or 0.9
    )
    ev.force_max_discharge_power = convert_to_boolean(
        get_config_value(config_entry, "hsem_ev_charger_force_max_discharge_power")
    )
    _ev_max_discharge = convert_to_int(
        get_config_value(config_entry, "hsem_ev_charger_max_discharge_power")
    )
    ev.max_discharge_power = _ev_max_discharge if _ev_max_discharge is not None else 0
    cfg.ev = ev

    # Second EV charger
    ev2 = EVChargerConfig()
    ev2.status_entity = _optional_entity(
        get_config_value(config_entry, "hsem_ev_second_charger_status")
    )
    ev2.power_entity = _optional_entity(
        get_config_value(config_entry, "hsem_ev_second_charger_power")
    )
    ev2.soc_entity = _optional_entity(
        get_config_value(config_entry, "hsem_ev_second_soc")
    )
    ev2.connected_entity = _optional_entity(
        get_config_value(config_entry, "hsem_ev_second_connected")
    )
    ev2.allow_charge_past_target_soc = convert_to_boolean(
        get_config_value(config_entry, "hsem_ev_second_allow_charge_past_target_soc")
    )
    ev2.past_target_confidence_factor = (
        convert_to_float(
            get_config_value(
                config_entry, "hsem_ev_second_past_target_confidence_factor"
            )
        )
        or 0.9
    )
    ev2.force_max_discharge_power = convert_to_boolean(
        get_config_value(
            config_entry, "hsem_ev_second_charger_force_max_discharge_power"
        )
    )
    _ev2_max_discharge = convert_to_int(
        get_config_value(config_entry, "hsem_ev_second_charger_max_discharge_power")
    )
    ev2.max_discharge_power = (
        _ev2_max_discharge if _ev2_max_discharge is not None else 0
    )
    cfg.ev_second = ev2

    # Battery economics
    cfg.batteries_charge_efficiency = (
        convert_to_float(
            get_config_value(config_entry, "hsem_batteries_charge_efficiency")
        )
        or 98.0
    )
    cfg.batteries_discharge_efficiency = (
        convert_to_float(
            get_config_value(config_entry, "hsem_batteries_discharge_efficiency")
        )
        or 98.0
    )
    cfg.batteries_purchase_price = (
        convert_to_float(
            get_config_value(config_entry, "hsem_batteries_purchase_price")
        )
        or 0.0
    )
    _expected_cycles = convert_to_int(
        get_config_value(config_entry, "hsem_batteries_expected_cycles")
    )
    cfg.batteries_expected_cycles = (
        _expected_cycles if _expected_cycles is not None else 6000
    )
    cfg.batteries_cycle_cost = (
        convert_to_float(get_config_value(config_entry, "hsem_batteries_cycle_cost"))
        or 0.0
    )
    cfg.batteries_capacity_loss_pct = (
        convert_to_float(
            get_config_value(config_entry, "hsem_batteries_capacity_loss_pct")
        )
        or 30.0
    )

    # Excess export
    cfg.batteries_enable_excess_export = bool(
        get_config_value(config_entry, "hsem_batteries_enable_excess_export")
    )
    cfg.batteries_excess_export_discharge_buffer = (
        convert_to_float(
            get_config_value(
                config_entry, "hsem_batteries_excess_export_discharge_buffer"
            )
        )
        or 10.0
    )
    # Per-slot hard floor for intentional battery-to-grid export (issue #752).
    # 0.0 = disabled — fully backward compatible.
    _battery_export_min_price = convert_to_float(
        get_config_value(config_entry, "hsem_batteries_export_min_price")
    )
    cfg.batteries_export_min_price = (
        _battery_export_min_price if _battery_export_min_price is not None else 0.0
    )

    # Wait mode behaviour
    _wait_mode_behavior = get_config_value(
        config_entry, "hsem_batteries_wait_mode_behavior"
    )
    cfg.batteries_wait_mode_behavior = (
        str(_wait_mode_behavior).lower()
        if _wait_mode_behavior is not None
        else "strict"
    )

    # EV planned load integration
    cfg.ev_planned_load_enabled = convert_to_boolean(
        get_config_value(config_entry, "hsem_ev_planned_load_enabled")
    )
    _bat_cap = convert_to_float(
        get_config_value(config_entry, "hsem_ev_planned_load_battery_capacity_kwh")
    )
    cfg.ev_planned_load_battery_capacity_kwh = _bat_cap if _bat_cap is not None else 0.0
    _chg_pwr = convert_to_float(
        get_config_value(config_entry, "hsem_ev_planned_load_charger_power_kw")
    )
    cfg.ev_planned_load_charger_power_kw = _chg_pwr if _chg_pwr is not None else 0.0
    _chg_eff = convert_to_float(
        get_config_value(config_entry, "hsem_ev_planned_load_charger_efficiency")
    )
    cfg.ev_planned_load_charger_efficiency_pct = (
        _chg_eff if _chg_eff is not None else 100.0
    )
    _min_pwr = convert_to_float(
        get_config_value(config_entry, "hsem_ev_planned_load_charger_min_power_w")
    )
    cfg.ev_planned_load_charger_min_power_w = (
        _min_pwr if _min_pwr is not None else 1380.0
    )

    # Second EV planned load integration
    cfg.ev_second_planned_load_enabled = convert_to_boolean(
        get_config_value(config_entry, "hsem_ev_second_planned_load_enabled")
    )
    _s2_cap = convert_to_float(
        get_config_value(
            config_entry, "hsem_ev_second_planned_load_battery_capacity_kwh"
        )
    )
    cfg.ev_second_planned_load_battery_capacity_kwh = (
        _s2_cap if _s2_cap is not None else 0.0
    )
    _s2_pwr = convert_to_float(
        get_config_value(config_entry, "hsem_ev_second_planned_load_charger_power_kw")
    )
    cfg.ev_second_planned_load_charger_power_kw = (
        _s2_pwr if _s2_pwr is not None else 0.0
    )
    _s2_eff = convert_to_float(
        get_config_value(config_entry, "hsem_ev_second_planned_load_charger_efficiency")
    )
    cfg.ev_second_planned_load_charger_efficiency_pct = (
        _s2_eff if _s2_eff is not None else 100.0
    )
    _s2_min = convert_to_float(
        get_config_value(
            config_entry, "hsem_ev_second_planned_load_charger_min_power_w"
        )
    )
    cfg.ev_second_planned_load_charger_min_power_w = (
        _s2_min if _s2_min is not None else 1380.0
    )

    # EV auto-Full on negative price (issue #609)
    cfg.ev_auto_full_negative_price = convert_to_boolean(
        get_config_value(config_entry, "hsem_ev_auto_full_negative_price")
    )

    # Daily plan-vs-actual tracking — optional cumulative energy meter entities.
    cfg.grid_import_energy_entity = _optional_entity(
        get_config_value(config_entry, "hsem_grid_import_energy_entity")
    )
    cfg.grid_export_energy_entity = _optional_entity(
        get_config_value(config_entry, "hsem_grid_export_energy_entity")
    )
    cfg.pv_energy_entity = _optional_entity(
        get_config_value(config_entry, "hsem_pv_energy_entity")
    )

    # ML consumption prediction — toggle and settings
    cfg.ml_consumption_enabled = bool(
        get_config_value(config_entry, "hsem_ml_consumption_enabled")
    )
    cfg.ml_consumption_energy_entity = _optional_entity(
        get_config_value(config_entry, "hsem_ml_consumption_energy_entity")
    )
    _ml_hist_days = convert_to_int(
        get_config_value(config_entry, "hsem_ml_consumption_history_days")
    )
    cfg.ml_consumption_history_days = _ml_hist_days if _ml_hist_days is not None else 14
    cfg.ml_consumption_net_consumption = bool(
        get_config_value(config_entry, "hsem_ml_consumption_net_consumption")
    )
    cfg.ml_consumption_sequential = bool(
        get_config_value(config_entry, "hsem_ml_consumption_sequential")
    )
    cfg.ml_consumption_temperature_entity = _optional_entity(
        get_config_value(config_entry, "hsem_ml_consumption_temperature_entity")
    )

    # Consumption weights
    _w1d = convert_to_int(
        get_config_value(config_entry, "hsem_house_consumption_energy_weight_1d")
    )
    cfg.house_consumption_energy_weight_1d = _w1d if _w1d is not None else 25
    _w3d = convert_to_int(
        get_config_value(config_entry, "hsem_house_consumption_energy_weight_3d")
    )
    cfg.house_consumption_energy_weight_3d = _w3d if _w3d is not None else 30
    _w7d = convert_to_int(
        get_config_value(config_entry, "hsem_house_consumption_energy_weight_7d")
    )
    cfg.house_consumption_energy_weight_7d = _w7d if _w7d is not None else 30
    _w14d = convert_to_int(
        get_config_value(config_entry, "hsem_house_consumption_energy_weight_14d")
    )
    cfg.house_consumption_energy_weight_14d = _w14d if _w14d is not None else 15

    return cfg


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _float_config(config_entry: Any, key: str, default: float) -> float:
    """Return a numeric config value while preserving an explicit zero."""
    value = convert_to_float(get_config_value(config_entry, key))
    return value if value is not None else default


def _optional_entity(
    value: Any,
) -> str | None:  # NOSONAR -- generic helper; type depends on caller
    """Return None if value is vol.UNDEFINED or falsy, else the string."""
    if value is vol.UNDEFINED or not value:
        return None
    return cast(str, value)
