"""Config-flow step for dedicated-load secondary storage."""

from __future__ import annotations

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    PERCENTAGE,
    UnitOfElectricCurrent,
    UnitOfElectricPotential,
    UnitOfEnergy,
    UnitOfPower,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.selector import selector

from custom_components.hsem.utils.config_validator import async_validate_entity_ids
from custom_components.hsem.utils.misc import get_config_value


def _number(
    minimum: float,
    maximum: float,
    step: float,
    unit: str | None = None,
) -> dict:
    """Return a compact HA number-selector config."""
    config: dict = {
        "min": minimum,
        "max": maximum,
        "step": step,
        "mode": "box",
    }
    if unit:
        config["unit_of_measurement"] = unit
    return {"number": config}


async def get_secondary_storage_step_schema(
    config_entry: ConfigEntry | None,
) -> vol.Schema:
    """Return the optional secondary-storage schema."""
    value = lambda key: get_config_value(config_entry, key)  # noqa: E731
    return vol.Schema(
        {
            vol.Required(
                "hsem_secondary_storage_enabled",
                default=value("hsem_secondary_storage_enabled"),
            ): selector({"boolean": {}}),
            vol.Required(
                "hsem_secondary_storage_control_enabled",
                default=value("hsem_secondary_storage_control_enabled"),
            ): selector({"boolean": {}}),
            vol.Optional(
                "hsem_secondary_storage_soc_entity",
                default=value("hsem_secondary_storage_soc_entity"),
            ): selector({"entity": {"domain": "sensor"}}),
            vol.Optional(
                "hsem_secondary_storage_battery_net_power_entity",
                default=value("hsem_secondary_storage_battery_net_power_entity"),
            ): selector({"entity": {"domain": "sensor"}}),
            vol.Optional(
                "hsem_secondary_storage_load_power_entity",
                default=value("hsem_secondary_storage_load_power_entity"),
            ): selector({"entity": {"domain": "sensor"}}),
            vol.Optional(
                "hsem_secondary_storage_output_source_priority_entity",
                default=value("hsem_secondary_storage_output_source_priority_entity"),
            ): selector({"entity": {"domain": "select"}}),
            vol.Optional(
                "hsem_secondary_storage_charger_source_priority_entity",
                default=value("hsem_secondary_storage_charger_source_priority_entity"),
            ): selector({"entity": {"domain": "select"}}),
            vol.Optional(
                "hsem_secondary_storage_max_charge_current_entity",
                default=value("hsem_secondary_storage_max_charge_current_entity"),
            ): selector({"entity": {"domain": "number"}}),
            vol.Required(
                "hsem_secondary_storage_capacity_kwh",
                default=value("hsem_secondary_storage_capacity_kwh"),
            ): selector(_number(0.1, 200.0, 0.1, UnitOfEnergy.KILO_WATT_HOUR)),
            vol.Required(
                "hsem_secondary_storage_min_soc_pct",
                default=value("hsem_secondary_storage_min_soc_pct"),
            ): selector(_number(0.0, 99.0, 1.0, PERCENTAGE)),
            vol.Required(
                "hsem_secondary_storage_max_soc_pct",
                default=value("hsem_secondary_storage_max_soc_pct"),
            ): selector(_number(1.0, 100.0, 1.0, PERCENTAGE)),
            vol.Required(
                "hsem_secondary_storage_nominal_voltage_v",
                default=value("hsem_secondary_storage_nominal_voltage_v"),
            ): selector(_number(1.0, 1000.0, 0.1, UnitOfElectricPotential.VOLT)),
            vol.Required(
                "hsem_secondary_storage_min_charge_current_a",
                default=value("hsem_secondary_storage_min_charge_current_a"),
            ): selector(_number(10.0, 80.0, 10.0, UnitOfElectricCurrent.AMPERE)),
            vol.Required(
                "hsem_secondary_storage_max_charge_current_a",
                default=value("hsem_secondary_storage_max_charge_current_a"),
            ): selector(_number(10.0, 80.0, 10.0, UnitOfElectricCurrent.AMPERE)),
            vol.Required(
                "hsem_secondary_storage_grid_phase",
                default=value("hsem_secondary_storage_grid_phase"),
            ): selector(_number(1.0, 3.0, 1.0)),
            vol.Required(
                "hsem_secondary_storage_charge_efficiency_pct",
                default=value("hsem_secondary_storage_charge_efficiency_pct"),
            ): selector(_number(1.0, 100.0, 0.1, PERCENTAGE)),
            vol.Required(
                "hsem_secondary_storage_discharge_efficiency_pct",
                default=value("hsem_secondary_storage_discharge_efficiency_pct"),
            ): selector(_number(1.0, 100.0, 0.1, PERCENTAGE)),
            vol.Required(
                "hsem_secondary_storage_inverter_standby_power_w",
                default=value("hsem_secondary_storage_inverter_standby_power_w"),
            ): selector(_number(0.0, 2000.0, 1.0, UnitOfPower.WATT)),
            vol.Required(
                "hsem_secondary_storage_cycle_cost_per_kwh",
                default=value("hsem_secondary_storage_cycle_cost_per_kwh"),
            ): selector(_number(0.0, 10.0, 0.001)),
            vol.Required(
                "hsem_secondary_storage_base_load_includes_dedicated_load",
                default=value(
                    "hsem_secondary_storage_base_load_includes_dedicated_load"
                ),
            ): selector({"boolean": {}}),
            vol.Required(
                "hsem_secondary_storage_allow_primary_battery_transfer",
                default=value("hsem_secondary_storage_allow_primary_battery_transfer"),
            ): selector({"boolean": {}}),
        }
    )


async def validate_secondary_storage_input(
    hass: HomeAssistant,
    user_input: dict,
) -> dict[str, str]:
    """Validate entity availability and physical limits when enabled."""
    if not bool(user_input.get("hsem_secondary_storage_enabled")):
        return {}

    required = [
        "hsem_secondary_storage_soc_entity",
        "hsem_secondary_storage_load_power_entity",
    ]
    control_fields = [
        "hsem_secondary_storage_output_source_priority_entity",
        "hsem_secondary_storage_charger_source_priority_entity",
        "hsem_secondary_storage_max_charge_current_entity",
    ]
    optional = ["hsem_secondary_storage_battery_net_power_entity"]
    if bool(user_input.get("hsem_secondary_storage_control_enabled")):
        required.extend(control_fields)
    else:
        optional.extend(control_fields)

    errors = await async_validate_entity_ids(hass, user_input, required, optional)
    min_soc = float(user_input.get("hsem_secondary_storage_min_soc_pct", 20.0))
    max_soc = float(user_input.get("hsem_secondary_storage_max_soc_pct", 100.0))
    if not 0.0 <= min_soc < max_soc <= 100.0:
        errors["hsem_secondary_storage_min_soc_pct"] = (
            "secondary_storage_invalid_soc_range"
        )

    min_current = float(
        user_input.get("hsem_secondary_storage_min_charge_current_a", 0.0)
    )
    max_current = float(
        user_input.get("hsem_secondary_storage_max_charge_current_a", 0.0)
    )
    if (
        not 10.0 <= min_current <= max_current <= 80.0
        or min_current % 10.0 != 0.0
        or max_current % 10.0 != 0.0
    ):
        errors["hsem_secondary_storage_min_charge_current_a"] = (
            "secondary_storage_invalid_charge_range"
        )
    grid_phase = int(user_input.get("hsem_secondary_storage_grid_phase", 3))
    if grid_phase not in {1, 2, 3}:
        errors["hsem_secondary_storage_grid_phase"] = (
            "secondary_storage_invalid_grid_phase"
        )
    return errors
