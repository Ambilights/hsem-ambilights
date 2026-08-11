"""Diagnostic sensor exposing the current and future PowMr shadow plan."""

from __future__ import annotations

from typing import Any, override

from homeassistant.components.sensor import SensorEntity
from homeassistant.components.sensor.const import SensorDeviceClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNAVAILABLE, EntityCategory

from custom_components.hsem.coordinator import (
    CoordinatorData,
    HSEMDataUpdateCoordinator,
)
from custom_components.hsem.entity import HSEMCoordinatorEntity, HSEMEntity
from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.planner.secondary_storage import (
    SECONDARY_MODE_CHARGE,
    SECONDARY_MODE_SBU,
    SECONDARY_MODE_UTILITY,
)
from custom_components.hsem.utils.sensornames.secondary import (
    get_secondary_storage_plan_sensor_entity_id,
    get_secondary_storage_plan_sensor_name,
    get_secondary_storage_plan_sensor_unique_id,
)

_DISABLED = "disabled"
_VALID_STATES = {
    _DISABLED,
    SECONDARY_MODE_CHARGE,
    SECONDARY_MODE_SBU,
    SECONDARY_MODE_UTILITY,
    STATE_UNAVAILABLE,
}


def _plan_windows(
    slots: list[HourlyRecommendation],
) -> list[dict[str, Any]]:
    """Coalesce consecutive secondary modes into compact UI windows."""
    windows: list[dict[str, Any]] = []
    for slot in slots:
        mode = slot.secondary_storage_mode
        if mode not in _VALID_STATES or mode in {_DISABLED, STATE_UNAVAILABLE}:
            continue
        if windows and windows[-1]["mode"] == mode:
            window = windows[-1]
            window["end"] = slot.end.isoformat()
            window["charged_kwh"] = round(
                window["charged_kwh"] + slot.secondary_storage_charged_kwh, 3
            )
            window["discharged_kwh"] = round(
                window["discharged_kwh"] + slot.secondary_storage_discharged_kwh,
                3,
            )
            window["grid_import_kwh"] = round(
                window["grid_import_kwh"] + slot.secondary_storage_grid_import_kwh,
                3,
            )
            window["soc_at_end_pct"] = slot.secondary_storage_estimated_soc_pct
            window["charge_current_a"] = max(
                window["charge_current_a"],
                slot.secondary_storage_charge_current_a,
            )
            continue
        windows.append(
            {
                "start": slot.start.isoformat(),
                "end": slot.end.isoformat(),
                "mode": mode,
                "charged_kwh": slot.secondary_storage_charged_kwh,
                "discharged_kwh": slot.secondary_storage_discharged_kwh,
                "grid_import_kwh": slot.secondary_storage_grid_import_kwh,
                "soc_at_end_pct": slot.secondary_storage_estimated_soc_pct,
                "charge_current_a": slot.secondary_storage_charge_current_a,
            }
        )
    return windows


class HSEMSecondaryStoragePlanSensor(
    HSEMCoordinatorEntity,
    SensorEntity,
    HSEMEntity,
):
    """Expose the optimiser's Utility/Charge/SBU plan without requiring writes."""

    _attr_icon = "mdi:battery-clock"
    _attr_has_entity_name = True
    _attr_translation_key = "secondary_storage_plan"
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = sorted(_VALID_STATES)
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(
        self,
        config_entry: ConfigEntry,
        coordinator: HSEMDataUpdateCoordinator,
    ) -> None:
        """Initialise the coordinator-driven plan sensor."""
        HSEMCoordinatorEntity.__init__(self, coordinator)
        HSEMEntity.__init__(self, config_entry)
        self._attr_unique_id = get_secondary_storage_plan_sensor_unique_id(
            config_entry.entry_id
        )
        self.entity_id = get_secondary_storage_plan_sensor_entity_id()
        self._name = get_secondary_storage_plan_sensor_name()

    @property
    @override
    def name(self) -> str:
        """Return the display name."""
        return self._name

    @property  # type: ignore[misc]  # HA stub declares state as final
    @override
    def state(self) -> str:
        """Return the active secondary mode or a diagnostic sentinel."""
        data: CoordinatorData | None = self.coordinator.data
        if data is None or data.cfg is None:
            return STATE_UNAVAILABLE
        if not data.cfg.secondary_storage.enabled:
            return _DISABLED
        recommendation = data.hourly_recommendation
        if recommendation is None:
            return STATE_UNAVAILABLE
        mode = recommendation.secondary_storage_mode
        return mode if mode in _VALID_STATES else STATE_UNAVAILABLE

    @property
    @override
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return live telemetry, the current target, and compact plan windows."""
        data: CoordinatorData | None = self.coordinator.data
        if data is None or data.cfg is None or data.live is None:
            return {}
        cfg = data.cfg.secondary_storage
        live = data.live.secondary_storage
        current = data.hourly_recommendation
        recommendations = data.hourly_recommendations
        last = next(
            (slot for slot in reversed(recommendations) if slot.secondary_storage_mode),
            None,
        )
        return {
            "enabled": cfg.enabled,
            "control_enabled": cfg.control_enabled,
            "read_only": data.cfg.read_only,
            "actual_soc_pct": live.soc_pct,
            "actual_battery_net_power_w": live.battery_net_power_w,
            "actual_load_power_w": live.load_power_w,
            "actual_output_source_priority": live.output_source_priority,
            "actual_charger_source_priority": live.charger_source_priority,
            "actual_max_charge_current_a": live.max_charge_current_a,
            "target_charge_current_a": (
                current.secondary_storage_charge_current_a if current else 0.0
            ),
            "target_soc_at_slot_end_pct": (
                current.secondary_storage_estimated_soc_pct if current else None
            ),
            "planned_soc_at_horizon_end_pct": (
                last.secondary_storage_estimated_soc_pct if last else None
            ),
            "planned_windows": _plan_windows(recommendations),
        }

    @property
    @override
    def available(self) -> bool:
        """Return True once coordinator data exists."""
        return self.coordinator.data is not None

    @property
    @override
    def should_poll(self) -> bool:
        """Return False because coordinator updates drive this entity."""
        return False
