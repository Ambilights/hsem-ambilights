"""Safety-gated PowMr adapter for the secondary-storage plan."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN

from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.planner.secondary_storage import (
    SECONDARY_MODE_CHARGE,
    SECONDARY_MODE_SBU,
    SECONDARY_MODE_UTILITY,
)
from custom_components.hsem.utils.degraded_mode import hardware_writes_allowed
from custom_components.hsem.utils.ha_helpers import (
    async_set_number_value,
    async_set_select_option,
)
from custom_components.hsem.utils.inverter_verify import (
    ApplyStatus,
    CycleApplySummary,
    async_write_and_verify,
    get_write_failure_backoff,
)
from custom_components.hsem.utils.logger import HSEM_LOGGER as _LOGGER
from custom_components.hsem.utils.phase_power import (
    POWMR_CHARGER_SOLAR_ONLY,
    POWMR_CHARGER_UTILITY,
    POWMR_OUTPUT_SBU,
    POWMR_OUTPUT_UTILITY,
)


@dataclass(frozen=True)
class SecondaryWrite:
    """One ordered HA service operation in a safe PowMr transition."""

    kind: str
    entity_id: str
    desired: str | float


def _quantize_current(value: float, minimum: float, maximum: float) -> float:
    """Clamp to PowMr's verified 10 A number step."""
    clamped = min(max(value, minimum), maximum)
    quantized = round(clamped / 10.0) * 10.0
    return min(max(quantized, minimum), maximum)


def build_secondary_write_plan(
    cfg: SensorConfig,
    live: LiveState,
    rec: HourlyRecommendation,
) -> list[SecondaryWrite]:
    """Translate a planned mode into fail-safe, ordered PowMr operations."""
    configured = cfg.secondary_storage
    measured = live.secondary_storage
    output_entity = configured.output_source_priority_entity
    charger_entity = configured.charger_source_priority_entity
    current_entity = configured.max_charge_current_entity
    if not output_entity or not charger_entity or not current_entity:
        return []

    mode = rec.secondary_storage_mode
    max_soc_charge_guard = False
    if mode == SECONDARY_MODE_SBU and (
        measured.soc_pct is None or measured.soc_pct <= configured.min_soc_pct + 0.1
    ):
        mode = SECONDARY_MODE_UTILITY
    if mode == SECONDARY_MODE_CHARGE and (
        measured.soc_pct is not None
        and measured.soc_pct >= configured.max_soc_pct - 0.1
    ):
        mode = SECONDARY_MODE_UTILITY
        max_soc_charge_guard = True
    if (
        mode == SECONDARY_MODE_CHARGE
        and rec.secondary_storage_charge_current_a
        < configured.min_charge_current_a - 1e-9
    ):
        mode = SECONDARY_MODE_UTILITY

    if mode == SECONDARY_MODE_SBU:
        return [
            SecondaryWrite("select", charger_entity, POWMR_CHARGER_SOLAR_ONLY),
            SecondaryWrite("select", output_entity, POWMR_OUTPUT_SBU),
        ]
    if mode == SECONDARY_MODE_CHARGE:
        target_current = _quantize_current(
            rec.secondary_storage_charge_current_a,
            configured.min_charge_current_a,
            configured.max_charge_current_a,
        )
        return [
            SecondaryWrite("select", output_entity, POWMR_OUTPUT_UTILITY),
            SecondaryWrite("number", current_entity, target_current),
            SecondaryWrite("select", charger_entity, POWMR_CHARGER_UTILITY),
        ]
    if mode == SECONDARY_MODE_UTILITY:
        if max_soc_charge_guard:
            return [
                SecondaryWrite("select", charger_entity, POWMR_CHARGER_SOLAR_ONLY),
                SecondaryWrite("select", output_entity, POWMR_OUTPUT_UTILITY),
            ]
        return [
            SecondaryWrite("select", output_entity, POWMR_OUTPUT_UTILITY),
            SecondaryWrite("select", charger_entity, POWMR_CHARGER_SOLAR_ONLY),
        ]
    return []


def _read_entity(sensor: Any, entity_id: str, numeric: bool) -> str | float | None:
    """Read a current HA state for write verification."""
    state = sensor.hass.states.get(entity_id)
    if state is None or state.state in {STATE_UNKNOWN, STATE_UNAVAILABLE}:
        return None
    if numeric:
        try:
            return float(state.state)
        except TypeError, ValueError:
            return None
    return str(state.state)


async def _execute_write(sensor: Any, operation: SecondaryWrite) -> None:
    """Execute one typed PowMr service operation."""
    if operation.kind == "number":
        await async_set_number_value(
            sensor,
            operation.entity_id,
            float(operation.desired),
        )
        return
    await async_set_select_option(
        sensor,
        operation.entity_id,
        str(operation.desired),
    )


async def async_apply_secondary_storage(
    sensor: Any,
    cfg: SensorConfig,
    live: LiveState,
    rec: HourlyRecommendation,
) -> CycleApplySummary:
    """Apply the current PowMr plan behind global and feature-specific gates."""
    summary = CycleApplySummary()
    if (
        cfg.read_only
        or not cfg.secondary_storage.enabled
        or not cfg.secondary_storage.control_enabled
    ):
        _LOGGER.debug(
            "PowMr writes skipped — read_only=%s enabled=%s control_enabled=%s",
            cfg.read_only,
            cfg.secondary_storage.enabled,
            cfg.secondary_storage.control_enabled,
        )
        return summary
    if not hardware_writes_allowed(live.degraded_mode):
        _LOGGER.warning(
            "PowMr writes blocked — degraded mode: %s",
            live.degraded_mode.value,
        )
        return summary
    measured = live.secondary_storage
    if (
        measured.soc_pct is None
        or not 0.0 <= measured.soc_pct <= 100.0
        or measured.load_power_w is None
        or measured.load_power_w < 0.0
    ):
        _LOGGER.warning("PowMr writes blocked — required live telemetry is invalid")
        return summary

    operations = build_secondary_write_plan(cfg, live, rec)
    if not operations:
        _LOGGER.warning("PowMr write plan is empty; no hardware changes applied")
        return summary

    for operation in operations:
        numeric = operation.kind == "number"
        result = await async_write_and_verify(
            entity_id=operation.entity_id,
            desired=operation.desired,
            writer=partial(_execute_write, sensor, operation),
            reader=partial(_read_entity, sensor, operation.entity_id, numeric),
            backoff=get_write_failure_backoff(sensor),
        )
        summary.results.append(result)
        if result.status not in {ApplyStatus.OK, ApplyStatus.SKIPPED}:
            break
    return summary
