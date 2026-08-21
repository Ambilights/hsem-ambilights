"""Safety-gated PowMr adapter for the secondary-storage plan."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any, Protocol

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
    ApplyResult,
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


class SecondaryControlWriteObserver(Protocol):
    """Observe the lifetime of one verified PowMr control write."""

    def secondary_control_write_started(
        self,
        entity_id: str,
        desired: str | float,
    ) -> int:
        """Return a token identifying the newly started write."""
        ...

    def secondary_control_write_finished(
        self,
        entity_id: str,
        desired: str | float,
        token: int,
        *,
        verified: bool,
        echo_expected: bool,
    ) -> None:
        """Resolve a write token after verification or cancellation."""
        ...

    def secondary_control_mode_started(
        self,
        slot_start: datetime,
        slot_end: datetime,
        mode: str,
    ) -> int:
        """Return a token for one complete ordered mode transition."""
        ...

    def secondary_control_mode_finished(
        self,
        slot_start: datetime,
        slot_end: datetime,
        mode: str,
        token: int,
        *,
        verified: bool,
    ) -> None:
        """Report whether the complete mode transition verified."""
        ...


def _quantize_current(value: float, minimum: float, maximum: float) -> float:
    """Clamp to PowMr's verified 10 A number step."""
    clamped = min(max(value, minimum), maximum)
    quantized = round(clamped / 10.0) * 10.0
    return min(max(quantized, minimum), maximum)


def resolve_secondary_control_mode(
    cfg: SensorConfig,
    live: LiveState,
    rec: HourlyRecommendation,
) -> str | None:
    """Resolve planner intent through the immediate PowMr safety guards."""
    configured = cfg.secondary_storage
    measured = live.secondary_storage
    mode = rec.secondary_storage_mode
    if mode not in {
        SECONDARY_MODE_CHARGE,
        SECONDARY_MODE_SBU,
        SECONDARY_MODE_UTILITY,
    }:
        return None
    if mode == SECONDARY_MODE_SBU and (
        measured.soc_pct is None or measured.soc_pct <= configured.min_soc_pct + 0.1
    ):
        return SECONDARY_MODE_UTILITY
    if mode == SECONDARY_MODE_CHARGE and (
        measured.soc_pct is not None
        and measured.soc_pct >= configured.max_soc_pct - 0.1
    ):
        return SECONDARY_MODE_UTILITY
    if (
        mode == SECONDARY_MODE_CHARGE
        and rec.secondary_storage_charge_current_a
        < configured.min_charge_current_a - 1e-9
    ):
        return SECONDARY_MODE_UTILITY
    return mode


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

    planned_mode = rec.secondary_storage_mode
    mode = resolve_secondary_control_mode(cfg, live, rec)
    max_soc_charge_guard = (
        planned_mode == SECONDARY_MODE_CHARGE
        and mode == SECONDARY_MODE_UTILITY
        and measured.soc_pct is not None
        and measured.soc_pct >= configured.max_soc_pct - 0.1
    )

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
        if (
            measured.charger_source_priority == POWMR_CHARGER_UTILITY
            and measured.max_charge_current_a is not None
            and target_current < measured.max_charge_current_a - 1e-9
        ):
            # Disarm grid charging before a downward current change.  If the
            # number write cannot be verified, the old higher current must not
            # remain live for the verifier retry/backoff interval.
            return [
                SecondaryWrite("select", charger_entity, POWMR_CHARGER_SOLAR_ONLY),
                SecondaryWrite("select", output_entity, POWMR_OUTPUT_UTILITY),
                SecondaryWrite("number", current_entity, target_current),
                SecondaryWrite("select", charger_entity, POWMR_CHARGER_UTILITY),
            ]
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
            SecondaryWrite("select", charger_entity, POWMR_CHARGER_SOLAR_ONLY),
            SecondaryWrite("select", output_entity, POWMR_OUTPUT_UTILITY),
        ]
    return []


def is_fail_closed_secondary_plan(
    cfg: SensorConfig,
    operations: list[SecondaryWrite],
) -> bool:
    """Return whether a plan can only stop PowMr grid charging and discharge."""
    configured = cfg.secondary_storage
    output_entity = configured.output_source_priority_entity
    charger_entity = configured.charger_source_priority_entity
    stopped_charger = False
    routed_load_to_utility = False
    for operation in operations:
        if operation.kind != "select":
            return False
        if operation.entity_id == charger_entity:
            if operation.desired != POWMR_CHARGER_SOLAR_ONLY:
                return False
            stopped_charger = True
        elif operation.entity_id == output_entity:
            if operation.desired != POWMR_OUTPUT_UTILITY:
                return False
            routed_load_to_utility = True
        else:
            return False
    return stopped_charger and routed_load_to_utility


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


async def _async_apply_secondary_write(
    sensor: Any,
    operation: SecondaryWrite,
    observer: SecondaryControlWriteObserver | None,
) -> ApplyResult:
    """Write and verify one PowMr control while acknowledging its state echo."""
    numeric = operation.kind == "number"
    token = (
        observer.secondary_control_write_started(
            operation.entity_id,
            operation.desired,
        )
        if observer is not None
        else None
    )
    verified = False
    echo_expected = False
    try:
        result = await async_write_and_verify(
            entity_id=operation.entity_id,
            desired=operation.desired,
            writer=partial(_execute_write, sensor, operation),
            reader=partial(_read_entity, sensor, operation.entity_id, numeric),
            backoff=get_write_failure_backoff(sensor),
        )
        verified = result.status in {ApplyStatus.OK, ApplyStatus.SKIPPED}
        echo_expected = result.status == ApplyStatus.OK
        return result
    finally:
        if observer is not None and token is not None:
            observer.secondary_control_write_finished(
                operation.entity_id,
                operation.desired,
                token,
                verified=verified,
                echo_expected=echo_expected,
            )


async def async_apply_secondary_storage(
    sensor: Any,
    cfg: SensorConfig,
    live: LiveState,
    rec: HourlyRecommendation,
    *,
    fail_closed_only: bool = False,
    control_write_observer: SecondaryControlWriteObserver | None = None,
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
    fail_closed_stop = is_fail_closed_secondary_plan(cfg, operations)
    if fail_closed_only and not fail_closed_stop:
        _LOGGER.warning(
            "PowMr enabling transition blocked after a failed/unverified Huawei write"
        )
        return summary

    resolved_mode = resolve_secondary_control_mode(cfg, live, rec)
    if resolved_mode is None:
        return summary
    mode_token = (
        control_write_observer.secondary_control_mode_started(
            rec.start,
            rec.end,
            resolved_mode,
        )
        if control_write_observer is not None
        else None
    )
    mode_verified = False
    try:
        charger_safely_disabled = False
        for operation in operations:
            result = await _async_apply_secondary_write(
                sensor,
                operation,
                control_write_observer,
            )
            summary.results.append(result)
            if (
                operation.entity_id
                == cfg.secondary_storage.charger_source_priority_entity
                and operation.desired == POWMR_CHARGER_SOLAR_ONLY
                and result.status in {ApplyStatus.OK, ApplyStatus.SKIPPED}
            ):
                charger_safely_disabled = True
            if result.status not in {ApplyStatus.OK, ApplyStatus.SKIPPED}:
                if operation.kind == "number" and not charger_safely_disabled:
                    # A current that cannot be verified is unsafe while utility
                    # charging may still be armed.  This installation has no PowMr
                    # PV input, so Only Solar is a complete charging stop.
                    fallback = SecondaryWrite(
                        "select",
                        cfg.secondary_storage.charger_source_priority_entity or "",
                        POWMR_CHARGER_SOLAR_ONLY,
                    )
                    fallback_result = await _async_apply_secondary_write(
                        sensor,
                        fallback,
                        control_write_observer,
                    )
                    summary.results.append(fallback_result)
                break
        mode_verified = len(summary.results) == len(operations) and all(
            result.status in {ApplyStatus.OK, ApplyStatus.SKIPPED}
            for result in summary.results
        )
    finally:
        if control_write_observer is not None and mode_token is not None:
            control_write_observer.secondary_control_mode_finished(
                rec.start,
                rec.end,
                resolved_mode,
                mode_token,
                verified=mode_verified,
            )
    return summary
