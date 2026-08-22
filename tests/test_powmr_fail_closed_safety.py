"""Regression tests for PowMr fail-closed hardware transitions.

The installation has no PV connected to the PowMr.  Selecting ``Only Solar``
therefore disables PowMr battery charging completely while ``Utility first``
continues to supply its dedicated load from the grid.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from custom_components.hsem.coordinator import CoordinatorData

from custom_components.hsem.custom_sensors.secondary_storage_applier import (
    async_apply_secondary_storage,
)
from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.planner.secondary_storage import (
    SECONDARY_MODE_CHARGE,
    SECONDARY_MODE_SBU,
    SECONDARY_MODE_UTILITY,
)
from custom_components.hsem.utils.degraded_mode import DegradedMode
from custom_components.hsem.utils.inverter_verify import (
    ApplyResult,
    ApplyStatus,
    CycleApplySummary,
)
from custom_components.hsem.utils.phase_power import (
    POWMR_CHARGER_SOLAR_ONLY,
    POWMR_CHARGER_UTILITY,
    POWMR_OUTPUT_UTILITY,
)

_NOW = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)
_OUTPUT_ENTITY = "select.powmr_output_source_priority"
_CHARGER_ENTITY = "select.powmr_charger_source_priority"
_CURRENT_ENTITY = "number.powmr_max_charge_current"


def _config() -> SensorConfig:
    """Return the minimum enabled PowMr hardware configuration."""
    cfg = SensorConfig()
    cfg.secondary_storage.enabled = True
    cfg.secondary_storage.control_enabled = True
    cfg.secondary_storage.output_source_priority_entity = _OUTPUT_ENTITY
    cfg.secondary_storage.charger_source_priority_entity = _CHARGER_ENTITY
    cfg.secondary_storage.max_charge_current_entity = _CURRENT_ENTITY
    cfg.secondary_storage.min_charge_current_a = 10.0
    cfg.secondary_storage.max_charge_current_a = 60.0
    return cfg


def _live() -> LiveState:
    """Return valid live telemetry for a PowMr currently charging at 60 A."""
    live = LiveState()
    live._degraded_mode = DegradedMode.OK
    live.secondary_storage.soc_pct = 50.0
    live.secondary_storage.load_power_w = 200.0
    live.secondary_storage.output_source_priority = POWMR_OUTPUT_UTILITY
    live.secondary_storage.charger_source_priority = POWMR_CHARGER_UTILITY
    live.secondary_storage.max_charge_current_a = 60.0
    return live


def _recommendation(mode: str, current_a: float = 0.0) -> HourlyRecommendation:
    """Return a complete recommendation with the requested PowMr command."""
    return HourlyRecommendation(
        start=_NOW,
        end=_NOW + timedelta(minutes=15),
        avg_house_consumption_kwh=0.0,
        avg_house_consumption_1d_kwh=0.0,
        avg_house_consumption_3d_kwh=0.0,
        avg_house_consumption_7d_kwh=0.0,
        avg_house_consumption_14d_kwh=0.0,
        batteries_charged_kwh=0.0,
        batteries_discharged_kwh=0.0,
        estimated_battery_capacity_kwh=0.0,
        estimated_battery_soc_pct=0.0,
        estimated_cost_currency=0.0,
        estimated_net_consumption_kwh=0.0,
        export_price=0.0,
        grid_export_kwh=0.0,
        grid_import_kwh=0.0,
        import_price=0.0,
        recommendation=None,
        solcast_pv_estimate_kwh=0.0,
        secondary_storage_charge_current_a=current_a,
        secondary_storage_mode=mode,
    )


def _huawei_failure(status: ApplyStatus) -> CycleApplySummary:
    """Return one failed or unverified Huawei apply result."""
    return CycleApplySummary(
        results=[
            ApplyResult(
                entity_id="select.batteries_working_mode",
                desired="target",
                actual=None,
                status=status,
                attempts=3,
            )
        ]
    )


async def _apply_top_level(
    status: ApplyStatus,
    mode: str,
    current_a: float,
) -> AsyncMock:
    """Run the top-level writer with an unconfirmed Huawei battery write."""
    from custom_components.hsem.custom_sensors.working_mode_sensor import (
        HSEMWorkingModeSensor,
    )

    cfg = _config()
    live = _live()
    rec = _recommendation(mode, current_a)
    data = cast(
        "CoordinatorData",
        SimpleNamespace(
            cfg=cfg,
            live=live,
            hourly_recommendation=rec,
            current_required_battery=0.0,
            apply_summary=None,
            state=None,
        ),
    )
    sensor = MagicMock(spec=HSEMWorkingModeSensor)
    sensor.hass = MagicMock()
    sensor.coordinator = MagicMock()

    with (
        patch(
            "custom_components.hsem.custom_sensors.working_mode_sensor.async_apply_inverter_power_control",
            new_callable=AsyncMock,
            return_value=CycleApplySummary(),
        ),
        patch(
            "custom_components.hsem.custom_sensors.working_mode_sensor.async_apply_battery_settings",
            new_callable=AsyncMock,
            return_value=_huawei_failure(status),
        ),
        patch(
            "custom_components.hsem.custom_sensors.working_mode_sensor.async_apply_secondary_storage",
            new_callable=AsyncMock,
            return_value=CycleApplySummary(),
        ) as secondary_applier,
    ):
        await HSEMWorkingModeSensor._async_apply_hardware_writes(sensor, data)

    assert data.apply_summary is not None
    assert data.apply_summary.overall_status == status
    return secondary_applier


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [ApplyStatus.FAILED, ApplyStatus.UNVERIFIED])
async def test_huawei_failure_allows_powmr_utility_zero_stop(
    status: ApplyStatus,
) -> None:
    """An unrelated Huawei failure must not suppress a fail-closed PowMr stop."""
    secondary_applier = await _apply_top_level(
        status,
        SECONDARY_MODE_UTILITY,
        0.0,
    )

    secondary_applier.assert_awaited_once()
    await_args = secondary_applier.await_args
    assert await_args is not None
    assert await_args.kwargs["fail_closed_only"] is True
    assert await_args.kwargs["control_write_observer"] is not None
    effective_rec = await_args.args[3]
    assert effective_rec.secondary_storage_mode == SECONDARY_MODE_UTILITY
    assert effective_rec.secondary_storage_charge_current_a == pytest.approx(0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [ApplyStatus.FAILED, ApplyStatus.UNVERIFIED])
@pytest.mark.parametrize(
    ("mode", "current_a"),
    [(SECONDARY_MODE_CHARGE, 20.0), (SECONDARY_MODE_SBU, 0.0)],
)
async def test_huawei_failure_blocks_unsafe_powmr_transition(
    status: ApplyStatus,
    mode: str,
    current_a: float,
) -> None:
    """An unconfirmed Huawei transition must still block PowMr Charge and SBU."""
    secondary_applier = await _apply_top_level(status, mode, current_a)

    secondary_applier.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [ApplyStatus.FAILED, ApplyStatus.UNVERIFIED])
async def test_failed_current_down_throttle_disables_powmr_grid_charge(
    status: ApplyStatus,
) -> None:
    """Stop grid charging before a failing 60-to-20 A current write.

    There is no PV attached to this PowMr, so a verified ``Only Solar`` state
    is a complete charging stop.  It must be established before changing the
    current limit, and utility charging must not be re-enabled when that limit
    write is failed or unverified.
    """
    cfg = _config()
    live = _live()
    calls: list[tuple[str, str | float]] = []

    async def _verify(**kwargs: object) -> ApplyResult:
        entity_id = str(kwargs["entity_id"])
        desired = kwargs["desired"]
        assert isinstance(desired, (str, float))
        calls.append((entity_id, desired))
        if entity_id == _OUTPUT_ENTITY:
            return ApplyResult(
                entity_id=entity_id,
                desired=desired,
                actual=POWMR_OUTPUT_UTILITY,
                status=ApplyStatus.SKIPPED,
            )
        if entity_id == _CHARGER_ENTITY:
            assert desired == POWMR_CHARGER_SOLAR_ONLY
            return ApplyResult(
                entity_id=entity_id,
                desired=desired,
                actual=POWMR_CHARGER_SOLAR_ONLY,
                status=ApplyStatus.OK,
                attempts=1,
            )
        if entity_id == _CURRENT_ENTITY:
            return ApplyResult(
                entity_id=entity_id,
                desired=desired,
                actual=None if status == ApplyStatus.UNVERIFIED else 60.0,
                status=status,
                attempts=3,
            )
        raise AssertionError(f"Unexpected PowMr entity: {entity_id}")

    with patch(
        "custom_components.hsem.custom_sensors.secondary_storage_applier.async_write_and_verify",
        new_callable=AsyncMock,
        side_effect=_verify,
    ):
        summary = await async_apply_secondary_storage(
            MagicMock(),
            cfg,
            live,
            _recommendation(SECONDARY_MODE_CHARGE, 20.0),
        )

    assert [entity_id for entity_id, _desired in calls] == [
        _CHARGER_ENTITY,
        _OUTPUT_ENTITY,
        _CURRENT_ENTITY,
        _CHARGER_ENTITY,
        _OUTPUT_ENTITY,
    ]
    assert calls[0][1] == POWMR_CHARGER_SOLAR_ONLY
    assert calls[1][1] == POWMR_OUTPUT_UTILITY
    assert calls[2][1] == pytest.approx(20.0)
    assert all(desired != POWMR_CHARGER_UTILITY for _entity_id, desired in calls)
    assert [result.status for result in summary.results] == [
        ApplyStatus.OK,
        ApplyStatus.SKIPPED,
        status,
        ApplyStatus.OK,
        ApplyStatus.SKIPPED,
    ]
    assert summary.overall_status == status


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [ApplyStatus.FAILED, ApplyStatus.UNVERIFIED])
@pytest.mark.parametrize("failed_entity", [_CHARGER_ENTITY, _OUTPUT_ENTITY])
async def test_utility_stop_is_charger_first_and_fail_closed(
    status: ApplyStatus,
    failed_entity: str,
) -> None:
    """A stop failure must never arm more L3 load or suppress the charger stop."""
    cfg = _config()
    live = _live()
    calls: list[tuple[str, str | float]] = []

    async def _verify(**kwargs: object) -> ApplyResult:
        entity_id = str(kwargs["entity_id"])
        desired = kwargs["desired"]
        assert isinstance(desired, (str, float))
        calls.append((entity_id, desired))
        if entity_id == failed_entity:
            return ApplyResult(
                entity_id=entity_id,
                desired=desired,
                actual=None,
                status=status,
                attempts=3,
            )
        return ApplyResult(
            entity_id=entity_id,
            desired=desired,
            actual=desired,
            status=ApplyStatus.OK,
            attempts=1,
        )

    with patch(
        "custom_components.hsem.custom_sensors.secondary_storage_applier.async_write_and_verify",
        new_callable=AsyncMock,
        side_effect=_verify,
    ):
        summary = await async_apply_secondary_storage(
            MagicMock(),
            cfg,
            live,
            _recommendation(SECONDARY_MODE_UTILITY),
        )

    assert calls == [
        (_CHARGER_ENTITY, POWMR_CHARGER_SOLAR_ONLY),
        (_OUTPUT_ENTITY, POWMR_OUTPUT_UTILITY),
    ]
    assert summary.overall_status == status


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [ApplyStatus.FAILED, ApplyStatus.UNVERIFIED])
async def test_other_failed_current_write_attempts_only_solar_fallback(
    status: ApplyStatus,
) -> None:
    """A non-decrease current failure must also disarm utility charging."""
    cfg = _config()
    live = _live()
    live.secondary_storage.max_charge_current_a = 10.0
    calls: list[tuple[str, str | float]] = []

    async def _verify(**kwargs: object) -> ApplyResult:
        entity_id = str(kwargs["entity_id"])
        desired = kwargs["desired"]
        assert isinstance(desired, (str, float))
        calls.append((entity_id, desired))
        if entity_id == _OUTPUT_ENTITY:
            return ApplyResult(
                entity_id=entity_id,
                desired=desired,
                actual=POWMR_OUTPUT_UTILITY,
                status=ApplyStatus.SKIPPED,
            )
        if entity_id == _CURRENT_ENTITY:
            return ApplyResult(
                entity_id=entity_id,
                desired=desired,
                actual=None if status == ApplyStatus.UNVERIFIED else 10.0,
                status=status,
                attempts=3,
            )
        if entity_id == _CHARGER_ENTITY:
            assert desired == POWMR_CHARGER_SOLAR_ONLY
            return ApplyResult(
                entity_id=entity_id,
                desired=desired,
                actual=POWMR_CHARGER_SOLAR_ONLY,
                status=ApplyStatus.OK,
                attempts=1,
            )
        raise AssertionError(f"Unexpected PowMr entity: {entity_id}")

    with patch(
        "custom_components.hsem.custom_sensors.secondary_storage_applier.async_write_and_verify",
        new_callable=AsyncMock,
        side_effect=_verify,
    ):
        summary = await async_apply_secondary_storage(
            MagicMock(),
            cfg,
            live,
            _recommendation(SECONDARY_MODE_CHARGE, 20.0),
        )

    assert calls == [
        (_OUTPUT_ENTITY, POWMR_OUTPUT_UTILITY),
        (_CURRENT_ENTITY, 20.0),
        (_CHARGER_ENTITY, POWMR_CHARGER_SOLAR_ONLY),
        (_OUTPUT_ENTITY, POWMR_OUTPUT_UTILITY),
    ]
    assert summary.overall_status == status


@pytest.mark.asyncio
async def test_stale_charge_lease_never_reenables_utility_charging() -> None:
    """A mismatch during current verification aborts the final enabling write."""
    cfg = _config()
    live = _live()
    calls: list[tuple[str, str | float]] = []
    observer = MagicMock()
    observer.secondary_control_mode_started.side_effect = [100, 200]
    observer.secondary_control_mode_is_valid.side_effect = [
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    observer.secondary_control_write_started.side_effect = [1, 2, 3, 4, 5]

    async def _verify(**kwargs: object) -> ApplyResult:
        entity_id = str(kwargs["entity_id"])
        desired = kwargs["desired"]
        assert isinstance(desired, (str, float))
        calls.append((entity_id, desired))
        return ApplyResult(
            entity_id=entity_id,
            desired=desired,
            actual=desired,
            status=ApplyStatus.OK,
            attempts=1,
        )

    with patch(
        "custom_components.hsem.custom_sensors.secondary_storage_applier.async_write_and_verify",
        new_callable=AsyncMock,
        side_effect=_verify,
    ):
        summary = await async_apply_secondary_storage(
            MagicMock(),
            cfg,
            live,
            _recommendation(SECONDARY_MODE_CHARGE, 20.0),
            control_write_observer=observer,
        )

    assert calls == [
        (_CHARGER_ENTITY, POWMR_CHARGER_SOLAR_ONLY),
        (_OUTPUT_ENTITY, POWMR_OUTPUT_UTILITY),
        (_CURRENT_ENTITY, 20.0),
        (_CHARGER_ENTITY, POWMR_CHARGER_SOLAR_ONLY),
        (_OUTPUT_ENTITY, POWMR_OUTPUT_UTILITY),
    ]
    assert all(desired != POWMR_CHARGER_UTILITY for _entity, desired in calls)
    assert summary.overall_status == ApplyStatus.UNVERIFIED
    revoked = [
        result
        for result in summary.results
        if result.error_message == "PowMr mode transition lease was superseded"
    ]
    assert len(revoked) == 1
    assert observer.secondary_control_mode_finished.call_args_list[-1].kwargs == {
        "verified": False
    }
