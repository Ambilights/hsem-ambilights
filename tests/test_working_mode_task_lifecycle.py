"""Regression tests for the working-mode hardware worker lifecycle.

Acceptance criteria (issue #369)
---------------------------------
1. ``_update_task`` is stored after ``_handle_coordinator_update`` is called.
2. The stored task is cancelled when ``async_will_remove_from_hass`` is called.
3. Cancellation does not raise or propagate outside the entity.
4. No inverter/battery write can occur after the entity is unloaded.
5. A completed task is NOT cancelled again (``cancel()`` is a no-op on done tasks).
6. Calling ``_cancel_update_task`` when ``_update_task`` is ``None`` is safe.
7. Routine coordinator pushes are coalesced without cancelling verification.
8. A changed safe intent completes the active transaction, then refreshes.
9. HA state is published before any potentially slow hardware operation.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.hsem.coordinator import CoordinatorData
from custom_components.hsem.custom_sensors.applier import FullyFedDischargeCapState
from custom_components.hsem.custom_sensors.working_mode_sensor import (
    HSEMWorkingModeSensor,
)
from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.utils.degraded_mode import DegradedMode
from custom_components.hsem.utils.inverter_verify import CycleApplySummary
from custom_components.hsem.utils.phase_power import (
    POWMR_CHARGER_SOLAR_ONLY,
    POWMR_OUTPUT_UTILITY,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config_entry() -> MagicMock:
    """Minimal mock config entry sufficient for HSEMWorkingModeSensor."""
    cfg = MagicMock()
    cfg.entry_id = "test_entry_id_p0_17"
    cfg.options = {}
    cfg.data = {}
    return cfg


def _make_coordinator() -> MagicMock:
    """Coordinator mock whose ``data`` is None by default."""
    coord = MagicMock()
    coord.data = None
    coord.last_update_success = True
    return coord


def _make_sensor() -> HSEMWorkingModeSensor:
    """Build a sensor instance with mocked coordinator and hass."""
    cfg = _make_config_entry()
    coord = _make_coordinator()

    sensor = HSEMWorkingModeSensor(cfg, coord)

    # Minimal hass mock — ``async_create_task`` returns a real asyncio.Task so
    # that cancellation tests work properly.
    hass = MagicMock()

    def _fake_create_task(coro, *, name=None):
        loop = asyncio.get_event_loop()
        return loop.create_task(coro, name=name)

    hass.async_create_task = MagicMock(side_effect=_fake_create_task)
    sensor.hass = hass
    object.__setattr__(sensor, "async_write_ha_state", MagicMock())
    return sensor


def _make_data(
    recommendation: str = "batteries_wait_mode",
    *,
    secondary_mode: str = "utility",
    primary_battery_hold: bool = False,
    batteries_charged_kwh: float = 0.0,
    batteries_discharged_kwh: float = 0.0,
    grid_import_kwh: float = 0.0,
    grid_export_kwh: float = 0.0,
    estimated_battery_capacity_kwh: float = 0.0,
    current_required_battery: float = 0.0,
) -> CoordinatorData:
    """Build a coordinator snapshot with a deterministic hardware intent."""
    cfg = SensorConfig()
    cfg.read_only = False
    cfg.batteries_wait_mode_behavior = "strict"
    cfg.phase_aware_charging_enabled = True
    cfg.main_fuse_amps = 16
    cfg.main_fuse_phases = 3
    cfg.secondary_storage.enabled = True
    cfg.secondary_storage.control_enabled = True
    live = LiveState(_degraded_mode=DegradedMode.OK)
    live.import_electricity_price = 1.0
    live.battery_current_capacity_kwh = 0.0
    live.huawei_batteries_rated_capacity_wh = 30000.0
    live.huawei_batteries_max_charge_power_w = 10000.0
    # Phase-aware grid charging needs this reading: without it the limiter
    # cannot separate the battery's own draw from real house load, so it
    # now refuses the charge rather than sizing it from a partial snapshot.
    live.huawei_batteries_charge_discharge_power_w = 0.0
    start = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
    rec = HourlyRecommendation(
        start=start,
        end=start + timedelta(minutes=15),
        avg_house_consumption_kwh=0.4,
        historical_avg_house_consumption_kwh=0.4,
        avg_house_consumption_1d_kwh=0.4,
        avg_house_consumption_3d_kwh=0.4,
        avg_house_consumption_7d_kwh=0.4,
        avg_house_consumption_14d_kwh=0.4,
        batteries_charged_kwh=batteries_charged_kwh,
        batteries_discharged_kwh=batteries_discharged_kwh,
        estimated_battery_capacity_kwh=estimated_battery_capacity_kwh,
        estimated_battery_soc_pct=0.0,
        estimated_cost_currency=0.0,
        estimated_net_consumption_kwh=0.0,
        export_price=0.0,
        grid_export_kwh=grid_export_kwh,
        grid_import_kwh=grid_import_kwh,
        import_price=1.0,
        recommendation=recommendation,
        solcast_pv_estimate_kwh=0.0,
        secondary_storage_mode=secondary_mode,
        primary_battery_hold=primary_battery_hold,
        import_price_available=True,
        export_price_available=True,
        price_actionable=True,
    )
    return CoordinatorData(
        cfg=cfg,
        live=live,
        hourly_recommendations=[rec],
        hourly_recommendation=rec,
        state=recommendation,
        current_required_battery=current_required_battery,
    )


async def _assert_changed_snapshot_marks_refresh(
    old_data: CoordinatorData,
    new_data: CoordinatorData,
) -> None:
    """Assert a changed safe intent waits for a post-write refresh."""
    sensor = _make_sensor()
    event = asyncio.Event()

    async def _hanging_coro() -> None:
        await event.wait()

    first_task = asyncio.get_event_loop().create_task(_hanging_coro())
    sensor._update_task = first_task
    sensor._active_hardware_intent = sensor._hardware_intent(old_data)
    await asyncio.sleep(0)

    sensor.coordinator.data = new_data
    sensor._handle_coordinator_update()
    await asyncio.sleep(0)

    assert not first_task.cancelled()
    assert sensor._post_write_refresh_needed
    assert sensor._pending_update_data is new_data
    first_task.cancel()
    await asyncio.gather(first_task, return_exceptions=True)


# ---------------------------------------------------------------------------
# Task lifecycle tests
# ---------------------------------------------------------------------------


class TestUpdateTaskTracking:
    """Background task is stored after _handle_coordinator_update."""

    def test_update_task_is_none_initially(self) -> None:
        """``_update_task`` must be ``None`` before any coordinator update."""
        sensor = _make_sensor()
        assert sensor._update_task is None
        assert isinstance(sensor._fully_fed_discharge_state, FullyFedDischargeCapState)

    @pytest.mark.asyncio
    async def test_update_task_stored_after_coordinator_update(self) -> None:
        """``_update_task`` is populated after ``_handle_coordinator_update``."""
        sensor = _make_sensor()
        sensor.coordinator.data = _make_data()

        with patch.object(
            sensor,
            "_async_on_coordinator_update",
            new_callable=AsyncMock,
        ):
            sensor._handle_coordinator_update()

        assert sensor._update_task is not None

    @pytest.mark.asyncio
    async def test_update_task_is_asyncio_task(self) -> None:
        """The stored task must be an ``asyncio.Task`` instance."""
        sensor = _make_sensor()
        sensor.coordinator.data = _make_data()

        with patch.object(
            sensor,
            "_async_on_coordinator_update",
            new_callable=AsyncMock,
        ):
            sensor._handle_coordinator_update()

        assert isinstance(sensor._update_task, asyncio.Task)

        # Clean up
        await asyncio.gather(sensor._update_task, return_exceptions=True)


class TestTaskCancellationOnUnload:
    """Task is cancelled cleanly when the entity is unloaded."""

    @pytest.mark.asyncio
    async def test_done_callback_ignores_cancelled_task(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A cancelled task must not be logged as an unhandled exception.

        Regression for issue #736: ``task.exception()`` raises
        ``asyncio.CancelledError`` when the task was cancelled.  The done
        callback must detect this and return silently.
        """
        sensor = _make_sensor()
        event = asyncio.Event()

        async def _hanging_coro():
            await event.wait()

        task = asyncio.get_event_loop().create_task(_hanging_coro())
        task.add_done_callback(sensor._on_update_task_done)

        # Let the task start and block.
        await asyncio.sleep(0)

        task.cancel()

        # Must not raise through the done callback.
        await asyncio.gather(task, return_exceptions=True)

        assert task.cancelled()
        assert "Unhandled exception" not in caplog.text

    @pytest.mark.asyncio
    async def test_unload_cancels_pending_task(self) -> None:
        """A pending task must be cancelled on ``async_will_remove_from_hass``."""
        sensor = _make_sensor()

        # Create a coroutine that never returns so the task stays pending.
        event = asyncio.Event()

        async def _hanging_coro():
            await event.wait()  # Blocks until set — simulates in-flight work.

        sensor._update_task = asyncio.get_event_loop().create_task(_hanging_coro())

        # Yield once so the task can start and reach the ``await`` inside.
        await asyncio.sleep(0)

        await sensor.async_will_remove_from_hass()

        # Yield again so the event loop can transition the task to cancelled.
        await asyncio.sleep(0)

        assert sensor._update_task.cancelled()

    @pytest.mark.asyncio
    async def test_unload_does_not_raise_on_cancellation(self) -> None:
        """``async_will_remove_from_hass`` must not raise even if task is pending."""
        sensor = _make_sensor()

        event = asyncio.Event()

        async def _hanging_coro():
            await event.wait()

        sensor._update_task = asyncio.get_event_loop().create_task(_hanging_coro())

        # Must complete without raising any exception.
        await sensor.async_will_remove_from_hass()

    @pytest.mark.asyncio
    async def test_completed_task_not_cancelled_again(self) -> None:
        """Unload must not attempt to cancel an already-completed task."""
        sensor = _make_sensor()

        async def _quick_coro():
            return None

        task = asyncio.get_event_loop().create_task(_quick_coro())
        await task  # Ensure the task finishes before unload.
        sensor._update_task = task

        # cancel() on a done task is a no-op and must not raise.
        await sensor.async_will_remove_from_hass()

        # Task state must still be done, not cancelled.
        assert task.done()
        assert not task.cancelled()

    @pytest.mark.asyncio
    async def test_cancel_task_when_none_is_safe(self) -> None:
        """Calling ``_cancel_update_task`` with no stored task must be a no-op."""
        sensor = _make_sensor()
        assert sensor._update_task is None

        # Must not raise.
        sensor._cancel_update_task()

    @pytest.mark.asyncio
    async def test_unload_when_no_task_is_safe(self) -> None:
        """``async_will_remove_from_hass`` with no stored task must not raise."""
        sensor = _make_sensor()
        sensor._fully_fed_discharge_state.commanded_cap_w = 1800

        # Must complete without error and must discard stale control state.
        await sensor.async_will_remove_from_hass()

        assert sensor._fully_fed_discharge_state.commanded_cap_w is None


class TestNoWriteAfterUnload:
    """No inverter/battery write can occur after the entity is unloaded."""

    @pytest.mark.asyncio
    async def test_no_hardware_write_after_unload(self) -> None:
        """Hardware-write helpers must NOT be called after the entity is unloaded.

        Scenario:
        1. A coordinator update fires, creating a background task.
        2. The entity is unloaded immediately (``async_will_remove_from_hass``).
        3. The task is cancelled before it can execute the hardware-write path.
        """
        sensor = _make_sensor()
        write_called = False

        async def _spy_write(data):
            nonlocal write_called
            write_called = True

        sensor._async_apply_hardware_writes = _spy_write  # type: ignore[method-assign]  # test monkey-patch
        sensor.coordinator.data = _make_data()

        # Create a task that yields control once, giving us the window to cancel.
        event = asyncio.Event()

        async def _slow_update():
            await event.wait()  # Yields; allows cancellation before write.
            await sensor._async_apply_hardware_writes(None)

        with patch.object(
            sensor,
            "_async_on_coordinator_update",
            side_effect=_slow_update,
        ):
            sensor._handle_coordinator_update()

        # Unload immediately — must cancel before the write executes.
        await sensor.async_will_remove_from_hass()

        # Give the event loop a chance to run cancelled callbacks.
        await asyncio.sleep(0)

        assert not write_called, (
            "Hardware write was called after entity unload — stale task not cancelled."
        )

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates_out_of_update_coro(self) -> None:
        """``CancelledError`` must propagate from ``_async_on_coordinator_update``.

        asyncio requires that ``CancelledError`` is re-raised so the task
        machinery can correctly transition the task to the cancelled state.
        """
        sensor = _make_sensor()
        event = asyncio.Event()

        async def _hanging_apply(_data):
            await event.wait()

        sensor._async_apply_hardware_writes = _hanging_apply  # type: ignore[method-assign]  # test monkey-patch
        data = _make_data()
        sensor.coordinator.data = data
        sensor._pending_update_data = data

        task = asyncio.get_event_loop().create_task(
            sensor._async_on_coordinator_update()
        )

        # Allow the task to start and block inside _async_apply_hardware_writes.
        await asyncio.sleep(0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


class TestSingleTaskInFlight:
    """Coordinator pushes are safely coalesced through one worker."""

    @pytest.mark.asyncio
    async def test_runtime_override_uses_detached_slot_and_clears(self) -> None:
        """A live EV label must not become the next refresh's planner baseline."""
        sensor = _make_sensor()
        accepted = _make_data("batteries_wait_mode")
        accepted_rec = accepted.hourly_recommendation
        assert accepted_rec is not None
        assert accepted.live is not None
        accepted_rec.ev_charger_calculated_power = 7000.0
        accepted.live.ev.is_charging = True

        with patch.object(sensor, "_start_update_task"):
            sensor.coordinator.data = accepted
            sensor._handle_coordinator_update()

            overridden = sensor._pending_update_data
            assert overridden is not None
            assert overridden.hourly_recommendation is not None
            assert overridden.hourly_recommendation.recommendation == (
                "ev_smart_charging"
            )
            assert overridden.hourly_recommendation is not accepted_rec
            assert overridden.hourly_recommendations[0] is (
                overridden.hourly_recommendation
            )
            assert sensor.state == "ev_smart_charging"
            attributes = sensor.extra_state_attributes
            assert attributes["hourly_recommendation"] is (
                overridden.hourly_recommendation
            )
            assert attributes["hourly_recommendations"][0] is (
                overridden.hourly_recommendation
            )

            # The accepted planner snapshot remains the baseline for a later
            # live-only publication after the transient EV condition clears.
            assert accepted_rec.recommendation == "batteries_wait_mode"
            assert accepted.hourly_recommendations[0].recommendation == (
                "batteries_wait_mode"
            )
            cleared_live = deepcopy(accepted.live)
            cleared_live.ev.is_charging = False
            live_refresh = replace(accepted, live=cleared_live)
            sensor._pending_update_data = None
            sensor.coordinator.data = live_refresh
            sensor._handle_coordinator_update()

        cleared = sensor._pending_update_data
        assert cleared is not None
        assert cleared.hourly_recommendation is accepted_rec
        assert cleared.hourly_recommendation.recommendation == "batteries_wait_mode"
        assert sensor.state == "batteries_wait_mode"
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_live_override_cannot_mutate_inflight_snapshot(self) -> None:
        """A new shared live snapshot cannot rewrite an active transaction."""
        sensor = _make_sensor()
        accepted = _make_data("batteries_wait_mode")
        accepted_rec = accepted.hourly_recommendation
        assert accepted_rec is not None
        assert accepted.live is not None
        accepted_rec.ev_charger_calculated_power = 7000.0
        started = asyncio.Event()
        release = asyncio.Event()
        applied: list[CoordinatorData] = []

        async def _apply(snapshot: CoordinatorData | None) -> None:
            assert snapshot is not None
            applied.append(snapshot)
            started.set()
            await release.wait()

        object.__setattr__(sensor, "_async_apply_hardware_writes", _apply)
        sensor.coordinator.data = accepted
        sensor._handle_coordinator_update()
        task = sensor._update_task
        assert task is not None
        await started.wait()

        active = applied[0]
        try:
            assert active.hourly_recommendation is accepted_rec
            updated_live = deepcopy(accepted.live)
            updated_live.ev.is_charging = True
            live_refresh = replace(accepted, live=updated_live)
            sensor.coordinator.data = live_refresh
            sensor._handle_coordinator_update()

            pending = sensor._pending_update_data
            assert pending is not None
            assert pending.hourly_recommendation is not None
            assert pending.hourly_recommendation.recommendation == "ev_smart_charging"
            assert pending.hourly_recommendation is not active.hourly_recommendation
            assert active.hourly_recommendation is not None
            assert active.hourly_recommendation.recommendation == "batteries_wait_mode"
            assert accepted_rec.recommendation == "batteries_wait_mode"
        finally:
            sensor._unloading = True
            sensor._pending_update_data = None
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_same_intent_does_not_cancel_inflight_task(self) -> None:
        """A routine push must not reset an in-progress verification delay."""
        sensor = _make_sensor()
        event = asyncio.Event()

        async def _hanging_coro():
            await event.wait()

        data = _make_data()
        first_task = asyncio.get_event_loop().create_task(_hanging_coro())
        sensor._update_task = first_task
        sensor._active_hardware_intent = sensor._hardware_intent(data)

        await asyncio.sleep(0)
        sensor.coordinator.data = data
        sensor._handle_coordinator_update()
        await asyncio.sleep(0)

        assert not first_task.cancelled()
        assert sensor._update_task is first_task
        assert sensor._pending_update_data is data

        first_task.cancel()
        await asyncio.gather(first_task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_changed_intent_requests_post_write_refresh(self) -> None:
        """A changed recommendation waits for a coherent post-write refresh."""
        sensor = _make_sensor()
        event = asyncio.Event()

        async def _hanging_coro():
            await event.wait()

        old_data = _make_data("batteries_wait_mode")
        new_data = _make_data("batteries_charge_grid")
        first_task = asyncio.get_event_loop().create_task(_hanging_coro())
        sensor._update_task = first_task
        sensor._active_hardware_intent = sensor._hardware_intent(old_data)
        await asyncio.sleep(0)

        sensor.coordinator.data = new_data
        sensor._handle_coordinator_update()
        await asyncio.sleep(0)

        assert not first_task.cancelled()
        assert sensor._post_write_refresh_needed
        assert sensor._pending_update_data is new_data
        first_task.cancel()
        await asyncio.gather(first_task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_changed_primary_hold_requests_post_write_refresh(self) -> None:
        """A new optimiser hold waits for a coherent post-write refresh."""
        sensor = _make_sensor()
        event = asyncio.Event()

        async def _hanging_coro():
            await event.wait()

        old_data = _make_data(primary_battery_hold=False)
        new_data = _make_data(primary_battery_hold=True)
        first_task = asyncio.get_event_loop().create_task(_hanging_coro())
        sensor._update_task = first_task
        sensor._active_hardware_intent = sensor._hardware_intent(old_data)
        await asyncio.sleep(0)

        sensor.coordinator.data = new_data
        sensor._handle_coordinator_update()
        await asyncio.sleep(0)

        assert not first_task.cancelled()
        assert sensor._post_write_refresh_needed
        assert sensor._pending_update_data is new_data
        first_task.cancel()
        await asyncio.gather(first_task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_export_limit_change_requests_post_write_refresh(self) -> None:
        """A price crossing requests a fresh post-write snapshot."""
        old_data = _make_data()
        new_data = deepcopy(old_data)
        assert old_data.cfg is not None and new_data.cfg is not None
        assert old_data.live is not None and new_data.live is not None
        old_data.cfg.export_electricity_min_price = 0.5
        new_data.cfg.export_electricity_min_price = 0.5
        old_data.live.export_electricity_price = 1.0
        new_data.live.export_electricity_price = 0.0

        await _assert_changed_snapshot_marks_refresh(old_data, new_data)

    @pytest.mark.asyncio
    async def test_reduced_phase_headroom_requests_post_write_refresh(self) -> None:
        """Lower live fuse headroom requests a fresh post-write snapshot."""
        old_data = _make_data("batteries_charge_grid", batteries_charged_kwh=0.5)
        new_data = deepcopy(old_data)
        assert old_data.live is not None and new_data.live is not None
        old_data.live.grid_phase_power_w = (0.0, 0.0, 0.0)
        new_data.live.grid_phase_power_w = (3500.0, 3500.0, 3500.0)

        await _assert_changed_snapshot_marks_refresh(old_data, new_data)

    @pytest.mark.asyncio
    async def test_powmr_min_soc_guard_requests_post_write_refresh(self) -> None:
        """Crossing PowMr minimum SoC requests a fresh post-write snapshot."""
        old_data = _make_data(secondary_mode="sbu")
        assert old_data.cfg is not None and old_data.live is not None
        old_data.cfg.secondary_storage.output_source_priority_entity = (
            "select.powmr_output"
        )
        old_data.cfg.secondary_storage.charger_source_priority_entity = (
            "select.powmr_charger"
        )
        old_data.cfg.secondary_storage.max_charge_current_entity = (
            "number.powmr_current"
        )
        old_data.live.secondary_storage.soc_pct = 50.0
        new_data = deepcopy(old_data)
        assert new_data.live is not None
        new_data.live.secondary_storage.soc_pct = 20.0

        await _assert_changed_snapshot_marks_refresh(old_data, new_data)

    @pytest.mark.asyncio
    async def test_phase_safety_utility_revokes_active_powmr_charge_lease(
        self,
    ) -> None:
        """Lost L3 headroom supersedes Charge before its final enabling write."""
        sensor = _make_sensor()
        old_data = _make_data(secondary_mode="charge")
        assert old_data.cfg is not None and old_data.live is not None
        assert old_data.hourly_recommendation is not None
        old_data.cfg.secondary_storage.output_source_priority_entity = (
            "select.powmr_output"
        )
        old_data.cfg.secondary_storage.charger_source_priority_entity = (
            "select.powmr_charger"
        )
        old_data.cfg.secondary_storage.max_charge_current_entity = (
            "number.powmr_current"
        )
        old_data.live.secondary_storage.soc_pct = 50.0
        old_data.live.secondary_storage.load_power_w = 200.0
        old_data.live.secondary_storage.output_source_priority = POWMR_OUTPUT_UTILITY
        old_data.live.secondary_storage.charger_source_priority = (
            POWMR_CHARGER_SOLAR_ONLY
        )
        old_data.live.secondary_storage.max_charge_current_a = 10.0
        old_data.live.grid_phase_power_w = (0.0, 0.0, 0.0)
        old_data.hourly_recommendation.secondary_storage_charge_current_a = 20.0
        new_data = deepcopy(old_data)
        assert new_data.live is not None
        new_data.live.grid_phase_power_w = (0.0, 0.0, 3680.0)

        release = asyncio.Event()

        async def _hanging_coro() -> None:
            await release.wait()

        first_task = asyncio.create_task(_hanging_coro())
        sensor._update_task = first_task
        sensor._active_hardware_intent = sensor._hardware_intent(old_data)
        sensor._active_secondary_hardware_intent = sensor._secondary_hardware_intent(
            old_data
        )
        await asyncio.sleep(0)

        sensor.coordinator.data = new_data
        sensor._handle_coordinator_update()
        await asyncio.sleep(0)

        cast(
            MagicMock, sensor.coordinator.secondary_control_mode_superseded
        ).assert_called_with("newer PowMr hardware intent published")
        assert not first_task.cancelled()
        assert sensor._post_write_refresh_needed
        first_task.cancel()
        await asyncio.gather(first_task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_hardware_target_change_requests_post_write_refresh(self) -> None:
        """An entity-target change requests a fresh post-write snapshot."""
        old_data = _make_data()
        assert old_data.cfg is not None
        old_data.cfg.huawei_solar_batteries_working_mode = "select.old_mode"
        new_data = deepcopy(old_data)
        assert new_data.cfg is not None
        new_data.cfg.huawei_solar_batteries_working_mode = "select.new_mode"

        await _assert_changed_snapshot_marks_refresh(old_data, new_data)

    @pytest.mark.parametrize(
        ("field_name", "new_value"),
        [
            ("batteries_charged_kwh", 0.100),
            ("batteries_discharged_kwh", 0.100),
            ("grid_import_kwh", 0.100),
            ("grid_export_kwh", 0.100),
            ("estimated_battery_capacity_kwh", 1.000),
            ("avg_house_consumption_kwh", 0.500),
            ("historical_avg_house_consumption_kwh", 0.500),
            ("secondary_storage_charge_current_a", 10.0),
        ],
    )
    @pytest.mark.asyncio
    async def test_same_label_command_input_change_requests_post_write_refresh(
        self, field_name: str, new_value: float
    ) -> None:
        """Material same-label command changes require a fresh snapshot."""
        sensor = _make_sensor()
        event = asyncio.Event()

        async def _hanging_coro():
            await event.wait()

        old_data = _make_data("batteries_discharge_mode")
        new_data = _make_data("batteries_discharge_mode")
        assert new_data.hourly_recommendation is not None
        setattr(new_data.hourly_recommendation, field_name, new_value)
        first_task = asyncio.get_event_loop().create_task(_hanging_coro())
        sensor._update_task = first_task
        sensor._active_hardware_intent = sensor._hardware_intent(old_data)
        await asyncio.sleep(0)

        sensor.coordinator.data = new_data
        sensor._handle_coordinator_update()
        await asyncio.sleep(0)

        assert not first_task.cancelled()
        assert sensor._post_write_refresh_needed
        assert sensor._pending_update_data is new_data
        first_task.cancel()
        await asyncio.gather(first_task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_same_label_reserve_change_requests_post_write_refresh(self) -> None:
        """A new whole-Wh reserve requires a fresh post-write snapshot."""
        sensor = _make_sensor()
        event = asyncio.Event()

        async def _hanging_coro():
            await event.wait()

        old_data = _make_data(current_required_battery=1.000)
        new_data = _make_data(current_required_battery=1.001)
        first_task = asyncio.get_event_loop().create_task(_hanging_coro())
        sensor._update_task = first_task
        sensor._active_hardware_intent = sensor._hardware_intent(old_data)
        await asyncio.sleep(0)

        sensor.coordinator.data = new_data
        sensor._handle_coordinator_update()
        await asyncio.sleep(0)

        assert not first_task.cancelled()
        assert sensor._post_write_refresh_needed
        assert sensor._pending_update_data is new_data
        first_task.cancel()
        await asyncio.gather(first_task, return_exceptions=True)

    @pytest.mark.parametrize("gate", ["read_only", "error"])
    @pytest.mark.asyncio
    async def test_hard_no_write_gate_cancels_inflight_task(self, gate: str) -> None:
        """A deliberate safety gate still interrupts an active transaction."""
        sensor = _make_sensor()
        event = asyncio.Event()

        async def _hanging_coro() -> None:
            await event.wait()

        old_data = _make_data()
        gated_data = deepcopy(old_data)
        assert gated_data.cfg is not None
        if gate == "read_only":
            gated_data.cfg.read_only = True
        else:
            assert gated_data.live is not None
            gated_data.live._degraded_mode = DegradedMode.Error

        first_task = asyncio.create_task(_hanging_coro())
        sensor._update_task = first_task
        sensor._active_hardware_intent = sensor._hardware_intent(old_data)
        await asyncio.sleep(0)

        sensor.coordinator.data = gated_data
        sensor._handle_coordinator_update()
        await asyncio.sleep(0)

        assert first_task.cancelled()
        assert not sensor._post_write_refresh_needed
        assert sensor._pending_update_data is gated_data
        await asyncio.gather(first_task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_changed_intent_finishes_transaction_then_applies_fresh_data(
        self,
    ) -> None:
        """Stale B is discarded after A and only refreshed C is applied."""
        sensor = _make_sensor()
        data_a = _make_data("batteries_wait_mode")
        stale_b = _make_data("batteries_charge_grid", batteries_charged_kwh=0.5)
        fresh_c = _make_data("batteries_discharge_mode", batteries_discharged_kwh=0.4)
        started = asyncio.Event()
        release = asyncio.Event()
        applied: list[CoordinatorData] = []
        writes: list[str] = []
        concurrent = 0
        max_concurrent = 0

        async def _apply(snapshot: CoordinatorData | None) -> None:
            nonlocal concurrent, max_concurrent
            assert snapshot is not None
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
            try:
                applied.append(snapshot)
                writes.append(f"{snapshot.state}:first")
                if snapshot is data_a:
                    started.set()
                    await release.wait()
                    writes.append(f"{snapshot.state}:second")
            finally:
                concurrent -= 1

        async def _refresh() -> None:
            assert writes[-1] == "batteries_wait_mode:second"
            sensor.coordinator.last_update_success = True
            sensor.coordinator.data = fresh_c
            sensor._handle_coordinator_update()

        object.__setattr__(sensor, "_async_apply_hardware_writes", _apply)
        refresh = AsyncMock(side_effect=_refresh)
        object.__setattr__(sensor.coordinator, "async_request_refresh", refresh)
        sensor.coordinator.data = data_a
        sensor._handle_coordinator_update()
        task = sensor._update_task
        assert task is not None
        await started.wait()

        sensor.coordinator.data = stale_b
        sensor._handle_coordinator_update()
        assert not task.cancelled()
        assert sensor._post_write_refresh_needed
        release.set()
        await task

        assert applied == [data_a, fresh_c]
        assert stale_b not in applied
        assert max_concurrent == 1
        refresh.assert_awaited_once_with()

    @pytest.mark.parametrize(
        "refresh_outcome", ["no_generation", "failed_generation", "exception"]
    )
    @pytest.mark.asyncio
    async def test_failed_post_write_refresh_drops_stale_data_and_recovers(
        self, refresh_outcome: str
    ) -> None:
        """An untrusted refresh drops B without spinning; a later C recovers."""
        sensor = _make_sensor()
        data_a = _make_data("batteries_wait_mode")
        stale_b = _make_data("batteries_charge_grid", batteries_charged_kwh=0.5)
        fresh_c = _make_data("batteries_discharge_mode", batteries_discharged_kwh=0.4)
        started = asyncio.Event()
        release = asyncio.Event()
        applied: list[CoordinatorData] = []

        async def _apply(snapshot: CoordinatorData | None) -> None:
            assert snapshot is not None
            applied.append(snapshot)
            if snapshot is data_a:
                started.set()
                await release.wait()

        async def _failed_refresh() -> None:
            if refresh_outcome == "exception":
                raise RuntimeError("refresh failed")
            if refresh_outcome == "failed_generation":
                sensor.coordinator.last_update_success = False
                sensor._handle_coordinator_update()
            # ``no_generation`` deliberately returns with the old successful
            # status but without a listener callback. Generation proof must
            # still reject the stale pending snapshot.

        object.__setattr__(sensor, "_async_apply_hardware_writes", _apply)
        refresh = AsyncMock(side_effect=_failed_refresh)
        object.__setattr__(sensor.coordinator, "async_request_refresh", refresh)
        sensor.coordinator.data = data_a
        sensor._handle_coordinator_update()
        task = sensor._update_task
        assert task is not None
        await started.wait()

        sensor.coordinator.data = stale_b
        sensor._handle_coordinator_update()
        release.set()
        await task
        await asyncio.sleep(0)

        assert applied == [data_a]
        assert sensor._pending_update_data is None
        refresh.assert_awaited_once_with()

        sensor.coordinator.last_update_success = True
        sensor.coordinator.data = fresh_c
        sensor._handle_coordinator_update()
        recovery_task = sensor._update_task
        assert recovery_task is not None
        await recovery_task

        assert applied == [data_a, fresh_c]
        refresh.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_sub_wh_float_noise_does_not_cancel_inflight_task(self) -> None:
        """Float noise below the published whole-Wh precision is coalesced."""
        sensor = _make_sensor()
        event = asyncio.Event()

        async def _hanging_coro():
            await event.wait()

        old_data = _make_data(batteries_discharged_kwh=0.4000000000)
        new_data = _make_data(batteries_discharged_kwh=0.4000000001)
        first_task = asyncio.get_event_loop().create_task(_hanging_coro())
        sensor._update_task = first_task
        sensor._active_hardware_intent = sensor._hardware_intent(old_data)
        await asyncio.sleep(0)

        sensor.coordinator.data = new_data
        sensor._handle_coordinator_update()
        await asyncio.sleep(0)

        assert not first_task.cancelled()
        assert sensor._update_task is first_task
        assert sensor._pending_update_data is new_data
        assert not sensor._post_write_refresh_needed

        first_task.cancel()
        await asyncio.gather(first_task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_latest_snapshot_is_processed_after_inflight_write(self) -> None:
        """Many routine pushes collapse to the latest pending snapshot."""
        sensor = _make_sensor()
        first_data = _make_data()
        middle_data = _make_data()
        latest_data = _make_data()
        started = asyncio.Event()
        release = asyncio.Event()
        applied: list[CoordinatorData] = []
        request_refresh = AsyncMock()

        async def _apply(data: CoordinatorData | None) -> None:
            assert data is not None
            applied.append(data)
            if data is first_data:
                started.set()
                await release.wait()

        object.__setattr__(sensor, "_async_apply_hardware_writes", _apply)
        object.__setattr__(sensor.coordinator, "async_request_refresh", request_refresh)
        sensor.coordinator.data = first_data
        sensor._handle_coordinator_update()
        task = sensor._update_task
        assert task is not None
        await started.wait()

        sensor.coordinator.data = middle_data
        sensor._handle_coordinator_update()
        sensor.coordinator.data = latest_data
        sensor._handle_coordinator_update()
        release.set()
        await task

        assert applied == [first_data, latest_data]
        request_refresh.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_completed_apply_republishes_diagnostics_without_write_loop(
        self,
    ) -> None:
        """A completed summary refreshes all listeners exactly once.

        The diagnostics-only listener notification includes the working-mode
        sensor itself.  That callback must publish state but must not queue a
        second hardware apply.
        """
        sensor = _make_sensor()
        data = _make_data()
        summary = CycleApplySummary()
        apply_count = 0
        applier_status_refresh = MagicMock()
        degraded_mode_refresh = MagicMock()
        plan_explanation_refresh = MagicMock()

        async def _apply(snapshot: CoordinatorData | None) -> None:
            nonlocal apply_count
            assert snapshot is data
            assert snapshot is not None
            apply_count += 1
            snapshot.apply_summary = summary

        def _publish(published: CycleApplySummary) -> None:
            assert sensor.coordinator.data is data
            sensor.coordinator.data.apply_summary = published
            sensor.coordinator.async_update_listeners()

        object.__setattr__(sensor, "_async_apply_hardware_writes", _apply)
        publish_summary = MagicMock(side_effect=_publish)
        object.__setattr__(
            sensor.coordinator,
            "async_publish_apply_summary",
            publish_summary,
        )

        def _refresh_subscribers() -> None:
            sensor._handle_coordinator_update()
            applier_status_refresh()
            degraded_mode_refresh()
            plan_explanation_refresh()

        update_listeners = MagicMock(side_effect=_refresh_subscribers)
        object.__setattr__(
            sensor.coordinator,
            "async_update_listeners",
            update_listeners,
        )
        sensor.coordinator.data = data
        sensor._handle_coordinator_update()
        task = sensor._update_task
        assert task is not None
        await task

        assert apply_count == 1
        assert data.apply_summary is summary
        publish_summary.assert_called_once_with(summary)
        update_listeners.assert_called_once_with()
        applier_status_refresh.assert_called_once_with()
        degraded_mode_refresh.assert_called_once_with()
        plan_explanation_refresh.assert_called_once_with()
        assert sensor._pending_update_data is None

    @pytest.mark.asyncio
    async def test_unload_during_apply_does_not_publish_diagnostics(self) -> None:
        """An unloading entity must not notify HA after its apply returns."""
        sensor = _make_sensor()
        data = _make_data()

        async def _apply(snapshot: CoordinatorData | None) -> None:
            assert snapshot is data
            assert snapshot is not None
            snapshot.apply_summary = CycleApplySummary()
            sensor._unloading = True

        object.__setattr__(sensor, "_async_apply_hardware_writes", _apply)
        publish_summary = MagicMock()
        object.__setattr__(
            sensor.coordinator,
            "async_publish_apply_summary",
            publish_summary,
        )
        sensor.coordinator.data = data
        sensor._pending_update_data = data

        await sensor._async_on_coordinator_update()

        publish_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_stale_intent_does_not_publish_apply_summary(self) -> None:
        """A completed obsolete command cannot overwrite current diagnostics."""
        sensor = _make_sensor()
        old_data = _make_data("batteries_wait_mode")
        new_data = _make_data("batteries_charge_grid")

        async def _apply(snapshot: CoordinatorData | None) -> None:
            assert snapshot is old_data
            assert snapshot is not None
            snapshot.apply_summary = CycleApplySummary()
            sensor.coordinator.data = new_data

        object.__setattr__(sensor, "_async_apply_hardware_writes", _apply)
        publish_summary = MagicMock()
        object.__setattr__(
            sensor.coordinator,
            "async_publish_apply_summary",
            publish_summary,
        )
        sensor.coordinator.data = old_data
        sensor._pending_update_data = old_data

        await sensor._async_on_coordinator_update()

        publish_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancelled_apply_does_not_publish_apply_summary(self) -> None:
        """Cancellation during a hardware apply cannot publish a partial result."""
        sensor = _make_sensor()
        data = _make_data()
        started = asyncio.Event()
        release = asyncio.Event()

        async def _apply(snapshot: CoordinatorData | None) -> None:
            assert snapshot is data
            assert snapshot is not None
            snapshot.apply_summary = CycleApplySummary()
            started.set()
            await release.wait()

        object.__setattr__(sensor, "_async_apply_hardware_writes", _apply)
        publish_summary = MagicMock()
        object.__setattr__(
            sensor.coordinator,
            "async_publish_apply_summary",
            publish_summary,
        )
        sensor.coordinator.data = data
        sensor._pending_update_data = data
        task = asyncio.create_task(sensor._async_on_coordinator_update())
        await started.wait()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        publish_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_state_is_published_before_hardware_apply(self) -> None:
        """Slow or failed writes cannot leave the working-mode sensor stale."""
        sensor = _make_sensor()
        data = _make_data()
        events: list[str] = []

        object.__setattr__(
            sensor,
            "async_write_ha_state",
            MagicMock(side_effect=lambda: events.append("state")),
        )

        async def _apply(_data: CoordinatorData | None) -> None:
            events.append("apply")

        object.__setattr__(sensor, "_async_apply_hardware_writes", _apply)
        sensor.coordinator.data = data
        sensor._handle_coordinator_update()
        task = sensor._update_task
        assert task is not None
        await task

        assert events[0] == "state"
        assert "apply" in events

    @pytest.mark.asyncio
    async def test_published_state_uses_resolved_recommendation(self) -> None:
        """The first state flush contains the real-time override, not raw plan."""
        sensor = _make_sensor()
        data = _make_data("batteries_wait_mode")
        assert data.live is not None
        data.live.import_electricity_price = -0.1
        published: list[str | None] = []

        object.__setattr__(
            sensor,
            "async_write_ha_state",
            MagicMock(side_effect=lambda: published.append(sensor.state)),
        )

        async def _apply(_data: CoordinatorData | None) -> None:
            return None

        object.__setattr__(sensor, "_async_apply_hardware_writes", _apply)
        sensor.coordinator.data = data
        sensor._handle_coordinator_update()
        task = sensor._update_task
        assert task is not None
        await task

        assert published[0] == "force_export"
