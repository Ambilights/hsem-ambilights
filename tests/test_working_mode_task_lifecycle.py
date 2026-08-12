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
8. A changed hardware intent cancels the obsolete in-flight command.
9. HA state is published before any potentially slow hardware operation.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.hsem.coordinator import CoordinatorData
from custom_components.hsem.custom_sensors.applier import FullyFedDischargeCapState
from custom_components.hsem.custom_sensors.working_mode_sensor import (
    HSEMWorkingModeSensor,
)
from custom_components.hsem.utils.degraded_mode import DegradedMode

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
) -> CoordinatorData:
    """Build a coordinator snapshot with a deterministic hardware intent."""
    cfg = MagicMock()
    cfg.read_only = False
    live = MagicMock()
    live.degraded_mode = DegradedMode.OK
    live.import_electricity_price = 1.0
    live.ev.is_charging = False
    live.ev_second.is_charging = False
    live.battery_current_capacity_kwh = 0.0
    rec = MagicMock()
    rec.recommendation = recommendation
    rec.ev_charger_calculated_power = 0.0
    rec.ev_second_charger_calculated_power = 0.0
    rec.ev_total_planned_load_kwh = 0.0
    rec.ev_planned_load_kwh = 0.0
    rec.ev_accounted_load_kwh = 0.0
    rec.secondary_storage_mode = secondary_mode
    rec.secondary_storage_charge_current_a = 0.0
    return CoordinatorData(
        cfg=cfg,
        live=live,
        hourly_recommendation=rec,
        state=recommendation,
    )


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
        sensor._fully_fed_discharge_state.battery_capacity_sample_kwh = 20.0

        # Must complete without error and must discard stale control state.
        await sensor.async_will_remove_from_hass()

        assert sensor._fully_fed_discharge_state.commanded_cap_w is None
        assert sensor._fully_fed_discharge_state.battery_capacity_sample_kwh is None


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
    async def test_changed_intent_cancels_inflight_task(self) -> None:
        """A changed recommendation cancels an obsolete command immediately."""
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

        assert first_task.cancelled()
        assert sensor._pending_update_data is new_data
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

        async def _apply(data: CoordinatorData | None) -> None:
            assert data is not None
            applied.append(data)
            if data is first_data:
                started.set()
                await release.wait()

        object.__setattr__(sensor, "_async_apply_hardware_writes", _apply)
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
