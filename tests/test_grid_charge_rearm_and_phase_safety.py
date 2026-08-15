"""Grid-charge re-arm without phase-aware charging, and live phase refresh.

Two independent behaviours:

1. Leaving a grid-charge slot pre-disarms the hardware charge limit to 0 W.
   Restoring it on the next slot depends on the applier being able to *see*
   that 0 W, but ``huawei_batteries_grid_charge_max_power_w`` was only read
   when phase-aware charging was enabled.  For everyone else the field stayed
   ``None``, ``_grid_charge_needs_rearm`` returned ``False``, and grid charging
   stayed dead: TOU armed the full-day charge window while the power limit held
   the battery at zero.

2. Phase-aware charging sizes the grid-charge cap from a meter snapshot taken
   just before the write, so a load appearing mid-slot is invisible until the
   next coordinator cycle.  A debounced listener republishes the live snapshot
   while a charge is actually running, leaving the accepted plan untouched.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from custom_components.hsem.custom_sensors.applier import _grid_charge_needs_rearm
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.planner.secondary_storage import (
    SECONDARY_MODE_CHARGE,
    SECONDARY_MODE_SBU,
)
from custom_components.hsem.utils.recommendations import Recommendations


class TestGridChargeRearmGuard:
    """``_grid_charge_needs_rearm`` depends on the limit actually being read."""

    @staticmethod
    def _live(current_limit_w: float | None) -> LiveState:
        live = LiveState()
        live.huawei_batteries_max_charge_power_w = 10000.0
        live.huawei_batteries_grid_charge_max_power_w = current_limit_w
        return live

    def test_disarmed_limit_is_rearmed(self) -> None:
        assert _grid_charge_needs_rearm(self._live(0.0)) is True

    def test_unread_limit_cannot_be_rearmed(self) -> None:
        """The regression: an unread limit is indistinguishable from "fine"."""
        assert _grid_charge_needs_rearm(self._live(None)) is False

    def test_positive_user_limit_is_left_alone(self) -> None:
        assert _grid_charge_needs_rearm(self._live(5000.0)) is False

    def test_no_restore_power_means_no_rearm(self) -> None:
        """Never disarm-and-strand: without a known ceiling, do not re-arm."""
        live = self._live(0.0)
        live.huawei_batteries_max_charge_power_w = None

        assert _grid_charge_needs_rearm(live) is False


class TestGridChargeLimitIsAlwaysRead:
    """The collector must populate the limit regardless of phase-aware mode."""

    @staticmethod
    async def _collect(*, phase_aware: bool, limit_state: str) -> LiveState:
        """Run one dry-run cycle and return the live snapshot."""
        from unittest.mock import AsyncMock

        from tests.test_ha_mock_integration import (
            _BASE_ENTITY_STATES,
            _patch_all_ha_helpers,
            make_bare_coordinator,
            make_fake_config_entry,
            make_fake_hass,
        )

        states: dict[str, str | dict] = dict(_BASE_ENTITY_STATES)
        states["number.batteries_grid_charge_maximum_power"] = limit_state

        config_entry = make_fake_config_entry(
            {
                "hsem_read_only": True,
                "hsem_phase_aware_charging_enabled": phase_aware,
                "hsem_huawei_solar_batteries_grid_charge_maximum_power": (
                    "number.batteries_grid_charge_maximum_power"
                ),
            }
        )
        coordinator = make_bare_coordinator(
            hass=make_fake_hass(states), config_entry=config_entry
        )
        coordinator._set_update_interval = AsyncMock()  # type: ignore[method-assign]
        captured: list[object] = []
        coordinator.async_set_updated_data = captured.append  # type: ignore[method-assign,assignment]

        with _patch_all_ha_helpers():
            await coordinator._async_run_update_cycle()

        live = captured[0].live  # type: ignore[attr-defined]
        assert isinstance(live, LiveState)
        return live

    @pytest.mark.asyncio
    async def test_disarmed_limit_is_visible_without_phase_aware_charging(
        self,
    ) -> None:
        """The regression: this read back ``None``, so re-arm never fired."""
        live = await self._collect(phase_aware=False, limit_state="0")

        assert live.huawei_batteries_grid_charge_max_power_w == pytest.approx(0.0)
        live.huawei_batteries_max_charge_power_w = 10000.0
        assert _grid_charge_needs_rearm(live) is True

    @pytest.mark.asyncio
    async def test_limit_is_still_read_with_phase_aware_charging(self) -> None:
        live = await self._collect(phase_aware=True, limit_state="0")

        assert live.huawei_batteries_grid_charge_max_power_w == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_a_positive_limit_reads_back_unchanged(self) -> None:
        live = await self._collect(phase_aware=False, limit_state="5000")

        assert live.huawei_batteries_grid_charge_max_power_w == pytest.approx(5000.0)

    @pytest.mark.asyncio
    async def test_unreadable_limit_is_non_critical_without_phase_aware(self) -> None:
        """An unconfigured entity must not escalate to a write-blocking Error."""
        live = await self._collect(phase_aware=False, limit_state="unavailable")

        assert live.huawei_batteries_grid_charge_max_power_w is None
        assert _grid_charge_needs_rearm(live) is False


class TestPhaseSafetyRefreshTrigger:
    """The live refresh must fire only while a charge actuator is running."""

    @staticmethod
    def _coordinator(
        *, phase_aware: bool, recommendation: str, secondary_mode: str
    ) -> tuple[Any, list[object]]:
        from custom_components.hsem.coordinator import HSEMDataUpdateCoordinator

        coordinator = object.__new__(HSEMDataUpdateCoordinator)
        coordinator._cfg = MagicMock()
        coordinator._cfg.phase_aware_charging_enabled = phase_aware
        rec = MagicMock()
        rec.recommendation = recommendation
        rec.secondary_storage_mode = secondary_mode
        coordinator._hourly_recommendation = rec
        coordinator._phase_safety_update_pending = False
        coordinator._phase_safety_update_debounce_task = None
        coordinator._tearing_down = False
        created: list[object] = []

        def _create_task(coro: Any, **_: Any) -> MagicMock:
            """Record the scheduled coroutine and close it unawaited."""
            created.append(coro)
            coro.close()
            return MagicMock()

        coordinator.hass = MagicMock()
        coordinator.hass.async_create_task = _create_task
        return coordinator, created

    @pytest.mark.asyncio
    async def test_active_grid_charge_schedules_a_refresh(self) -> None:
        coordinator, created = self._coordinator(
            phase_aware=True,
            recommendation=Recommendations.BatteriesChargeGrid.value,
            secondary_mode=SECONDARY_MODE_SBU,
        )

        await coordinator._async_handle_phase_safety_change(MagicMock())

        assert coordinator._phase_safety_update_pending is True
        assert len(created) == 1

    @pytest.mark.asyncio
    async def test_active_secondary_charge_schedules_a_refresh(self) -> None:
        coordinator, created = self._coordinator(
            phase_aware=True,
            recommendation=Recommendations.BatteriesWaitMode.value,
            secondary_mode=SECONDARY_MODE_CHARGE,
        )

        await coordinator._async_handle_phase_safety_change(MagicMock())

        assert len(created) == 1

    @pytest.mark.asyncio
    async def test_idle_slot_does_not_schedule_a_refresh(self) -> None:
        """Nothing is charging, so the fuse cannot be overloaded by charging."""
        coordinator, created = self._coordinator(
            phase_aware=True,
            recommendation=Recommendations.BatteriesDischargeMode.value,
            secondary_mode=SECONDARY_MODE_SBU,
        )

        await coordinator._async_handle_phase_safety_change(MagicMock())

        assert coordinator._phase_safety_update_pending is False
        assert created == []

    @pytest.mark.asyncio
    async def test_phase_aware_disabled_does_not_schedule_a_refresh(self) -> None:
        coordinator, created = self._coordinator(
            phase_aware=False,
            recommendation=Recommendations.BatteriesChargeGrid.value,
            secondary_mode=SECONDARY_MODE_SBU,
        )

        await coordinator._async_handle_phase_safety_change(MagicMock())

        assert created == []

    @pytest.mark.asyncio
    async def test_a_trailing_change_during_the_cooldown_is_kept(self) -> None:
        """A meter that publishes only on change must not lose its last event."""
        coordinator, created = self._coordinator(
            phase_aware=True,
            recommendation=Recommendations.BatteriesChargeGrid.value,
            secondary_mode=SECONDARY_MODE_SBU,
        )
        running = asyncio.get_running_loop().create_future()
        coordinator._phase_safety_update_debounce_task = MagicMock()
        coordinator._phase_safety_update_debounce_task.done = lambda: False

        await coordinator._async_handle_phase_safety_change(MagicMock())

        # No second task, but the request survives for the running loop.
        assert created == []
        assert coordinator._phase_safety_update_pending is True
        running.cancel()


class TestPhaseSafetyListenerRegistration:
    """The meter entities must be tracked, but only when they matter."""

    @staticmethod
    def _source() -> str:
        import inspect

        from custom_components.hsem.custom_sensors import state_collector

        return inspect.getsource(state_collector)

    def test_listener_is_gated_on_phase_aware_charging(self) -> None:
        source = self._source()
        marker = "Starting phase-safety tracking for %s"
        assert marker in source

        block_start = source.index(marker)
        preceding = source[:block_start]
        assert preceding.rstrip().rfind("if cfg.phase_aware_charging_enabled:") > (
            preceding.rfind("async_track_state_change_event")
        )

    def test_battery_power_is_tracked_alongside_the_phase_meters(self) -> None:
        """The limiter needs battery power to remove the charger's own draw."""
        source = self._source()
        marker = source.index("Starting phase-safety tracking for %s")
        window = source[max(0, marker - 900) : marker]
        assert "huawei_solar_batteries_charge_discharge_power" in window
        for phase in ("a", "b", "c"):
            assert f"huawei_solar_power_meter_phase_{phase}_active_power" in window
