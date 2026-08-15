"""HSEM must keep optimising when a *non-critical* integration goes offline.

Regression coverage for the outage in which every ``kia_uvo`` entity went
``unavailable`` and HSEM stopped planning entirely: the battery sat idle at
its 5 % floor and imported through the most expensive hours of the day.

Three separate defects combined to produce that:

1. The planner pipeline, the corrective force-discharge replan and the
   excess-load monitor all gated on the *boolean* ``live.missing_entities``
   instead of the severity, contradicting :mod:`utils.degraded_mode`, which
   defines ``Degraded`` as "read-only calculations continue on best-effort
   values".
2. ``working_mode_sensor.extra_state_attributes`` dropped the entire payload —
   including ``hourly_recommendations`` — on any absence, so dashboards and
   automations went blind during an unrelated integration's outage.
3. ``state_collector`` coerced an unreadable EV state-of-charge to ``0.0``,
   which is indistinguishable from a genuinely empty car.  Relaxing (1) alone
   would therefore have made HSEM schedule a full charge against a phantom SoC.

All tests here are pure-Python; no Home Assistant runtime is required.
"""

from __future__ import annotations

import inspect

import pytest

from custom_components.hsem import coordinator as coordinator_module
from custom_components.hsem.coordinator import (
    CoordinatorData,
    HSEMDataUpdateCoordinator,
)
from custom_components.hsem.coordinator_builder import (
    _ev_soc_valid,
    build_planner_input,
)
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.planner_input import PlannerInput
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.utils.degraded_mode import DegradedMode

# Labels exactly as ``state_collector._read`` records them.
_EV_SOC_MISSING = (
    "Error reading ev_planned_load_soc "
    "(entity_id=sensor.kia_ev6_ev_battery_level): state unavailable or invalid"
)
_CRITICAL_MISSING = (
    "Error reading batteries_state_of_capacity "
    "(entity_id=sensor.batteries_state_of_capacity): state unavailable or invalid"
)


class TestOutageSeverityClassification:
    """A vehicle integration outage is Degraded, never Error."""

    def test_ev_soc_outage_is_degraded(self) -> None:
        live = LiveState()
        live.add_missing_entity(_EV_SOC_MISSING)

        assert live.degraded_mode is DegradedMode.Degraded

    def test_battery_outage_is_still_error(self) -> None:
        live = LiveState()
        live.add_missing_entity(_CRITICAL_MISSING)

        assert live.degraded_mode is DegradedMode.Error

    def test_ev_outage_alongside_battery_outage_is_error(self) -> None:
        """One critical absence must dominate any number of harmless ones."""
        live = LiveState()
        live.add_missing_entity(_EV_SOC_MISSING)
        live.add_missing_entity(_CRITICAL_MISSING)

        assert live.degraded_mode is DegradedMode.Error


class TestCoordinatorGatesOnSeverity:
    """The planner gates must test severity, not the bare boolean.

    These are source-level assertions.  Driving the full coordinator cycle
    requires a Home Assistant runtime, but the defect was precisely that a
    gate said ``not live.missing_entities`` — a condition cheap to state and
    cheap to check.
    """

    @staticmethod
    def _source() -> str:
        return inspect.getsource(coordinator_module)

    def test_no_gate_uses_the_bare_missing_entities_boolean(self) -> None:
        """``not live.missing_entities`` must not gate any calculation again."""
        assert "not live.missing_entities" not in self._source()

    def test_planner_pipeline_gates_on_error_only(self) -> None:
        assert (
            'live.force_working_mode_state == "auto"\n'
            "                and live.degraded_mode is not DegradedMode.Error\n"
            "                and consumption_ok" in self._source()
        )

    def test_excess_load_monitor_gates_on_error_only(self) -> None:
        assert (
            "if slot is None or live is None "
            "or live.degraded_mode is DegradedMode.Error:" in self._source()
        )


class TestEvSocValidity:
    """An EV SoC reading is usable only when present and in range."""

    @pytest.mark.parametrize("soc", [0.0, 0.5, 55.0, 100.0])
    def test_in_range_readings_are_valid(self, soc: float) -> None:
        assert _ev_soc_valid(soc) is True

    @pytest.mark.parametrize("soc", [None, -1.0, 100.1, 1000.0])
    def test_absent_or_out_of_range_readings_are_invalid(
        self, soc: float | None
    ) -> None:
        assert _ev_soc_valid(soc) is False


class TestEvOutageDoesNotFabricateACharge:
    """An unreadable EV SoC must disable EV planning, not read as 0 %."""

    @staticmethod
    def _cfg() -> SensorConfig:
        cfg = SensorConfig()
        cfg.ev_planned_load_enabled = True
        cfg.ev_second_planned_load_enabled = True
        return cfg

    @staticmethod
    def _build(cfg: SensorConfig, live: LiveState) -> PlannerInput:
        return build_planner_input(
            cfg=cfg,
            live=live,
            hourly_recommendations=[],
            batteries_schedules=[],
            previous_winner_name=None,
            previous_winner_score=0.0,
        )

    def test_unreadable_soc_disables_primary_ev(self) -> None:
        live = LiveState()
        live.ev_planned_load_current_soc_pct = None
        live.add_missing_entity(_EV_SOC_MISSING)

        assert self._build(self._cfg(), live).ev_planned_load_enabled is False

    def test_unreadable_soc_disables_second_ev(self) -> None:
        live = LiveState()
        live.ev_second_planned_load_current_soc_pct = None

        assert self._build(self._cfg(), live).ev_second_planned_load_enabled is False

    def test_readable_soc_keeps_ev_enabled(self) -> None:
        live = LiveState()
        live.ev_planned_load_current_soc_pct = 42.0
        live.ev_second_planned_load_current_soc_pct = 42.0

        planner_input = self._build(self._cfg(), live)

        assert planner_input.ev_planned_load_enabled is True
        assert planner_input.ev_planned_load_current_soc_pct == pytest.approx(42.0)

    def test_genuine_zero_percent_still_charges(self) -> None:
        """A real 0 % reading is a valid input, unlike an absent one."""
        live = LiveState()
        live.ev_planned_load_current_soc_pct = 0.0

        planner_input = self._build(self._cfg(), live)

        assert planner_input.ev_planned_load_enabled is True
        assert planner_input.ev_planned_load_current_soc_pct == pytest.approx(0.0)

    def test_one_ev_offline_does_not_disable_the_other(self) -> None:
        live = LiveState()
        live.ev_planned_load_current_soc_pct = None
        live.ev_second_planned_load_current_soc_pct = 30.0

        planner_input = self._build(self._cfg(), live)

        assert planner_input.ev_planned_load_enabled is False
        assert planner_input.ev_second_planned_load_enabled is True

    def test_disabled_feature_stays_disabled_with_a_valid_reading(self) -> None:
        cfg = SensorConfig()
        cfg.ev_planned_load_enabled = False
        live = LiveState()
        live.ev_planned_load_current_soc_pct = 42.0

        assert self._build(cfg, live).ev_planned_load_enabled is False


class TestLiveStateDefaults:
    """The unknown-SoC sentinel must survive as ``None``."""

    def test_soc_defaults_to_unknown_not_zero(self) -> None:
        live = LiveState()

        assert live.ev_planned_load_current_soc_pct is None
        assert live.ev_second_planned_load_current_soc_pct is None


# ---------------------------------------------------------------------------
# End-to-end: a full dry-run cycle with a dead vehicle integration
# ---------------------------------------------------------------------------


class TestFullCycleSurvivesVehicleOutage:
    """The scenario that motivated this module, driven through the coordinator.

    ``kia_uvo`` went ``unavailable`` at 04:45; by 05:44 HSEM had stopped
    planning, the last hardware write was nearly three hours old and the
    battery sat idle at its floor while importing at peak price.
    """

    @staticmethod
    async def _run_cycle(
        entity_states: dict[str, object],
    ) -> tuple[CoordinatorData, HSEMDataUpdateCoordinator]:
        """Run one read-only cycle and return ``(data, coordinator)``."""
        from unittest.mock import AsyncMock

        from tests.test_ha_mock_integration import (
            _patch_all_ha_helpers,
            make_bare_coordinator,
            make_fake_config_entry,
            make_fake_hass,
        )

        config_entry = make_fake_config_entry(
            {
                "hsem_read_only": True,
                "hsem_ev_smart_charging": True,
                "hsem_ev_soc": "sensor.kia_ev6_ev_battery_level",
                "hsem_ev_target_soc": 80.0,
            }
        )
        coordinator = make_bare_coordinator(
            hass=make_fake_hass(entity_states),  # type: ignore[arg-type]
            config_entry=config_entry,
        )
        coordinator._set_update_interval = AsyncMock()  # type: ignore[method-assign]
        captured: list[CoordinatorData] = []
        coordinator.async_set_updated_data = captured.append  # type: ignore[method-assign,assignment]

        with _patch_all_ha_helpers():
            await coordinator._async_run_update_cycle()

        assert len(captured) == 1
        assert captured[0].live is not None
        return captured[0], coordinator

    @staticmethod
    def _states(**overrides: object) -> dict[str, object]:
        from tests.test_ha_mock_integration import _BASE_ENTITY_STATES

        states: dict[str, object] = dict(_BASE_ENTITY_STATES)
        states.update(overrides)
        return states

    @pytest.mark.asyncio
    async def test_dead_ev_sensor_still_runs_the_planner(self) -> None:
        """The exact outage: the EV SoC entity is simply not there.

        Before the fix this cycle produced no planner output at all — the
        MILP never ran, so 191 of the 192 slots carried no recommendation and
        only the current slot held a price-outage fail-safe.
        """
        data, coordinator = await self._run_cycle(self._states())

        assert data.live is not None
        assert data.live.degraded_mode is DegradedMode.Degraded
        assert coordinator._last_planner_output is not None
        assert coordinator._previous_planner_winner_name is not None

    @pytest.mark.asyncio
    async def test_dead_ev_sensor_leaves_no_slot_unplanned(self) -> None:
        """Every slot in the horizon must carry a recommendation."""
        data, _ = await self._run_cycle(self._states())

        assert len(data.hourly_recommendations) == 192
        assert all(slot.recommendation for slot in data.hourly_recommendations)

    @pytest.mark.asyncio
    async def test_dead_ev_sensor_plans_no_ev_charging(self) -> None:
        """No slot may charge a car whose state of charge is unknown."""
        data, _ = await self._run_cycle(self._states())

        assert not any(
            abs(slot.ev_total_planned_load_kwh) > 1e-9
            for slot in data.hourly_recommendations
        )

    @pytest.mark.asyncio
    async def test_dead_battery_sensor_still_halts_planning(self) -> None:
        """The fail-closed path must survive the relaxation.

        ``state`` is not asserted here: the price-outage fail-safe hold runs
        before either gate and legitimately overrides it.  What must not
        happen is the planner running on incomplete battery data.
        """
        data, coordinator = await self._run_cycle(
            self._states(**{"sensor.batteries_state_of_capacity": "unavailable"})
        )

        assert data.live is not None
        assert data.live.degraded_mode is DegradedMode.Error
        assert coordinator._last_planner_output is None
        assert coordinator._previous_planner_winner_name is None
