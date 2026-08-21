"""Tests for coordinator_builder helper functions.

Covers ``_resolve_max_discharge_power_w`` — the guard against the EV
discharge-cap feedback loop (issue #592, beta7).
"""

from __future__ import annotations

import pytest

from custom_components.hsem.coordinator_builder import (
    _normalize_live_house_for_secondary,
    _resolve_live_solar_measurement,
    _resolve_max_discharge_power_w,
    build_planner_input,
)
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.secondary_storage_config import (
    SecondaryStorageConfig,
)
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.planner.secondary_storage import SECONDARY_MODE_SBU
from custom_components.hsem.utils.phase_power import (
    POWMR_CHARGER_UTILITY,
    POWMR_OUTPUT_SBU,
    POWMR_OUTPUT_UTILITY,
    secondary_site_power_delta_w,
)


class TestNormalizeLiveHouseForSecondary:
    """Current-slot live load must match the configured history topology."""

    @staticmethod
    def _secondary(*, includes_load: bool) -> SecondaryStorageConfig:
        return SecondaryStorageConfig(
            enabled=True,
            capacity_kwh=15.0,
            current_soc_pct=60.0,
            min_soc_pct=40.0,
            max_soc_pct=100.0,
            nominal_voltage_v=48.0,
            load_power_w=200.0,
            max_charge_current_a=100.0,
            min_charge_current_a=10.0,
            base_load_includes_dedicated_load=includes_load,
        )

    @pytest.mark.parametrize(
        ("includes_load", "mode", "expected_w"),
        [
            (True, POWMR_OUTPUT_SBU, 1200.0),
            (True, POWMR_OUTPUT_UTILITY, 1000.0),
            (False, POWMR_OUTPUT_SBU, 1000.0),
            (False, POWMR_OUTPUT_UTILITY, 800.0),
        ],
    )
    def test_live_mode_is_normalized_exactly_once(
        self,
        includes_load: bool,
        mode: str,
        expected_w: float,
    ) -> None:
        secondary = self._secondary(includes_load=includes_load)
        delta_w = secondary_site_power_delta_w(
            battery_net_power_w=0.0,
            load_power_w=secondary.load_power_w,
            charge_efficiency_pct=secondary.charge_efficiency_pct,
            base_load_includes_dedicated_load=includes_load,
            output_source_priority=mode,
            charger_source_priority=None,
        )
        assert _normalize_live_house_for_secondary(1000.0, delta_w) == pytest.approx(
            expected_w
        )

    def test_unknown_mode_is_not_guessed(self) -> None:
        assert _normalize_live_house_for_secondary(1000.0, 0.0) == pytest.approx(1000.0)

    @pytest.mark.parametrize(
        ("includes_load", "mode", "raw_house_w", "expected_base_w"),
        [
            (True, POWMR_OUTPUT_UTILITY, 1700.0, 1200.0),
            (True, POWMR_OUTPUT_SBU, 1500.0, 1200.0),
            (False, POWMR_OUTPUT_UTILITY, 1700.0, 1000.0),
            (False, POWMR_OUTPUT_SBU, 1500.0, 1000.0),
        ],
    )
    def test_live_powmr_charge_is_removed_before_milp_reallocation(
        self,
        includes_load: bool,
        mode: str,
        raw_house_w: float,
        expected_base_w: float,
    ) -> None:
        """Actual AC charger draw cannot be counted again by the new solve."""
        secondary = self._secondary(includes_load=includes_load)
        secondary.charge_efficiency_pct = 96.0
        delta_w = secondary_site_power_delta_w(
            battery_net_power_w=480.0,
            load_power_w=secondary.load_power_w,
            charge_efficiency_pct=secondary.charge_efficiency_pct,
            base_load_includes_dedicated_load=includes_load,
            output_source_priority=mode,
            charger_source_priority=POWMR_CHARGER_UTILITY,
        )

        assert _normalize_live_house_for_secondary(
            raw_house_w, delta_w
        ) == pytest.approx(expected_base_w)

    def test_builder_removes_live_powmr_charge_from_current_slot_input(self) -> None:
        cfg = SensorConfig()
        cfg.secondary_storage.enabled = True
        cfg.secondary_storage.base_load_includes_dedicated_load = False
        cfg.secondary_storage.charge_efficiency_pct = 96.0
        live = LiveState()
        live.house_consumption_power_w = 1700.0
        live.secondary_storage.soc_pct = 60.0
        live.secondary_storage.load_power_w = 200.0
        live.secondary_storage.battery_net_power_w = 480.0
        live.secondary_storage.output_source_priority = POWMR_OUTPUT_UTILITY
        live.secondary_storage.charger_source_priority = POWMR_CHARGER_UTILITY

        planner_input = build_planner_input(
            cfg=cfg,
            live=live,
            hourly_recommendations=[],
            previous_winner_name=None,
            previous_winner_score=0.0,
        )

        assert planner_input.live_house_consumption_w == pytest.approx(1000.0)

    def test_builder_carries_verified_current_slot_mode_lock(self) -> None:
        """The coordinator's transient lock reaches the pure MILP input."""
        cfg = SensorConfig()
        cfg.secondary_storage.enabled = True
        live = LiveState()
        live.secondary_storage.soc_pct = 60.0
        live.secondary_storage.load_power_w = 200.0

        planner_input = build_planner_input(
            cfg=cfg,
            live=live,
            hourly_recommendations=[],
            previous_winner_name=None,
            previous_winner_score=0.0,
            secondary_current_slot_mode_lock=SECONDARY_MODE_SBU,
        )

        assert planner_input.secondary_storage.current_slot_mode_lock == (
            SECONDARY_MODE_SBU
        )


class TestResolveLiveSolarMeasurement:
    """Distinguish a real zero-PV reading from the unavailable-state fallback."""

    @staticmethod
    def _cfg(entity_id: str | None = "sensor.solar") -> SensorConfig:
        cfg = SensorConfig()
        cfg.solar_production_power = entity_id
        return cfg

    def test_configured_zero_measurement_is_available(self) -> None:
        live = LiveState()
        live.solar_production_power_w = 0.0

        value, available = _resolve_live_solar_measurement(self._cfg(), live)

        assert value == pytest.approx(0.0)
        assert available is True

    def test_read_failure_fallback_zero_is_unavailable(self) -> None:
        live = LiveState()
        live.solar_production_power_w = 0.0
        live.add_missing_entity(
            "Error reading solar_production_power (entity_id=sensor.solar): "
            "state unavailable or invalid"
        )

        value, available = _resolve_live_solar_measurement(self._cfg(), live)

        assert value == pytest.approx(0.0)
        assert available is False

    def test_unconfigured_default_zero_is_unavailable(self) -> None:
        live = LiveState()

        value, available = _resolve_live_solar_measurement(self._cfg(None), live)

        assert value == pytest.approx(0.0)
        assert available is False

    def test_builder_plumbs_available_zero_into_planner_input(self) -> None:
        cfg = self._cfg()
        live = LiveState()
        live.solar_production_power_w = 0.0

        planner_input = build_planner_input(
            cfg=cfg,
            live=live,
            hourly_recommendations=[],
            previous_winner_name=None,
            previous_winner_score=0.0,
        )

        assert planner_input.live_solar_production_w == pytest.approx(0.0)
        assert planner_input.live_solar_production_available is True

    def test_builder_plumbs_read_failure_as_unavailable(self) -> None:
        cfg = self._cfg()
        live = LiveState()
        live.solar_production_power_w = 0.0
        live.add_missing_entity("Error reading solar_production_power: unavailable")

        planner_input = build_planner_input(
            cfg=cfg,
            live=live,
            hourly_recommendations=[],
            previous_winner_name=None,
            previous_winner_score=0.0,
        )

        assert planner_input.live_solar_production_w == pytest.approx(0.0)
        assert planner_input.live_solar_production_available is False


class TestResolveMaxDischargePowerW:
    """The planner must see the battery's physical capability, not the
    applier's EV-capped write-back value."""

    @staticmethod
    def _live(
        *,
        ev_charging: bool,
        max_discharge_w: float,
        rated_wh: int = 10000,
    ) -> LiveState:
        live = LiveState()
        live.ev.is_charging = ev_charging
        live.huawei_batteries_max_discharge_power_w = max_discharge_w
        live.huawei_batteries_rated_capacity_wh = rated_wh
        return live

    def test_no_ev_charging_uses_rated_capability(self) -> None:
        """The planner always gets the physical capability derived from the
        rated capacity — the live entity is a commanded value, never a
        capability (issue #592)."""
        live = self._live(ev_charging=False, max_discharge_w=321.0)
        assert _resolve_max_discharge_power_w(live) == pytest.approx(5000.0)

    def test_ev_charging_uses_rated_capability(self) -> None:
        """During an EV session the applier caps the entity (e.g. 321 W).
        The planner must get the physical capability (5000 W for a
        10 kWh battery), not the capped value — otherwise the entire
        planning horizon is limited to the EV cap (issue #592)."""
        live = self._live(ev_charging=True, max_discharge_w=321.0)
        assert _resolve_max_discharge_power_w(live) == pytest.approx(5000.0)

    def test_ev_status_flicker_does_not_leak_cap(self) -> None:
        """A single-cycle EV boolean flicker must not let the still-capped
        entity poison an off-schedule planner run (beta8 regression)."""
        live = self._live(ev_charging=False, max_discharge_w=40.0)
        assert _resolve_max_discharge_power_w(live) == pytest.approx(5000.0)

    def test_missing_rated_capacity_falls_back_to_live(self) -> None:
        """Missing rated capacity → keep the live value (degraded but safe)."""
        live = self._live(ev_charging=True, max_discharge_w=321.0, rated_wh=0)
        assert _resolve_max_discharge_power_w(live) == pytest.approx(321.0)

    def test_missing_rated_capacity_and_live_value_returns_none(self) -> None:
        live = self._live(ev_charging=False, max_discharge_w=0.0, rated_wh=0)
        assert _resolve_max_discharge_power_w(live) is None
