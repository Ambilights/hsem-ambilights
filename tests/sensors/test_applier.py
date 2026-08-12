"""Tests for custom_sensors/applier.py.

The :func:`_parse_power_control_pct` helper is pure Python and fully testable
without Home Assistant.  The async hardware-write functions are covered by
integration tests; here we only test the deterministic helper.
"""

from __future__ import annotations

from custom_components.hsem.custom_sensors.applier import _parse_power_control_pct


class TestParsePowerControlPct:
    """Unit tests for the inverter power control state parser."""

    def test_unlimited_returns_100(self):
        assert _parse_power_control_pct("Unlimited") == 100

    def test_unlimited_case_insensitive(self):
        assert _parse_power_control_pct("unlimited") == 100
        assert _parse_power_control_pct("UNLIMITED") == 100

    def test_limited_to_80_percent(self):
        assert _parse_power_control_pct("Limited to 80%") == 80

    def test_limited_to_0_percent(self):
        assert _parse_power_control_pct("Limited to 0%") == 0

    def test_fractional_rounds_to_int(self):
        assert _parse_power_control_pct("Limited to 79.6%") == 80

    def test_none_returns_none(self):
        assert _parse_power_control_pct(None) is None

    def test_integer_returns_none(self):
        assert _parse_power_control_pct(100) is None  # type: ignore[arg-type]  # test passes mock where real type expected

    def test_empty_string_returns_none(self):
        assert _parse_power_control_pct("") is None

    def test_unknown_string_returns_none(self):
        assert _parse_power_control_pct("some other value") is None

    def test_whitespace_stripped(self):
        assert _parse_power_control_pct("  Limited to 50%  ") == 50

    # --- localization regression tests (bug fix) ---

    def test_danish_unlimited(self):
        """Danish HA translation of 'Unlimited'."""
        assert _parse_power_control_pct("Ikke begrænset") == 100

    def test_dutch_unlimited(self):
        """Dutch HA translation of 'Unlimited'."""
        assert _parse_power_control_pct("Onbeperkt") == 100

    def test_german_unlimited(self):
        """German HA translation of 'Unlimited'."""
        assert _parse_power_control_pct("Unbegrenzt") == 100

    def test_german_limited(self):
        """German 'Begrenzt auf 80 %' should yield 80."""
        assert _parse_power_control_pct("Begrenzt auf 80 %") == 80

    def test_dutch_limited(self):
        """Dutch 'Beperkt tot 75%' should yield 75."""
        assert _parse_power_control_pct("Beperkt tot 75%") == 75

    def test_fractional_localized(self):
        """Localized percentage with decimal rounds correctly."""
        assert _parse_power_control_pct("Begrenzt auf 79.6 %") == 80


# ---------------------------------------------------------------------------
# compute_ev_discharge_cap_w — EV discharge cap selection (issue #592, beta8)
# ---------------------------------------------------------------------------


class TestComputeEvDischargeCapW:
    """With history present, the cap IS the historical baseline — the live
    reading must not move it in either direction (issue #592)."""

    @staticmethod
    def _cap(**kwargs):
        from custom_components.hsem.custom_sensors.applier import (
            compute_ev_discharge_cap_w,
        )

        return compute_ev_discharge_cap_w(**kwargs)

    def test_live_below_history_holds_baseline(self):
        """Sensor drift pulling live below the baseline must NOT lower the
        cap (beta8 ratchet: 363→289→…→40 W over one night)."""
        cap = self._cap(
            live_net_w=200.0,  # drifted below the true 400 W baseline
            ev_power_available=True,
            historical_w=400,
            sub_window_ws=[400],
        )
        assert cap == 400

    def test_live_above_history_does_not_raise_cap(self):
        """House noise (cooking, heat pump) must NOT raise the cap —
        swinging with live demand for hours drains the battery into a
        grid-served EV session (v6.2.0-beta1: 652→1968→928 W swings)."""
        cap = self._cap(
            live_net_w=900.0,
            ev_power_available=True,
            historical_w=400,
            sub_window_ws=[400],
        )
        assert cap == 400

    def test_live_spike_does_not_raise_cap(self):
        """Even a huge live spike leaves the cap at the baseline."""
        cap = self._cap(
            live_net_w=5000.0,
            ev_power_available=True,
            historical_w=400,
            sub_window_ws=[400],
        )
        assert cap == 400

    def test_no_history_trusts_live(self):
        cap = self._cap(
            live_net_w=350.0,
            ev_power_available=True,
            historical_w=0,
            sub_window_ws=[],
        )
        assert cap == 350

    def test_no_ev_power_sensor_uses_min_sub_window(self):
        """Boolean-only EV sensor: fall back to the smallest sub-window."""
        cap = self._cap(
            live_net_w=None,
            ev_power_available=False,
            historical_w=400,
            sub_window_ws=[280, 520, 480, 400],
        )
        assert cap == 280

    def test_negative_live_treated_as_zero(self):
        """CT clamp < EV sensor (slight over-read) → live is negative;
        cap still holds the historical baseline."""
        cap = self._cap(
            live_net_w=-150.0,
            ev_power_available=True,
            historical_w=400,
            sub_window_ws=[400],
        )
        assert cap == 400

    def test_everything_missing_returns_zero(self):
        cap = self._cap(
            live_net_w=None,
            ev_power_available=False,
            historical_w=0,
            sub_window_ws=[],
        )
        assert cap == 0


# ---------------------------------------------------------------------------
# _wait_mode_self_consumption_cap_w — reserve-preserving discharge cap (issue #742)
# ---------------------------------------------------------------------------


class TestWaitModeSelfConsumptionCapW:
    """Unit tests for the wait-mode self-consumption discharge cap."""

    @staticmethod
    def _cap(**kwargs):
        from custom_components.hsem.custom_sensors.applier import (
            _wait_mode_self_consumption_cap_w,
        )

        return _wait_mode_self_consumption_cap_w(**kwargs)

    def test_no_surplus_returns_zero(self):
        cap = self._cap(
            battery_capacity_kwh=2.0,
            required_capacity_kwh=2.0,
            slot_hours=0.25,
            max_discharge_power_w=5000,
        )
        assert cap == 0

    def test_below_reserve_returns_zero(self):
        cap = self._cap(
            battery_capacity_kwh=1.5,
            required_capacity_kwh=2.0,
            slot_hours=0.25,
            max_discharge_power_w=5000,
        )
        assert cap == 0

    def test_surplus_converted_to_power(self):
        """1 kWh surplus over a 1-hour slot → 1000 W cap."""
        cap = self._cap(
            battery_capacity_kwh=3.0,
            required_capacity_kwh=2.0,
            slot_hours=1.0,
            max_discharge_power_w=5000,
        )
        assert cap == 1000

    def test_surplus_over_short_slot(self):
        """1 kWh surplus over a 15-minute slot → 4000 W cap."""
        cap = self._cap(
            battery_capacity_kwh=3.0,
            required_capacity_kwh=2.0,
            slot_hours=0.25,
            max_discharge_power_w=5000,
        )
        assert cap == 4000

    def test_cap_limited_by_max_discharge_power(self):
        cap = self._cap(
            battery_capacity_kwh=10.0,
            required_capacity_kwh=0.0,
            slot_hours=0.25,
            max_discharge_power_w=2500,
        )
        assert cap == 2500

    def test_zero_slot_hours_returns_zero(self):
        cap = self._cap(
            battery_capacity_kwh=5.0,
            required_capacity_kwh=0.0,
            slot_hours=0.0,
            max_discharge_power_w=5000,
        )
        assert cap == 0


# ---------------------------------------------------------------------------
# Fully Fed export power cap
# ---------------------------------------------------------------------------


class TestFullyFedDischargeCapW:
    """The Fully Fed battery contribution follows the plan without overshoot."""

    @staticmethod
    def _cap(**kwargs):
        from custom_components.hsem.custom_sensors.applier import (
            _fully_fed_discharge_cap_w,
        )

        return _fully_fed_discharge_cap_w(**kwargs)

    def test_planned_slot_energy_becomes_power_cap(self):
        cap = self._cap(
            planned_discharge_kwh=0.45,
            slot_hours=0.25,
            remaining_slot_hours=0.25,
            battery_capacity_kwh=20.0,
            planned_end_capacity_kwh=19.55,
            max_discharge_power_w=10000,
        )
        assert cap == 1800

    def test_hardware_maximum_is_respected(self):
        cap = self._cap(
            planned_discharge_kwh=2.5,
            slot_hours=0.25,
            remaining_slot_hours=0.25,
            battery_capacity_kwh=20.0,
            planned_end_capacity_kwh=17.5,
            max_discharge_power_w=5000,
        )
        assert cap == 5000

    def test_proportional_energy_and_time_do_not_ratchet_cap(self):
        initial_cap = self._cap(
            planned_discharge_kwh=2.5,
            slot_hours=0.25,
            remaining_slot_hours=0.25,
            battery_capacity_kwh=28.2,
            planned_end_capacity_kwh=25.7,
            max_discharge_power_w=10000,
        )
        three_minutes_later_cap = self._cap(
            planned_discharge_kwh=2.5,
            slot_hours=0.25,
            remaining_slot_hours=0.20,
            battery_capacity_kwh=27.7,
            planned_end_capacity_kwh=25.7,
            max_discharge_power_w=10000,
        )
        assert initial_cap == 10000
        assert three_minutes_later_cap == 10000

    def test_remaining_energy_tapers_power_near_target(self):
        cap = self._cap(
            planned_discharge_kwh=2.5,
            slot_hours=0.25,
            remaining_slot_hours=0.10,
            battery_capacity_kwh=25.8,
            planned_end_capacity_kwh=25.7,
            max_discharge_power_w=10000,
        )
        assert cap == 1000

    def test_near_zero_remaining_time_cannot_exceed_planned_power(self):
        cap = self._cap(
            planned_discharge_kwh=2.5,
            slot_hours=0.25,
            remaining_slot_hours=1.0 / 3600.0,
            battery_capacity_kwh=25.8,
            planned_end_capacity_kwh=25.7,
            max_discharge_power_w=30000,
        )
        assert cap == 10000

    def test_callback_before_slot_start_uses_full_slot_duration(self):
        cap = self._cap(
            planned_discharge_kwh=2.5,
            slot_hours=0.25,
            remaining_slot_hours=0.50,
            battery_capacity_kwh=28.2,
            planned_end_capacity_kwh=25.7,
            max_discharge_power_w=10000,
        )
        assert cap == 10000

    def test_live_capacity_at_plan_target_stops_battery(self):
        cap = self._cap(
            planned_discharge_kwh=0.5,
            slot_hours=0.25,
            remaining_slot_hours=0.10,
            battery_capacity_kwh=10.25,
            planned_end_capacity_kwh=10.25,
            max_discharge_power_w=10000,
        )
        assert cap == 0

    def test_invalid_or_finished_duration_fails_closed(self):
        invalid_slot_cap = self._cap(
            planned_discharge_kwh=0.5,
            slot_hours=0.0,
            remaining_slot_hours=0.10,
            battery_capacity_kwh=10.5,
            planned_end_capacity_kwh=10.0,
            max_discharge_power_w=10000,
        )
        finished_slot_cap = self._cap(
            planned_discharge_kwh=0.5,
            slot_hours=0.25,
            remaining_slot_hours=0.0,
            battery_capacity_kwh=10.5,
            planned_end_capacity_kwh=10.0,
            max_discharge_power_w=10000,
        )
        assert invalid_slot_cap == 0
        assert finished_slot_cap == 0

    def test_non_positive_plan_fails_closed(self):
        cap = self._cap(
            planned_discharge_kwh=0.0,
            slot_hours=0.25,
            remaining_slot_hours=0.25,
            battery_capacity_kwh=10.5,
            planned_end_capacity_kwh=10.0,
            max_discharge_power_w=10000,
        )
        assert cap == 0


class TestForcibleDischargeState:
    """Legacy cleanup runs only for a genuinely active command."""

    @staticmethod
    def _active(state):
        from custom_components.hsem.custom_sensors.applier import (
            _is_forcible_discharge_active,
        )

        return _is_forcible_discharge_active(state)

    def test_stopped_and_missing_are_inactive(self):
        assert self._active(None) is False
        assert self._active("") is False
        assert self._active("Stopped") is False
        assert self._active("unknown") is False
        assert self._active("unavailable") is False

    def test_active_discharge_summary_is_detected(self):
        assert self._active("Discharging at 10000W until 5.0%") is True
