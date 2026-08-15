"""Three defects found by external review of v6.2.2-powmr.23.

1. **Missing battery power was not fail-closed.**  ``compute_phase_charge_limits``
   reconstructs house load by removing the battery's own contribution from the
   meter snapshot.  Substituting ``0.0`` for an unreadable reading is only
   conservative while the battery *charges*.  While it *discharges* it is
   suppressing metered import, so the omission makes the remaining import look
   like spare fuse capacity — and a grid-charge slot opens exactly during the
   discharge-to-charge transition.

2. **Live sessions were classified per slot, not per EV.**  A slot belonging to
   one EV's fixed session froze a second EV's flexible allocation, which then
   kept the minimum-power zeroing and was commanded 0 W while the plan still
   counted its energy.

3. **Redistribution ignored surplus-only constraints.**  For an EV charging
   past target the LP carries explicit surplus-only rows; concentrating energy
   into one slot could manufacture grid import there.
"""

from __future__ import annotations

import math

import pytest

from custom_components.hsem.custom_sensors.phase_charge_limiter import (
    PhaseAwareChargeCommands,
    build_phase_aware_charge_commands,
)
from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.planner.milp._write_results import (
    _redistribute_below_minimum_power,
)
from custom_components.hsem.utils.phase_power import compute_phase_charge_limits
from custom_components.hsem.utils.recommendations import Recommendations

FUSE_A = 16.0
PHASE_LIMIT_W = FUSE_A * 230.0  # 3680 W


# ---------------------------------------------------------------------------
# 1. Missing battery power
# ---------------------------------------------------------------------------


def _authorised(meter_per_phase: float, battery_sensor_w: float) -> float:
    """Return the charge power the limiter authorises for one meter snapshot."""
    return compute_phase_charge_limits(
        measured_phase_power_w=(meter_per_phase, meter_per_phase, meter_per_phase),
        fuse_amps=FUSE_A,
        desired_primary_charge_power_w=10000.0,
        primary_is_controlled=True,
        primary_actual_battery_power_w=battery_sensor_w,
        primary_charge_efficiency_pct=98.0,
        primary_discharge_efficiency_pct=98.0,
        desired_secondary_charge_current_a=0.0,
        secondary_actual_site_delta_w=0.0,
        secondary_desired_noncharge_site_delta_w=0.0,
        secondary_grid_phase=3,
        secondary_nominal_voltage_v=48.0,
        secondary_charge_efficiency_pct=100.0,
        secondary_min_charge_current_a=10.0,
        secondary_max_charge_current_a=100.0,
        secondary_charge_current_step_a=10.0,
    ).primary_charge_power_w


class TestDischargingBatteryMasksTheLoad:
    """Why ``or 0.0`` could not stand: it is unsafe in one direction."""

    # House 3000 W/phase; battery discharging 9 kW covers 2940 W/phase of it,
    # so the meter shows only 60 W/phase.
    METER_WHILE_DISCHARGING = 60.0

    def test_a_known_reading_keeps_the_charge_under_the_fuse(self) -> None:
        authorised = _authorised(self.METER_WHILE_DISCHARGING, -9000.0)
        after_transition = 3000.0 + authorised / 3

        assert after_transition <= PHASE_LIMIT_W

    def test_substituting_zero_would_blow_past_the_fuse(self) -> None:
        """Pins the defect: the same snapshot, sensor unavailable."""
        authorised = _authorised(self.METER_WHILE_DISCHARGING, 0.0)
        after_transition = 3000.0 + authorised / 3

        assert authorised == pytest.approx(10000.0)
        assert after_transition / 230.0 > FUSE_A * 1.5

    def test_a_charging_battery_errs_the_safe_way(self) -> None:
        """The case that made ``or 0.0`` look harmless."""
        meter = 3000.0 + 3000.0 / 0.98 / 3  # house plus its own charge draw

        assert _authorised(meter, 0.0) <= _authorised(meter, 3000.0)


class TestLimiterRefusesWithoutBatteryPower:
    """The limiter must decline rather than size from a partial snapshot."""

    @staticmethod
    def _commands(
        battery_power_w: float | None, recommendation: str
    ) -> PhaseAwareChargeCommands:
        from datetime import UTC, datetime, timedelta

        cfg = SensorConfig()
        cfg.phase_aware_charging_enabled = True
        cfg.main_fuse_amps = int(FUSE_A)
        cfg.main_fuse_phases = 3
        live = LiveState()
        live.grid_phase_power_w = (60.0, 60.0, 60.0)
        live.huawei_batteries_charge_discharge_power_w = battery_power_w
        live.huawei_batteries_max_charge_power_w = 10000.0
        start = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)
        rec = HourlyRecommendation(
            start=start,
            end=start + timedelta(minutes=15),
            avg_house_consumption_kwh=0.75,
            avg_house_consumption_1d_kwh=0.75,
            avg_house_consumption_3d_kwh=0.75,
            avg_house_consumption_7d_kwh=0.75,
            avg_house_consumption_14d_kwh=0.75,
            batteries_charged_kwh=2.5,
            batteries_discharged_kwh=0.0,
            estimated_battery_capacity_kwh=6.0,
            estimated_battery_soc_pct=25.0,
            estimated_cost_currency=0.0,
            estimated_net_consumption_kwh=0.75,
            export_price=0.1,
            grid_export_kwh=0.0,
            grid_import_kwh=0.75,
            import_price=1.0,
            recommendation=recommendation,
            solcast_pv_estimate_kwh=0.0,
        )
        return build_phase_aware_charge_commands(cfg, live, rec)

    def test_grid_charge_is_blocked_at_zero_watts(self) -> None:
        cmds = self._commands(None, Recommendations.BatteriesChargeGrid.value)

        assert cmds.primary_grid_charge_power_w == 0.0

    def test_a_non_finite_reading_is_also_refused(self) -> None:
        cmds = self._commands(math.nan, Recommendations.BatteriesChargeGrid.value)

        assert cmds.primary_grid_charge_power_w == 0.0

    def test_a_readable_battery_still_gets_a_limit(self) -> None:
        cmds = self._commands(-9000.0, Recommendations.BatteriesChargeGrid.value)

        assert cmds.primary_grid_charge_power_w is not None
        assert cmds.primary_grid_charge_power_w > 0.0

    def test_a_non_charging_slot_is_unaffected(self) -> None:
        """Nothing is being armed, so the reading is not needed."""
        cmds = self._commands(None, Recommendations.BatteriesWaitMode.value)

        assert cmds.primary_grid_charge_power_w is None


# ---------------------------------------------------------------------------
# 3. Redistribution must respect surplus-only slots
# ---------------------------------------------------------------------------


class TestRedistributionCeiling:
    """A slot may only absorb what its own constraints can fund."""

    HOURS = {0: 0.25, 1: 0.25}
    EFF = 0.9
    MIN_W = 3000.0
    RATED_W = 11000.0

    def _run(self, ceilings: dict[int, float] | None) -> tuple[dict[int, float], float]:
        return _redistribute_below_minimum_power(
            {0: 0.45, 1: 0.45},
            slot_hours=self.HOURS,
            charger_efficiency=self.EFF,
            charger_min_power_w=self.MIN_W,
            rated_ac_power_w=self.RATED_W,
            max_extra_dc=ceilings,
        )

    def test_without_a_ceiling_energy_concentrates(self) -> None:
        """Target-driven charging: importing is allowed, so concentrate."""
        placed, unplaceable = self._run(None)

        assert placed == {0: pytest.approx(0.90)}
        assert unplaceable == pytest.approx(0.0)

    def test_a_zero_ceiling_prevents_manufacturing_import(self) -> None:
        """Surplus-only: neither slot may take more than the solver gave it."""
        placed, unplaceable = self._run({0: 0.0, 1: 0.0})

        assert placed == {}
        assert unplaceable == pytest.approx(0.90)

    def test_a_partial_ceiling_caps_what_a_slot_absorbs(self) -> None:
        """0.45 solved + 0.30 headroom clears the 0.675 minimum, but no more."""
        placed, unplaceable = self._run({0: 0.30, 1: 0.0})

        assert placed[0] == pytest.approx(0.75)
        assert unplaceable == pytest.approx(0.15)

    def test_a_ceiling_too_small_to_reach_the_minimum_drops_the_slot(self) -> None:
        """Better to place nothing than command a power the charger ignores."""
        placed, unplaceable = self._run({0: 0.20, 1: 0.0})

        assert placed == {}
        assert unplaceable == pytest.approx(0.90)

    def test_an_infinite_ceiling_matches_no_ceiling(self) -> None:
        assert self._run({0: math.inf, 1: math.inf}) == self._run(None)
