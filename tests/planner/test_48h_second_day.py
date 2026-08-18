"""Regression tests for second-day correctness in a 48-hour plan.

Verifies that the full horizon is populated, second-day recommendations remain
well-formed and diverse, and day-two PV surplus can still charge the battery.
"""

from __future__ import annotations

from custom_components.hsem.models.hourly_consumption_average import (
    HourlyConsumptionAverage,
)
from custom_components.hsem.models.planner_input import PlannerInput
from custom_components.hsem.models.price_point import PricePoint
from custom_components.hsem.models.solcast_slot import SolcastSlot
from custom_components.hsem.planner import run_planner
from custom_components.hsem.utils.recommendations import Recommendations

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DISCHARGE_VALUES = {
    Recommendations.BatteriesDischargeMode.value,
    Recommendations.ForceBatteriesDischarge.value,
}
_CHARGE_VALUES = {
    Recommendations.BatteriesChargeGrid.value,
    Recommendations.BatteriesChargeSolar.value,
}


def _make_48h_input(
    *,
    now_iso: str = "2024-06-15T00:00:00+02:00",
    battery_soc_pct: float = 50.0,
    pv_kwh_per_hour: float = 0.0,
    load_kwh_per_hour: float = 0.5,
    months_winter: list[int] | None = None,
) -> PlannerInput:
    """Return a 48-hour summer planning input for second-day regression tests."""
    # Varying prices: cheap night (00-06), moderate day, expensive evening peak
    import_prices_24h = [
        0.08,
        0.06,
        0.05,
        0.05,
        0.06,
        0.09,  # 00-06 cheap
        0.15,
        0.22,
        0.26,
        0.24,
        0.12,
        0.08,  # 06-12
        0.06,
        0.07,
        0.10,
        0.25,
        0.30,
        0.32,  # 12-18
        0.29,
        0.24,
        0.18,
        0.14,
        0.11,
        0.09,  # 18-24
    ]

    prices = [
        PricePoint(
            hour=h,
            import_price=import_prices_24h[h],
            export_price=max(import_prices_24h[h] - 0.02, 0.0),
        )
        for h in range(24)
    ]
    solar = [SolcastSlot(hour=h, pv_estimate=pv_kwh_per_hour) for h in range(24)]
    consumption = [
        HourlyConsumptionAverage(
            hour=h,
            avg_1d=load_kwh_per_hour,
            avg_3d=load_kwh_per_hour,
            avg_7d=load_kwh_per_hour,
            avg_14d=load_kwh_per_hour,
        )
        for h in range(24)
    ]

    return PlannerInput(
        now_iso=now_iso,
        interval_minutes=60,
        interval_length_hours=48,
        battery_soc_pct=battery_soc_pct,
        battery_rated_capacity_kwh=10.0,
        battery_end_of_discharge_soc_pct=10.0,
        battery_max_charge_power_w=5000.0,
        battery_purchase_price=0.0,
        battery_expected_cycles=6000,
        weight_1d=25,
        weight_3d=30,
        weight_7d=30,
        weight_14d=15,
        consumption_averages=consumption,
        price_points=prices,
        solcast_slots=solar,
        excess_export_enabled=False,
        excess_export_discharge_buffer_pct=10.0,
        excess_export_price_threshold=0.10,
        months_winter=(
            months_winter if months_winter is not None else [1, 2, 3, 4, 10, 11, 12]
        ),
        house_power_includes_ev=True,
        is_read_only=True,
    )


# ===========================================================================
# Basic 48-hour output contract
# ===========================================================================


class TestBasic48hContract:
    """The planner must return 48 slots and assign all of them."""

    def test_48h_produces_48_slots(self):
        result = run_planner(_make_48h_input())
        assert len(result.slots) == 48

    def test_all_slots_have_recommendation(self):
        result = run_planner(_make_48h_input())
        for slot in result.slots:
            assert slot.recommendation is not None, (
                f"Slot {slot.start.isoformat()} has no recommendation in 48h plan"
            )

    def test_slots_span_two_calendar_days(self):
        result = run_planner(_make_48h_input())
        dates = {s.start.date() for s in result.slots}
        assert len(dates) == 2, f"Expected slots on exactly 2 dates, got {dates}"


# ===========================================================================
# Day-2 slots must not all be BatteriesDischargeMode
# ===========================================================================


class TestDay2NotAllDischarge:
    """The second 24 hours must not be entirely discharge recommendations.

    Before the fix, `apply_optimization_strategy` would assign
    ``BatteriesDischargeMode`` to every unscheduled summer slot because:
    - The solar charging pass only processed today's (day-1) slots.
    - No charge windows were recognised for day-2 (schedules only fired once).
    """

    def test_day2_has_non_discharge_slots(self):
        """At least some day-2 slots should not be BatteriesDischargeMode."""
        result = run_planner(_make_48h_input())
        day1 = result.slots[0].start.date()
        day2 = day1.replace(day=day1.day + 1)

        day2_slots = [s for s in result.slots if s.start.date() == day2]
        non_discharge = [
            s for s in day2_slots if s.recommendation not in _DISCHARGE_VALUES
        ]
        assert len(non_discharge) > 0, (
            "Every day-2 slot is BatteriesDischargeMode — regression from the "
            "second-day planning bug. Day-2 recommendations: "
            f"{[(s.start.hour, s.recommendation) for s in day2_slots]}"
        )

    def test_day2_cheap_night_slots_are_not_discharge(self):
        """Cheap night slots (00:00-06:00) on day 2 must not be discharge."""
        result = run_planner(_make_48h_input())
        day1 = result.slots[0].start.date()
        day2 = day1.replace(day=day1.day + 1)

        cheap_night_day2 = [
            s for s in result.slots if s.start.date() == day2 and s.start.hour < 6
        ]
        assert cheap_night_day2, "No cheap-night day-2 slots found (unexpected)"

        all_discharge = all(
            s.recommendation in _DISCHARGE_VALUES for s in cheap_night_day2
        )
        assert not all_discharge, (
            "Cheap night slots (00:00-06:00) on day 2 are all BatteriesDischargeMode. "
            f"Recommendations: {[(s.start.hour, s.recommendation) for s in cheap_night_day2]}"
        )


# ===========================================================================
# Day-2 solar charging
# ===========================================================================


class TestDay2SolarCharging:
    """Day-two daytime PV can charge the battery for later local demand."""

    def test_day2_pv_surplus_gets_charge_solar(self):
        """A daytime-only PV surplus should charge before evening demand.

        An all-day PV surplus has no future battery use and is correctly
        exported directly by the MILP. This profile instead creates a real
        daytime-to-evening storage opportunity on both delivery days.
        """
        inp = _make_48h_input(
            pv_kwh_per_hour=0.0,
            load_kwh_per_hour=0.8,
            battery_soc_pct=10.0,
        )
        inp.solcast_slots = [
            SolcastSlot(hour=hour, pv_estimate=5.0 if 10 <= hour < 15 else 0.0)
            for hour in range(24)
        ]
        result = run_planner(inp)
        day2 = result.slots[0].start.date().replace(day=result.slots[0].start.day + 1)

        day2_solar_charge = [
            s
            for s in result.slots
            if s.start.date() == day2
            and s.recommendation == Recommendations.BatteriesChargeSolar.value
        ]
        assert day2_solar_charge, (
            "No day-two BatteriesChargeSolar slot despite daytime PV surplus "
            "and later local demand."
        )
