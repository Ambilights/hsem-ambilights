"""EV energy must not be lost to the charger's minimum operating power.

The MILP models EV charge as a continuous variable, so it can spread a small
requirement thinly across many slots.  A charger cannot run below its minimum
operating power, and the writer used to respond by zeroing that slot's power
command — while leaving the energy in ``ev_total_planned_load_kwh``.

Observed on live hardware: a 4.41 kWh requirement was spread across five
15-minute slots at ~2.6 kW each against a 3600 W charger minimum.  Four slots
had their command zeroed, so ``sensor.hsem_ev_optimal_charging_plan`` reported
one slot totalling 2.078 kWh, its state read ``waiting`` for the other four,
and the charge automation kept the charger off.  The plan silently scheduled
less than half of what it said it needed.

Energy is now carried from later slots into earlier ones — never the reverse,
so nothing is pushed past the deadline the MILP respected — and the per-slot
AC load map is built from the *placed* allocation so grid import/export stay
consistent with the schedule that is actually commanded.
"""

from __future__ import annotations

import pytest

from custom_components.hsem.models.ev_config import EVConfig
from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.planner.milp._write_results import (
    _redistribute_below_minimum_power,
)
from custom_components.hsem.planner.milp_optimizer import (
    is_scipy_available,
    solve_milp,
)
from tests.planner.test_session_ev import _NOW, _build_slots

# A 15-minute slot, a 90 %-efficient 11 kW charger with a 3600 W minimum —
# the live configuration that exposed the defect.
SLOT_HOURS = 0.25
EFF = 0.9
MIN_W = 3600.0
RATED_W = 11000.0

# DC energy corresponding to the minimum and the rating over a full slot.
MIN_DC = MIN_W * SLOT_HOURS * EFF / 1000.0  # 0.81 kWh
MAX_DC = RATED_W * SLOT_HOURS * EFF / 1000.0  # 2.475 kWh


def _redistribute(
    dc_by_slot: dict[int, float], **kwargs: float
) -> tuple[dict[int, float], float]:
    hours = {i: SLOT_HOURS for i in dc_by_slot}
    return _redistribute_below_minimum_power(
        dc_by_slot,
        slot_hours=hours,
        charger_efficiency=EFF,
        charger_min_power_w=float(kwargs.get("min_w", MIN_W)),
        rated_ac_power_w=float(kwargs.get("rated_w", RATED_W)),
    )


class TestEnergyIsConserved:
    """Nothing may be silently dropped from the schedule."""

    def test_live_case_keeps_every_kwh(self) -> None:
        """The exact allocation observed: 5 slots, 4 below the minimum."""
        solved = {10: 0.601, 11: 0.577, 12: 0.577, 13: 0.577, 14: 2.078}

        placed, unplaceable = _redistribute(solved)

        assert unplaceable == pytest.approx(0.0, abs=1e-9)
        assert sum(placed.values()) == pytest.approx(sum(solved.values()))

    def test_live_case_leaves_no_slot_below_the_minimum(self) -> None:
        solved = {10: 0.601, 11: 0.577, 12: 0.577, 13: 0.577, 14: 2.078}

        placed, _ = _redistribute(solved)

        assert placed
        for dc in placed.values():
            assert dc >= MIN_DC - 1e-9

    def test_bounded_recovery_fills_existing_commandable_headroom(self) -> None:
        """Recover a sub-minimum residue into selected slots with spare capacity.

        Four 60-minute solved allocations total 6.000 DC kWh.  At 90%
        efficiency, the 1380 W operating minimum is 1.242 DC kWh and the
        approximately 3333 W rating is 3.000 DC kWh.  Backward concentration
        leaves the two 0.450 kWh fragments below the minimum, so the bounded
        recovery pass must fill the already-commandable 2.100 kWh slot to
        3.000 rather than silently dropping 0.900 kWh.
        """
        solved = {0: 0.45, 2: 0.45, 3: 3.0, 4: 2.1}
        slot_hours = dict.fromkeys(solved, 1.0)
        efficiency = 0.9
        minimum_power_w = 1380.0
        rated_power_w = 10_000.0 / 3.0
        minimum_dc_kwh = minimum_power_w * efficiency / 1000.0
        maximum_dc_kwh = rated_power_w * efficiency / 1000.0

        placed, unplaceable = _redistribute_below_minimum_power(
            solved,
            slot_hours=slot_hours,
            charger_efficiency=efficiency,
            charger_min_power_w=minimum_power_w,
            rated_ac_power_w=rated_power_w,
        )

        assert unplaceable == pytest.approx(0.0, abs=1e-9)
        assert sum(placed.values()) == pytest.approx(6.0, abs=1e-9)
        assert placed == {3: pytest.approx(3.0), 4: pytest.approx(3.0)}
        assert set(placed).issubset(solved)
        assert max(placed) <= max(solved)
        for dc_kwh in placed.values():
            assert dc_kwh >= minimum_dc_kwh - 1e-9
            assert dc_kwh <= maximum_dc_kwh + 1e-9

    def test_energy_moves_earlier_never_later(self) -> None:
        """Moving energy later could push it past the deadline."""
        solved = {10: 0.601, 11: 0.577, 12: 0.577, 13: 0.577, 14: 2.078}

        placed, _ = _redistribute(solved)

        assert max(placed) <= max(solved)
        assert min(placed) >= min(solved)

    def test_an_already_feasible_allocation_is_untouched(self) -> None:
        solved = {3: 2.0, 4: 1.5, 5: 0.9}

        placed, unplaceable = _redistribute(solved)

        assert placed == solved
        assert unplaceable == pytest.approx(0.0)

    def test_a_single_feasible_slot_is_untouched(self) -> None:
        placed, unplaceable = _redistribute({7: 2.078})

        assert placed == {7: 2.078}
        assert unplaceable == pytest.approx(0.0)


class TestCapacityLimits:
    """No slot may be filled beyond what the charger can deliver."""

    def test_no_slot_exceeds_the_charger_rating(self) -> None:
        solved = {1: 2.4, 2: 2.4, 3: 0.2, 4: 0.2}

        placed, _ = _redistribute(solved)

        for dc in placed.values():
            assert dc <= MAX_DC + 1e-9

    def test_overflow_beyond_the_earliest_slot_is_reported(self) -> None:
        """Energy no slot can absorb must surface, not vanish silently."""
        solved = {1: 2.4, 2: 2.4}

        placed, unplaceable = _redistribute(solved, rated_w=RATED_W)

        assert sum(placed.values()) + unplaceable == pytest.approx(4.8)

    def test_a_lone_sub_minimum_slot_becomes_unplaceable(self) -> None:
        """With nowhere earlier to move it, the energy is reported, not lost."""
        placed, unplaceable = _redistribute({5: 0.2})

        assert placed == {}
        assert unplaceable == pytest.approx(0.2)


class TestDegenerateInputs:
    """Guards that must not change behaviour."""

    def test_no_minimum_configured_is_a_passthrough(self) -> None:
        solved = {1: 0.1, 2: 0.2}

        placed, unplaceable = _redistribute(solved, min_w=0.0)

        assert placed == solved
        assert unplaceable == pytest.approx(0.0)

    def test_empty_allocation_is_a_passthrough(self) -> None:
        placed, unplaceable = _redistribute({})

        assert placed == {}
        assert unplaceable == pytest.approx(0.0)

    def test_the_returned_mapping_is_a_copy(self) -> None:
        """Callers must be free to mutate without touching the LP solution."""
        solved = {1: 2.0}

        placed, _ = _redistribute(solved, min_w=0.0)
        placed[1] = 99.0

        assert solved[1] == pytest.approx(2.0)


class TestPartiallyElapsedSlot:
    """The current slot offers fewer minutes, so it needs more power."""

    def test_a_short_current_slot_needs_proportionally_more_energy(self) -> None:
        """5 minutes at the 3600 W minimum is 0.27 kWh, not 0.81."""
        five_minutes = 5.0 / 60.0

        placed, unplaceable = _redistribute_below_minimum_power(
            {0: 0.3},
            slot_hours={0: five_minutes},
            charger_efficiency=EFF,
            charger_min_power_w=MIN_W,
            rated_ac_power_w=RATED_W,
        )

        assert placed == {0: pytest.approx(0.3)}
        assert unplaceable == pytest.approx(0.0)

    def test_a_short_current_slot_still_rejects_too_little(self) -> None:
        five_minutes = 5.0 / 60.0

        placed, unplaceable = _redistribute_below_minimum_power(
            {0: 0.05},
            slot_hours={0: five_minutes},
            charger_efficiency=EFF,
            charger_min_power_w=MIN_W,
            rated_ac_power_w=RATED_W,
        )

        assert placed == {}
        assert unplaceable == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# End-to-end through the real solver
# ---------------------------------------------------------------------------


_pytestmark_scipy = pytest.mark.skipif(
    not is_scipy_available(), reason="scipy not available in this environment"
)


@_pytestmark_scipy
class TestThroughTheSolver:
    """The live defect, reproduced and fixed through ``solve_milp``."""

    @staticmethod
    def _solve() -> list[PlannedSlot]:
        # PV surplus spread thinly across many slots is what makes a thin
        # allocation optimal: each slot offers ~0.65 kWh of free solar, well
        # under the 0.81 kWh a 3600 W charger needs to run for a full slot.
        # Grid is priced high so the solver prefers the surplus.
        slots = _build_slots(
            12,
            start_hour=14,
            import_price=2.00,
            pv_kwh=0.75,
            consumption_kwh=0.10,
            interval_minutes=15,
        )
        ev = EVConfig(
            enabled=True,
            initial_soc_kwh=33.39,
            target_kwh=37.80,  # 4.41 kWh needed
            capacity_kwh=63.0,
            max_charge_per_slot=2.475,
            charger_efficiency=0.90,
            charger_min_power_w=3600.0,
            deadline_slot=9,
        )
        result = solve_milp(
            slots,
            _NOW,
            current_kwh=0.0,
            usable_kwh=10.0,
            max_charge_per_slot=5.0,
            max_discharge_per_slot=None,
            ev_configs=[ev],
        )
        assert result is not None
        return result[0]

    def test_every_scheduled_slot_is_commandable(self) -> None:
        """The defect: slots carried energy but a zero charger command."""
        out_slots = self._solve()

        scheduled = [s for s in out_slots if s.ev_total_planned_load_kwh > 1e-9]
        assert scheduled, "expected the solver to schedule some EV charging"
        for slot in scheduled:
            assert slot.ev_charger_calculated_power >= 3600.0, (
                f"slot {slot.start} carries "
                f"{slot.ev_total_planned_load_kwh} kWh with a "
                f"{slot.ev_charger_calculated_power} W command"
            )

    def test_the_full_requirement_is_scheduled(self) -> None:
        out_slots = self._solve()

        scheduled_dc = sum(s.ev_total_planned_load_kwh * 0.90 for s in out_slots)
        assert scheduled_dc == pytest.approx(4.41, abs=0.05)

    def test_energy_balance_holds_in_every_scheduled_slot(self) -> None:
        """Redistribution must not desynchronise the derived grid flows."""
        out_slots = self._solve()

        for slot in out_slots:
            if slot.ev_total_planned_load_kwh <= 1e-9:
                continue
            supply = slot.grid_import_kwh + slot.solcast_pv_estimate_kwh
            demand = (
                slot.avg_house_consumption_kwh
                + slot.ev_total_planned_load_kwh
                + slot.grid_export_kwh
            )
            assert supply == pytest.approx(demand, abs=0.05), (
                f"slot {slot.start}: supply {supply:.3f} != demand {demand:.3f}"
            )


class TestSurplusOnlyHeadroom:
    """Surplus-only EVs may use unused PV, never grid import."""

    def test_an_importing_slot_offers_no_surplus(self) -> None:
        """Granting inf there funded concentration from the grid."""
        placed, unplaceable = _redistribute_below_minimum_power(
            {0: 0.45, 1: 0.45},
            slot_hours={0: 0.25, 1: 0.25},
            charger_efficiency=EFF,
            charger_min_power_w=MIN_W,
            rated_ac_power_w=RATED_W,
            max_extra_dc={0: 0.0, 1: 0.0},
        )

        assert placed == {}
        assert unplaceable == pytest.approx(0.90)

    def test_real_surplus_is_still_usable(self) -> None:
        placed, _ = _redistribute_below_minimum_power(
            {0: 0.45, 1: 0.45},
            slot_hours={0: 0.25, 1: 0.25},
            charger_efficiency=EFF,
            charger_min_power_w=MIN_W,
            rated_ac_power_w=RATED_W,
            max_extra_dc={0: 0.45, 1: 0.0},
        )

        assert placed[0] == pytest.approx(0.90)
