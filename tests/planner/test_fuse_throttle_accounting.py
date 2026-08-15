"""The published plan must agree with what the fuse throttle will command.

``run_planner``'s post-hoc main-fuse check lowers EV charger power and battery
charge energy when a slot would exceed the fuse.  It previously left every
*derived* field at its pre-throttle value, so the slot went on advertising the
original grid import, EV load, net consumption and cost while the hardware was
about to be driven at less — the plan and the command disagreed.

Battery state of charge is deliberately still not re-simulated: ``simulate_soc``
runs inside the candidate selector so a winner's score matches its slots, and
re-running it afterwards would decouple the two.  A throttled slot therefore
still reports the SoC the unthrottled plan would have reached.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.planner_input import PlannerInput
from custom_components.hsem.planner.engine_core import _apply_main_fuse_throttle
from custom_components.hsem.utils.prices import SlotPrice

# 16 A three-phase = 11040 W = 2.76 kWh per 15-minute slot.
FUSE_A = 16
SLOT_MIN = 15
SLOT_H = SLOT_MIN / 60.0
FUSE_KWH = FUSE_A * 3 * 230.0 * SLOT_H / 1000.0


def _slot(start: datetime, *, ev_w: float, import_kwh: float) -> PlannedSlot:
    """A slot already over the fuse, driven by EV charging."""
    slot = PlannedSlot(
        start=start,
        end=start + timedelta(minutes=SLOT_MIN),
        price=SlotPrice(import_price=1.0, export_price=0.1),
        solcast_pv_estimate_kwh=0.0,
        avg_house_consumption_kwh=0.5,
        estimated_net_consumption_kwh=0.5,
        recommendation="batteries_wait_mode",
    )
    slot.ev_charger_calculated_power = ev_w
    ev_kwh = ev_w / 1000.0 * SLOT_H
    slot.ev_planned_load_kwh = ev_kwh
    slot.ev_accounted_load_kwh = 0.0
    slot.ev_total_planned_load_kwh = ev_kwh
    slot.estimated_net_consumption_kwh = 0.5 + ev_kwh
    slot.estimated_cost_currency = round(slot.estimated_net_consumption_kwh * 1.0, 4)
    slot.grid_import_kwh = import_kwh
    slot.price_actionable = True
    return slot


class TestThrottledSlotAccounting:
    """Every derived field must follow the command down."""

    @staticmethod
    def _throttle(ev_w: float, import_kwh: float) -> PlannedSlot:
        """Run only the fuse block over one over-limit slot."""
        start = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)
        slot = _slot(start, ev_w=ev_w, import_kwh=import_kwh)
        inp = PlannerInput(
            now_iso=start.isoformat(),
            main_fuse_amps=float(FUSE_A),
            main_fuse_phases=3,
            interval_minutes=SLOT_MIN,
        )
        _apply_main_fuse_throttle([slot], inp, [])
        return slot

    def test_grid_import_follows_the_throttle_down(self) -> None:
        """4 kWh import against a 2.76 kWh fuse: 1.24 kWh must come off."""
        slot = self._throttle(ev_w=11000.0, import_kwh=4.0)

        assert slot.grid_import_kwh == pytest.approx(FUSE_KWH, abs=0.01)

    def test_ev_energy_follows_the_command(self) -> None:
        slot = self._throttle(ev_w=11000.0, import_kwh=4.0)
        commanded_kwh = slot.ev_charger_calculated_power / 1000.0 * SLOT_H

        assert slot.ev_total_planned_load_kwh == pytest.approx(commanded_kwh, abs=0.01)

    def test_net_consumption_and_cost_are_recomputed(self) -> None:
        slot = self._throttle(ev_w=11000.0, import_kwh=4.0)
        expected_net = 0.5 + slot.ev_planned_load_kwh

        assert slot.estimated_net_consumption_kwh == pytest.approx(expected_net)
        assert slot.estimated_cost_currency == pytest.approx(
            round(expected_net * 1.0, 4)
        )

    def test_the_plan_and_the_command_agree(self) -> None:
        """The invariant the defect broke."""
        slot = self._throttle(ev_w=11000.0, import_kwh=4.0)
        commanded_kwh = slot.ev_charger_calculated_power / 1000.0 * SLOT_H

        assert slot.grid_import_kwh <= FUSE_KWH + 1e-9
        assert slot.ev_total_planned_load_kwh == pytest.approx(commanded_kwh, abs=0.01)
        assert slot.estimated_net_consumption_kwh == pytest.approx(
            0.5 + commanded_kwh, abs=0.01
        )

    def test_a_slot_inside_the_fuse_is_untouched(self) -> None:
        slot = self._throttle(ev_w=2000.0, import_kwh=1.0)

        assert slot.ev_charger_calculated_power == pytest.approx(2000.0)
        assert slot.grid_import_kwh == pytest.approx(1.0)
        assert slot.estimated_cost_currency == pytest.approx(
            round((0.5 + 2000.0 / 1000.0 * SLOT_H) * 1.0, 4)
        )
