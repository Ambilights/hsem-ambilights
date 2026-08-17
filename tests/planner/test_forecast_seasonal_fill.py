"""Regression tests for forecast-driven idle-slot fill.

All fixtures use 15-minute slots and energy values in kWh.  These tests stop
at the recommendation pass, before SoC simulation, so battery trajectory and
grid import/export remain intentionally unset; the hand-calculated assertion
is the refill-headroom decision itself.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.planner.discharge_scheduler import (
    apply_optimization_strategy,
)
from custom_components.hsem.utils.prices import SlotPrice
from custom_components.hsem.utils.recommendations import Recommendations

_WINTER_MONTHS = [1, 2, 3, 4, 10, 11, 12]
_WAIT = Recommendations.BatteriesWaitMode.value
_DISCHARGE = Recommendations.BatteriesDischargeMode.value
_CHARGE_SOLAR = Recommendations.BatteriesChargeSolar.value
_CHARGE_GRID = Recommendations.BatteriesChargeGrid.value
_FORCED_DISCHARGE = Recommendations.ForceBatteriesDischarge.value


def _slot(
    start: datetime,
    *,
    net_kwh: float,
    pv_kwh: float,
    recommendation: str | None = None,
) -> PlannedSlot:
    """Build one 15-minute slot with non-arbitrage prices."""
    return PlannedSlot(
        start=start,
        end=start + timedelta(minutes=15),
        price=SlotPrice(import_price=0.20, export_price=0.05),
        solcast_pv_estimate_kwh=pv_kwh,
        estimated_net_consumption_kwh=net_kwh,
        recommendation=recommendation,
    )


def _apply(
    slots: list[PlannedSlot],
    now: datetime,
    *,
    current_kwh: float,
    required_kwh: float,
    mode: str = "forecast",
) -> None:
    apply_optimization_strategy(
        slots,
        now,
        current_capacity=current_kwh,
        usable_capacity=10.0,
        required_capacity=required_kwh,
        months_winter=_WINTER_MONTHS,
        seasonal_fill_mode=mode,
    )


def test_sunny_winter_uses_positive_refill_headroom_and_logs_inputs() -> None:
    """1.2 kWh refill - (2.5 - 2.0) kWh reserve = +0.7 kWh: discharge."""
    now = datetime(2026, 1, 15, 8, 0, tzinfo=UTC)
    slots = [
        _slot(now, net_kwh=0.4, pv_kwh=0.0),
        _slot(now + timedelta(hours=2), net_kwh=-1.2, pv_kwh=1.6),
    ]

    with patch(
        "custom_components.hsem.planner.discharge_scheduler.log_planner"
    ) as planner_log:
        _apply(slots, now, current_kwh=2.0, required_kwh=2.5)

    assert slots[0].recommendation == _DISCHARGE
    assert slots[1].recommendation == _CHARGE_SOLAR
    seasonal_calls = [
        call
        for call in planner_log.call_args_list
        if len(call.args) > 1 and "[disch] seasonal_fill slot=%s" in call.args[1]
    ]
    assert any(
        call.args[4] == pytest.approx(1.2)
        and call.args[7] == pytest.approx(0.7)
        and call.args[8] == _DISCHARGE
        for call in seasonal_calls
    )


def test_overcast_summer_with_usable_forecast_waits() -> None:
    """Positive Solcast proves availability, but no net surplus means -1 kWh headroom."""
    now = datetime(2026, 6, 15, 8, 0, tzinfo=UTC)
    slots = [
        _slot(now, net_kwh=0.4, pv_kwh=0.0),
        _slot(now + timedelta(minutes=15), net_kwh=0.3, pv_kwh=0.05),
    ]

    _apply(slots, now, current_kwh=2.0, required_kwh=3.0)

    assert slots[0].recommendation == _WAIT


@pytest.mark.parametrize(
    ("month", "expected"),
    [(1, _WAIT), (6, _DISCHARGE)],
)
def test_all_zero_solcast_preserves_legacy_month_rule(
    month: int,
    expected: str,
) -> None:
    """An unavailable/all-zero forecast must not be treated as real zero PV."""
    now = datetime(2026, month, 15, 8, 0, tzinfo=UTC)
    slots = [
        _slot(now, net_kwh=0.4, pv_kwh=0.0),
        _slot(now + timedelta(minutes=15), net_kwh=0.3, pv_kwh=0.0),
    ]

    _apply(slots, now, current_kwh=2.0, required_kwh=3.0)

    assert slots[0].recommendation == expected


def test_months_mode_preserves_calendar_rule_with_sunny_forecast() -> None:
    """The opt-out keeps winter Wait even when future surplus is ample."""
    now = datetime(2026, 1, 15, 8, 0, tzinfo=UTC)
    slots = [
        _slot(now, net_kwh=0.4, pv_kwh=0.0),
        _slot(now + timedelta(hours=2), net_kwh=-2.0, pv_kwh=2.4),
    ]

    _apply(slots, now, current_kwh=2.0, required_kwh=2.5, mode="months")

    assert slots[0].recommendation == _WAIT


def test_invalid_mode_warns_and_falls_back_to_months() -> None:
    """A manually corrupted enum must fail safe without a silent default."""
    now = datetime(2026, 1, 15, 8, 0, tzinfo=UTC)
    slots = [
        _slot(now, net_kwh=0.4, pv_kwh=0.0),
        _slot(now + timedelta(hours=2), net_kwh=-2.0, pv_kwh=2.4),
    ]

    with patch(
        "custom_components.hsem.planner.discharge_scheduler.log_planner"
    ) as planner_log:
        _apply(
            slots,
            now,
            current_kwh=2.0,
            required_kwh=2.5,
            mode="automatic",
        )

    assert slots[0].recommendation == _WAIT
    planner_log.assert_any_call(
        "warning",
        "[disch] invalid seasonal_fill_mode=%s; falling back to %s",
        "automatic",
        "months",
    )


def test_preassigned_milp_slot_is_untouched() -> None:
    """A non-None recommendation and its allocated energy remain authoritative."""
    now = datetime(2026, 1, 15, 8, 0, tzinfo=UTC)
    assigned = _slot(
        now,
        net_kwh=0.4,
        pv_kwh=0.0,
        recommendation=_CHARGE_GRID,
    )
    assigned.batteries_charged_kwh = 0.75
    slots = [
        assigned,
        _slot(now + timedelta(hours=2), net_kwh=-1.2, pv_kwh=1.6),
    ]

    _apply(slots, now, current_kwh=2.0, required_kwh=2.5)

    assert assigned.recommendation == _CHARGE_GRID
    assert assigned.batteries_charged_kwh == pytest.approx(0.75)


def test_future_forced_discharge_reserve_branch_keeps_priority() -> None:
    """The existing reserve branch still waits when capacity exceeds the reserve."""
    now = datetime(2026, 1, 15, 8, 0, tzinfo=UTC)
    forced = _slot(
        now + timedelta(minutes=15),
        net_kwh=0.4,
        pv_kwh=0.0,
        recommendation=_FORCED_DISCHARGE,
    )
    slots = [
        _slot(now, net_kwh=0.4, pv_kwh=0.0),
        forced,
        _slot(now + timedelta(minutes=30), net_kwh=-3.0, pv_kwh=3.4),
    ]

    _apply(slots, now, current_kwh=4.0, required_kwh=2.0)

    assert slots[0].recommendation == _WAIT
    assert forced.recommendation == _FORCED_DISCHARGE


def test_refill_suffix_stops_before_next_forced_discharge() -> None:
    """PV after a forced-discharge boundary cannot justify earlier discharge."""
    now = datetime(2026, 1, 15, 8, 0, tzinfo=UTC)
    slots = [
        _slot(now, net_kwh=0.4, pv_kwh=0.0),
        _slot(
            now + timedelta(minutes=15),
            net_kwh=0.4,
            pv_kwh=0.0,
            recommendation=_FORCED_DISCHARGE,
        ),
        _slot(now + timedelta(minutes=30), net_kwh=-3.0, pv_kwh=3.4),
    ]

    _apply(slots, now, current_kwh=1.0, required_kwh=2.0)

    assert slots[0].recommendation == _WAIT


def test_unknown_tail_pv_cannot_justify_actionable_prefix_discharge() -> None:
    """A Hold-only tail cannot promise refill energy to an earlier slot."""
    now = datetime(2026, 1, 15, 8, 0, tzinfo=UTC)

    def decisions(tail_pv_kwh: float) -> list[str | None]:
        slots = [
            _slot(now, net_kwh=0.4, pv_kwh=0.0),
            _slot(
                now + timedelta(minutes=15),
                net_kwh=0.2 - tail_pv_kwh,
                pv_kwh=tail_pv_kwh,
            ),
        ]
        slots[1].price_actionable = False
        _apply(slots, now, current_kwh=1.0, required_kwh=2.0)
        return [slot.recommendation for slot in slots]

    assert decisions(0.0)[0] == _WAIT
    assert decisions(100.0)[0] == _WAIT
