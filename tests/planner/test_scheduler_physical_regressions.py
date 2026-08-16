"""Hand-calculated regressions for heuristic battery scheduling physics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from inspect import signature

import pytest

import custom_components.hsem.planner.candidate_generator as candidate_generator
from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.planner_input import PlannerInput
from custom_components.hsem.planner import run_planner
from custom_components.hsem.planner.discharge_scheduler import (
    apply_optimization_strategy,
    calculate_required_battery_until_solar,
)
from custom_components.hsem.planner.engine_scheduling import _schedule_slots
from custom_components.hsem.planner.soc_simulation import simulate_soc
from custom_components.hsem.utils.prices import SlotPrice
from custom_components.hsem.utils.recommendations import Recommendations
from tests.planner.test_arbitrage_grid_charge import _make_arbitrage_input

_NOW = datetime(2026, 8, 16, tzinfo=UTC)
_BCS = Recommendations.BatteriesChargeSolar.value
_BDM = Recommendations.BatteriesDischargeMode.value
_FBD = Recommendations.ForceBatteriesDischarge.value
_FORCE_EXPORT = Recommendations.ForceExport.value
_WAIT = Recommendations.BatteriesWaitMode.value


def _slot(
    hour: int,
    *,
    net: float,
    import_price: float = 0.5,
    export_price: float = 0.0,
    recommendation: str | None = None,
    charged: float = 0.0,
    discharged: float = 0.0,
    ev_load: float = 0.0,
    primary_hold: bool = False,
) -> PlannedSlot:
    """Build a one-hour slot whose house/PV fields reproduce ``net``."""
    start = _NOW + timedelta(hours=hour)
    return PlannedSlot(
        start=start,
        end=start + timedelta(hours=1),
        price=SlotPrice(import_price, export_price),
        solcast_pv_estimate_kwh=max(-net, 0.0),
        avg_house_consumption_kwh=max(net, 0.0),
        ev_planned_load_kwh=ev_load,
        ev_total_planned_load_kwh=ev_load,
        estimated_net_consumption_kwh=net,
        batteries_charged_kwh=charged,
        batteries_discharged_kwh=discharged,
        recommendation=recommendation,
        primary_battery_hold=primary_hold,
    )


def _optimize(
    slots: list[PlannedSlot],
    *,
    current: float,
    required: float,
    usable: float,
    max_charge: float,
    max_discharge: float,
    winter: bool = False,
) -> None:
    if "max_charge_per_slot" in signature(apply_optimization_strategy).parameters:
        apply_optimization_strategy(
            slots,
            _NOW,
            current_capacity=current,
            usable_capacity=usable,
            required_capacity=required,
            months_winter=[_NOW.month] if winter else [],
            max_charge_per_slot=max_charge,
            max_discharge_per_slot=max_discharge,
        )
    else:  # pragma: no cover - exercised only against the published .34 code
        apply_optimization_strategy(
            slots,
            _NOW,
            current_capacity=current,
            usable_capacity=usable,
            required_capacity=required,
            months_winter=[_NOW.month] if winter else [],
        )


def _simulate(slots: list[PlannedSlot], *, current: float, usable: float) -> None:
    simulate_soc(
        slots,
        _NOW,
        current_kwh=current,
        usable_kwh=usable,
        max_capacity_kwh=usable,
        max_charge_per_slot=usable,
        max_discharge_per_slot=usable,
    )


def _solver_failure_input(*, excess_export: bool) -> PlannerInput:
    """Return the deterministic high-export fallback scenario."""
    inp = _make_arbitrage_input()
    inp.excess_export_enabled = excess_export
    for solar in inp.solcast_slots:
        if solar.hour == 13:
            solar.pv_estimate = 2.0
    for price in inp.price_points:
        if price.hour >= 14:
            price.export_price = 10.0
    return inp


def _force_solver_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_solution(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(candidate_generator, "solve_milp", _no_solution)


def test_regression_force_export_pv_is_not_seasonal_refill() -> None:
    """PV explicitly sent to grid cannot justify earlier battery discharge."""
    slots = [
        _slot(0, net=1.0, import_price=1.0, export_price=0.2),
        _slot(1, net=-1.0, import_price=0.1, export_price=1.0),
    ]

    _optimize(
        slots,
        current=1.0,
        required=1.0,
        usable=2.0,
        max_charge=2.0,
        max_discharge=2.0,
        winter=True,
    )

    assert [slot.recommendation for slot in slots] == [_WAIT, _FORCE_EXPORT]


def test_regression_preassigned_discharge_projection_stops_at_empty() -> None:
    """An oversized BDM cannot create negative capacity before later PV."""
    slots = [
        _slot(0, net=5.0, recommendation=_BDM),
        _slot(1, net=-5.0),
        _slot(2, net=4.0),
        _slot(3, net=-1.0),
    ]

    _optimize(
        slots,
        current=1.0,
        required=0.0,
        usable=5.0,
        max_charge=5.0,
        max_discharge=5.0,
    )

    assert [slot.recommendation for slot in slots] == [_BDM, _BCS, _BDM, _BCS]


def test_regression_force_discharge_projects_exact_target() -> None:
    """A 1 kWh forced export leaves four, so the next 1 kWh load may discharge."""
    slots = [
        _slot(0, net=-0.1, recommendation=_FBD, discharged=1.0),
        _slot(1, net=1.0),
        _slot(2, net=-1.0),
    ]

    _optimize(
        slots,
        current=5.0,
        required=0.0,
        usable=5.0,
        max_charge=5.0,
        max_discharge=1.0,
    )

    assert [slot.recommendation for slot in slots] == [_FBD, _BDM, _BCS]

    _simulate(slots[:1], current=5.0, usable=5.0)
    assert slots[0].batteries_discharged_kwh == pytest.approx(1.0)
    assert slots[0].estimated_battery_capacity_kwh == pytest.approx(4.0)


@pytest.mark.parametrize("suppression", ["hold", "ev"])
def test_regression_force_discharge_respects_runtime_suppression(
    suppression: str,
) -> None:
    """A primary hold or active EV cancels the battery-export target."""
    slot = _slot(
        0,
        net=1.0,
        recommendation=_FBD,
        discharged=1.0,
        primary_hold=suppression == "hold",
        ev_load=1.0 if suppression == "ev" else 0.0,
    )

    _simulate([slot], current=5.0, usable=5.0)

    assert slot.recommendation == _WAIT
    assert slot.batteries_discharged_kwh == pytest.approx(0.0)
    assert slot.estimated_battery_capacity_kwh == pytest.approx(5.0)


def test_regression_preassigned_solar_charge_projects_all_absorbed_pv() -> None:
    """A 1 kWh BCS target still absorbs 4 kWh additional available PV."""
    slots = [
        _slot(0, net=-5.0, recommendation=_BCS, charged=1.0),
        _slot(1, net=4.0),
        _slot(2, net=-1.0),
    ]
    physical_slot = _slot(0, net=-5.0, recommendation=_BCS, charged=1.0)

    _simulate([physical_slot], current=0.0, usable=5.0)
    _optimize(
        slots,
        current=0.0,
        required=0.0,
        usable=5.0,
        max_charge=5.0,
        max_discharge=5.0,
    )

    assert physical_slot.estimated_battery_capacity_kwh == pytest.approx(5.0)
    assert [slot.recommendation for slot in slots] == [_BCS, _BDM, _BCS]


def test_regression_small_future_export_does_not_hold_all_prior_load() -> None:
    """Five stored kWh cover load, reserve, and a 1 kWh export target."""
    slots = [
        _slot(0, net=1.0),
        _slot(1, net=0.0, recommendation=_FBD, discharged=1.0),
        _slot(2, net=-1.0),
    ]

    _optimize(
        slots,
        current=5.0,
        required=1.0,
        usable=5.0,
        max_charge=5.0,
        max_discharge=5.0,
    )

    assert slots[0].recommendation == _BDM


def test_regression_reserve_ignores_ev_and_obeys_discharge_power() -> None:
    """Reserve is 1 kWh with EV hold, and 2 kWh when both loads cap at 1."""
    ev_slots = [
        _slot(0, net=1.0),
        _slot(1, net=4.0, ev_load=4.0),
        _slot(2, net=-1.0),
    ]
    capped_slots = [
        _slot(0, net=1.0),
        _slot(1, net=4.0),
        _slot(2, net=-1.0),
    ]

    if (
        "max_discharge_per_slot"
        in signature(calculate_required_battery_until_solar).parameters
    ):
        ev_required = calculate_required_battery_until_solar(
            ev_slots,
            _NOW,
            usable_capacity=5.0,
            discharge_buffer_pct=0.0,
            max_discharge_per_slot=5.0,
        )
        capped_required = calculate_required_battery_until_solar(
            capped_slots,
            _NOW,
            usable_capacity=5.0,
            discharge_buffer_pct=0.0,
            max_discharge_per_slot=1.0,
        )
    else:  # pragma: no cover - exercised only against the published .34 code
        ev_required = calculate_required_battery_until_solar(
            ev_slots,
            _NOW,
            usable_capacity=5.0,
            discharge_buffer_pct=0.0,
        )
        capped_required = calculate_required_battery_until_solar(
            capped_slots,
            _NOW,
            usable_capacity=5.0,
            discharge_buffer_pct=0.0,
        )

    assert ev_required == pytest.approx(1.0)
    assert capped_required == pytest.approx(2.0)


def test_regression_ordinary_wait_does_not_hide_later_refill() -> None:
    """An ordinary priced wait does not forbid later PV absorption."""
    slots = [
        _slot(0, net=1.0),
        _slot(1, net=0.0, recommendation=_WAIT),
        _slot(2, net=-1.0),
    ]

    _optimize(
        slots,
        current=1.0,
        required=0.0,
        usable=1.0,
        max_charge=1.0,
        max_discharge=1.0,
    )

    assert [slot.recommendation for slot in slots] == [_BDM, _WAIT, _BCS]


def test_regression_refill_nets_future_forced_discharge_commitment() -> None:
    """Intervening PV can refill energy reserved for a later exact export."""
    slots = [
        _slot(0, net=1.0),
        _slot(1, net=-5.0),
        _slot(2, net=0.0, recommendation=_FBD, discharged=5.0),
    ]

    _optimize(
        slots,
        current=1.0,
        required=0.0,
        usable=5.0,
        max_charge=5.0,
        max_discharge=5.0,
    )

    assert [slot.recommendation for slot in slots] == [_BDM, _BCS, _FBD]


def test_regression_force_export_is_classified_before_reserve() -> None:
    """PV routed to grid is skipped, so reserve includes the following load."""
    slots = [
        _slot(0, net=1.0),
        _slot(1, net=-1.0, import_price=0.1, export_price=1.0),
        _slot(2, net=1.0),
        _slot(3, net=-1.0, import_price=1.0, export_price=0.1),
    ]
    inp = _make_arbitrage_input(schedules=[])
    inp.excess_export_discharge_buffer_pct = 0.0
    inp.export_min_price = 0.0
    inp.battery_max_discharge_power_w = 5000.0

    *_, required_capacity, _warnings = _schedule_slots(
        slots,
        inp,
        _NOW,
        current_kwh=5.0,
        usable_kwh=5.0,
        rt=0.0,
        effective_cycle_cost=0.0,
        warnings=[],
    )

    assert slots[1].recommendation == _FORCE_EXPORT
    assert required_capacity == pytest.approx(2.0)


@pytest.mark.parametrize("excess_export", [False, True])
def test_solver_failure_cannot_export_battery(
    monkeypatch: pytest.MonkeyPatch,
    excess_export: bool,
) -> None:
    """The passive fallback never introduces intentional battery export."""
    _force_solver_failure(monkeypatch)
    result = run_planner(_solver_failure_input(excess_export=excess_export))

    battery_exports = [
        slot
        for slot in result.slots
        if slot.estimated_net_consumption_kwh >= 0.0 and slot.grid_export_kwh > 1e-9
    ]
    assert battery_exports == []
