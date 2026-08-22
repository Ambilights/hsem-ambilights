"""Regression tests for immutable MILP post-processing.

The MILP writer owns battery and grid energy fields as one solved balance.
Heuristic seasonal filling must not add charge to idle solved slots, and the
SoC projection must not clamp one field without rebuilding the others.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.planner.candidate_generator import (
    CANDIDATE_MILP,
    CANDIDATE_PASSIVE,
    CandidatePlan,
)
from custom_components.hsem.planner.candidate_selector import select_best_candidate
from custom_components.hsem.planner.cost_function import CostWeights
from custom_components.hsem.planner.milp._solver_execution import (
    _validate_primary_postwrite_inventory,
)
from custom_components.hsem.planner.milp._write_results import (
    _write_milp_results_to_slots,
)
from custom_components.hsem.utils.prices import SlotPrice
from custom_components.hsem.utils.recommendations import Recommendations

_TZ = ZoneInfo("Europe/Stockholm")
_NOW = datetime(2026, 8, 14, 8, 0, tzinfo=_TZ)


def _slot(
    hour: int,
    *,
    house: float,
    pv: float,
    recommendation: str | None = None,
    charge: float = 0.0,
    discharge: float = 0.0,
    grid_import: float = 0.0,
    grid_export: float = 0.0,
    import_price: float = 2.0,
    export_price: float = 1.0,
) -> PlannedSlot:
    start = datetime(2026, 8, 14, hour, 0, tzinfo=_TZ)
    return PlannedSlot(
        start=start,
        end=start + timedelta(hours=1),
        price=SlotPrice(import_price=import_price, export_price=export_price),
        avg_house_consumption_kwh=house,
        solcast_pv_estimate_kwh=pv,
        estimated_net_consumption_kwh=house - pv,
        recommendation=recommendation,
        batteries_charged_kwh=charge,
        batteries_discharged_kwh=discharge,
        grid_import_kwh=grid_import,
        grid_export_kwh=grid_export,
    )


def _select(
    candidate: CandidatePlan,
    *,
    current_kwh: float,
    usable_kwh: float,
    charge_efficiency_pct: float = 100.0,
    discharge_efficiency_pct: float = 100.0,
    max_discharge_per_slot: float | None = 2.0,
) -> CandidatePlan:
    winner, _, _ = select_best_candidate(
        [candidate],
        now=_NOW,
        current_kwh=current_kwh,
        usable_kwh=usable_kwh,
        max_soc_capacity_kwh=usable_kwh,
        max_charge_per_slot=2.0,
        max_discharge_per_slot=max_discharge_per_slot,
        rated_kwh=usable_kwh,
        end_of_discharge_soc_pct=0.0,
        cost_weights=CostWeights(
            min_soc_pct=0.0,
            max_soc_pct=100.0,
            charge_efficiency_pct=charge_efficiency_pct,
            discharge_efficiency_pct=discharge_efficiency_pct,
        ),
        slot_duration_hours=1.0,
        charge_efficiency_pct=charge_efficiency_pct,
        discharge_efficiency_pct=discharge_efficiency_pct,
        months_winter=[],
    )
    return winner


def _primary_flows(slot: PlannedSlot) -> tuple[float, float, float, float]:
    return (
        slot.batteries_charged_kwh,
        slot.batteries_discharged_kwh,
        slot.grid_import_kwh,
        slot.grid_export_kwh,
    )


def test_postwrite_inventory_validation_is_cumulative() -> None:
    """Individually plausible slot discharges cannot overdraw the horizon."""
    slots = [
        _slot(8, house=0.0, pv=0.0, discharge=0.6),
        _slot(9, house=0.0, pv=0.0, discharge=0.6),
    ]

    validation = _validate_primary_postwrite_inventory(
        slots,
        [0, 1],
        current_kwh=1.0,
        usable_kwh=1.0,
    )

    assert validation["valid"] is False
    assert validation["reason"] == "primary_inventory_below_floor"
    assert validation["slot"] == 1


def _write_milp(
    slots: list[PlannedSlot],
    *,
    charges: list[float],
    discharges: list[float],
    current_kwh: float,
    usable_kwh: float,
    charge_efficiency: float = 1.0,
) -> list[PlannedSlot]:
    """Write a deterministic primary-only MILP allocation into slots."""
    count = len(slots)
    return _write_milp_results_to_slots(
        slots=slots,
        future_idx=list(range(count)),
        now=_NOW,
        ec_sol=np.array(charges),
        ed_sol=np.array(discharges),
        result_x=np.array([slot.grid_export_kwh for slot in slots]),
        m=count,
        ge_off=0,
        active_evs=[],
        ev_var_offsets=[],
        pv_avail=np.array(
            [
                max(slot.solcast_pv_estimate_kwh - slot.avg_house_consumption_kwh, 0.0)
                for slot in slots
            ]
        ),
        base_load=np.array(
            [
                max(slot.avg_house_consumption_kwh - slot.solcast_pv_estimate_kwh, 0.0)
                for slot in slots
            ]
        ),
        charge_eff=charge_efficiency,
        discharge_eff=1.0,
        p_exp=np.array([slot.price.export_price for slot in slots]),
        min_export_price=0.0,
        _has_session_demand=False,
        session_slots_set=set(),
        current_kwh=current_kwh,
        usable_kwh=usable_kwh,
        curt_sol_full=np.zeros(count),
    )


def _assert_primary_balance(
    slot: PlannedSlot,
    *,
    charge_efficiency: float = 1.0,
    discharge_efficiency: float = 1.0,
) -> None:
    supply = (
        slot.grid_import_kwh
        + slot.solcast_pv_estimate_kwh
        + slot.batteries_discharged_kwh * discharge_efficiency
    )
    demand = (
        slot.avg_house_consumption_kwh
        + slot.batteries_charged_kwh / charge_efficiency
        + slot.grid_export_kwh
    )
    assert supply == pytest.approx(demand, abs=1e-6)


def test_idle_milp_pv_export_gets_label_without_phantom_charge() -> None:
    """An idle solved export must not also become a heuristic solar charge."""
    slot = _slot(9, house=1.0, pv=2.0, grid_export=1.0)
    slot.secondary_storage_mode = "sbu"
    slot.secondary_storage_discharged_kwh = 0.2
    slot.secondary_storage_estimated_soc_pct = 75.0
    solved = _write_milp(
        [slot],
        charges=[0.0],
        discharges=[0.0],
        current_kwh=0.0,
        usable_kwh=2.0,
    )[0]
    before = _primary_flows(solved)
    secondary_before = (
        solved.secondary_storage_mode,
        solved.secondary_storage_discharged_kwh,
        solved.secondary_storage_estimated_soc_pct,
    )

    winner = _select(
        CandidatePlan(name=CANDIDATE_MILP, slots=[solved]),
        current_kwh=0.0,
        usable_kwh=2.0,
    )
    result = winner.slots[0]

    assert result.recommendation == Recommendations.BatteriesWaitMode.value
    assert result.primary_battery_hold is True
    assert _primary_flows(result) == before
    assert (
        result.secondary_storage_mode,
        result.secondary_storage_discharged_kwh,
        result.secondary_storage_estimated_soc_pct,
    ) == secondary_before
    assert result.estimated_battery_capacity_kwh == 0.0
    _assert_primary_balance(result)


def test_milp_solar_charge_and_later_flow_survive_selector_unchanged() -> None:
    """Idle PV must not fill the battery and clamp a later solved charge."""
    idle_export = _slot(9, house=1.0, pv=2.0, grid_export=1.0)
    # At 98% efficiency, 1.020408163 kWh AC surplus stores exactly 1 kWh.
    solved_charge = _slot(
        10,
        house=1.0,
        pv=2.020408163265306,
        recommendation=Recommendations.BatteriesChargeSolar.value,
        charge=1.0,
    )
    solved_slots = _write_milp(
        [idle_export, solved_charge],
        charges=[0.0, 1.0],
        discharges=[0.0, 0.0],
        current_kwh=1.0,
        usable_kwh=2.0,
        charge_efficiency=0.98,
    )
    before = [_primary_flows(slot) for slot in solved_slots]

    winner = _select(
        CandidatePlan(name=CANDIDATE_MILP, slots=solved_slots),
        current_kwh=1.0,
        usable_kwh=2.0,
        charge_efficiency_pct=98.0,
    )

    assert [_primary_flows(s) for s in winner.slots] == before
    assert winner.slots[0].recommendation == Recommendations.BatteriesWaitMode.value
    assert winner.slots[0].primary_battery_hold is True
    assert winner.slots[1].primary_battery_hold is False
    assert winner.slots[1].recommendation == Recommendations.BatteriesChargeSolar.value
    assert winner.slots[0].estimated_battery_capacity_kwh == 1.0
    assert winner.slots[1].estimated_battery_capacity_kwh == 2.0
    for result in winner.slots:
        _assert_primary_balance(result, charge_efficiency=0.98)


def test_milp_discharge_is_not_relabelled_by_heuristic_concentration() -> None:
    """Solved discharge labels and fields remain owned by the MILP."""
    slots = [
        _slot(
            9,
            house=1.0,
            pv=0.0,
            recommendation=Recommendations.BatteriesDischargeMode.value,
            discharge=0.5,
            grid_import=0.5,
            import_price=3.0,
        ),
        _slot(
            10,
            house=1.0,
            pv=0.0,
            recommendation=Recommendations.BatteriesDischargeMode.value,
            discharge=0.5,
            grid_import=0.5,
            import_price=2.0,
        ),
    ]
    before = [_primary_flows(slot) for slot in slots]

    winner = _select(
        CandidatePlan(name=CANDIDATE_MILP, slots=slots),
        current_kwh=1.0,
        usable_kwh=1.0,
        max_discharge_per_slot=1.0,
    )

    assert [_primary_flows(s) for s in winner.slots] == before
    assert all(
        s.recommendation == Recommendations.BatteriesDischargeMode.value
        for s in winner.slots
    )


def test_non_milp_candidate_keeps_heuristic_solar_fill() -> None:
    """The ownership gate must not change passive/fallback candidates."""
    slot = _slot(9, house=1.0, pv=2.0)

    winner = _select(
        CandidatePlan(name=CANDIDATE_PASSIVE, slots=[slot]),
        current_kwh=0.0,
        usable_kwh=2.0,
    )
    result = winner.slots[0]

    assert result.recommendation == Recommendations.BatteriesChargeSolar.value
    assert result.primary_battery_hold is False
    assert result.batteries_charged_kwh == 1.0
    assert result.grid_export_kwh == 0.0
    _assert_primary_balance(result)
