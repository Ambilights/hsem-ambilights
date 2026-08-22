"""Regressions for contiguous price authority in optimisation and scoring."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.price_point import PricePoint
from custom_components.hsem.models.solcast_slot import SolcastSlot
from custom_components.hsem.models.time_series import TimeSeriesIndex
from custom_components.hsem.planner.candidates._soc_plan import _apply_soc_plan
from custom_components.hsem.planner.cost_function import (
    CostWeights,
    PlanCostBreakdown,
    score_plan,
)
from custom_components.hsem.planner.discharge_scheduler import (
    apply_excess_export,
    calculate_required_battery_until_solar,
    concentrate_discharge_on_expensive_slots,
)
from custom_components.hsem.planner.milp._constraints import _build_constraints
from custom_components.hsem.planner.milp._layout import (
    MilpBoundsBuilder,
    MilpColumnLayout,
)
from custom_components.hsem.planner.milp_optimizer import (
    is_scipy_available,
    solve_milp,
)
from custom_components.hsem.planner.slot_population import (
    populate_prices,
    populate_solcast,
)
from custom_components.hsem.utils.prices import SlotPrice
from custom_components.hsem.utils.recommendations import Recommendations

_NOW = datetime(2026, 8, 14, 0, 0, tzinfo=UTC)


def _slot(
    index: int,
    *,
    import_price: float,
    export_price: float,
    actionable: bool,
    consumption_kwh: float = 0.5,
    pv_kwh: float = 0.0,
) -> PlannedSlot:
    start = _NOW + timedelta(hours=index)
    return PlannedSlot(
        start=start,
        end=start + timedelta(hours=1),
        price=SlotPrice(import_price, export_price),
        import_price_available=actionable,
        export_price_available=actionable,
        price_actionable=actionable,
        avg_house_consumption_kwh=consumption_kwh,
        solcast_pv_estimate_kwh=pv_kwh,
        estimated_net_consumption_kwh=consumption_kwh - pv_kwh,
        estimated_battery_soc_pct=50.0,
    )


def _solve_tail(import_price: float, export_price: float) -> list[PlannedSlot]:
    slots = [
        _slot(0, import_price=0.10, export_price=0.05, actionable=True),
        _slot(1, import_price=0.0, export_price=0.0, actionable=False),
        _slot(
            2,
            import_price=import_price,
            export_price=export_price,
            actionable=False,
            consumption_kwh=2.0,
        ),
    ]
    solved = solve_milp(
        slots,
        _NOW,
        current_kwh=0.0,
        usable_kwh=4.0,
        max_charge_per_slot=2.0,
        max_discharge_per_slot=2.0,
        cycle_cost_per_kwh=0.02,
        charge_efficiency_pct=100.0,
        discharge_efficiency_pct=100.0,
        replacement_price_per_kwh=1.0,
        main_fuse_amps=16.0,
        main_fuse_phases=1,
    )
    assert solved is not None
    return solved[0]


@pytest.mark.skipif(not is_scipy_available(), reason="scipy not available")
def test_isolated_tail_prices_cannot_change_milp_actions_or_penalty_scale() -> None:
    """A raw price after the first missing slot is economically neutral."""
    ordinary = _solve_tail(0.20, 0.10)
    extreme = _solve_tail(10_000.0, 9_000.0)
    nonfinite = _solve_tail(float("inf"), float("-inf"))

    def decisions(slots: list[PlannedSlot]) -> list[tuple[object, ...]]:
        return [
            (
                slot.recommendation,
                slot.batteries_charged_kwh,
                slot.batteries_discharged_kwh,
                slot.grid_import_kwh,
                slot.grid_export_kwh,
                slot.estimated_cost_currency,
            )
            for slot in slots
        ]

    assert decisions(ordinary) == decisions(extreme) == decisions(nonfinite)
    assert ordinary[0].batteries_charged_kwh > 0.0
    for slot in ordinary[1:]:
        assert slot.batteries_charged_kwh == 0.0
        assert slot.batteries_discharged_kwh == 0.0
        assert slot.estimated_cost_currency == 0.0
        assert slot.recommendation not in {
            Recommendations.BatteriesChargeGrid.value,
            Recommendations.ForceBatteriesDischarge.value,
            Recommendations.ForceExport.value,
        }


def test_nonactionable_prices_are_neutral_in_primary_and_secondary_scorer() -> None:
    """Raw tail prices cannot alter any monetary scorer component."""
    weights = CostWeights(
        cycle_cost_per_kwh=0.03,
        charge_efficiency_pct=90.0,
        discharge_efficiency_pct=90.0,
        secondary_storage_enabled=True,
        secondary_storage_charge_efficiency_pct=90.0,
        secondary_storage_discharge_efficiency_pct=90.0,
        secondary_storage_cycle_cost_per_kwh=0.02,
        secondary_storage_replacement_price_per_kwh=0.50,
    )

    def scored(
        import_price: float, export_price: float, actionable: bool
    ) -> PlanCostBreakdown:
        slot = _slot(
            0,
            import_price=import_price,
            export_price=export_price,
            actionable=actionable,
        )
        slot.grid_import_kwh = 1.0
        slot.grid_export_kwh = 0.25
        slot.batteries_charged_kwh = 0.5
        slot.batteries_discharged_kwh = 0.1
        slot.secondary_storage_charged_kwh = 0.4
        slot.secondary_storage_discharged_kwh = 0.2
        return score_plan([slot], weights, now=_NOW)

    low = scored(0.2, 0.1, False)
    extreme = scored(2_000.0, -1_000.0, False)
    nonfinite = scored(float("inf"), float("-inf"), False)
    assert low == extreme == nonfinite
    assert low.secondary_terminal_soc_value == 0.0
    assert low.secondary_cycle_cost > 0.0
    assert scored(2_000.0, -1_000.0, True) != low


def test_genuine_actionable_zero_and_negative_prices_remain_economic() -> None:
    """Neutral-tail handling does not erase published zero/negative values."""
    zero = _slot(0, import_price=0.0, export_price=0.0, actionable=True)
    zero.grid_import_kwh = 1.0
    assert score_plan([zero], now=_NOW).import_cost == 0.0

    negative = _slot(0, import_price=0.0, export_price=-0.25, actionable=True)
    negative.grid_export_kwh = 1.0
    assert score_plan([negative], now=_NOW).total_cost == pytest.approx(0.25)


def test_curtailment_variable_is_bounded_by_available_pv() -> None:
    """The LP cannot fabricate grid import and send it to curtailment."""
    layout = MilpColumnLayout(slot_count=2)
    for name in (
        "primary_charge",
        "primary_discharge",
        "grid_import",
        "grid_export",
        "pv",
        "primary_throughput",
        "soc_max_penalty",
        "soc_min_penalty",
        "curtailment",
    ):
        layout.add(name, 2)
    bounds_builder = MilpBoundsBuilder(layout)

    _build_constraints(
        2,
        18,
        0,
        2,
        4,
        6,
        8,
        10,
        16,
        18,
        12,
        14,
        [],
        [],
        [],
        np.array([False, False]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.zeros(2),
        1.0,
        1.0,
        0.0,
        4.0,
        2.0,
        2.0,
        0.0,
        False,
        False,
        set(),
        [],
        0,
        1.0,
        np.ones(2),
        np.zeros(2),
        False,
        bounds_builder,
    )
    bounds = bounds_builder.finalize()

    for lower, upper in bounds[0:4]:
        assert lower == pytest.approx(0.0)
        assert upper == pytest.approx(0.0)
    expected_curtailment_upper = (1.0, 0.0)
    for bound, expected_upper in zip(
        bounds[16:18],
        expected_curtailment_upper,
        strict=True,
    ):
        lower, upper = bound
        assert lower == pytest.approx(0.0)
        assert upper == pytest.approx(expected_upper)


@pytest.mark.skipif(not is_scipy_available(), reason="scipy not available")
@pytest.mark.parametrize(
    ("consumption_kwh", "pv_kwh", "current_kwh", "expected_import"),
    [(1.0, 2.0, 4.0, 0.0), (2.0, 1.0, 0.0, 1.0)],
)
def test_unknown_price_physical_balance_never_imports_to_curtail(
    consumption_kwh: float,
    pv_kwh: float,
    current_kwh: float,
    expected_import: float,
) -> None:
    slot = _slot(
        0,
        import_price=999.0,
        export_price=999.0,
        actionable=False,
        consumption_kwh=consumption_kwh,
        pv_kwh=pv_kwh,
    )
    solved = solve_milp(
        [slot],
        _NOW,
        current_kwh=current_kwh,
        usable_kwh=4.0,
        max_charge_per_slot=2.0,
        max_discharge_per_slot=2.0,
        cycle_cost_per_kwh=0.02,
        charge_efficiency_pct=100.0,
        discharge_efficiency_pct=100.0,
    )
    assert solved is not None
    slots, diagnostics = solved
    result = slots[0]
    assert result.recommendation == Recommendations.BatteriesWaitMode.value
    assert result.primary_battery_hold is True
    assert result.batteries_charged_kwh == 0.0
    assert result.batteries_discharged_kwh == 0.0
    assert result.grid_import_kwh == pytest.approx(expected_import, abs=1e-3)
    assert diagnostics["total_curtailment_kwh"] <= max(pv_kwh - consumption_kwh, 0.0)
    if pv_kwh > consumption_kwh:
        assert result.grid_export_kwh + diagnostics["total_curtailment_kwh"] == (
            pytest.approx(pv_kwh - consumption_kwh, abs=1e-3)
        )


def test_nonactionable_extreme_cannot_evict_actionable_discharge() -> None:
    """Discharge concentration ranks only slots with published authority."""
    actionable = _slot(0, import_price=1.0, export_price=0.0, actionable=True)
    island = _slot(1, import_price=10_000.0, export_price=0.0, actionable=False)
    for slot in (actionable, island):
        slot.recommendation = Recommendations.BatteriesDischargeMode.value
        slot.estimated_net_consumption_kwh = 1.0

    concentrate_discharge_on_expensive_slots(
        [actionable, island],
        _NOW,
        current_kwh=1.0,
        usable_kwh=1.0,
        max_discharge_per_slot=1.0,
    )

    assert actionable.recommendation == Recommendations.BatteriesDischargeMode.value
    assert island.recommendation == Recommendations.BatteriesDischargeMode.value


def test_nonactionable_extreme_cannot_trigger_force_export() -> None:
    """An isolated raw export value cannot create an export recommendation."""
    island = _slot(
        0,
        import_price=0.0,
        export_price=10_000.0,
        actionable=False,
        consumption_kwh=0.0,
        pv_kwh=1.0,
    )
    apply_excess_export(
        [island],
        _NOW,
        current_capacity=4.0,
        required_capacity=0.0,
        export_price_threshold=0.0,
        warnings=[],
    )
    assert island.recommendation is None


def test_unknown_tail_pv_cannot_replace_earlier_soc_plan_grid_charge() -> None:
    """Forbidden tail solar cannot satisfy an actionable discharge target."""

    def planned(tail_pv_kwh: float) -> list[tuple[str | None, float]]:
        cheap = _slot(0, import_price=0.05, export_price=0.0, actionable=True)
        discharge = _slot(
            1,
            import_price=1.0,
            export_price=0.0,
            actionable=True,
            consumption_kwh=1.0,
        )
        discharge.recommendation = Recommendations.BatteriesDischargeMode.value
        tail = _slot(
            2,
            import_price=10_000.0,
            export_price=10_000.0,
            actionable=False,
            consumption_kwh=0.0,
            pv_kwh=tail_pv_kwh,
        )
        slots = [cheap, discharge, tail]
        target = _apply_soc_plan(
            slots,
            _NOW,
            2.0,
            current_kwh=0.0,
            usable_kwh=4.0,
            cycle_cost_per_kwh=0.0,
            charge_fraction=1.0,
            charge_efficiency_pct=100.0,
            discharge_efficiency_pct=100.0,
        )
        assert target == pytest.approx(1.0)
        return [(slot.recommendation, slot.batteries_charged_kwh) for slot in slots]

    no_tail_solar = planned(0.0)
    extreme_tail_solar = planned(100.0)
    assert no_tail_solar[:2] == extreme_tail_solar[:2]
    assert no_tail_solar[0] == (Recommendations.BatteriesChargeGrid.value, 1.0)
    assert extreme_tail_solar[2] == (None, 0.0)


def test_unknown_tail_pv_cannot_reduce_required_capacity_before_solar() -> None:
    """Reserve forecasting ignores solar that strict Hold cannot absorb."""
    prefix = _slot(
        0,
        import_price=0.2,
        export_price=0.1,
        actionable=True,
        consumption_kwh=1.0,
    )
    tail = _slot(
        1,
        import_price=10_000.0,
        export_price=10_000.0,
        actionable=False,
        consumption_kwh=0.0,
        pv_kwh=100.0,
    )

    required = calculate_required_battery_until_solar(
        [prefix, tail],
        _NOW,
        usable_capacity=4.0,
        discharge_buffer_pct=0.0,
    )

    assert required == pytest.approx(1.0)


def test_price_population_rejects_all_nonfinite_but_preserves_zero_negative() -> None:
    """Advertised NaN/infinities are missing, not actionable price signals."""
    tsi = TimeSeriesIndex.from_now(_NOW, interval_minutes=60, horizon_hours=5)
    slots = [PlannedSlot(meta.start, meta.end) for meta in tsi.slots]
    values = [float("nan"), float("inf"), float("-inf"), 0.0, -0.25]
    points = [
        PricePoint(
            hour=meta.hour,
            day_offset=meta.key.day_offset,
            import_price=value,
            export_price=value,
        )
        for meta, value in zip(tsi.slots, values)
    ]

    populate_prices(slots, points, tsi)

    for slot in slots[:3]:
        assert slot.price == SlotPrice(0.0, 0.0)
        assert slot.import_price_available is False
        assert slot.export_price_available is False
        assert slot.price_actionable is False
    assert slots[3].price == SlotPrice(0.0, 0.0)
    assert slots[3].price_actionable is True
    assert slots[4].price == SlotPrice(-0.25, -0.25)
    assert slots[4].price_actionable is True


def test_solcast_population_rejects_all_nonfinite_values() -> None:
    """Direct SolcastSlot infinities cannot poison the physical LP arrays."""
    tsi = TimeSeriesIndex.from_now(_NOW, interval_minutes=60, horizon_hours=4)
    slots = [PlannedSlot(meta.start, meta.end) for meta in tsi.slots]
    values = [float("nan"), float("inf"), float("-inf"), 1.25]
    solcast = [
        SolcastSlot(
            hour=meta.hour,
            pv_estimate=value,
            day_offset=meta.key.day_offset,
        )
        for meta, value in zip(tsi.slots, values)
    ]

    populate_solcast(slots, solcast, 60, tsi)

    assert [slot.solcast_pv_estimate_kwh for slot in slots] == [0.0, 0.0, 0.0, 1.25]
