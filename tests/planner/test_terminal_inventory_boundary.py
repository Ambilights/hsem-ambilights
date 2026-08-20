"""Regressions for bounded primary-battery terminal inventory valuation.

Unless stated otherwise, energy values are battery-side DC kWh, grid/load
values are AC kWh, and prices are local currency per AC kWh.  The fixtures use
explicit 15- or 60-minute slots so quantity caps, efficiency, wear, and the
published-price boundary remain hand-calculable.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.planner_input import PlannerInput
from custom_components.hsem.models.price_forecast import (
    ForecastPricePoint,
    PriceForecast,
)
from custom_components.hsem.models.terminal_cost_to_go import TerminalCostToGo
from custom_components.hsem.planner.cost_function import CostWeights, score_plan
from custom_components.hsem.planner.engine_explanation import _build_explanation
from custom_components.hsem.planner.future_value import build_terminal_cost_to_go
from custom_components.hsem.planner.milp_optimizer import (
    is_scipy_available,
    solve_milp,
)
from custom_components.hsem.utils.prices import SlotPrice
from custom_components.hsem.utils.recommendations import Recommendations

_NOW = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


def _slot(
    index: int,
    *,
    duration_minutes: int = 15,
    import_price: float = 1.0,
    export_price: float = 0.0,
    house_kwh: float = 0.0,
    pv_kwh: float = 0.0,
    actionable: bool = True,
) -> PlannedSlot:
    """Build one future slot with explicit physical and price authority."""
    start = _NOW + timedelta(minutes=(index + 1) * duration_minutes)
    return PlannedSlot(
        start=start,
        end=start + timedelta(minutes=duration_minutes),
        price=SlotPrice(import_price=import_price, export_price=export_price),
        import_price_available=actionable,
        export_price_available=actionable,
        price_actionable=actionable,
        avg_house_consumption_kwh=house_kwh,
        solcast_pv_estimate_kwh=pv_kwh,
        estimated_net_consumption_kwh=house_kwh - pv_kwh,
    )


def _forecast(
    *slots_and_prices: tuple[PlannedSlot, float],
    mae: float = 0.0,
    margin: float = 0.0,
) -> PriceForecast:
    """Build an enabled forecast aligned exactly to the supplied slot starts."""
    return PriceForecast(
        points=tuple(
            ForecastPricePoint(start=slot.start, value=price)
            for slot, price in slots_and_prices
        ),
        mae=mae,
        margin=margin,
        enabled=True,
    )


def _cost_to_go(
    slots: list[PlannedSlot],
    forecast: PriceForecast | None,
    *,
    usable_kwh: float = 25.5,
    max_discharge_per_slot: float = 2.5,
    discharge_efficiency_pct: float = 100.0,
    cycle_cost_per_kwh: float = 0.0,
) -> TerminalCostToGo:
    """Build terminal value with all physical quantities explicit."""
    return build_terminal_cost_to_go(
        slots,
        _NOW,
        forecast=forecast,
        usable_kwh=usable_kwh,
        max_discharge_per_slot=max_discharge_per_slot,
        discharge_efficiency_pct=discharge_efficiency_pct,
        cycle_cost_per_kwh=cycle_cost_per_kwh,
    )


def _solve(
    slots: list[PlannedSlot],
    terminal_cost_to_go: TerminalCostToGo,
    *,
    current_kwh: float,
    usable_kwh: float,
    max_charge_per_slot: float = 2.5,
    max_discharge_per_slot: float = 2.5,
    charge_efficiency_pct: float = 100.0,
    discharge_efficiency_pct: float = 100.0,
    cycle_cost_per_kwh: float = 0.0,
    no_export: bool = True,
    excess_export_discharge_buffer_pct: float = 0.0,
) -> tuple[list[PlannedSlot], dict]:
    """Solve a deterministic primary-battery boundary scenario."""
    solved = solve_milp(
        slots,
        _NOW,
        current_kwh=current_kwh,
        usable_kwh=usable_kwh,
        max_charge_per_slot=max_charge_per_slot,
        max_discharge_per_slot=max_discharge_per_slot,
        cycle_cost_per_kwh=cycle_cost_per_kwh,
        charge_efficiency_pct=charge_efficiency_pct,
        discharge_efficiency_pct=discharge_efficiency_pct,
        time_discount_rate=1.0,
        replacement_price_per_kwh=None,
        terminal_cost_to_go=terminal_cost_to_go,
        no_export=no_export,
        excess_export_discharge_buffer_pct=excess_export_discharge_buffer_pct,
    )
    assert solved is not None
    return solved


def test_one_kwh_forecast_demand_cannot_value_all_25_5_kwh() -> None:
    """Cap terminal salvage to one post-boundary demand opportunity.

    The sole unpublished 1.0 AC-kWh load is forecast at 4.00. MAE 0.50 and
    margin 0.25 leave 3.25 currency/kWh. At 100% efficiency its DC quantity
    is exactly 1.0 kWh: the first stored kWh is worth 3.25, while the other
    24.5 kWh receive no invented terminal credit.
    """
    published = _slot(0, import_price=2.0)
    tail = _slot(1, house_kwh=1.0, actionable=False)
    model = _cost_to_go(
        [published, tail],
        _forecast((tail, 4.0), mae=0.5, margin=0.25),
    )

    assert model.source == "forecast"
    assert model.boundary == published.end
    assert model.total_quantity_kwh == pytest.approx(1.0)
    assert len(model.tiers) == 1
    assert model.tiers[0].quantity_kwh == pytest.approx(1.0)
    assert model.tiers[0].forecast_price_per_kwh == pytest.approx(3.25)
    assert model.tiers[0].value_per_kwh == pytest.approx(3.25)
    assert model.inventory_value(0.0) == pytest.approx(0.0)
    assert model.inventory_value(0.5) == pytest.approx(1.625)
    assert model.inventory_value(1.0) == pytest.approx(3.25)
    assert model.inventory_value(25.5) == pytest.approx(3.25)


def test_forecast_must_be_unpublished_exactly_aligned_and_haircut() -> None:
    """Exclude official overlap and off-cadence predictions at the boundary."""
    published = _slot(0, import_price=3.0, house_kwh=1.0)
    tail = _slot(1, house_kwh=1.0, actionable=False)
    off_cadence = PlannedSlot(
        start=tail.start + timedelta(minutes=5),
        end=tail.end + timedelta(minutes=5),
    )
    forecast = PriceForecast(
        points=(
            ForecastPricePoint(start=published.start, value=99.0),
            ForecastPricePoint(start=off_cadence.start, value=99.0),
            ForecastPricePoint(start=tail.start, value=4.0),
        ),
        mae=0.5,
        margin=0.25,
        enabled=True,
    )

    model = _cost_to_go([published, tail], forecast)

    assert len(model.tiers) == 1
    assert model.tiers[0].start == tail.start
    assert model.tiers[0].forecast_price_per_kwh == pytest.approx(3.25)

    overlap_only = _cost_to_go(
        [published, tail],
        _forecast((published, 99.0)),
    )
    assert overlap_only.source == "hardware_floor_only"
    assert overlap_only.total_quantity_kwh == pytest.approx(0.0)
    assert overlap_only.inventory_value(25.5) == pytest.approx(0.0)


def test_efficiency_wear_and_power_cap_are_hand_calculable() -> None:
    """Price 2.0, eta 0.9, and wear 0.1 yield 1.5/DC-kWh.

    The post-boundary load asks for 1.8 AC kWh, or 2.0 DC kWh at 90%, but the
    per-slot discharge cap admits only 1.5 DC kWh. Marginal terminal value is
    2.0 * (2 * 0.9 - 1) - 0.1 = 1.5 currency per DC kWh.
    """
    published = _slot(0)
    tail = _slot(1, house_kwh=1.8, actionable=False)
    model = _cost_to_go(
        [published, tail],
        _forecast((tail, 2.0)),
        usable_kwh=10.0,
        max_discharge_per_slot=1.5,
        discharge_efficiency_pct=90.0,
        cycle_cost_per_kwh=0.1,
    )

    assert model.total_quantity_kwh == pytest.approx(1.5)
    assert model.tiers[0].quantity_kwh == pytest.approx(1.5)
    assert model.tiers[0].value_per_kwh == pytest.approx(1.5)
    assert model.inventory_value(1.5) == pytest.approx(2.25)


@pytest.mark.skipif(not is_scipy_available(), reason="scipy not available")
def test_rolling_boundary_releases_known_peak_instead_of_relocating_hold() -> None:
    """Move publication one slot and keep both known peak blocks executable.

    Before publication, two 1.0 kWh forecast loads create only 2.0 kWh of
    bounded terminal demand. After the first forecast slot publishes it is
    removed from cost-to-go, leaving 1.0 kWh. A full 5 kWh battery therefore
    serves every overlapping known 3.2+ price load in both solves; terminal
    value does not freeze the final published block and move with the boundary.
    """

    def scenario(actionable_count: int) -> tuple[list[PlannedSlot], TerminalCostToGo]:
        slots = [
            _slot(
                index,
                duration_minutes=60,
                import_price=(1.0, 3.2, 3.25, 0.0)[index],
                house_kwh=(0.0, 1.0, 1.0, 1.0)[index],
                actionable=index < actionable_count,
            )
            for index in range(4)
        ]
        forecast = _forecast((slots[2], 4.0), (slots[3], 4.0))
        return slots, _cost_to_go(
            slots,
            forecast,
            usable_kwh=5.0,
            max_discharge_per_slot=2.0,
        )

    early_slots, early_model = scenario(2)
    later_slots, later_model = scenario(3)

    assert early_model.boundary == early_slots[1].end
    assert early_model.total_quantity_kwh == pytest.approx(2.0)
    assert later_model.boundary == later_slots[2].end
    assert later_model.total_quantity_kwh == pytest.approx(1.0)
    assert [tier.start for tier in later_model.tiers] == [later_slots[3].start]

    early_plan, _early_diagnostics = _solve(
        early_slots,
        early_model,
        current_kwh=5.0,
        usable_kwh=5.0,
        max_charge_per_slot=2.0,
        max_discharge_per_slot=2.0,
    )
    later_plan, _later_diagnostics = _solve(
        later_slots,
        later_model,
        current_kwh=5.0,
        usable_kwh=5.0,
        max_charge_per_slot=2.0,
        max_discharge_per_slot=2.0,
    )

    assert early_plan[1].batteries_discharged_kwh == pytest.approx(1.0, abs=0.001)
    assert early_plan[1].grid_import_kwh == pytest.approx(0.0, abs=0.001)
    for index in (1, 2):
        assert later_plan[index].batteries_discharged_kwh == pytest.approx(
            1.0,
            abs=0.001,
        )
        assert later_plan[index].grid_import_kwh == pytest.approx(0.0, abs=0.001)


@pytest.mark.skipif(not is_scipy_available(), reason="scipy not available")
def test_live_45_slot_full_battery_serves_terminal_peak_down_to_floor() -> None:
    """Reproduce the observed 20-slot tail inside a 45-slot full-battery run.

    Huawei has 25.5 kWh above its configured 15% floor (30 kWh nameplate).
    The final 20 quarter-hours carry the observed 7.704 kWh load shape and
    2.959-3.292 import prices. Earlier synthetic demand brings total known
    load to 25.0 AC kWh. With 98% discharge efficiency the battery can deliver
    24.99 kWh before reaching its hard floor, so only about 0.01 kWh imports;
    it must not remain full through the final peak.
    """
    tail_loads = [
        *([0.253] * 4),
        *([0.471] * 4),
        *([0.432] * 4),
        *([0.401] * 4),
        *([0.369] * 4),
    ]
    tail_prices = [
        3.253,
        3.257,
        3.261,
        3.268,
        3.252,
        3.261,
        3.274,
        3.284,
        3.239,
        3.255,
        3.275,
        3.292,
        2.959,
        3.020,
        3.100,
        3.188,
        3.086,
        3.120,
        3.160,
        3.194,
    ]
    assert sum(tail_loads) == pytest.approx(7.704)

    early_load = (25.0 - sum(tail_loads)) / 25
    slots = [
        _slot(index, import_price=2.5, house_kwh=early_load) for index in range(25)
    ]
    slots.extend(
        _slot(
            index + 25,
            import_price=price,
            house_kwh=load,
        )
        for index, (load, price) in enumerate(zip(tail_loads, tail_prices, strict=True))
    )
    model = _cost_to_go(
        slots,
        PriceForecast(),
        usable_kwh=25.5,
        max_discharge_per_slot=2.5,
        discharge_efficiency_pct=98.0,
        cycle_cost_per_kwh=0.092593,
    )
    planned, diagnostics = _solve(
        slots,
        model,
        current_kwh=25.5,
        usable_kwh=25.5,
        discharge_efficiency_pct=98.0,
        charge_efficiency_pct=98.0,
        cycle_cost_per_kwh=0.092593,
    )

    assert model.source == "hardware_floor_only"
    assert model.total_quantity_kwh == pytest.approx(0.0)
    tail = planned[-20:]
    assert all(slot.batteries_discharged_kwh > 0.0 for slot in tail)
    assert all(slot.primary_battery_hold is False for slot in tail)
    assert sum(slot.grid_import_kwh for slot in tail) == pytest.approx(
        0.0,
        abs=0.02,
    )
    assert sum(slot.batteries_discharged_kwh for slot in tail) == pytest.approx(
        sum(tail_loads) / 0.98,
        abs=0.02,
    )

    discharged_kwh = sum(slot.batteries_discharged_kwh for slot in planned)
    remaining_above_floor_kwh = 25.5 - discharged_kwh
    assert remaining_above_floor_kwh >= -0.03
    assert remaining_above_floor_kwh <= 0.03
    final_absolute_soc_pct = 15.0 + max(remaining_above_floor_kwh, 0.0) / 30.0 * 100
    assert final_absolute_soc_pct >= 15.0
    assert final_absolute_soc_pct <= 15.1
    assert diagnostics["terminal_cost_to_go_source"] == "hardware_floor_only"
    assert diagnostics["terminal_cost_to_go_total_quantity_kwh"] == pytest.approx(0.0)


@pytest.mark.skipif(not is_scipy_available(), reason="scipy not available")
def test_terminal_model_cannot_bypass_nonactionable_or_ev_guards() -> None:
    """Keep unpublished storage idle and exclude accounted EV demand."""
    actionable = _slot(0)
    unpublished = _slot(
        1,
        import_price=99.0,
        house_kwh=1.0,
        actionable=False,
    )
    empty = _cost_to_go([actionable, unpublished], PriceForecast(), usable_kwh=1.0)
    unpublished_plan, _diagnostics = _solve(
        [actionable, unpublished],
        empty,
        current_kwh=1.0,
        usable_kwh=1.0,
    )
    assert unpublished_plan[1].batteries_discharged_kwh == pytest.approx(0.0)
    assert unpublished_plan[1].batteries_charged_kwh == pytest.approx(0.0)

    accounted_ev = _slot(0, import_price=10.0, house_kwh=1.0)
    accounted_ev.ev_accounted_load_kwh = 1.0
    ev_model = _cost_to_go([accounted_ev], PriceForecast(), usable_kwh=1.0)
    ev_plan, _diagnostics = _solve(
        [accounted_ev],
        ev_model,
        current_kwh=1.0,
        usable_kwh=1.0,
    )
    assert ev_plan[0].batteries_discharged_kwh == pytest.approx(0.0)
    assert ev_plan[0].grid_import_kwh == pytest.approx(1.0, abs=0.001)


@pytest.mark.skipif(not is_scipy_available(), reason="scipy not available")
def test_terminal_model_preserves_no_export_and_conditional_export_reserve() -> None:
    """Keep PV source attribution and the 50% conditional export reserve."""
    pv_slot = _slot(0, export_price=10.0, pv_kwh=1.0)
    empty = _cost_to_go([pv_slot], PriceForecast(), usable_kwh=1.0)
    no_export_plan, _diagnostics = _solve(
        [pv_slot],
        empty,
        current_kwh=1.0,
        usable_kwh=1.0,
        no_export=True,
    )
    no_export_slot = no_export_plan[0]
    assert no_export_slot.batteries_discharged_kwh == pytest.approx(0.0)
    assert no_export_slot.primary_battery_export_kwh == pytest.approx(0.0)
    assert no_export_slot.pv_export_kwh == pytest.approx(1.0, abs=0.001)
    assert no_export_slot.grid_export_kwh == pytest.approx(1.0, abs=0.001)

    export_slot = _slot(0, export_price=10.0)
    reserve_model = _cost_to_go([export_slot], PriceForecast(), usable_kwh=1.0)
    reserve_plan, _diagnostics = _solve(
        [export_slot],
        reserve_model,
        current_kwh=1.0,
        usable_kwh=1.0,
        no_export=False,
        excess_export_discharge_buffer_pct=50.0,
    )
    reserve_slot = reserve_plan[0]
    assert reserve_slot.batteries_discharged_kwh == pytest.approx(0.5, abs=0.001)
    assert reserve_slot.primary_battery_export_kwh == pytest.approx(0.5, abs=0.001)
    assert 1.0 - reserve_slot.batteries_discharged_kwh == pytest.approx(
        0.5,
        abs=0.001,
    )


def test_equal_export_discharge_and_refill_cancel_terminal_value() -> None:
    """Make bounded terminal accounting path-independent and source-aware."""
    boundary_slot = _slot(10, duration_minutes=60)
    tail = _slot(
        11,
        duration_minutes=60,
        house_kwh=0.9,
        actionable=False,
    )
    model = _cost_to_go(
        [boundary_slot, tail],
        _forecast((tail, 3.0)),
        usable_kwh=1.0,
        max_discharge_per_slot=1.0,
        discharge_efficiency_pct=90.0,
        cycle_cost_per_kwh=0.05,
    )

    discharge = _slot(
        0,
        duration_minutes=60,
        import_price=3.0,
        export_price=2.0,
    )
    discharge.recommendation = Recommendations.ForceBatteriesDischarge.value
    discharge.batteries_discharged_kwh = 1.0
    discharge.grid_export_kwh = 0.9
    discharge.primary_battery_export_kwh = 0.9
    discharge.pv_export_kwh = 0.0

    refill = _slot(1, duration_minutes=60, import_price=0.5)
    refill.recommendation = Recommendations.BatteriesChargeGrid.value
    refill.batteries_charged_kwh = 1.0
    refill.grid_import_kwh = 1.0 / 0.9

    breakdown = score_plan(
        [discharge, refill],
        CostWeights(
            min_soc_pct=0.0,
            max_soc_pct=100.0,
            soc_low_penalty_weight=0.0,
            soc_high_penalty_weight=0.0,
            cycle_cost_per_kwh=0.05,
            charge_efficiency_pct=90.0,
            discharge_efficiency_pct=90.0,
        ),
        slot_duration_hours=1.0,
        now=_NOW,
        initial_battery_kwh=1.0,
        replacement_price_per_kwh=99.0,
        terminal_cost_to_go=model,
    )

    assert discharge.primary_battery_export_kwh + discharge.pv_export_kwh == (
        pytest.approx(discharge.grid_export_kwh)
    )
    assert model.inventory_value(1.0) - model.inventory_value(1.0) == pytest.approx(0.0)
    assert breakdown.terminal_soc_value == pytest.approx(0.0)
    assert breakdown.score == pytest.approx(
        breakdown.total_cost + breakdown.primary_action_tiebreak
    )


@pytest.mark.skipif(not is_scipy_available(), reason="scipy not available")
def test_milp_diagnostics_and_scorer_share_exact_terminal_value() -> None:
    """Expose one auditable cost-to-go calculation in every score view."""
    action = _slot(
        0,
        duration_minutes=60,
        import_price=10.0,
        house_kwh=0.5,
    )
    tail = _slot(
        1,
        duration_minutes=60,
        house_kwh=1.0,
        actionable=False,
    )
    model = _cost_to_go(
        [action, tail],
        _forecast((tail, 4.0)),
        usable_kwh=1.0,
        max_discharge_per_slot=1.0,
    )
    planned, diagnostics = _solve(
        [action, tail],
        model,
        current_kwh=1.0,
        usable_kwh=1.0,
        max_charge_per_slot=1.0,
        max_discharge_per_slot=1.0,
    )

    assert planned[0].batteries_discharged_kwh == pytest.approx(0.5, abs=0.001)
    assert planned[0].grid_import_kwh == pytest.approx(0.0, abs=0.001)
    assert planned[1].batteries_discharged_kwh == pytest.approx(0.0)

    breakdown = score_plan(
        planned,
        CostWeights(
            min_soc_pct=0.0,
            max_soc_pct=100.0,
            soc_low_penalty_weight=0.0,
            soc_high_penalty_weight=0.0,
            cycle_cost_per_kwh=0.0,
            charge_efficiency_pct=100.0,
            discharge_efficiency_pct=100.0,
        ),
        slot_duration_hours=1.0,
        now=_NOW,
        initial_battery_kwh=1.0,
        replacement_price_per_kwh=99.0,
        terminal_cost_to_go=model,
    )

    expected_terminal_value = model.inventory_value(1.0) - model.inventory_value(0.5)
    assert expected_terminal_value == pytest.approx(2.0)
    assert breakdown.terminal_soc_value == pytest.approx(expected_terminal_value)
    assert diagnostics["terminal_inventory_value"] == pytest.approx(
        breakdown.terminal_soc_value,
        abs=1e-6,
    )
    assert diagnostics["terminal_cost_to_go_source"] == "forecast"
    assert diagnostics["terminal_cost_to_go_boundary"] == action.end.isoformat()
    assert diagnostics["terminal_cost_to_go_tier_count"] == 1
    assert diagnostics["terminal_cost_to_go_total_quantity_kwh"] == pytest.approx(1.0)
    [tier_diagnostics] = diagnostics["terminal_cost_to_go_tiers"]
    assert tier_diagnostics["start"] == tail.start.isoformat()
    assert tier_diagnostics["quantity_kwh"] == pytest.approx(1.0)
    assert tier_diagnostics["value_per_kwh"] == pytest.approx(4.0)
    assert tier_diagnostics["forecast_price_per_kwh"] == pytest.approx(4.0)
    assert diagnostics["terminal_cost_to_go_initial_inventory_kwh"] == pytest.approx(
        1.0
    )
    assert diagnostics["terminal_cost_to_go_final_inventory_kwh"] == pytest.approx(0.5)
    assert diagnostics[
        "terminal_cost_to_go_initial_valued_quantity_kwh"
    ] == pytest.approx(1.0)
    assert diagnostics[
        "terminal_cost_to_go_final_valued_quantity_kwh"
    ] == pytest.approx(0.5)
    assert diagnostics["terminal_cost_to_go_initial_value"] == pytest.approx(4.0)
    assert diagnostics["terminal_cost_to_go_final_value"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("mae", "margin"),
    [
        (float("nan"), 0.0),
        (0.0, float("nan")),
        (float("inf"), 0.0),
        (0.0, float("inf")),
        (float("-inf"), 0.0),
        (0.0, float("-inf")),
    ],
)
def test_nonfinite_forecast_haircut_fails_closed(
    mae: float,
    margin: float,
) -> None:
    """Never let invalid forecast uncertainty poison terminal valuation."""
    published = _slot(0)
    tail = _slot(1, house_kwh=1.0, actionable=False)
    model = _cost_to_go(
        [published, tail],
        _forecast((tail, 5.0), mae=mae, margin=margin),
    )

    assert model.source == "hardware_floor_only"
    assert model.tiers == ()
    assert model.total_quantity_kwh == pytest.approx(0.0)
    assert model.inventory_value(25.5) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("avg_house_consumption_kwh", None),
        ("avg_house_consumption_kwh", float("nan")),
        ("avg_house_consumption_kwh", float("inf")),
        ("avg_house_consumption_kwh", float("-inf")),
        ("solcast_pv_estimate_kwh", None),
        ("solcast_pv_estimate_kwh", float("nan")),
        ("solcast_pv_estimate_kwh", float("inf")),
        ("solcast_pv_estimate_kwh", float("-inf")),
        ("ev_accounted_load_kwh", None),
        ("ev_accounted_load_kwh", float("nan")),
        ("ev_accounted_load_kwh", float("inf")),
        ("ev_accounted_load_kwh", float("-inf")),
    ],
)
def test_invalid_tail_load_inputs_cannot_create_terminal_tier(
    field_name: str,
    invalid_value: object,
) -> None:
    """Treat unavailable or nonfinite post-boundary load inputs as no value."""
    published = _slot(0)
    tail = _slot(1, house_kwh=1.0, actionable=False)
    setattr(tail, field_name, invalid_value)

    model = _cost_to_go(
        [published, tail],
        _forecast((tail, 5.0)),
    )

    assert model.source == "hardware_floor_only"
    assert model.tiers == ()
    assert model.total_quantity_kwh == pytest.approx(0.0)


def test_tail_quantity_excludes_pv_and_accounted_ev_load() -> None:
    """Value only residual house demand that the primary battery may serve."""
    published = _slot(0)
    tail = _slot(
        1,
        house_kwh=4.0,
        pv_kwh=1.0,
        actionable=False,
    )
    tail.ev_accounted_load_kwh = 1.0

    model = _cost_to_go(
        [published, tail],
        _forecast((tail, 3.0)),
        usable_kwh=10.0,
        max_discharge_per_slot=10.0,
        discharge_efficiency_pct=80.0,
    )

    # (4.0 house - 1.0 PV - 1.0 accounted EV) / 0.8 efficiency.
    assert model.total_quantity_kwh == pytest.approx(2.5)
    assert model.tiers[0].quantity_kwh == pytest.approx(2.5)
    assert model.tiers[0].start == tail.start


def test_duplicate_forecasts_collapse_conservatively_and_deterministically() -> None:
    """Use the lower duplicate and order tiers by value then physical start."""
    published = _slot(0)
    earlier = _slot(1, house_kwh=1.0, actionable=False)
    later = _slot(2, house_kwh=1.0, actionable=False)
    points = (
        ForecastPricePoint(start=earlier.start, value=5.0),
        ForecastPricePoint(start=earlier.start, value=3.0),
        ForecastPricePoint(start=later.start, value=4.0),
    )

    def build(points_in_order: tuple[ForecastPricePoint, ...]) -> TerminalCostToGo:
        return _cost_to_go(
            [published, earlier, later],
            PriceForecast(
                points=points_in_order,
                mae=0.5,
                enabled=True,
            ),
        )

    forward = build(points)
    reverse = build(tuple(reversed(points)))

    assert forward == reverse
    assert [tier.start for tier in forward.tiers] == [later.start, earlier.start]
    assert [tier.forecast_price_per_kwh for tier in forward.tiers] == pytest.approx(
        [3.5, 2.5]
    )
    assert [tier.value_per_kwh for tier in forward.tiers] == pytest.approx([3.5, 2.5])


def test_charge_only_explanation_never_promises_in_window_discharge_savings() -> None:
    """Reconcile charge-only wording and money over two 15-minute slots.

    The selected plan imports 1.2 kWh at 1.0 and 0.4 kWh at 3.2, for a
    monetary cost of 2.48.  An idle battery would import only the two house
    loads, costing 1.48.  The negative 1.00 explanation score is therefore
    valid, but with zero discharge in both slots it must not claim that
    discharge savings occur inside this scheduling window.
    """
    charge = _slot(0, import_price=1.0, house_kwh=0.2)
    charge.recommendation = Recommendations.BatteriesChargeGrid.value
    charge.batteries_charged_kwh = 1.0
    charge.grid_import_kwh = 1.2
    charge.estimated_cost_currency = 1.2

    hold = _slot(1, import_price=3.2, house_kwh=0.4)
    hold.recommendation = Recommendations.BatteriesWaitMode.value
    hold.primary_battery_hold = True
    hold.grid_import_kwh = 0.4
    hold.estimated_cost_currency = 1.28

    explanation = _build_explanation(
        PlannerInput(
            now_iso=_NOW.isoformat(),
            interval_minutes=15,
            interval_length_hours=1,
            battery_soc_pct=90.0,
            battery_rated_capacity_kwh=30.0,
            battery_end_of_discharge_soc_pct=15.0,
            battery_max_soc_pct=100.0,
            battery_purchase_price=0.0,
            battery_expected_cycles=6000,
            months_winter=[1, 2, 3, 4, 10, 11, 12],
            is_read_only=True,
        ),
        [charge, hold],
        battery_soc_at_end=93.3,
        now=_NOW,
    )

    assert explanation.selected_strategy == "opportunistic_charge"
    assert "no scheduled discharge window" in explanation.summary
    assert explanation.estimated_total_cost == pytest.approx(2.48)
    assert explanation.score == pytest.approx(-1.0)

    do_nothing = next(
        rejected
        for rejected in explanation.rejected_plans
        if rejected.name == "do_nothing"
    )
    assert do_nothing.estimated_cost == pytest.approx(1.48)
    assert "discharge savings" not in do_nothing.reason.lower()
    assert "within the current scheduling window" not in do_nothing.reason.lower()
