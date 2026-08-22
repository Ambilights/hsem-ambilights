"""Regression tests for authoritative signed-price planner economics."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

import custom_components.hsem.planner.candidate_generator as candidate_generator
from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.planner_input import PlannerInput
from custom_components.hsem.models.secondary_storage_config import (
    SecondaryStorageConfig,
)
from custom_components.hsem.planner import run_planner
from custom_components.hsem.planner.cost_function import CostWeights, score_plan
from custom_components.hsem.planner.cost_helpers import slot_grid_cash_flow_cost
from custom_components.hsem.planner.engine_explanation import _build_explanation
from custom_components.hsem.planner.milp._secondary_diagnostics import (
    _sanitised_import_price,
)
from custom_components.hsem.planner.milp_optimizer import (
    is_scipy_available,
    solve_milp,
)
from custom_components.hsem.planner.slot_population import populate_estimated_cost
from custom_components.hsem.planner.soc_simulation import simulate_soc
from custom_components.hsem.utils.misc import calculate_recommended_threshold
from custom_components.hsem.utils.prices import SlotPrice
from custom_components.hsem.utils.recommendations import Recommendations
from tests.planner.fixtures import make_summer_day_input

_TZ = ZoneInfo("Europe/Copenhagen")
_NOW = datetime(2024, 6, 15, 0, 0, tzinfo=_TZ)


def _slot(
    *,
    import_price: float = 0.20,
    export_price: float = 0.05,
    consumption_kwh: float = 0.0,
    pv_kwh: float = 0.0,
) -> PlannedSlot:
    """Build one actionable one-hour slot with authoritative price inputs."""
    slot = PlannedSlot(
        start=_NOW,
        end=_NOW + timedelta(hours=1),
        price=SlotPrice(import_price=import_price, export_price=export_price),
    )
    slot.avg_house_consumption_kwh = consumption_kwh
    slot.solcast_pv_estimate_kwh = pv_kwh
    slot.estimated_net_consumption_kwh = consumption_kwh - pv_kwh
    return slot


def test_final_slot_cost_uses_grid_flow_and_signed_import_price() -> None:
    """Battery charging is billed from final import, not forecast net load."""
    slot = _slot(import_price=-0.05)
    slot.recommendation = Recommendations.BatteriesChargeGrid.value
    slot.batteries_charged_kwh = 1.0

    simulate_soc(
        [slot],
        _NOW,
        current_kwh=0.0,
        usable_kwh=2.0,
        max_capacity_kwh=2.0,
        max_charge_per_slot=1.0,
        max_discharge_per_slot=1.0,
        rated_kwh=2.0,
        end_of_discharge_soc_pct=0.0,
        charge_efficiency_pct=100.0,
        discharge_efficiency_pct=100.0,
    )

    assert slot.estimated_net_consumption_kwh == 0.0
    assert slot.grid_import_kwh == pytest.approx(1.0)
    assert slot.estimated_cost_currency == pytest.approx(-0.05)
    assert slot_grid_cash_flow_cost(slot) == pytest.approx(-0.05)
    assert score_plan([slot], CostWeights()).import_cost == pytest.approx(-0.05)


def test_preflow_baseline_cost_preserves_signed_import_and_export_prices() -> None:
    """The initial net-load ledger keeps both kinds of negative market rate."""
    import_slot = _slot(import_price=-0.05, consumption_kwh=2.0)
    export_slot = _slot(export_price=-0.10, pv_kwh=3.0)

    populate_estimated_cost([import_slot, export_slot])

    assert import_slot.estimated_cost_currency == pytest.approx(-0.10)
    assert export_slot.estimated_cost_currency == pytest.approx(0.30)


@pytest.mark.skipif(not is_scipy_available(), reason="scipy not available")
def test_milp_uses_bounded_negative_import_credit() -> None:
    """A negative rate can justify finite charging despite battery wear."""
    result = solve_milp(
        [_slot(import_price=-0.05, export_price=0.0)],
        _NOW,
        current_kwh=0.0,
        usable_kwh=2.0,
        max_charge_per_slot=2.0,
        max_discharge_per_slot=2.0,
        cycle_cost_per_kwh=0.02,
        charge_efficiency_pct=100.0,
        discharge_efficiency_pct=100.0,
    )

    assert result is not None
    slots, _diagnostics = result
    assert slots[0].batteries_charged_kwh == pytest.approx(2.0)
    assert slots[0].grid_import_kwh == pytest.approx(2.0)
    assert slots[0].grid_export_kwh == pytest.approx(0.0)
    assert slots[0].estimated_cost_currency == pytest.approx(-0.10)


@pytest.mark.skipif(not is_scipy_available(), reason="scipy not available")
def test_milp_preserves_export_rate_above_import_rate() -> None:
    """Real export revenue, not an import-price clamp, clears battery wear."""
    result = solve_milp(
        [_slot(import_price=0.05, export_price=0.10)],
        _NOW,
        current_kwh=1.0,
        usable_kwh=1.0,
        max_charge_per_slot=1.0,
        max_discharge_per_slot=1.0,
        cycle_cost_per_kwh=0.08,
        charge_efficiency_pct=100.0,
        discharge_efficiency_pct=100.0,
    )

    assert result is not None
    slots, _diagnostics = result
    assert slots[0].primary_battery_export_kwh == pytest.approx(1.0)
    assert slots[0].grid_export_kwh == pytest.approx(1.0)
    assert slots[0].grid_import_kwh == pytest.approx(0.0)


@pytest.mark.skipif(not is_scipy_available(), reason="scipy not available")
def test_extreme_export_price_cannot_discharge_below_floor() -> None:
    """No finite signed export price may buy energy that does not exist."""
    result = solve_milp(
        [_slot(import_price=0.10, export_price=100.0)],
        _NOW,
        current_kwh=0.0,
        usable_kwh=1.0,
        max_charge_per_slot=10.0,
        max_discharge_per_slot=10.0,
        cycle_cost_per_kwh=0.0,
        charge_efficiency_pct=100.0,
        discharge_efficiency_pct=100.0,
    )

    assert result is not None
    slots, diagnostics = result
    assert slots[0].batteries_discharged_kwh == pytest.approx(0.0)
    assert slots[0].grid_export_kwh == pytest.approx(0.0)
    assert diagnostics["total_violation_kwh"] == pytest.approx(0.0)
    assert diagnostics["primary_postwrite_inventory_validation"]["valid"] is True


@pytest.mark.skipif(not is_scipy_available(), reason="scipy not available")
def test_extreme_negative_import_cannot_charge_above_ceiling() -> None:
    """No finite signed import credit may create battery headroom."""
    result = solve_milp(
        [_slot(import_price=-100.0, export_price=0.0)],
        _NOW,
        current_kwh=1.0,
        usable_kwh=1.0,
        max_charge_per_slot=10.0,
        max_discharge_per_slot=10.0,
        cycle_cost_per_kwh=0.0,
        charge_efficiency_pct=100.0,
        discharge_efficiency_pct=100.0,
    )

    assert result is not None
    slots, diagnostics = result
    assert slots[0].batteries_charged_kwh == pytest.approx(0.0)
    assert slots[0].grid_import_kwh == pytest.approx(0.0)
    assert diagnostics["total_violation_kwh"] == pytest.approx(0.0)
    assert diagnostics["primary_postwrite_inventory_validation"]["valid"] is True


def test_candidate_depreciation_floor_is_battery_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Depreciation must not become the site floor that suppresses PV export."""
    captured: dict[str, object] = {}

    def fake_solve(*_args: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(candidate_generator, "is_scipy_available", lambda: True)
    monkeypatch.setattr(candidate_generator, "solve_milp", fake_solve)
    inp = make_summer_day_input()
    inp.export_min_price = 0.04
    inp.battery_export_min_price = 0.05
    inp.battery_purchase_price = 100_000.0
    usable_kwh = 9.0

    candidate_generator.generate_candidates(
        [_slot()],
        inp,
        _NOW,
        max_charge_per_slot=1.0,
        current_kwh=1.0,
        usable_kwh=usable_kwh,
    )

    depreciation_floor = calculate_recommended_threshold(
        purchase_price=inp.battery_purchase_price,
        expected_cycles=inp.battery_expected_cycles,
        usable_capacity=usable_kwh,
        capacity_loss_pct=inp.battery_capacity_loss_pct,
    )
    assert captured["min_export_price"] == pytest.approx(inp.export_min_price)
    assert captured["battery_export_min_price"] == pytest.approx(
        max(inp.battery_export_min_price, depreciation_floor)
    )
    assert depreciation_floor > inp.export_min_price


def test_explanation_cash_flow_reconciles_with_plan_cost() -> None:
    """Explanation money is import cost minus export revenue, excluding wear."""
    output = run_planner(make_summer_day_input())
    assert output.plan_cost is not None
    expected_cash_flow = round(
        output.plan_cost.import_cost - output.plan_cost.export_revenue,
        4,
    )
    assert output.explanation.estimated_total_cost == pytest.approx(expected_cash_flow)
    assert sum(slot.estimated_cost_currency for slot in output.slots) == pytest.approx(
        expected_cash_flow,
        abs=0.001,
    )


@pytest.mark.parametrize(
    ("base_includes_dedicated_load", "expected_idle_cost"),
    [(False, 3.0), (True, 2.0)],
)
def test_explanation_idle_cost_accounts_for_powmr_utility_load_once(
    base_includes_dedicated_load: bool,
    expected_idle_cost: float,
) -> None:
    """The idle comparator adds only a dedicated load absent from base load."""
    slot = _slot(import_price=2.0, consumption_kwh=1.0)
    slot.secondary_storage_load_kwh = 0.5
    secondary = SecondaryStorageConfig(
        enabled=True,
        capacity_kwh=10.0,
        current_soc_pct=50.0,
        min_soc_pct=20.0,
        max_soc_pct=100.0,
        nominal_voltage_v=24.0,
        load_power_w=500.0,
        min_charge_current_a=10.0,
        max_charge_current_a=40.0,
        charge_current_step_a=10.0,
        base_load_includes_dedicated_load=base_includes_dedicated_load,
    )
    explanation = _build_explanation(
        PlannerInput(
            now_iso=_NOW.isoformat(),
            interval_minutes=60,
            secondary_storage=secondary,
        ),
        [slot],
        battery_soc_at_end=0.0,
        now=_NOW,
    )

    idle = next(
        plan for plan in explanation.rejected_plans if plan.name == "do_nothing"
    )
    assert idle.estimated_cost == pytest.approx(expected_idle_cost)


def test_secondary_diagnostics_preserve_negative_import_price() -> None:
    """PowMr result diagnostics use the same signed rate as PlanCost."""
    slot = _slot(import_price=-0.05)
    assert _sanitised_import_price(slot) == pytest.approx(-0.05)
    slot.price_actionable = False
    assert _sanitised_import_price(slot) == 0.0
