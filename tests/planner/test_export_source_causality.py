"""Regression tests for causal battery-export and exact primary actions."""

from __future__ import annotations

import pytest

from custom_components.hsem.models.ev_config import EVConfig
from custom_components.hsem.planner.cost_function import CostWeights, score_plan
from custom_components.hsem.planner.milp_optimizer import (
    is_scipy_available,
    solve_milp,
)
from tests.planner.test_terminal_inventory_export import _NOW, _slot

pytestmark = pytest.mark.skipif(
    not is_scipy_available(),
    reason="scipy not available in this environment",
)


def test_pv_cannot_hide_battery_export_behind_forced_ev_load() -> None:
    """Hold 1 DC kWh when its 1.0 export cannot replace its 1.5 value.

    One 60-minute slot has 2.0 AC kWh PV and a fixed 1.0 AC kWh EV session.
    With no Huawei discharge, PV serves the EV and exports 1.0 kWh. Discharging
    Huawei would raise export to 2.0 kWh, so that extra export is causally
    battery-origin even if an arbitrary electron routing labels Huawei as the
    EV source. Its 1.0 revenue is below R=1.5, so the 1.0 DC kWh inventory must
    be held and the source split must remain 0.0 primary + 1.0 PV export.
    """
    ev = EVConfig(
        enabled=True,
        initial_soc_kwh=0.0,
        target_kwh=1.0,
        capacity_kwh=10.0,
        max_charge_per_slot=1.0,
        charger_efficiency=1.0,
        charger_min_power_w=0.0,
        deadline_slot=0,
        session_charge_kw=1.0,
    )
    solved = solve_milp(
        [_slot(0, import_price=1.0, export_price=1.0, pv_kwh=2.0)],
        _NOW,
        current_kwh=1.0,
        usable_kwh=1.0,
        max_charge_per_slot=1.0,
        max_discharge_per_slot=1.0,
        charge_efficiency_pct=100.0,
        discharge_efficiency_pct=100.0,
        cycle_cost_per_kwh=0.0,
        replacement_price_per_kwh=1.5,
        time_discount_rate=1.0,
        ev_configs=[ev],
    )
    assert solved is not None
    planned, diagnostics = solved
    slot = planned[0]

    assert slot.ev_total_planned_load_kwh == pytest.approx(1.0, abs=0.001)
    assert slot.batteries_charged_kwh == pytest.approx(0.0, abs=0.001)
    assert slot.batteries_discharged_kwh == pytest.approx(0.0, abs=0.001)
    assert slot.grid_export_kwh == pytest.approx(1.0, abs=0.001)
    assert slot.primary_battery_export_kwh == pytest.approx(0.0, abs=0.001)
    assert slot.pv_export_kwh == pytest.approx(1.0, abs=0.001)
    assert 1.0 + slot.batteries_charged_kwh - slot.batteries_discharged_kwh == (
        pytest.approx(1.0, abs=0.001)
    )
    assert diagnostics["grid_import_export_overlap_max_kwh"] == pytest.approx(
        0.0,
        abs=1e-9,
    )


def test_primary_action_is_exact_and_diagnostic_components_match_scorer() -> None:
    """Choose a strictly economic 1 DC kWh discharge with one exact action.

    One 60-minute slot has 1.0 AC kWh load, no PV, import price 1.0, R=0.8,
    unit efficiencies, and zero wear. Discharging saves 1.0 while forfeiting
    only 0.8 of inventory, so the strict result is ec=0 and ed=1. The exact
    action binary forbids simultaneous charge/discharge, and the published
    terminal and structural-tiebreak terms must match direct scoring.
    """
    solved = solve_milp(
        [_slot(0, import_price=1.0, export_price=0.0, house_kwh=1.0)],
        _NOW,
        current_kwh=1.0,
        usable_kwh=1.0,
        max_charge_per_slot=1.0,
        max_discharge_per_slot=1.0,
        charge_efficiency_pct=100.0,
        discharge_efficiency_pct=100.0,
        cycle_cost_per_kwh=0.0,
        replacement_price_per_kwh=0.8,
        time_discount_rate=1.0,
    )
    assert solved is not None
    planned, diagnostics = solved
    slot = planned[0]

    assert not (
        slot.batteries_charged_kwh > 0.001 and slot.batteries_discharged_kwh > 0.001
    )
    assert slot.batteries_charged_kwh == pytest.approx(0.0, abs=0.001)
    assert slot.batteries_discharged_kwh == pytest.approx(1.0, abs=0.001)
    assert slot.grid_import_kwh == pytest.approx(0.0, abs=0.001)
    assert slot.grid_export_kwh == pytest.approx(0.0, abs=0.001)

    breakdown = score_plan(
        planned,
        CostWeights(
            min_soc_pct=0.0,
            max_soc_pct=100.0,
            cycle_cost_per_kwh=0.0,
            charge_efficiency_pct=100.0,
            discharge_efficiency_pct=100.0,
        ),
        now=_NOW,
        initial_battery_kwh=1.0,
        replacement_price_per_kwh=0.8,
    )
    assert diagnostics["terminal_inventory_value"] == pytest.approx(
        breakdown.terminal_soc_value,
        abs=1e-6,
    )
    assert diagnostics["primary_action_tiebreak"] == pytest.approx(
        breakdown.primary_action_tiebreak,
        abs=1e-6,
    )
    assert breakdown.terminal_soc_value == pytest.approx(0.8, abs=1e-6)
    assert breakdown.primary_action_tiebreak == pytest.approx(-0.000005, abs=1e-9)
    assert breakdown.score == pytest.approx(0.799995, abs=1e-6)


def test_flat_local_tie_is_stable_without_creating_a_primary_cycle() -> None:
    """Prefer direct local discharge in five identical true economic ties.

    Each solve has one 60-minute, 1.0 kWh house-load slot at import 1.0,
    R=1.0, unit efficiencies, and zero wear. Importing and consuming the
    initial 1.0 DC kWh have the same economic score. The -0.5e-5/kWh local
    action tiebreak must select ed=1 consistently, with ec=0 and no export.
    """
    for _attempt in range(5):
        solved = solve_milp(
            [_slot(0, import_price=1.0, export_price=0.0, house_kwh=1.0)],
            _NOW,
            current_kwh=1.0,
            usable_kwh=1.0,
            max_charge_per_slot=1.0,
            max_discharge_per_slot=1.0,
            charge_efficiency_pct=100.0,
            discharge_efficiency_pct=100.0,
            cycle_cost_per_kwh=0.0,
            replacement_price_per_kwh=1.0,
            time_discount_rate=1.0,
        )
        assert solved is not None
        planned, diagnostics = solved
        slot = planned[0]

        assert slot.batteries_charged_kwh == pytest.approx(0.0, abs=0.001)
        assert slot.batteries_discharged_kwh == pytest.approx(1.0, abs=0.001)
        assert slot.grid_import_kwh == pytest.approx(0.0, abs=0.001)
        assert slot.grid_export_kwh == pytest.approx(0.0, abs=0.001)
        assert diagnostics["primary_action_tiebreak"] == pytest.approx(
            -0.000005,
            abs=1e-9,
        )
        assert diagnostics["grid_import_export_overlap_max_kwh"] == pytest.approx(
            0.0,
            abs=1e-9,
        )


def test_flat_grid_charge_then_local_discharge_cycle_loses_tiebreak() -> None:
    """Reject a zero-spread 1 kWh grid-charge/local-discharge cycle.

    Two 60-minute slots both import at 1.0, R=1.0, efficiencies are 100%,
    wear is zero, and the battery starts empty. Charging 1.0 DC kWh in slot 0
    only to serve the 1.0 kWh slot-1 load has the same economic cost as direct
    grid supply, but its +0.5e-5 action tiebreak is worse. All battery flows
    therefore remain zero and slot 1 imports exactly 1.0 AC kWh.
    """
    solved = solve_milp(
        [
            _slot(0, import_price=1.0, export_price=0.0),
            _slot(1, import_price=1.0, export_price=0.0, house_kwh=1.0),
        ],
        _NOW,
        current_kwh=0.0,
        usable_kwh=1.0,
        max_charge_per_slot=1.0,
        max_discharge_per_slot=1.0,
        charge_efficiency_pct=100.0,
        discharge_efficiency_pct=100.0,
        cycle_cost_per_kwh=0.0,
        replacement_price_per_kwh=1.0,
        time_discount_rate=1.0,
    )
    assert solved is not None
    planned, diagnostics = solved

    assert sum(slot.batteries_charged_kwh for slot in planned) == pytest.approx(
        0.0,
        abs=0.001,
    )
    assert sum(slot.batteries_discharged_kwh for slot in planned) == pytest.approx(
        0.0,
        abs=0.001,
    )
    assert planned[1].grid_import_kwh == pytest.approx(1.0, abs=0.001)
    assert diagnostics["primary_action_tiebreak"] == pytest.approx(0.0, abs=1e-9)
    assert diagnostics["grid_import_export_overlap_max_kwh"] == pytest.approx(
        0.0,
        abs=1e-9,
    )


def test_fixed_ev_load_has_no_raw_grid_import_export_overlap() -> None:
    """Keep a fixed 5 kWh EV on grid while Huawei serves only 1 kWh house.

    One 60-minute slot contains 6.0 AC kWh measured load: 5.0 kWh accounted
    EV demand and 1.0 kWh house demand. With 97% discharge efficiency the
    EV guard permits at most 1/0.97=1.031 DC kWh Huawei discharge. The raw
    MILP must import more than 4 kWh, export zero, and report zero simultaneous
    grid-direction overlap; publication-time netting is not the proof.
    """
    slot = _slot(0, import_price=1.0, export_price=0.5, house_kwh=6.0)
    slot.ev_accounted_load_kwh = 5.0
    slot.ev_total_planned_load_kwh = 5.0
    solved = solve_milp(
        [slot],
        _NOW,
        current_kwh=10.0,
        usable_kwh=10.0,
        max_charge_per_slot=5.0,
        max_discharge_per_slot=5.0,
        charge_efficiency_pct=97.0,
        discharge_efficiency_pct=97.0,
        cycle_cost_per_kwh=0.0,
        replacement_price_per_kwh=None,
        time_discount_rate=1.0,
        ev_configs=None,
    )
    assert solved is not None
    planned, diagnostics = solved
    out = planned[0]

    assert out.batteries_discharged_kwh <= 1.0 / 0.97 + 0.01
    assert out.grid_import_kwh > 4.0
    assert out.grid_export_kwh == pytest.approx(0.0, abs=0.001)
    assert diagnostics["grid_import_export_overlap_max_kwh"] == pytest.approx(
        0.0,
        abs=1e-9,
    )
    assert "grid_flow_mode" in diagnostics["model_integral_blocks"]


def test_site_export_floor_blocks_direct_pv_export() -> None:
    """The hardware-wide floor suppresses PV export as well as battery export."""
    solved = solve_milp(
        [_slot(0, import_price=0.20, export_price=0.10, pv_kwh=1.0)],
        _NOW,
        current_kwh=1.0,
        usable_kwh=1.0,
        max_charge_per_slot=1.0,
        max_discharge_per_slot=0.0,
        min_export_price=0.15,
        battery_export_min_price=0.0,
    )
    assert solved is not None
    planned, _diagnostics = solved
    assert planned[0].grid_export_kwh == pytest.approx(0.0)
    assert planned[0].pv_export_kwh == pytest.approx(0.0)


def test_battery_only_export_floor_does_not_block_direct_pv_export() -> None:
    """The battery floor remains source-specific when the site floor is off."""
    solved = solve_milp(
        [_slot(0, import_price=0.20, export_price=0.10, pv_kwh=1.0)],
        _NOW,
        current_kwh=1.0,
        usable_kwh=1.0,
        max_charge_per_slot=1.0,
        max_discharge_per_slot=0.0,
        min_export_price=0.0,
        battery_export_min_price=0.15,
    )
    assert solved is not None
    planned, _diagnostics = solved
    assert planned[0].grid_export_kwh == pytest.approx(1.0)
    assert planned[0].primary_battery_export_kwh == pytest.approx(0.0)
    assert planned[0].pv_export_kwh == pytest.approx(1.0)
