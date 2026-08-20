"""Regression tests for conditional primary-battery export reserve checkpoints."""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.planner.cost_function import (
    CostWeights,
    PlanCostBreakdown,
    score_plan,
)
from custom_components.hsem.planner.milp._export_reserve import (
    _next_solar_refill_checkpoints,
)
from custom_components.hsem.planner.milp_optimizer import (
    is_scipy_available,
    solve_milp,
)
from custom_components.hsem.utils.prices import SlotPrice

_TZ = ZoneInfo("Europe/Stockholm")
_NOW = datetime(2024, 6, 15, 0, 0, tzinfo=_TZ)


def _make_slot(
    index: int,
    *,
    import_price: float,
    export_price: float,
    consumption_kwh: float,
    pv_kwh: float,
    interval_minutes: int = 60,
) -> PlannedSlot:
    """Build one fully actionable optimizer slot."""
    start = _NOW + timedelta(minutes=index * interval_minutes)
    slot = PlannedSlot(
        start=start,
        end=start + timedelta(minutes=interval_minutes),
        price=SlotPrice(import_price=import_price, export_price=export_price),
    )
    slot.avg_house_consumption_kwh = consumption_kwh
    slot.solcast_pv_estimate_kwh = pv_kwh
    slot.estimated_net_consumption_kwh = consumption_kwh - pv_kwh
    return slot


@pytest.mark.parametrize(
    ("pv_avail", "expected"),
    [
        pytest.param(
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0, 4, 4, 4, 4, 6, 6],
            id="contiguous-runs",
        ),
        pytest.param(
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0, 2, 2, 4, 4],
            id="separated-runs",
        ),
        pytest.param(
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [1, 1, 4, 4, 4],
            id="isolated-surplus",
        ),
        pytest.param(
            [1.0, 1.0, 1.0],
            [2, 2, 2],
            id="all-surplus",
        ),
        pytest.param(
            [0.0, 0.0, 0.0],
            [2, 2, 2],
            id="no-surplus",
        ),
    ],
)
def test_next_solar_refill_checkpoints_group_contiguous_surplus_runs(
    pv_avail: list[float], expected: list[int]
) -> None:
    """Every slot in one physical surplus run shares its later checkpoint."""
    checkpoints = _next_solar_refill_checkpoints(np.asarray(pv_avail, dtype=float))

    assert checkpoints.tolist() == expected


def _solve_adjacent_surplus_case(
    *,
    following_demand_kwh: float,
    no_export: bool = False,
) -> tuple[list[PlannedSlot], dict[str, Any]]:
    """Solve three 1 h slots with kWh energy and currency/kWh prices.

    The battery starts with 5 kWh of usable energy. The first two slots each
    have 0.26 kWh direct-PV surplus; the third has the caller-provided demand.
    Charge/discharge efficiency is 100%, cycle wear is zero, and a 20% export
    buffer requires 1 kWh at the shared end-of-demand checkpoint.
    """
    slots = [
        _make_slot(
            0,
            import_price=2.90,
            export_price=2.90,
            consumption_kwh=0.0,
            pv_kwh=0.26,
        ),
        _make_slot(
            1,
            import_price=3.00,
            export_price=3.00,
            consumption_kwh=0.0,
            pv_kwh=0.26,
        ),
        _make_slot(
            2,
            import_price=10.0,
            export_price=0.0,
            consumption_kwh=following_demand_kwh,
            pv_kwh=0.0,
        ),
    ]

    result = solve_milp(
        slots,
        _NOW,
        current_kwh=5.0,
        usable_kwh=5.0,
        max_charge_per_slot=0.01,
        max_discharge_per_slot=5.0,
        charge_efficiency_pct=100.0,
        discharge_efficiency_pct=100.0,
        excess_export_discharge_buffer_pct=20.0,
        no_export=no_export,
    )

    assert result is not None
    return result


def _battery_trajectory(planned: list[PlannedSlot]) -> list[float]:
    """Return end-of-slot battery energy from the known 5 kWh initial state."""
    energy_kwh = 5.0
    trajectory: list[float] = []
    for slot in planned:
        energy_kwh += slot.batteries_charged_kwh - slot.batteries_discharged_kwh
        trajectory.append(energy_kwh)
    return trajectory


def _money_cost(planned: list[PlannedSlot]) -> PlanCostBreakdown:
    """Score authoritative money flows at 100% efficiency and zero wear."""
    return score_plan(
        planned,
        CostWeights(
            cycle_cost_per_kwh=0.0,
            charge_efficiency_pct=100.0,
            discharge_efficiency_pct=100.0,
        ),
        slot_duration_hours=1.0,
    )


@pytest.mark.skipif(
    not is_scipy_available(), reason="scipy not available in this environment"
)
def test_adjacent_surplus_slots_cannot_bypass_shared_export_reserve() -> None:
    """A 4 kWh demand yields 5→5→1 kWh and only direct-PV revenue."""
    planned, diagnostics = _solve_adjacent_surplus_case(following_demand_kwh=4.0)
    money = _money_cost(planned)

    assert diagnostics["battery_export_reserve_active"] is True
    assert _battery_trajectory(planned) == pytest.approx([5.0, 5.0, 1.0])
    assert [slot.batteries_charged_kwh for slot in planned] == pytest.approx(
        [0.0, 0.0, 0.0]
    )
    assert [slot.batteries_discharged_kwh for slot in planned] == pytest.approx(
        [0.0, 0.0, 4.0]
    )
    assert [slot.primary_battery_export_kwh for slot in planned] == pytest.approx(
        [0.0, 0.0, 0.0]
    )
    assert [slot.pv_export_kwh for slot in planned] == pytest.approx([0.26, 0.26, 0.0])
    assert [slot.grid_export_kwh for slot in planned] == pytest.approx(
        [0.26, 0.26, 0.0]
    )
    assert [slot.grid_import_kwh for slot in planned] == pytest.approx([0.0, 0.0, 0.0])
    # Revenue = 0.26×2.90 + 0.26×3.00 = 1.534 currency; no import or loss.
    assert money.import_cost == pytest.approx(0.0)
    assert money.export_revenue == pytest.approx(1.534)
    assert money.conversion_loss_cost == pytest.approx(0.0)
    assert money.cycle_cost == pytest.approx(0.0)
    assert money.total_cost == pytest.approx(-1.534)


@pytest.mark.skipif(
    not is_scipy_available(), reason="scipy not available in this environment"
)
def test_adjacent_surplus_slots_export_at_higher_peak_when_reserve_is_feasible() -> (
    None
):
    """A 1 kWh demand yields 5→2→1 kWh with 3 kWh exported at the peak."""
    planned, diagnostics = _solve_adjacent_surplus_case(following_demand_kwh=1.0)
    money = _money_cost(planned)

    assert _battery_trajectory(planned) == pytest.approx([5.0, 2.0, 1.0])
    assert [slot.batteries_charged_kwh for slot in planned] == pytest.approx(
        [0.0, 0.0, 0.0]
    )
    assert [slot.batteries_discharged_kwh for slot in planned] == pytest.approx(
        [0.0, 3.0, 1.0]
    )
    assert [slot.primary_battery_export_kwh for slot in planned] == pytest.approx(
        [0.0, 3.0, 0.0]
    )
    assert [slot.pv_export_kwh for slot in planned] == pytest.approx([0.26, 0.26, 0.0])
    assert [slot.grid_export_kwh for slot in planned] == pytest.approx(
        [0.26, 3.26, 0.0]
    )
    assert [slot.grid_import_kwh for slot in planned] == pytest.approx([0.0, 0.0, 0.0])
    assert diagnostics[
        "battery_export_reserve_min_checkpoint_soc_kwh"
    ] == pytest.approx(1.0, abs=1e-5)
    # Revenue = 0.26×2.90 + (3.00+0.26)×3.00 = 10.534 currency.
    assert money.import_cost == pytest.approx(0.0)
    assert money.export_revenue == pytest.approx(10.534)
    assert money.conversion_loss_cost == pytest.approx(0.0)
    assert money.cycle_cost == pytest.approx(0.0)
    assert money.total_cost == pytest.approx(-10.534)


@pytest.mark.skipif(
    not is_scipy_available(), reason="scipy not available in this environment"
)
def test_grouped_export_reserve_does_not_reclassify_direct_pv_export() -> None:
    """No-export yields 5→5→4 kWh while preserving direct-PV revenue."""
    planned, _diagnostics = _solve_adjacent_surplus_case(
        following_demand_kwh=1.0,
        no_export=True,
    )
    money = _money_cost(planned)

    assert _battery_trajectory(planned) == pytest.approx([5.0, 5.0, 4.0])
    assert [slot.batteries_discharged_kwh for slot in planned] == pytest.approx(
        [0.0, 0.0, 1.0]
    )
    assert [slot.primary_battery_export_kwh for slot in planned] == pytest.approx(
        [0.0, 0.0, 0.0]
    )
    assert [slot.pv_export_kwh for slot in planned] == pytest.approx([0.26, 0.26, 0.0])
    assert [slot.grid_export_kwh for slot in planned] == pytest.approx(
        [0.26, 0.26, 0.0]
    )
    assert [slot.grid_import_kwh for slot in planned] == pytest.approx([0.0, 0.0, 0.0])
    assert money.import_cost == pytest.approx(0.0)
    assert money.export_revenue == pytest.approx(1.534)
    assert money.total_cost == pytest.approx(-1.534)


@pytest.mark.skipif(
    not is_scipy_available(), reason="scipy not available in this environment"
)
@pytest.mark.parametrize(
    ("slot_count", "performance_budget_seconds"),
    [
        pytest.param(96, 5.0, id="96-slots"),
        pytest.param(192, 15.0, id="192-slots"),
    ],
)
def test_reserve_active_milp_scales_across_supported_horizons(
    slot_count: int, performance_budget_seconds: float
) -> None:
    """Reserve-active 15-minute MILPs stay practical at 24 h and 48 h."""
    slots: list[PlannedSlot] = []
    for index in range(slot_count):
        daily_phase = index % 96
        pv_kwh = max(0.0, 0.45 * math.sin((daily_phase - 24) * math.pi / 48))
        import_price = 0.25 + 0.15 * abs(math.sin(index * math.pi / 48))
        slots.append(
            _make_slot(
                index,
                import_price=import_price,
                export_price=import_price * 0.8,
                consumption_kwh=0.15,
                pv_kwh=pv_kwh,
                interval_minutes=15,
            )
        )

    started = time.perf_counter()
    result = solve_milp(
        slots,
        _NOW,
        current_kwh=5.0,
        usable_kwh=9.0,
        max_charge_per_slot=1.25,
        max_discharge_per_slot=1.25,
        excess_export_discharge_buffer_pct=15.0,
    )
    elapsed = time.perf_counter() - started

    assert result is not None
    _planned, diagnostics = result
    assert diagnostics["battery_export_reserve_active"] is True
    assert elapsed < performance_budget_seconds, (
        f"reserve-active {slot_count}-slot solve took {elapsed:.3f}s; "
        f"budget is {performance_budget_seconds:.3f}s"
    )
