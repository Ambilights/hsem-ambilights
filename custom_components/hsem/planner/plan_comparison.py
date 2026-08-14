"""Comparison helper for scored HSEM candidate plans."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.planner.cost_types import CostWeights, PlanCostBreakdown
from custom_components.hsem.utils.logger import log_planner


def compare_plans(
    plan_a: Sequence[PlannedSlot],
    plan_b: Sequence[PlannedSlot],
    weights: CostWeights | None = None,
    *,
    slot_duration_hours: float = 1.0,
    now: datetime | None = None,
    initial_battery_kwh: float | None = None,
    replacement_price_per_kwh: float | None = None,
) -> tuple[PlanCostBreakdown, PlanCostBreakdown, str]:
    """Score two candidate plans and return which one wins.

    The winner is the plan with the lower selector score.  Scores within
    ``1e-9`` are reported as a tie.  ``score_plan`` is imported lazily to
    keep this helper independent while ``cost_function`` re-exports it.
    """
    from custom_components.hsem.planner.cost_function import score_plan

    bd_a = score_plan(
        plan_a,
        weights,
        slot_duration_hours=slot_duration_hours,
        now=now,
        initial_battery_kwh=initial_battery_kwh,
        replacement_price_per_kwh=replacement_price_per_kwh,
    )
    bd_b = score_plan(
        plan_b,
        weights,
        slot_duration_hours=slot_duration_hours,
        now=now,
        initial_battery_kwh=initial_battery_kwh,
        replacement_price_per_kwh=replacement_price_per_kwh,
    )

    diff = bd_a.score - bd_b.score
    if abs(diff) < 1e-9:
        winner = "tie"
    elif diff < 0:
        winner = "plan_a"
    else:
        winner = "plan_b"

    log_planner(
        "debug",
        "[cost] compare_plans  a_score=%.6f  b_score=%.6f  diff=%.6f  winner=%s",
        bd_a.score,
        bd_b.score,
        diff,
        winner,
    )

    return bd_a, bd_b, winner
