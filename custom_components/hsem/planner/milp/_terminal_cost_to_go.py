"""MILP extension for bounded primary terminal inventory value."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from custom_components.hsem.models.terminal_cost_to_go import TerminalValueTier

if TYPE_CHECKING:
    from custom_components.hsem.planner.milp._layout import MilpBoundsBuilder


def add_terminal_cost_to_go_constraints(
    constraints: dict[str, Any],
    *,
    bounds_builder: MilpBoundsBuilder,
    n_vars: int,
    m: int,
    ec_off: int,
    ed_off: int,
    terminal_value_off: int,
    tiers: Sequence[TerminalValueTier],
    current_kwh: float,
) -> dict[str, Any]:
    """Bound valued tier allocations by physical final battery inventory."""
    old_a_ub = constraints["A_ub"]
    old_b_ub = constraints["b_ub"]
    old_rows = old_a_ub.shape[0]
    a_ub = np.zeros((old_rows + 1, n_vars))
    b_ub = np.zeros(old_rows + 1)
    a_ub[:old_rows, : old_a_ub.shape[1]] = old_a_ub
    b_ub[:old_rows] = old_b_ub

    # sum(valued tier inventory) <= current + sum(charge - discharge)
    row = old_rows
    for tier_i in range(len(tiers)):
        a_ub[row, terminal_value_off + tier_i] = 1.0
    for t in range(m):
        a_ub[row, ec_off + t] = -1.0
        a_ub[row, ed_off + t] = 1.0
    b_ub[row] = max(current_kwh, 0.0)

    constraints["A_ub"] = a_ub
    constraints["b_ub"] = b_ub
    bounds_builder.set(
        "primary_terminal_inventory",
        [(0.0, max(float(tier.quantity_kwh), 0.0)) for tier in tiers],
    )
    return constraints
