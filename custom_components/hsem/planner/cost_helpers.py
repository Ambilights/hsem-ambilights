"""Cost-function helpers for grid cash flow and battery valuation."""

from __future__ import annotations

import math

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.planner.cost_types import CostWeights
from custom_components.hsem.utils.logger import log_planner
from custom_components.hsem.utils.misc import resolve_cycle_cost
from custom_components.hsem.utils.units import usable_kwh_from_rated

PRIMARY_ACTION_TIEBREAK_COST = 1e-5


def resolve_slot_money_prices(
    import_price: float,
    export_price: float,
    *,
    price_actionable: bool,
    export_min_price: float = 0.0,
) -> tuple[float, float]:
    """Return finite authoritative import/export rates for money accounting.

    Finite negative prices are real market signals and are preserved. A
    non-actionable slot has no price authority, while the configured site
    export floor values physically blocked export at zero without changing
    the raw diagnostic price stored on the slot.
    """
    if not price_actionable:
        return (0.0, 0.0)

    effective_import = import_price if math.isfinite(import_price) else 0.0
    effective_export = export_price if math.isfinite(export_price) else 0.0
    if export_min_price > 1e-9 and effective_export < export_min_price:
        effective_export = 0.0
    return (effective_import, effective_export)


def grid_cash_flow_cost(
    grid_import_kwh: float,
    grid_export_kwh: float,
    import_price: float,
    export_price: float,
    *,
    price_actionable: bool,
    export_min_price: float = 0.0,
) -> float:
    """Return auditable meter cash flow; positive is cost, negative income."""
    effective_import, effective_export = resolve_slot_money_prices(
        import_price,
        export_price,
        price_actionable=price_actionable,
        export_min_price=export_min_price,
    )
    return (
        max(grid_import_kwh, 0.0) * effective_import
        - max(grid_export_kwh, 0.0) * effective_export
    )


def slot_grid_cash_flow_cost(
    slot: PlannedSlot,
    *,
    export_min_price: float = 0.0,
) -> float:
    """Return one slot's auditable meter cash flow from published grid flows.

    Positive values are a net cost and negative values are net income. This
    deliberately excludes battery wear and every selector-only term; those
    remain separately itemised by :func:`score_plan`.
    """
    return grid_cash_flow_cost(
        slot.grid_import_kwh,
        slot.grid_export_kwh,
        slot.price.import_price,
        slot.price.export_price,
        price_actionable=slot.price_actionable,
        export_min_price=export_min_price,
    )


# ---------------------------------------------------------------------------
# Cycle cost helper
# ---------------------------------------------------------------------------


def _resolve_cycle_cost(weights: CostWeights) -> float:
    """Return the battery cycle depreciation cost per kWh cycled.

    Uses usable capacity (rated × DoD fraction) in the denominator, not
    rated capacity, because battery degradation is driven by cycling within
    the usable SoC range.

    The ``2×`` factor in the denominator accounts for the fact that one full
    battery cycle involves energy flow in *both* directions::

        throughput_per_cycle = 2 × usable_kwh
                              (charge once + discharge once)

    Since ``purchase_price / expected_cycles`` is the cost *per full cycle*
    and the cycle cost is expressed *per kWh of throughput*, the cost must
    be spread over the total lifetime throughput:

        cycle_cost_per_kwh = purchase_price / expected_cycles / (2 × usable_kwh)

    This is mathematically equivalent to:

        purchase_price / (2 × usable_kwh × expected_cycles)

    When ``weights.cycle_cost_per_kwh`` is explicitly set (not ``None``), that
    value is used directly — the caller is responsible for resolving auto vs.
    user margin.  When ``None``, the value is auto-calculated from the battery
    economics fields.  Returns 0.0 when any required value is non-positive
    or missing.

    Args:
        weights: Configuration object from which to resolve the cost.

    Returns:
        Depreciation cost in local currency per kWh.
    """
    if weights.cycle_cost_per_kwh is not None:
        result = weights.cycle_cost_per_kwh
        log_planner(
            "debug",
            "[cost] _resolve_cycle_cost  explicit=%.6f",
            result,
        )
        return result

    if (
        weights.battery_purchase_price > 1e-9
        and weights.battery_rated_capacity_kwh > 1e-9
        and weights.battery_expected_cycles > 0
    ):
        usable_kwh = usable_kwh_from_rated(
            weights.battery_rated_capacity_kwh,
            weights.min_soc_pct,
            weights.max_soc_pct,
        )
        if usable_kwh < 1e-9:
            usable_kwh = weights.battery_rated_capacity_kwh
        result = resolve_cycle_cost(
            purchase_price=weights.battery_purchase_price,
            usable_kwh=usable_kwh,
            expected_cycles=weights.battery_expected_cycles,
            capacity_loss_pct=weights.battery_capacity_loss_pct,
        )
        log_planner(
            "debug",
            "[cost] _resolve_cycle_cost  purchase=%.2f  usable=%.3f  cycles=%d  "
            "cycle_cost=%.6f",
            weights.battery_purchase_price,
            usable_kwh,
            weights.battery_expected_cycles,
            result,
        )
        return result

    log_planner("debug", "[cost] _resolve_cycle_cost  return 0 (insufficient data)")
    return 0.0
