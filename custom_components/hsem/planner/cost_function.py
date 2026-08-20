"""Plan cost function for the HSEM planner (issues #295, #413).

This module scores a candidate plan (a fully-populated list of
:class:`~custom_components.hsem.models.planner_outputs.PlannedSlot` objects)
and exposes two distinct aggregate numbers:

- :attr:`PlanCostBreakdown.total_cost` — the **real-money outcome** of the
  plan within the horizon.  Sum of grid import cost minus export revenue
  plus battery cycle (depreciation) cost plus round-trip conversion loss
  cost.  Auditable; directly comparable to an electricity bill.
- :attr:`PlanCostBreakdown.score` — the **selector objective**.  Equals
  ``total_cost`` plus every synthetic penalty (SoC guard and grid limit),
  terminal inventory value, and primary-action tiebreak. The candidate
  selector picks the **lowest score**, not lowest money cost.

Cost components
---------------
The cost function aggregates eight independently-tunable terms:

Money terms (sum to ``total_cost``):

1. **Import cost** — energy imported from the grid × the sanitised
   (non-negative) import price.  Negative spot prices are clamped to 0 for
   this term — mirrors ``milp_optimizer.py``'s ``p_imp_obj`` clamp, so a
   negative price is never scored as a profit for importing (issue #655).
2. **Export revenue** — energy exported to the grid × export price
   (negative contribution, i.e. revenue reduces total cost).
3. **Battery conversion loss** — energy lost during a charge/discharge cycle,
   priced at import for local delivery and at export for battery-origin
   grid export.
4. **Battery cycle cost** — depreciation per kWh cycled, derived from the
   battery's purchase price, rated capacity, and expected lifetime cycles.

Selector-only terms (added on top of ``total_cost`` to produce ``score``):

5. **SoC penalties** — quadratic penalty when the end-of-slot SoC is too low
   (below the configured ``min_soc_pct`` guard) or too high (above the
   configured ``max_soc_pct`` guard), multiplied by a configurable weight.
6. **Grid limit penalty** — penalty when grid import or export in any slot
   exceeds the configured grid power limit, proportional to the excess energy.
7. **Terminal inventory value** — uniform replacement value times net
   battery discharge minus charge. Equal discharge/refill cancels regardless
   of slot position, so the term depends only on final inventory.
8. **Primary-action tiebreak** — a microscopic selector-only preference for
   direct local self-consumption over an exact economic tie, while charging,
   battery export, and a charge/discharge cycle remain slightly disfavoured.

All monetary values are in the caller's local currency.

Design constraints
------------------
- **Pure Python, no Home Assistant imports** — testable with plain pytest.
- **Additive, independently-disableable terms** — any weight set to 0 disables
  that penalty without touching the others.
- **Float-safe** — non-finite prices are treated as 0.0 rather than propagating.
- **Immutable input** — slots are *never* mutated; the function is a pure
  read-only scan.
- **Money / selector split** — ``total_cost`` never includes synthetic
  penalties; ``score`` always does.  The selector minimises ``score``.

Backward compatibility
----------------------
:attr:`PlanCostBreakdown.total` is preserved as a deprecated alias for
``score`` so existing code and tests that compared plans by ``.total``
keep selecting the same winner.  New code should use ``total_cost`` (money)
or ``score`` (selector) explicitly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.terminal_cost_to_go import TerminalCostToGo
from custom_components.hsem.planner.cost_helpers import (
    PRIMARY_ACTION_TIEBREAK_COST,
    _resolve_cycle_cost,
)
from custom_components.hsem.planner.cost_types import (  # noqa: F401
    CostWeights,
    PlanCostBreakdown,
)
from custom_components.hsem.planner.plan_comparison import compare_plans
from custom_components.hsem.planner.secondary_cost import SecondaryCostAccumulator
from custom_components.hsem.utils.datetime_utils import utc_key
from custom_components.hsem.utils.logger import log_planner
from custom_components.hsem.utils.misc import clamp_efficiency
from custom_components.hsem.utils.recommendations import Recommendations
from custom_components.hsem.utils.units import hours_ahead

# Re-export CostWeights and PlanCostBreakdown so existing importers don't break.
__all__ = ["CostWeights", "PlanCostBreakdown", "compare_plans", "score_plan"]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_plan(
    slots: Sequence[PlannedSlot],
    weights: CostWeights | None = None,
    *,
    slot_duration_hours: float = 1.0,
    grid_limit_kw: float | None = None,
    now: datetime | None = None,
    initial_battery_kwh: float | None = None,
    replacement_price_per_kwh: float | None = None,
    terminal_cost_to_go: TerminalCostToGo | None = None,
) -> PlanCostBreakdown:
    """Score a candidate plan and return a full cost breakdown.

    This is a **pure read-only function** — the slot list is never mutated.
    Non-finite price values are treated as ``0.0`` to avoid silent propagation.

    The grid limit can be passed either via ``weights.grid_limit_kw`` or via
    the keyword argument ``grid_limit_kw``; the keyword argument takes
    precedence when not ``None``.

    Past slots are skipped entirely.  When *now* is provided a slot is
    considered past when ``slot.end <= now``.  When *now* is ``None`` the
    function falls back to checking
    ``slot.recommendation == Recommendations.TimePassed.value``, which is
    the sentinel written by the slot-population step on completed slots.
    Either way, including past slots in the SoC-guard penalty would
    generate a false ``soc_low_penalty`` because the simulator zeros
    ``estimated_battery_soc_pct`` on past slots as a sentinel value.

    Two aggregate numbers are returned (issue #413):

    - ``total_cost`` — money outcome only.  Equals
      ``import_cost − export_revenue + cycle_cost + conversion_loss_cost``.
    - ``score`` — selector objective.  Equals ``total_cost`` plus all
      synthetic penalties (SoC guard and grid limit), the terminal inventory
      value, and ``primary_action_tiebreak``. The candidate selector
      minimises this value.

    Terminal inventory accounting is enabled when ``initial_battery_kwh``
    and either the bounded ``terminal_cost_to_go`` model or the legacy
    uniform replacement price are provided. Production primary planning uses
    F(initial inventory) - F(final inventory), where F is quantity-capped. This
    prevents a candidate looking better merely by consuming justified valued
    tier inventory at horizon end. With no tiers, the hard discharge floor is
    the only terminal protection. The scalar path remains for compatible
    callers.

    Args:
        slots:
            Ordered list of :class:`PlannedSlot` objects representing one
            candidate plan.  Typically the ``slots`` field of a
            :class:`~custom_components.hsem.models.planner_outputs.PlannerOutput`.
        weights:
            Cost weights and configuration.  Defaults to
            :class:`CostWeights` with all-default values when ``None``.
        slot_duration_hours:
            Duration of each slot in hours.  Used to convert per-slot energy
            (kWh) to power (kW) for the grid-limit check.  Defaults to 1.0
            (hourly slots).
        grid_limit_kw:
            Override for the grid power limit in kW.  When provided, it
            supersedes ``weights.grid_limit_kw``.  ``None`` leaves the
            weights value unchanged.
        now:
            Timezone-aware current datetime.  When provided, any slot whose
            ``end`` is at or before *now* is skipped.  When ``None`` the
            fallback sentinel check (``recommendation == TimePassed``) is
            used instead.
        initial_battery_kwh:
            Energy stored above the discharge floor (kWh) at the start of
            the horizon. Required whenever either terminal-value policy is
            enabled.
            ``None`` disables the term.
        replacement_price_per_kwh:
            Legacy uniform value per kWh of stored inventory. Ignored when
            ``terminal_cost_to_go`` is supplied.
        terminal_cost_to_go:
            Bounded piecewise value of primary inventory at the actionable
            boundary. Applied as F(initial inventory) - F(final inventory);
            an empty model contributes zero economic terminal value.

    Returns:
        A :class:`PlanCostBreakdown` containing every cost component, the
        money ``total_cost``, and the selector ``score``.
        **Lower ``score`` = better plan** (this is what the selector
        minimises).

    Examples:
        >>> from datetime import datetime
        >>> from zoneinfo import ZoneInfo
        >>> from custom_components.hsem.models.planner_outputs import PlannedSlot
        >>> from custom_components.hsem.utils.prices import SlotPrice
        >>> tz = ZoneInfo("Europe/Copenhagen")
        >>> start = datetime(2024, 6, 15, 0, 0, tzinfo=tz)
        >>> from datetime import timedelta
        >>> slot = PlannedSlot(
        ...     start=start,
        ...     end=start + timedelta(hours=1),
        ...     price=SlotPrice(import_price=0.20, export_price=0.05),
        ...     grid_import_kwh=1.0,
        ...     grid_export_kwh=0.0,
        ...     estimated_battery_soc=50.0,
        ... )
        >>> bd = score_plan([slot])
        >>> bd.import_cost
        0.2
        >>> bd.total_cost
        0.2
        >>> bd.score
        0.2
        >>> bd.total  # deprecated alias for score
        0.2
    """
    if weights is None:
        weights = CostWeights()

    log_planner(
        "debug",
        "[cost] score_plan  slots=%d  initial_battery=%s  repl_price=%s",
        len(slots),
        f"{initial_battery_kwh:.3f}" if initial_battery_kwh is not None else "None",
        (
            f"{replacement_price_per_kwh:.6f}"
            if replacement_price_per_kwh is not None
            else "None"
        ),
    )

    # Resolve grid limit (keyword arg takes precedence)
    effective_grid_limit_kw: float | None = (
        grid_limit_kw if grid_limit_kw is not None else weights.grid_limit_kw
    )

    cycle_cost_kwh = _resolve_cycle_cost(weights)

    # Resolve the effective roundtrip loss fraction.
    # When separate charge/discharge efficiencies are provided (both non-default),
    # we compute the roundtrip loss from them:
    #   roundtrip_loss = 1 - (charge_eff × discharge_eff)
    # Compute roundtrip loss from charge/discharge efficiencies.
    charge_eff = clamp_efficiency(weights.charge_efficiency_pct)
    discharge_eff = clamp_efficiency(weights.discharge_efficiency_pct)

    import_cost = 0.0
    export_revenue = 0.0
    conversion_loss_cost = 0.0
    cycle_cost_total = 0.0
    soc_penalty = 0.0
    grid_limit_penalty = 0.0
    terminal_soc_value = 0.0
    primary_action_tiebreak = 0.0
    primary_inventory_change_kwh = 0.0
    secondary_costs = SecondaryCostAccumulator()

    # Discounted versions for the selector score (total_cost stays raw).
    # time_discount_rate < 1.0 means future savings are worth less.
    discount_rate = weights.time_discount_rate
    use_discount = discount_rate < 1.0 - 1e-9 and now is not None
    import_cost_disc = 0.0
    export_revenue_disc = 0.0
    conversion_loss_cost_disc = 0.0
    cycle_cost_total_disc = 0.0
    soc_penalty_disc = 0.0
    grid_limit_penalty_disc = 0.0

    _time_passed_value = Recommendations.TimePassed.value

    for slot in slots:
        # Skip past slots entirely.  The SoC simulation zeros
        # estimated_battery_soc_pct on past slots as a sentinel, which would
        # falsely trigger the SoC-low penalty on every past slot.
        # Energy-flow fields (grid_import_kwh, grid_export_kwh, etc.) are
        # no longer zeroed for past slots (they are preserved for the daily
        # plan-vs-actual tracker), but skipping past slots here has no
        # effect on import cost, cycle cost, or any other term since they
        # belong to a completed time period.
        #
        # Primary guard: slot.end <= now (time-based, no string coupling).
        # Fallback guard: recommendation == TimePassed (used when now is None,
        # e.g. in unit tests that call score_plan without a clock).
        if now is not None:
            if utc_key(slot.end) <= utc_key(now):
                continue
        elif slot.recommendation == _time_passed_value:
            continue

        # Compute time discount for this slot.
        # discount = discount_rate ^ hours_from_now
        # Past slots are already skipped above, so hours_ahead >= 0.
        if use_discount:
            assert (
                now is not None
            )  # guarded by use_discount = discount_rate < 1.0 and now is not None
            start_utc = utc_key(slot.start)
            slot_mid = start_utc + (utc_key(slot.end) - start_utc) / 2
            hours_ahead_val = hours_ahead(now, slot_mid)
            discount = discount_rate**hours_ahead_val
        else:
            discount = 1.0

        imp_price = slot.price.import_price
        exp_price = slot.price.export_price

        # Treat all non-finite values as zero.  Directly constructed planner
        # inputs must fail closed just like source-populated inputs.
        if not math.isfinite(imp_price):
            imp_price = 0.0
        if not math.isfinite(exp_price):
            exp_price = 0.0

        # A numeric value outside the contiguous published-price prefix is
        # diagnostic data, not economic authority.  Neutralise every monetary
        # use while leaving physical cycle, SoC, and grid penalties intact.
        if not slot.price_actionable:
            imp_price = 0.0
            exp_price = 0.0

        # Sanitised (non-negative) import price — mirrors milp_optimizer.py's
        # p_imp_obj clamp.  A negative spot price must never be scored as a
        # profit for importing or for lossy conversion: the MILP's own
        # objective never rewards those events (its gi[t]/ec[t]/ed[t]
        # coefficients use p_imp_obj = max(p_imp, 0)), so the selector must
        # value the identical physical decisions the same way, or its score
        # no longer matches what the LP actually optimised for (issue #655).
        # The raw (possibly negative) imp_price is still used for export
        # clamping logic elsewhere and is unaffected by this sanitisation.
        imp_price_obj = max(imp_price, 0.0)

        # Production plans carry an explicit source split. Keep a bounded
        # fallback for older callers that construct aggregate-only slots.
        grid_export = max(slot.grid_export_kwh, 0.0)
        explicit_primary = max(slot.primary_battery_export_kwh, 0.0)
        explicit_pv = max(slot.pv_export_kwh, 0.0)
        if abs(explicit_primary + explicit_pv - grid_export) <= 0.002:
            primary_export_ac = min(explicit_primary, grid_export)
            pv_export_ac = max(grid_export - primary_export_ac, 0.0)
        else:
            primary_export_ac = min(
                grid_export,
                max(slot.batteries_discharged_kwh * discharge_eff, 0.0),
            )
            pv_export_ac = max(grid_export - primary_export_ac, 0.0)
        primary_export_dc = min(
            max(slot.batteries_discharged_kwh, 0.0),
            primary_export_ac / discharge_eff,
        )
        local_discharge_dc = max(
            slot.batteries_discharged_kwh - primary_export_dc,
            0.0,
        )
        primary_action_tiebreak += PRIMARY_ACTION_TIEBREAK_COST * (
            slot.batteries_charged_kwh
            + slot.batteries_discharged_kwh
            - 1.5 * local_discharge_dc
        )
        # 1. Import cost — grid_import_kwh already reflects the extra grid draw
        #    needed to store energy through the charge efficiency (i.e. the
        #    simulation writes grid_import_kwh = charge_stored / charge_eff).
        if slot.grid_import_kwh > 1e-9:
            cost = slot.grid_import_kwh * imp_price_obj
            import_cost += cost
            import_cost_disc += cost * discount

        # 2. Export revenue. The site floor applies to both sources; the
        # battery floor applies only to intentional primary-battery export.
        effective_exp_price = exp_price
        if (
            weights.export_min_price > 1e-9
            and effective_exp_price < weights.export_min_price
        ):
            effective_exp_price = 0.0
        battery_exp_price = effective_exp_price
        if (
            weights.battery_export_min_price > 1e-9
            and exp_price < weights.battery_export_min_price
        ):
            battery_exp_price = 0.0
        if grid_export > 1e-9:
            rev = (
                pv_export_ac * effective_exp_price
                + primary_export_ac * battery_exp_price
            )
            export_revenue += rev
            export_revenue_disc += rev * discount

        # 3. Conversion losses use the same source/destination split as MILP:
        # local discharge loss is valued at import, exported loss at export.
        charge_loss_fraction = 1.0 - charge_eff
        discharge_loss_fraction = 1.0 - discharge_eff
        if slot.batteries_charged_kwh > 1e-9 and charge_loss_fraction > 1e-9:
            lost_kwh_charge = slot.batteries_charged_kwh * charge_loss_fraction
            conv = lost_kwh_charge * imp_price_obj
            conversion_loss_cost += conv
            conversion_loss_cost_disc += conv * discount
        if slot.batteries_discharged_kwh > 1e-9 and discharge_loss_fraction > 1e-9:
            lost_local = local_discharge_dc * discharge_loss_fraction
            lost_export = primary_export_dc * discharge_loss_fraction
            conv = lost_local * imp_price_obj + lost_export * max(
                battery_exp_price, 0.0
            )
            conversion_loss_cost += conv
            conversion_loss_cost_disc += conv * discount

        # 4. Battery cycle depreciation
        throughput_kwh = max(slot.batteries_charged_kwh, slot.batteries_discharged_kwh)
        if throughput_kwh > 1e-9 and cycle_cost_kwh > 1e-9:
            cycle = throughput_kwh * cycle_cost_kwh
            cycle_cost_total += cycle
            cycle_cost_total_disc += cycle * discount

        if weights.secondary_storage_enabled:
            secondary_conv, secondary_cycle, secondary_terminal = (
                secondary_costs.add_slot(
                    slot,
                    weights,
                    import_price=imp_price,
                    export_price=exp_price,
                )
            )
            conversion_loss_cost += secondary_conv
            conversion_loss_cost_disc += secondary_conv * discount
            cycle_cost_total += secondary_cycle
            cycle_cost_total_disc += secondary_cycle * discount
            terminal_soc_value += secondary_terminal

        # 5. SoC guard penalties (quadratic in the violation magnitude).
        soc = slot.estimated_battery_soc_pct
        if soc < weights.min_soc_pct:
            violation = weights.min_soc_pct - soc
            pen = weights.soc_low_penalty_weight * violation**2
            soc_penalty += pen
            soc_penalty_disc += pen * discount
        elif soc > weights.max_soc_pct:
            violation = soc - weights.max_soc_pct
            pen = weights.soc_high_penalty_weight * violation**2
            soc_penalty += pen
            soc_penalty_disc += pen * discount

        # 6. Grid limit penalty
        if effective_grid_limit_kw is not None and slot_duration_hours > 1e-9:
            import_kw = slot.grid_import_kwh / slot_duration_hours
            export_kw = slot.grid_export_kwh / slot_duration_hours
            for kw in (import_kw, export_kw):
                excess_kw = kw - effective_grid_limit_kw
                if excess_kw > 1e-9:
                    pen = (
                        excess_kw
                        * slot_duration_hours
                        * weights.grid_limit_penalty_per_kwh
                    )
                    grid_limit_penalty += pen
                    grid_limit_penalty_disc += pen * discount

        if slot.price_actionable:
            primary_inventory_change_kwh += (
                slot.batteries_charged_kwh - slot.batteries_discharged_kwh
            )

        # 7. Legacy uniform terminal inventory value.
        if (
            terminal_cost_to_go is None
            and initial_battery_kwh is not None
            and replacement_price_per_kwh is not None
            and slot.price_actionable
        ):
            replacement_value = (
                max(replacement_price_per_kwh, 0.0)
                if math.isfinite(replacement_price_per_kwh)
                else 0.0
            )
            terminal_soc_value += (
                slot.batteries_discharged_kwh - slot.batteries_charged_kwh
            ) * replacement_value

    if initial_battery_kwh is not None and terminal_cost_to_go is not None:
        final_inventory_kwh = max(
            initial_battery_kwh + primary_inventory_change_kwh,
            0.0,
        )
        terminal_soc_value += terminal_cost_to_go.inventory_value(
            initial_battery_kwh
        ) - terminal_cost_to_go.inventory_value(final_inventory_kwh)

    # ``total_cost`` is money only — never includes synthetic penalties.
    total_cost = import_cost - export_revenue + conversion_loss_cost + cycle_cost_total

    # ``score`` is the selector objective.  It uses discounted values when
    # time_discount_rate < 1.0 so that uncertain distant savings are weighted
    # less than near-term certain savings.  ``total_cost`` is always raw
    # (undiscounted) so it remains auditable as real money.
    if use_discount:
        score = (
            import_cost_disc
            - export_revenue_disc
            + conversion_loss_cost_disc
            + cycle_cost_total_disc
            + soc_penalty_disc
            + grid_limit_penalty_disc
            + terminal_soc_value
            + primary_action_tiebreak
        )
    else:
        score = (
            total_cost
            + soc_penalty
            + grid_limit_penalty
            + terminal_soc_value
            + primary_action_tiebreak
        )

    score_rounded = round(score, 6)

    result = PlanCostBreakdown(
        import_cost=round(import_cost, 6),
        export_revenue=round(export_revenue, 6),
        conversion_loss_cost=round(conversion_loss_cost, 6),
        cycle_cost=round(cycle_cost_total, 6),
        soc_penalty=round(soc_penalty, 6),
        grid_limit_penalty=round(grid_limit_penalty, 6),
        terminal_soc_value=round(terminal_soc_value, 6),
        primary_action_tiebreak=round(primary_action_tiebreak, 6),
        secondary_conversion_loss_cost=round(
            secondary_costs.conversion_loss_cost,
            6,
        ),
        secondary_cycle_cost=round(secondary_costs.cycle_cost, 6),
        secondary_terminal_soc_value=round(
            secondary_costs.terminal_soc_value,
            6,
        ),
        total_cost=round(total_cost, 6),
        score=score_rounded,
        # ``total`` is a deprecated alias for ``score`` (issue #413).
        total=score_rounded,
    )

    log_planner(
        "debug",
        "[cost] score_plan DONE  total_cost=%.6f  score=%.6f  "
        "import=%.6f  export_rev=%.6f  conv_loss=%.6f  "
        "cycle=%.6f  soc_pen=%.6f  grid=%.6f  term_soc=%.6f  tie=%.6f",
        result.total_cost,
        result.score,
        result.import_cost,
        result.export_revenue,
        result.conversion_loss_cost,
        result.cycle_cost,
        result.soc_penalty,
        result.grid_limit_penalty,
        result.terminal_soc_value,
        result.primary_action_tiebreak,
    )

    return result
