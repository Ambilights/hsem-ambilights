"""Dataclass for a human-readable explanation of the selected plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from homeassistant.const import STATE_UNKNOWN

from custom_components.hsem.models.rejected_plan import RejectedPlan


@dataclass
class PlanExplanation:
    """Human-readable explanation of the plan selected by the HSEM planner.

    Attached to :class:`PlannerOutput` after each planning run.  Suitable for
    surfacing as a Home Assistant sensor attribute so users can understand *why*
    a particular strategy was chosen and what alternatives were rejected.

    Attributes:
        selected_strategy:
            Short identifier for the active strategy
            (e.g. ``"charge_grid_discharge_peak"``).
        summary:
            One-sentence human-readable summary of the selected plan.
        score:
            Estimated savings of the selected plan versus doing nothing
            (battery fully idle) in the actionable published-price window.
            Positive means the plan saves money there; negative means it costs
            more there and may still be selected because bounded post-boundary
            inventory value favours charging without an in-window discharge.
            Units are local currency.
        estimated_total_cost:
            Estimated net grid cost for the planning horizon (local currency).
            Positive = net import cost; negative = net export revenue.
        price_spread:
            Difference between the maximum and minimum import price in the
            planning horizon (local currency/kWh).  A larger spread indicates
            more arbitrage potential.
        peak_import_price:
            Maximum import price seen across all future slots.
        off_peak_import_price:
            Minimum import price seen across all future slots.
        forecast_pv_kwh:
            Total PV production forecast for the planning horizon (kWh).
        forecast_net_consumption_kwh:
            Total estimated net consumption (load minus PV) for the planning
            horizon (kWh).  Negative means net solar surplus.
        battery_soc_pct:
            Battery state-of-charge at the start of the planning run (%).
        battery_soc_at_end_pct:
            Estimated battery state-of-charge at the end of the planning
            horizon (%).
        constraints:
            List of active constraints or flags that influenced the decision
            (e.g. ``"winter_month"``, ``"no_price_spread"``,
            ``"excess_export_enabled"``).
        rejected_plans:
            Alternative plans that were evaluated and rejected, each with a
            name, reason, and estimated cost.
        solver_status:
            ``optimal``, ``time_limit_feasible_incumbent``, or a specific
            failure/skip status from the most recent HiGHS attempt.
        solver_optimal:
            Whether HiGHS proved the selected MILP solution optimal.
        incumbent_used:
            Whether a fully validated time-limit incumbent supplied the MILP
            candidate. The candidate name remains ``milp`` by design.
        fallback_reason:
            Machine-readable reason a non-MILP plan was used. Empty when the
            selected plan is a successful MILP result.
        hysteresis_active:
            ``True`` when plan-level hysteresis was applied and the previous
            plan was kept despite a new candidate having a slightly better score.
        hysteresis_reason:
            Human-readable explanation of the hysteresis decision, or ``""``
            when hysteresis is inactive or the plan was switched.
        previous_plan_name:
            Name of the winning plan from the previous planner run, or
            ``""`` on first run.
    """

    selected_strategy: str = STATE_UNKNOWN
    winner_name: str = ""  # e.g. "milp", "passive" — matches rejected_plans
    summary: str = ""
    score: float = 0.0
    estimated_total_cost: float = 0.0
    price_spread: float = 0.0
    peak_import_price: float = 0.0
    off_peak_import_price: float = 0.0
    forecast_pv_kwh: float = 0.0
    forecast_net_consumption_kwh: float = 0.0
    battery_soc_pct: float = 0.0
    battery_soc_at_end_pct: float = 0.0
    constraints: list[str] = field(default_factory=list)
    rejected_plans: list[RejectedPlan] = field(default_factory=list)
    # MILP solve/fallback observability.
    solver_status: str = "not_run"
    solver_optimal: bool = False
    solver_time_limit_seconds: float = 0.0
    solver_elapsed_seconds: float = 0.0
    solver_mip_gap: float | None = None
    solver_message: str = ""
    incumbent_used: bool = False
    incumbent_validation: str = ""
    fallback_reason: str = ""
    # Compact bounded terminal-value observability (full tiers stay in solver
    # diagnostics so recorder-backed attributes remain small).
    terminal_cost_to_go_source: str = "hardware_floor_only"
    terminal_cost_to_go_boundary: str | None = None
    terminal_cost_to_go_tier_count: int = 0
    terminal_cost_to_go_total_quantity_kwh: float = 0.0
    terminal_cost_to_go_highest_value_per_kwh: float = 0.0
    terminal_cost_to_go_lowest_value_per_kwh: float = 0.0
    terminal_cost_to_go_initial_valued_quantity_kwh: float = 0.0
    terminal_cost_to_go_final_valued_quantity_kwh: float = 0.0
    terminal_cost_to_go_initial_value: float = 0.0
    terminal_cost_to_go_final_value: float = 0.0
    # Hysteresis fields (issue #372)
    hysteresis_active: bool = False
    hysteresis_reason: str = ""
    previous_plan_name: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Serialise the explanation to a plain dict for HA attributes.

        Returns:
            A JSON-safe dictionary representation of the explanation.
        """
        return {
            "selected_strategy": self.selected_strategy,
            "winner_name": self.winner_name,
            "summary": self.summary,
            "score": round(self.score, 4),
            "estimated_total_cost": round(self.estimated_total_cost, 4),
            "price_spread": round(self.price_spread, 4),
            "peak_import_price": round(self.peak_import_price, 4),
            "off_peak_import_price": round(self.off_peak_import_price, 4),
            "forecast_pv_kwh": round(self.forecast_pv_kwh, 3),
            "forecast_net_consumption_kwh": round(self.forecast_net_consumption_kwh, 3),
            "battery_soc_pct": round(self.battery_soc_pct, 1),
            "battery_soc_at_end_pct": round(self.battery_soc_at_end_pct, 1),
            "solver_status": self.solver_status,
            "solver_optimal": self.solver_optimal,
            "solver_time_limit_seconds": round(self.solver_time_limit_seconds, 3),
            "solver_elapsed_seconds": round(self.solver_elapsed_seconds, 3),
            "solver_mip_gap": (
                round(self.solver_mip_gap, 6)
                if self.solver_mip_gap is not None
                else None
            ),
            "solver_message": self.solver_message,
            "incumbent_used": self.incumbent_used,
            "incumbent_validation": self.incumbent_validation,
            "fallback_reason": self.fallback_reason,
            "terminal_cost_to_go_source": self.terminal_cost_to_go_source,
            "terminal_cost_to_go_boundary": self.terminal_cost_to_go_boundary,
            "terminal_cost_to_go_tier_count": self.terminal_cost_to_go_tier_count,
            "terminal_cost_to_go_total_quantity_kwh": round(
                self.terminal_cost_to_go_total_quantity_kwh,
                6,
            ),
            "terminal_cost_to_go_highest_value_per_kwh": round(
                self.terminal_cost_to_go_highest_value_per_kwh,
                6,
            ),
            "terminal_cost_to_go_lowest_value_per_kwh": round(
                self.terminal_cost_to_go_lowest_value_per_kwh,
                6,
            ),
            "terminal_cost_to_go_initial_valued_quantity_kwh": round(
                self.terminal_cost_to_go_initial_valued_quantity_kwh,
                6,
            ),
            "terminal_cost_to_go_final_valued_quantity_kwh": round(
                self.terminal_cost_to_go_final_valued_quantity_kwh,
                6,
            ),
            "terminal_cost_to_go_initial_value": round(
                self.terminal_cost_to_go_initial_value,
                6,
            ),
            "terminal_cost_to_go_final_value": round(
                self.terminal_cost_to_go_final_value,
                6,
            ),
            "constraints": list(self.constraints),
            "rejected_plans": [
                {
                    "name": rp.name,
                    "reason": rp.reason,
                    "estimated_cost": round(rp.estimated_cost, 4),
                    "import_cost": round(rp.import_cost, 4),
                    "export_revenue": round(rp.export_revenue, 4),
                    "conversion_loss": round(rp.conversion_loss, 4),
                    "cycle_cost": round(rp.cycle_cost, 4),
                    "score": round(rp.score, 4),
                }
                for rp in self.rejected_plans
            ],
        }
