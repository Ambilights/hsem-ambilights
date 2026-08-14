"""MILP diagnostic and winner-warning helpers for the planner engine."""

from __future__ import annotations

from custom_components.hsem.models.plan_explanation import PlanExplanation
from custom_components.hsem.planner.candidate_generator import (
    CANDIDATE_BASELINE,
    CANDIDATE_MILP,
    CandidatePlan,
)
from custom_components.hsem.utils.logger import log_planner


def extract_solver_diagnostics(candidates: list[CandidatePlan]) -> dict:
    """Return successful MILP diagnostics or the retained failed attempt."""
    for candidate in candidates:
        if candidate.name == CANDIDATE_MILP and candidate.diagnostics is not None:
            return candidate.diagnostics
    for candidate in candidates:
        diagnostics = candidate.diagnostics
        if diagnostics is None:
            continue
        attempt = diagnostics.get("milp_attempt")
        if isinstance(attempt, dict):
            return attempt
    return {}


def _diagnostic_number(diagnostics: dict, key: str, default: float) -> float:
    """Read a finite numeric diagnostic, returning *default* otherwise."""
    value = diagnostics.get(key)
    if not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if number == number and abs(number) != float("inf") else default


def publish_selected_candidate_warnings(
    warnings: list[str],
    baseline_warnings: list[str],
    winner_name: str,
) -> None:
    """Surface heuristic scheduling warnings only when that baseline wins."""
    if winner_name == CANDIDATE_BASELINE:
        warnings.extend(baseline_warnings)
    elif baseline_warnings:
        log_planner(
            "debug",
            "[core] Suppressed %d discarded-baseline warning(s); winner=%s",
            len(baseline_warnings),
            winner_name,
        )


def populate_solver_explanation(
    explanation: PlanExplanation,
    candidates: list[CandidatePlan],
    winner_name: str,
    configured_timeout: float,
) -> None:
    """Expose MILP outcome without changing the candidate control-flow name."""
    diagnostics = extract_solver_diagnostics(candidates)
    explanation.solver_status = str(diagnostics.get("solver_status", "not_run"))
    explanation.solver_optimal = bool(diagnostics.get("solver_optimal", False))
    explanation.solver_time_limit_seconds = _diagnostic_number(
        diagnostics, "solver_time_limit_seconds", configured_timeout
    )
    explanation.solver_elapsed_seconds = _diagnostic_number(
        diagnostics, "solver_elapsed_seconds", 0.0
    )
    raw_gap = diagnostics.get("solver_mip_gap")
    explanation.solver_mip_gap = (
        float(raw_gap) if isinstance(raw_gap, (int, float)) else None
    )
    explanation.solver_message = str(diagnostics.get("solver_message", ""))
    explanation.incumbent_used = bool(diagnostics.get("incumbent_used", False))
    explanation.incumbent_validation = str(diagnostics.get("incumbent_validation", ""))
    explanation.fallback_reason = str(diagnostics.get("fallback_reason", ""))

    if winner_name != CANDIDATE_MILP and not explanation.fallback_reason:
        explanation.fallback_reason = (
            "milp_candidate_not_selected"
            if explanation.solver_status in {"optimal", "time_limit_feasible_incumbent"}
            else "milp_candidate_unavailable"
        )
    if explanation.fallback_reason:
        explanation.constraints.append("milp_fallback")
    elif explanation.incumbent_used:
        explanation.constraints.append("milp_time_limit_incumbent")
