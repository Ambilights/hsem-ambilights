"""Candidate plan generator for the HSEM planner (issues #296, #416).

Successful solves expose the MILP plan plus two diagnostic comparators.
Solver failure deliberately leaves only the passive executable fallback; the
retired heuristic baseline does not enforce the MILP's full safety constraints.

Design principles
-----------------
- **Pure Python, no Home Assistant imports** — testable with plain pytest.
- Each candidate is built from an independent copy of the populated slots so
  strategies cannot interfere with each other.
- ``no_action`` remains diagnostic and can never win selection.
- ``passive`` clears all forced actions and is the only solver-failure
  fallback.
- Normal successful solves remain MILP-owned; no heuristic mutation matrix is
  revived.

Candidates produced
-------------------
1. ``no_action`` — diagnostic comparator with active storage actions
   cleared before normal completion and simulation.
2. ``passive`` — solar absorption without grid charge or forced discharge.
3. ``milp`` — solver-owned plan, present only after a successful solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from custom_components.hsem.models.ev_config import EVConfig
from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.planner_input import PlannerInput
from custom_components.hsem.models.terminal_cost_to_go import TerminalCostToGo
from custom_components.hsem.planner.candidates._mutations import (
    _apply_passive_solar,
    _clear_all_charge_discharge,
    _copy_slots,
)
from custom_components.hsem.planner.cost_function import PlanCostBreakdown
from custom_components.hsem.planner.milp_optimizer import (
    CANDIDATE_MILP,
    is_scipy_available,
    solve_milp,
)
from custom_components.hsem.utils.logger import log_planner
from custom_components.hsem.utils.misc import (
    calculate_recommended_threshold,
    resolve_cycle_cost,
)
from custom_components.hsem.utils.recommendations import (
    CHARGE_RECS as _CHARGE_RECS,
    DISCHARGE_RECS as _DISCHARGE_RECS,
)

# ---------------------------------------------------------------------------
# Candidate name constants — shared with selector so both sides speak the
# same identifiers without re-defining strings.
# ---------------------------------------------------------------------------

CANDIDATE_BASELINE = "baseline"
CANDIDATE_NO_ACTION = "no_action"
CANDIDATE_PASSIVE = "passive"
CANDIDATE_GRID_CHARGE = "grid_charge"
CANDIDATE_SOLAR_ONLY = "solar_only"
CANDIDATE_DISCHARGE_ONLY = "discharge_only"
CANDIDATE_AGGRESSIVE = "aggressive"

# Partial-SoC candidates (BatPred-inspired) — each charges a different
# fraction of the energy needed for the upcoming discharge windows so
# the selector can find the optimal charge level.
CANDIDATE_SOC_PLAN = "soc_plan"
CANDIDATE_SOC_25 = "soc_plan_25"
CANDIDATE_SOC_50 = "soc_plan_50"
CANDIDATE_SOC_75 = "soc_plan_75"
CANDIDATE_SOC_100 = "soc_plan_100"
CANDIDATE_SOC_125 = "soc_plan_125"
CANDIDATE_SOC_FULL = "soc_plan_full"

# Charge fractions for partial-SoC candidates — each is a multiplier
# applied to the calculated energy needed for the discharge windows.
# A fraction of 1.0 means "charge exactly what's needed."
_SOC_FRACTIONS: dict[str, float] = {
    CANDIDATE_SOC_25: 0.25,
    CANDIDATE_SOC_50: 0.50,
    CANDIDATE_SOC_75: 0.75,
    CANDIDATE_SOC_100: 1.00,
    CANDIDATE_SOC_125: 1.25,
    CANDIDATE_SOC_FULL: 2.00,  # fill to max usable capacity
}

# Re-export MILP candidate name so callers only need to import from here
__all__ = [
    "CANDIDATE_BASELINE",
    "CANDIDATE_NO_ACTION",
    "CANDIDATE_PASSIVE",
    "CANDIDATE_GRID_CHARGE",
    "CANDIDATE_SOLAR_ONLY",
    "CANDIDATE_DISCHARGE_ONLY",
    "CANDIDATE_AGGRESSIVE",
    "CANDIDATE_SOC_PLAN",
    "CANDIDATE_SOC_25",
    "CANDIDATE_SOC_50",
    "CANDIDATE_SOC_75",
    "CANDIDATE_SOC_100",
    "CANDIDATE_SOC_125",
    "CANDIDATE_SOC_FULL",
    "CANDIDATE_MILP",
    "CandidatePlan",
    "generate_candidates",
]


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class CandidatePlan:
    """A single candidate plan ready for scoring.

    Attributes:
        name:
            Short, machine-readable identifier for this candidate strategy.
        slots:
            Fully populated :class:`PlannedSlot` list.  ``batteries_charged``
            and ``recommendation`` have been written by the generator;
            ``batteries_discharged``, ``grid_import_kwh``, ``grid_export_kwh``,
            and ``estimated_battery_soc`` are written by :func:`simulate_soc`
            **after** the caller receives this object.
        is_valid:
            ``True`` once the plan has passed validity checks (e.g. SoC never
            drops below the end-of-discharge floor).  Set by the selector after
            the SoC simulation runs.
        rejection_reason:
            Human-readable reason when ``is_valid`` is ``False``.
        diagnostics:
            Optional diagnostics dict from the candidate generator (e.g., MILP
            penalty violations).  Set by the generator during candidate creation.
    """

    name: str
    slots: list[PlannedSlot]
    is_valid: bool = True
    rejection_reason: str = ""
    diagnostics: dict | None = field(default=None, repr=False, compare=False)
    _cost: PlanCostBreakdown | None = field(default=None, repr=False, compare=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_candidates(
    baseline_slots: list[PlannedSlot],
    inp: PlannerInput,
    now: datetime,
    max_charge_per_slot: float,
    current_kwh: float = 0.0,
    usable_kwh: float = 0.0,
    max_discharge_per_slot: float | None = None,
    replacement_price_per_kwh: float | None = None,
    terminal_cost_to_go: TerminalCostToGo | None = None,
    ev_configs: list[EVConfig] | None = None,
) -> list[CandidatePlan]:
    """Generate all candidate plans from the already-populated baseline slots.

    The *baseline_slots* list must already contain populated prices, load, PV,
    and EV demand, plus the baseline safety and context decisions applied
    before candidate construction. The SoC simulation has **not** yet been
    applied; the selector runs it separately for each non-MILP candidate and
    verifies the solver-owned flows for the MILP candidate.

    Args:
        baseline_slots:
            Fully scheduled slots (pre-SoC-simulation) representing the
            current HSEM planning output.  This list is **not** mutated; each
            candidate receives its own deep copy.
        inp:
            The planner input for this run. Supplies the economic, physical,
            and safety parameters used by the MILP.
        now:
            Timezone-aware current datetime.
        max_charge_per_slot:
            Maximum battery-side energy (kWh) storable per slot.
        current_kwh:
            Current battery energy above the discharge floor (kWh).
        usable_kwh:
            Maximum usable battery capacity above the discharge floor (kWh).
        max_discharge_per_slot:
            Maximum energy dischargeable per slot (kWh) passed through to the
            MILP optimizer.  ``None`` means unlimited.
        replacement_price_per_kwh:
            Legacy uniform terminal value passed through for compatible direct
            callers. Ignored when ``terminal_cost_to_go`` is supplied.
        terminal_cost_to_go:
            Bounded piecewise primary-inventory value passed to the MILP.
            Production planning supplies this even when it has no tiers.
        ev_configs:
            Optional list of :class:`EVConfig` objects (one per EV).  When
            provided, the MILP co-optimises EV charging alongside the battery.
            The engine computes the deadline slot mapping before passing the
            configs here.  ``None`` means no EV co-optimisation
            (backward-compatible behaviour).

    Returns:
        Ordered candidate list. Passive is the sole executable fallback after
        solver failure; a successful solve also adds the MILP candidate.
    """
    candidates: list[CandidatePlan] = []

    # MILP-only mode: no_action is diagnostic, passive is fail-closed, and
    # the solver is the only authority for active optimization.

    # 1. No-action — clear active storage actions for a non-winning diagnostic.
    # Normal completion and SoC simulation may still model passive energy flow.
    no_action = _copy_slots(baseline_slots)
    _clear_all_charge_discharge(no_action)
    candidates.append(CandidatePlan(name=CANDIDATE_NO_ACTION, slots=no_action))

    # 2. Passive — solar charging only, no grid charge/discharge.
    passive = _copy_slots(baseline_slots)
    _apply_passive_solar(passive, now)
    candidates.append(CandidatePlan(name=CANDIDATE_PASSIVE, slots=passive))

    # 3. MILP — globally optimal LP solution when the solver succeeds.
    if is_scipy_available():
        # Use the canonical resolve_cycle_cost() — same as engine_core and
        # cost_helpers.py — so the MILP optimises against the same value.
        effective_cycle_cost = resolve_cycle_cost(
            purchase_price=inp.battery_purchase_price,
            usable_kwh=usable_kwh,
            expected_cycles=inp.battery_expected_cycles,
            capacity_loss_pct=inp.battery_capacity_loss_pct,
            user_margin=inp.battery_cycle_cost_per_kwh,
        )

        # Depreciation is a battery-throughput concern, not a site-export
        # concern. Keep the configured inverter/site floor independent so
        # direct PV export remains eligible below the battery wear threshold.
        depreciation_export_floor = calculate_recommended_threshold(
            purchase_price=inp.battery_purchase_price,
            expected_cycles=inp.battery_expected_cycles,
            usable_capacity=usable_kwh,
            capacity_loss_pct=inp.battery_capacity_loss_pct,
        )
        effective_battery_export_floor = max(
            inp.battery_export_min_price, depreciation_export_floor
        )
        milp_attempt: dict = {}
        milp_result = solve_milp(
            baseline_slots,
            now,
            current_kwh=current_kwh,
            usable_kwh=usable_kwh,
            max_charge_per_slot=max_charge_per_slot,
            max_discharge_per_slot=max_discharge_per_slot,
            cycle_cost_per_kwh=effective_cycle_cost,
            charge_efficiency_pct=inp.battery_charge_efficiency_pct,
            discharge_efficiency_pct=inp.battery_discharge_efficiency_pct,
            time_discount_rate=inp.time_discount_rate,
            replacement_price_per_kwh=replacement_price_per_kwh,
            terminal_cost_to_go=terminal_cost_to_go,
            min_export_price=inp.export_min_price,
            ev_configs=ev_configs,
            no_export=not inp.excess_export_enabled,
            main_fuse_amps=inp.main_fuse_amps,
            main_fuse_phases=inp.main_fuse_phases,
            phase_power_imbalance_w=(
                inp.grid_phase_power_imbalance_w
                if inp.phase_aware_charging_enabled
                else None
            ),
            max_grid_export_power_kw=inp.max_grid_export_power_kw,
            secondary_storage=inp.secondary_storage,
            battery_export_min_price=effective_battery_export_floor,
            excess_export_discharge_buffer_pct=(inp.excess_export_discharge_buffer_pct),
            solver_time_limit_seconds=inp.milp_solver_timeout_seconds,
            attempt_diagnostics=milp_attempt,
        )
        log_planner(
            "debug",
            "[gen] MILP solve called  cycle_cost=%.6f  no_export=%s  "
            "excess_export_enabled=%s  site_export_min_price=%.4f  "
            "battery_export_min_price=%.4f  depreciation_floor=%.4f  "
            "ev_configs=%d",
            effective_cycle_cost,
            not inp.excess_export_enabled,
            inp.excess_export_enabled,
            inp.export_min_price,
            effective_battery_export_floor,
            depreciation_export_floor,
            len(ev_configs) if ev_configs else 0,
        )
        if milp_result is not None:
            milp_slots, milp_diag = milp_result
            candidates.append(
                CandidatePlan(
                    name=CANDIDATE_MILP, slots=milp_slots, diagnostics=milp_diag
                )
            )
            log_planner(
                "debug",
                "[gen] MILP candidate added (scipy available and solver succeeded)"
                "  status=%s  penalty_violations=%s  total_violation=%.4f",
                milp_diag.get("solver_status", "unknown"),
                milp_diag.get("has_violations", False),
                milp_diag.get("total_violation_kwh", 0.0),
            )
        else:
            for candidate in candidates:
                candidate.diagnostics = {"milp_attempt": dict(milp_attempt)}
            status = str(milp_attempt.get("solver_status", "unknown"))
            fallback_reason = str(
                milp_attempt.get("fallback_reason", "milp_candidate_unavailable")
            )
            log_planner(
                ("warning" if status.startswith(("time_limit", "solver")) else "debug"),
                "[gen] MILP candidate skipped status=%s fallback=%s",
                status,
                fallback_reason,
            )
    else:
        unavailable_attempt = {
            "solver_status": "scipy_unavailable",
            "solver_optimal": False,
            "solver_status_code": None,
            "solver_message": "scipy is not available",
            "solver_time_limit_seconds": inp.milp_solver_timeout_seconds,
            "solver_elapsed_seconds": 0.0,
            "solver_mip_gap": None,
            "incumbent_used": False,
            "incumbent_validation": "not_run",
            "fallback_reason": "scipy_unavailable",
        }
        for candidate in candidates:
            candidate.diagnostics = {"milp_attempt": dict(unavailable_attempt)}
        log_planner("warning", "[gen] MILP unavailable — passive fallback only")

    # Log candidate slot-level recommendations for debugging
    log_planner(
        "debug",
        "[gen] Generated %d candidates: %s",
        len(candidates),
        ", ".join(c.name for c in candidates),
    )
    for cand in candidates:
        charge_slots = [
            s.start.strftime("%d %H:%M")
            for s in cand.slots
            if s.recommendation in _CHARGE_RECS
        ]
        discharge_slots = [
            s.start.strftime("%d %H:%M")
            for s in cand.slots
            if s.recommendation in _DISCHARGE_RECS
        ]
        total_charge = sum(s.batteries_charged_kwh for s in cand.slots)
        log_planner(
            "debug",
            "[gen] %s: charge_slots=%d (%s)  discharge_slots=%d (%s)  "
            "total_charge=%.3f kWh",
            cand.name,
            len(charge_slots),
            ", ".join(charge_slots) if charge_slots else "—",
            len(discharge_slots),
            ", ".join(discharge_slots) if discharge_slots else "—",
            total_charge,
        )

    return candidates
