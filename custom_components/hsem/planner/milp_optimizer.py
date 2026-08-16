"""HiGHS optimisation for Huawei storage, EVs, and optional secondary storage."""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING, Any

from custom_components.hsem.const import (
    MILP_SOLVER_TIMEOUT_DEFAULT_SECONDS,
    MILP_SOLVER_TIMEOUT_MAX_SECONDS,
    MILP_SOLVER_TIMEOUT_MIN_SECONDS,
)
from custom_components.hsem.models.ev_config import EVConfig
from custom_components.hsem.utils.datetime_utils import utc_key
from custom_components.hsem.utils.logger import log_planner
from custom_components.hsem.utils.misc import clamp_efficiency
from custom_components.hsem.utils.phase_power import PhasePowers
from custom_components.hsem.utils.units import (
    fuse_max_energy_per_slot_kwh,
    slot_duration_hours,
)

if TYPE_CHECKING:
    from custom_components.hsem.models.planned_slot import PlannedSlot
    from custom_components.hsem.models.secondary_storage_config import (
        SecondaryStorageConfig,
    )

# Name exported so the engine and tests can reference it without re-defining
CANDIDATE_MILP = "milp"

# Minimum energy threshold below which a slot is treated as zero-charge/discharge
# to avoid writing tiny floating-point artefacts into recommendations.
_MIN_ACTION_KWH = 1e-4

# Select the zero-valued export-mode binary when both binary values are
# feasible. This is far below any material energy cost but makes diagnostics
# deterministic and prevents reserve rows activating without battery export.
_EXPORT_MODE_TIEBREAK_COST = 1e-6


def _normalise_solver_timeout(value: float) -> float:
    """Clamp a configured solver budget to the supported safe range."""
    try:
        timeout = float(value)
    except TypeError, ValueError:
        timeout = MILP_SOLVER_TIMEOUT_DEFAULT_SECONDS
    if not math.isfinite(timeout):
        timeout = MILP_SOLVER_TIMEOUT_DEFAULT_SECONDS
    return min(
        max(timeout, MILP_SOLVER_TIMEOUT_MIN_SECONDS),
        MILP_SOLVER_TIMEOUT_MAX_SECONDS,
    )


def _sync_attempt_diagnostics(
    attempt: dict[str, Any],
    target: dict[str, Any] | None,
    **updates: Any,
) -> None:
    """Update local diagnostics and the optional caller-owned snapshot."""
    attempt.update(updates)
    if target is not None:
        target.clear()
        target.update(attempt)


def solve_milp(
    slots: list[PlannedSlot],
    now: datetime,
    current_kwh: float,
    usable_kwh: float,
    max_charge_per_slot: float,
    max_discharge_per_slot: float | None,
    cycle_cost_per_kwh: float = 0.0,
    charge_efficiency_pct: float = 97.0,
    discharge_efficiency_pct: float = 97.0,
    time_discount_rate: float = 1.0,
    replacement_price_per_kwh: float | None = None,
    *,
    min_export_price: float = 0.0,
    ev_configs: list[EVConfig] | None = None,
    no_export: bool = False,
    main_fuse_amps: float | None = None,
    main_fuse_phases: int = 3,
    phase_power_imbalance_w: PhasePowers | None = None,
    max_grid_export_power_kw: float | None = None,
    battery_export_min_price: float = 0.0,
    excess_export_discharge_buffer_pct: float = 0.0,
    secondary_storage: SecondaryStorageConfig | None = None,
    solver_time_limit_seconds: float = MILP_SOLVER_TIMEOUT_DEFAULT_SECONDS,
    attempt_diagnostics: dict[str, Any] | None = None,
) -> tuple[list[PlannedSlot], dict] | None:
    """Solve the planning horizon with HiGHS and return copied result slots.

    The model co-optimises Huawei battery flow, optional EV charging, main-fuse
    and export limits, and optional secondary storage. Past slots remain fixed;
    future slots receive internally consistent energy-flow, EV-load, and cost
    fields. The caller subsequently runs SoC simulation in pre-populated mode.

    Price availability, efficiency losses, terminal battery value, conditional
    export reserve, and session-aware EV constraints share one model. A
    time-limited incumbent is accepted only after complete feasibility checks.

    attempt_diagnostics is updated for unavailable dependencies, invalid input,
    solver failure, or a rejected candidate. Successful calls return a tuple of
    slots and diagnostics; failures return None for the heuristic fallback.
    """
    solver_time_limit = _normalise_solver_timeout(solver_time_limit_seconds)
    attempt: dict[str, Any] = {
        "solver_status": "not_started",
        "solver_optimal": False,
        "solver_status_code": None,
        "solver_message": "",
        "solver_time_limit_seconds": solver_time_limit,
        "solver_elapsed_seconds": 0.0,
        "solver_mip_gap": None,
        "incumbent_used": False,
        "incumbent_validation": "not_run",
        "fallback_reason": "",
    }
    _sync_attempt_diagnostics(attempt, attempt_diagnostics)

    secondary_active = secondary_storage is not None and secondary_storage.valid
    if secondary_active:
        import copy

        from custom_components.hsem.planner.secondary_storage import (
            populate_secondary_storage_load,
        )

        slots = [copy.copy(slot) for slot in slots]
        assert secondary_storage is not None
        populate_secondary_storage_load(slots, secondary_storage, now)

    log_planner(
        "debug",
        "[milp] solve_milp  slots=%d  current=%.3f  usable=%.3f  "
        "max_chg=%.3f  max_dis=%s  cycle_cost=%.6f  "
        "chg_eff=%.2f  dis_eff=%.2f  discount=%.4f  repl_price=%s  "
        "no_export=%s  min_export_price=%.4f  battery_export_min_price=%.4f  "
        "export_buffer=%.2f%%  "
        "fuse=%s  phase_aware=%s  secondary=%s  timeout=%.1fs",
        len(slots),
        current_kwh,
        usable_kwh,
        max_charge_per_slot,
        f"{max_discharge_per_slot:.3f}" if max_discharge_per_slot is not None else "∞",
        cycle_cost_per_kwh,
        charge_efficiency_pct,
        discharge_efficiency_pct,
        time_discount_rate,
        (
            f"{replacement_price_per_kwh:.6f}"
            if replacement_price_per_kwh is not None
            else "None"
        ),
        no_export,
        min_export_price,
        battery_export_min_price,
        excess_export_discharge_buffer_pct,
        (
            f"{main_fuse_amps:.1f}A/{main_fuse_phases}ph"
            if main_fuse_amps is not None
            else "disabled"
        ),
        phase_power_imbalance_w is not None,
        secondary_active,
        solver_time_limit,
    )

    try:
        import numpy as np
        from scipy.optimize import linprog
    except ImportError:
        _sync_attempt_diagnostics(
            attempt,
            attempt_diagnostics,
            solver_status="scipy_unavailable",
            fallback_reason="scipy_unavailable",
        )
        log_planner("debug", "[milp] scipy/numpy not available — MILP disabled")
        return None

    if usable_kwh <= 0 or max_charge_per_slot <= 0:
        log_planner(
            "debug",
            "[milp] Skipping — usable_kwh=%.3f max_charge_per_slot=%.3f",
            usable_kwh,
            max_charge_per_slot,
        )
        _sync_attempt_diagnostics(
            attempt,
            attempt_diagnostics,
            solver_status="skipped_invalid_battery",
            fallback_reason="invalid_battery_limits",
        )
        return None

    n = len(slots)
    if n == 0:
        _sync_attempt_diagnostics(
            attempt,
            attempt_diagnostics,
            solver_status="skipped_empty_horizon",
            fallback_reason="empty_horizon",
        )
        return None

    max_dis = (
        max_discharge_per_slot if max_discharge_per_slot is not None else usable_kwh
    )

    # ------------------------------------------------------------------
    # Identify future (active) vs. past (fixed-zero) slot indices
    # ------------------------------------------------------------------
    future_mask = [utc_key(s.end) > utc_key(now) for s in slots]
    # Indices of future slots in the full slot list
    future_idx = [i for i, m in enumerate(future_mask) if m]

    if not future_idx:
        _sync_attempt_diagnostics(
            attempt,
            attempt_diagnostics,
            solver_status="skipped_no_future_slots",
            fallback_reason="no_future_slots",
        )
        return None

    # Economic optimisation is authoritative only for the contiguous price
    # prefix populated by the planner input pipeline.  Later slots may still
    # carry numeric placeholders or stale/raw source values for diagnostics,
    # but those values must not influence any objective or penalty magnitude.
    price_actionable = np.array(
        [slots[i].price_actionable for i in future_idx], dtype=bool
    )

    # ------------------------------------------------------------------
    # Build per-slot data arrays (future slots only)
    # ------------------------------------------------------------------
    p_imp_raw = np.array([slots[i].price.import_price for i in future_idx], dtype=float)
    p_exp_raw = np.array([slots[i].price.export_price for i in future_idx], dtype=float)

    # Fail closed for direct model construction as well as source-populated
    # inputs: NaN and either infinity are not economic signals.
    p_imp_raw = np.nan_to_num(p_imp_raw, nan=0.0, posinf=0.0, neginf=0.0)
    p_exp_raw = np.nan_to_num(p_exp_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Per-slot hard floor for intentional battery-to-grid export (issue
    # #752). 0.0 (default) → mask all-False, backward compatible.
    battery_export_blocked = np.zeros(len(future_idx), dtype=bool)
    if battery_export_min_price > 1e-9:
        battery_export_blocked = p_exp_raw < battery_export_min_price

    min_export_blocked = np.zeros(len(future_idx), dtype=bool)
    # Clamp export prices below min_export_price to 0.
    # The applier physically sets the inverter to GRID_EXPORT_LIMIT_WATT
    # for these slots, blocking export entirely.  The LP must not optimise
    # around a price signal that will never be realised.
    #
    # Negative export prices are NOT clamped — the LP has a curt[t]
    # variable with zero objective cost that naturally handles them:
    # when p_exp < 0, export costs money (p_exp is negative, so
    # -p_exp·ge becomes a positive cost), and the LP prefers curtailment
    # (cost 0) over export (cost > 0).
    if min_export_price > 1e-9:
        blocked = p_exp_raw < min_export_price
        min_export_blocked = blocked
        n_blocked = int(np.sum(blocked))
        if n_blocked > 0:
            log_planner(
                "debug",
                "[milp] Clamping %d export prices below min_price (%.4f) to 0 "
                "(max clamped=%.4f)",
                n_blocked,
                min_export_price,
                float(np.max(p_exp_raw[blocked])),
            )

    # Both export floors are physical execution guards. Apply their union to
    # the primary discharge cap and to the later Huawei/PowMr coupling rows so
    # the solved energy balance cannot rely on an export the applier blocks.
    primary_export_blocked = np.logical_or(battery_export_blocked, min_export_blocked)

    # Keep raw prices above only for physical export masks and diagnostics.
    # All economic terms use a zero-valued tail once price authority ends.
    # This preserves genuine actionable zero/negative prices while preventing
    # a later isolated price from changing current decisions or penalty scale.
    p_imp = np.where(price_actionable, p_imp_raw, 0.0)
    p_exp = np.where(price_actionable, p_exp_raw, 0.0)
    if min_export_price > 1e-9:
        p_exp = np.where(min_export_blocked, 0.0, p_exp)

    # Clamp export price to never exceed import price for the same slot.
    # Without this, slots where p_exp[t] > p_imp[t] create an unbounded LP
    # (HiGHS status=3): both gi[t] and ge[t] are [0, ∞) and linked only
    # through the energy-balance equality, so the LP can drive both to
    # infinity (import cheap, export expensive) while the terms cancel in
    # the balance equation.  This is economically correct — no rational
    # agent imports and exports the same commodity in the same instant for
    # profit — and capping the achievable arbitrage spread removes the
    # unbounded direction without changing any other behavior.
    export_exceeds_import = p_exp > p_imp
    n_clamped = int(np.sum(export_exceeds_import))
    if n_clamped > 0:
        deltas = p_exp[export_exceeds_import] - p_imp[export_exceeds_import]
        log_planner(
            "debug",
            "[milp] Clamping %d export prices that exceed import price "
            "(max delta=%.4f)",
            n_clamped,
            float(np.max(deltas)),
        )
        p_exp = np.minimum(p_exp, p_imp)

    # Clamp negative import prices to 0 for objective coefficients.
    # When p_imp[t] < 0, the gi[t] objective coefficient becomes
    # negative, incentivising the LP to import infinite energy
    # (HiGHS status=3, unbounded LP).  curt[t] has zero objective
    # cost but participates in the energy balance, so the LP can
    # import-and-curtail for unbounded profit even without p_exp>p_imp.
    #
    # Clamping to 0 here removes that unbounded direction while
    # Raw finite arrays are retained separately only for physical export
    # masks and diagnostics; objective and penalty scaling use the economic
    # arrays whose nonactionable tail is neutral.
    #
    # This is the companion to the export-≤-import clamp above:
    # together they close both unbounded-LP directions identified in
    # issue #635.
    p_imp_obj = np.maximum(p_imp, 0.0)

    # Net load = house consumption + EV extra load − PV estimate.
    # A positive value means the battery/grid must supply extra energy.
    # A negative value means there is PV surplus.
    # Split into base_load (positive demand) and pv_avail (PV surplus after load).
    # pv_avail[t] is added as an explicit LP variable to prevent infeasibility
    # when net_load is strongly negative and SoC limits constrain charge.
    #
    # EV adjustment: when EV charging is active, the EV consumes PV surplus
    # first (before the battery).  This reduces the PV surplus available to
    # the battery by the EV's total planned load (which includes both
    # ev_planned_load_kwh and ev_accounted_load_kwh).  base_load is NOT
    # increased because the battery never feeds the EV — any remaining EV
    # demand after PV goes to the grid.
    net_load = np.array(
        [
            slots[i].avg_house_consumption_kwh
            + slots[i].ev_planned_load_kwh
            - slots[i].solcast_pv_estimate_kwh
            for i in future_idx
        ],
        dtype=float,
    )
    pv_avail = np.maximum(-net_load, 0.0)  # PV surplus after house consumption
    base_load = np.maximum(net_load, 0.0)  # remaining demand after PV

    # ------------------------------------------------------------------
    # EV accounted load: when base_load_includes_ev=True, ev_accounted_load_kwh
    # is the EV load already captured in avg_house_consumption_kwh.  The battery
    # must not discharge to cover this load — it is the EV's own demand served
    # by the grid (or PV).  Without this cap, the live-injected current-slot
    # house consumption (which includes EV power when the CT clamp is upstream
    # of the charger) causes the MILP to discharge the house battery into the EV,
    # which provides zero financial benefit when EV charging is reimbursed
    # (issue #592).
    # ------------------------------------------------------------------
    ev_accounted = np.array(
        [slots[i].ev_accounted_load_kwh for i in future_idx], dtype=float
    )

    # ------------------------------------------------------------------
    # EV co-optimisation: when ev_configs is provided, the MILP decides EV
    # charging alongside the battery.  Recompute net_load/pv_avail/base_load
    # WITHOUT the pre-computed EV planned loads (the LP will decide allocation).
    # Otherwise keep the pre-existing EV adjustment (backward-compatible).
    # ------------------------------------------------------------------
    active_evs: list[EVConfig] = []
    if ev_configs:
        for ev in ev_configs:
            if ev.enabled and ev.capacity_kwh > 1e-9 and ev.max_charge_per_slot > 1e-9:
                active_evs.append(ev)
        if active_evs:
            # Recompute net_load without EV planned loads
            net_load = np.array(
                [
                    slots[i].avg_house_consumption_kwh
                    - slots[i].solcast_pv_estimate_kwh
                    for i in future_idx
                ],
                dtype=float,
            )
            pv_avail = np.maximum(-net_load, 0.0)
            base_load = np.maximum(net_load, 0.0)
            log_planner(
                "debug",
                "[milp] EV co-optimisation enabled: %d active EV(s), "
                "net_load rebuilt without pre-computed EV loads",
                len(active_evs),
            )
        else:
            active_evs = []
    if not active_evs and ev_configs:
        log_planner(
            "debug",
            "[milp] EV configs provided but no valid active EVs — "
            "falling back to fixed EV loads",
        )

    m = len(future_idx)  # number of active LP slots

    # ------------------------------------------------------------------
    # Session-aware EV demand (issue #615).
    # When an EV is actively charging (session_charge_kw is set), treat
    # the first 2 hours as certain demand at that power level.
    # Grid-charging the battery is blocked during these slots to avoid
    # stacking battery charge on top of the EV draw.
    # ------------------------------------------------------------------
    slot_hours = (
        slot_duration_hours(slots[future_idx[0]].start, slots[future_idx[0]].end)
        if future_idx
        else 0.0
    )
    SESSION_HOURS = 2.0
    if slot_hours > 1e-9:
        SESSION_SLOTS = min(round(SESSION_HOURS / slot_hours), m)
    else:
        SESSION_SLOTS = min(8, m)  # fallback guard, should not normally trigger
    session_ev_indices: list[int] = []  # indices into active_evs
    session_slots_set: set[int] = set()
    if active_evs and slot_hours > 0:
        for ev_idx, ev in enumerate(active_evs):
            if ev.session_charge_kw is not None and ev.session_charge_kw > 1e-9:
                session_ev_indices.append(ev_idx)
        if session_ev_indices:
            session_slots_set = set(range(SESSION_SLOTS))
    _has_session_demand = bool(session_ev_indices)

    # ------------------------------------------------------------------
    # Variable layout:
    #   x = [ec(0..m-1), ed(0..m-1), gi(0..m-1), ge(0..m-1),
    #        pv(0..m-1), m(0..m-1),
    #        s_max_pen(0..m-1), s_min_pen(0..m-1),
    #        curt(0..m-1)]
    #   + [evN_c(0..m-1) for each active EV]      ← EV DC charge per slot
    #   + [evN_target_pen for each active EV]      ← deadline target slack
    # ------------------------------------------------------------------
    ec_off, ed_off, gi_off, ge_off, pv_off, m_off = 0, m, 2 * m, 3 * m, 4 * m, 5 * m
    s_max_off = 6 * m
    s_min_off = 7 * m
    curt_off = 8 * m
    n_vars = 9 * m

    # --- EV variable layout ---
    ev_var_offsets: list[int] = []  # start of ev_c[t] block per EV
    ev_pen_offsets: list[int] = []  # index of deadline penalty per EV
    for _ev_idx, _ev in enumerate(active_evs):
        ev_var_offsets.append(n_vars)
        n_vars += m  # ev_c[0..m-1] per EV
        ev_pen_offsets.append(n_vars)
        n_vars += 1  # single penalty per EV

    # --- Fuse constraint variables ---
    # When main_fuse_amps is provided and > 0, add gi_pen[t] penalty
    # variables that absorb grid import exceeding the fuse rating.
    fuse_active = main_fuse_amps is not None and main_fuse_amps > 1e-9
    if fuse_active:
        gi_pen_off = n_vars
        n_vars += m  # gi_pen[0..m-1] per slot
        # Calculate max grid import per slot in kWh (single source of truth
        # shared with the post-hoc EV/battery throttle in engine_core).
        # We derive interval_minutes from the first slot's duration.
        first_slot = slots[future_idx[0]]
        interval_minutes = slot_duration_hours(first_slot.start, first_slot.end) * 60.0
        assert main_fuse_amps is not None  # guarded by fuse_active
        max_grid_import_per_slot_kwh = fuse_max_energy_per_slot_kwh(
            main_fuse_amps,
            main_fuse_phases,
            interval_minutes / 60.0,
        )
        log_planner(
            "debug",
            "[milp] Main fuse constraint active: %d A × %d-phase → max %.3f kWh/slot "
            "(interval=%.0f min)",
            main_fuse_amps,
            main_fuse_phases,
            max_grid_import_per_slot_kwh,
            interval_minutes,
        )
    else:
        gi_pen_off = 0  # unused when fuse is inactive
        max_grid_import_per_slot_kwh = 0.0

    # Grid export power cap (issue #726): hard per-slot bound on ge[t].
    from custom_components.hsem.planner.milp._export_cap import _resolve_export_cap

    export_limit_active, max_grid_export_per_slot_kwh = _resolve_export_cap(
        max_grid_export_power_kw, slots, future_idx
    )
    # Resolve charge/discharge efficiencies for the energy balance equation.
    # The MILP must account for real-world conversion losses so its solution
    # matches the cost function's total_cost (which includes conversion loss
    # via the conversion_loss_cost term).
    charge_eff = clamp_efficiency(charge_efficiency_pct)
    discharge_eff = clamp_efficiency(discharge_efficiency_pct)
    charge_loss = 1.0 - charge_eff
    discharge_loss = 1.0 - discharge_eff

    # A user-configured excess-export buffer is a conditional reserve, not a
    # global minimum SoC. Binary variables distinguish intentional battery
    # export from ordinary self-consumption; constraints are added below.
    export_reserve_pct = min(max(excess_export_discharge_buffer_pct, 0.0), 100.0)
    export_reserve_kwh = usable_kwh * export_reserve_pct / 100.0
    export_reserve_active = bool(
        not no_export and export_reserve_kwh > 1e-9 and max_dis > 1e-9
    )
    export_mode_off: int | None = None
    export_reserve_checkpoints = None
    if export_reserve_active:
        from custom_components.hsem.planner.milp._export_reserve import (
            _next_solar_refill_checkpoints,
        )

        export_mode_off = n_vars
        n_vars += m
        export_reserve_checkpoints = _next_solar_refill_checkpoints(pv_avail)

    base_n_vars = n_vars
    secondary_layout = None
    if secondary_active:
        from custom_components.hsem.planner.milp._secondary_storage import (
            _allocate_secondary_variables,
        )

        secondary_layout, n_vars = _allocate_secondary_variables(base_n_vars, m)

    from custom_components.hsem.planner.milp._solver_execution import (
        _build_solve_and_finalize,
    )

    return _build_solve_and_finalize(
        locals()
        | {
            "candidate_milp": CANDIDATE_MILP,
            "export_mode_tiebreak_cost": _EXPORT_MODE_TIEBREAK_COST,
            "has_session_demand": _has_session_demand,
            "linprog": linprog,
            "min_action_kwh": _MIN_ACTION_KWH,
            "np": np,
            "session_slots": SESSION_SLOTS,
            "sync_attempt_diagnostics": _sync_attempt_diagnostics,
        }
    )


def is_scipy_available() -> bool:
    """Return ``True`` if scipy is importable in the current environment.

    The import result is cached at module level so that the blocking
    ``import scipy.optimize`` happens exactly once at import time rather
    than on every planner run inside the Home Assistant event loop.
    """
    return _SCIPY_AVAILABLE


# --- Module-level cache: computed once at import time --------------------
def _check_scipy() -> bool:
    """Check whether scipy is importable.  Called once at module load."""
    try:
        import scipy.optimize  # noqa: F401

        return True
    except ImportError:
        return False


_SCIPY_AVAILABLE: bool = _check_scipy()
