"""HiGHS optimisation for Huawei storage, EVs, and optional secondary storage."""

from __future__ import annotations

import math
from datetime import datetime
from time import perf_counter
from typing import TYPE_CHECKING, Any

from custom_components.hsem.const import (
    MILP_SOLVER_TIMEOUT_DEFAULT_SECONDS,
    MILP_SOLVER_TIMEOUT_MAX_SECONDS,
    MILP_SOLVER_TIMEOUT_MIN_SECONDS,
)
from custom_components.hsem.models.ev_config import EVConfig
from custom_components.hsem.utils.datetime_utils import as_tz
from custom_components.hsem.utils.logger import log_planner
from custom_components.hsem.utils.misc import clamp_efficiency
from custom_components.hsem.utils.phase_power import PhasePowers
from custom_components.hsem.utils.units import (
    fuse_max_energy_per_slot_kwh,
    slot_duration_hours,
    timedelta_to_hours,
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
    """Solve the horizon and return independent result slots plus diagnostics.

    Huawei storage and EVs use the existing continuous formulation. When
    ``secondary_storage.valid`` is true, HiGHS also receives binary charge
    and SBU mode variables for the dedicated-load battery. The secondary
    output can remove only its configured load from the site balance and can
    never create grid export.

    - ``recommendation``  — one of ``BatteriesChargeGrid``, ``BatteriesDischargeMode``,
      ``ForceBatteriesDischarge``, or ``None`` (idle).
    - ``batteries_charged_kwh`` — energy entering the battery this slot (kWh).
    - ``batteries_discharged_kwh`` — energy discharged from the battery this slot
      (kWh).  Derived from the **resolved** ed after mutex resolution — this is the
      source of truth and must not be re-derived by the SoC simulation.
    - ``grid_import_kwh`` — grid import this slot (kWh), derived from the energy
      balance equation using the resolved ec/ed values.
    - ``grid_export_kwh`` — grid export this slot (kWh), derived from the energy
      balance equation using the resolved ec/ed values.
    - ``ev_planned_load_kwh`` — EV AC load that must be added to base consumption
      (when ``ev_configs`` is provided and ``base_load_includes_ev`` is False).
    - ``ev_accounted_load_kwh`` — EV AC load already captured in house consumption
      (when ``ev_configs`` is provided and ``base_load_includes_ev`` is True).
    - ``ev_total_planned_load_kwh`` — total EV AC load (sum of planned + accounted).
    - ``ev_charger_calculated_power`` — target AC power (W) for the primary EV charger.
    - ``ev_second_charger_calculated_power`` — target AC power (W) for the second EV.
    - ``estimated_net_consumption_kwh`` — recomputed after EV decisions.
    - ``estimated_cost_currency`` — recomputed after EV decisions.

    The SoC simulation (:func:`~soc_simulation.simulate_soc`) must be run
    by the caller **after** receiving these slots with
    ``milp_prepopulated=True`` to populate ``estimated_battery_soc``
    and ``estimated_battery_capacity_kwh`` while preserving the LP-derived
    energy flow fields.

    The MILP objective now includes conversion loss costs so its optimisation
    matches the cost function's ``total_cost``.  The energy balance equation
    accounts for charge/discharge efficiencies so ``gi[t]`` reflects real grid
    import (not the idealised lossless value).

    Args:
        slots:
            Fully populated (pre-SoC-simulation) slot list from the engine.
            Past slots with recommendation ``TimePassed`` are treated as fixed
            (zero charge/discharge) and excluded from the LP.
        now:
            Timezone-aware current datetime used to identify past slots.
        current_kwh:
            Battery energy above the discharge floor at the start of the horizon
            (kWh).  This is the LP's initial SoC state.
        usable_kwh:
            Maximum usable energy (max_soc − min_soc, kWh).  Acts as the SoC
            upper bound.
        max_charge_per_slot:
            Maximum energy chargeable per slot (kWh, post-conversion-loss).
        max_discharge_per_slot:
            Maximum energy dischargeable per slot (kWh).  ``None`` means unlimited;
            the LP uses ``usable_kwh`` as the effective ceiling in that case.
        cycle_cost_per_kwh:
            Battery cycle (depreciation) cost per kWh cycled.  Defaults to 0.0.
        charge_efficiency_pct:
            Charge-side efficiency as a percentage (0-100).  Energy stored in
            the battery equals input energy x (charge_efficiency_pct / 100).
            Defaults to 97 % (3 % charge-side loss).
        discharge_efficiency_pct:
            Discharge-side efficiency as a percentage (0-100).  Energy delivered
            to the house equals battery energy removed x (discharge_efficiency_pct / 100).
            Defaults to 97 % (3 % discharge-side loss).
        replacement_price_per_kwh:
            Terminal-SoC replacement price (currency/kWh) used to value the
            opportunity cost of ending the horizon with less stored energy.
            Passed from the engine (computed from the next discharge window).
            ``None`` disables the terminal-SoC credit term.
        min_export_price:
            Minimum export price (local currency/kWh) for the combined
            threshold below which export is not worthwhile.  Set by the
            caller to ``max(export_min_price, recommended_threshold)``
            where ``export_min_price`` is the inverter's physical block
            threshold and ``recommended_threshold`` is the
            depreciation-based discharge minimum.  Used for:
            - Clamping export prices to 0 before the LP solves (export
              below this price is physically blocked).
            - Deciding between ``ForceBatteriesDischarge`` and
              ``BatteriesDischargeMode`` in post-processing.
            Defaults to 0.0.
        ev_configs:
            Optional list of :class:`EVConfig` objects (one per EV).  When
            provided, the MILP co-optimises EV charging alongside the battery.
            EV loads are treated as decision variables with deadline-target
            soft constraints.  The ``ev_planned_load_kwh`` field on the input
            slots is ignored for EV-enabled slots (the MILP decides allocation).
            ``None`` (default) uses pre-computed ``ev_planned_load_kwh`` as
            fixed inputs (backward-compatible behaviour).
        no_export:
            When ``True``, caps battery discharge per slot so the battery
            never exports to the grid — it only serves house load.  The
            per-slot cap is ``ed[t] ≤ base_load[t] / discharge_eff``.
        main_fuse_amps:
            Main fuse/breaker rating in amps.  When provided and > 0, a soft
            constraint limits total grid import power per slot to
            ``main_fuse_amps * 230 * main_fuse_phases / 1000 * (interval_minutes / 60)`` kWh.
            A penalty variable ``gi_pen[t]`` absorbs any excess, preventing
            infeasibility when house base load alone exceeds the fuse rating.
            ``None`` or 0 disables the constraint (identical to current behaviour).
        main_fuse_phases:
            Electrical phase count (1 or 3).  Used as the multiplier in the
            max-grid-import formula above.  Defaults to 3 (three-phase).
            Single-phase installations MUST use 1.
        phase_power_imbalance_w:
            Optional signed per-phase offsets in Watts.  When supplied for a
            three-phase installation with a configured main fuse, the MILP
            adds hard per-phase import limits.  Huawei charge is modelled as
            balanced; secondary-storage charge and load switching are placed
            entirely on ``secondary_storage.grid_phase``.
        max_grid_export_power_kw:
            DNO/inverter grid export cap in kW (issue #726).  When > 0, the
            per-slot ``ge[t]`` is hard-bounded to
            ``max_grid_export_power_kw * slot_hours`` kWh so the plan never
            exceeds the site limit.  ``None`` or 0 disables the bound.
        battery_export_min_price:
            Per-slot hard floor below which intentional battery-to-grid
            discharge is forbidden (issue #752). `0.0` disables it.
            Caps `ed[t]` to `base_load[t]/discharge_eff` on blocked slots.
        excess_export_discharge_buffer_pct:
            Percentage of usable primary-battery capacity that an intentional
            battery-to-grid export must leave at the next forecast solar-refill
            checkpoint. The reserve is conditional and does not restrict
            ordinary house self-consumption. `0.0` disables it.
        solver_time_limit_seconds:
            Maximum wall-clock budget passed to HiGHS, clamped to the
            supported 1-60 second range. A time-limited solution is used only
            when its full decision vector passes HSEM's feasibility checks.
        attempt_diagnostics:
            Optional caller-owned dictionary populated with solver status,
            elapsed time, MIP gap, incumbent validation, and fallback reason.
            This remains available when no MILP candidate can be returned.

    Returns:
        A tuple ``(slots, diagnostics)`` where:
        - ``slots`` is a list of :class:`PlannedSlot` copies with MILP-derived
          recommendations.
        - ``diagnostics`` is a dict with keys ``"s_max_pen"``, ``"s_min_pen"``,
          ``"has_violations"``, ``"total_violation_kwh"``,
          ``"discharge_loss_cost_destination_aware"``.
        Returns ``None`` if the solver fails (unrelated to constraint
        violations — e.g., solver crash or numerical issue).
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
        populate_secondary_storage_load(slots, secondary_storage)

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
    future_mask = [as_tz(s.end, now.tzinfo) > now for s in slots]
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

    # ------------------------------------------------------------------
    # Build per-slot data arrays (future slots only)
    # ------------------------------------------------------------------
    p_imp = np.array([slots[i].price.import_price for i in future_idx], dtype=float)
    p_exp = np.array([slots[i].price.export_price for i in future_idx], dtype=float)

    # Replace NaN prices with 0 to prevent solver numerical issues
    p_imp = np.nan_to_num(p_imp, nan=0.0)
    p_exp = np.nan_to_num(p_exp, nan=0.0)

    # Per-slot hard floor for intentional battery-to-grid export (issue
    # #752). 0.0 (default) → mask all-False, backward compatible.
    battery_export_blocked = np.zeros(len(future_idx), dtype=bool)
    if battery_export_min_price > 1e-9:
        battery_export_blocked = p_exp < battery_export_min_price

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
        blocked = p_exp < min_export_price
        n_blocked = int(np.sum(blocked))
        if n_blocked > 0:
            log_planner(
                "debug",
                "[milp] Clamping %d export prices below min_price (%.4f) to 0 "
                "(max clamped=%.4f)",
                n_blocked,
                min_export_price,
                float(np.max(p_exp[blocked])),
            )
        p_exp = np.where(blocked, 0.0, p_exp)

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
    # keeping the original p_imp for the export-≤-import clamp and
    # penalty scaling (both need the real market signal).
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
        interval_minutes = timedelta_to_hours(first_slot.end - first_slot.start) * 60.0
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

    # ------------------------------------------------------------------
    # Build objective vector and constraint matrices
    # ------------------------------------------------------------------
    p_imp_max = float(np.max(p_imp)) if m > 0 else 0.1
    p_soc = max(p_imp_max, 0.1) * 100.0

    from custom_components.hsem.planner.milp._constraints import _build_constraints
    from custom_components.hsem.planner.milp._objective import _build_objective

    c_obj = _build_objective(
        slots,
        future_idx,
        now,
        m,
        n_vars,
        ec_off,
        ed_off,
        gi_off,
        ge_off,
        m_off,
        s_max_off,
        s_min_off,
        gi_pen_off,
        ev_var_offsets,
        ev_pen_offsets,
        active_evs,
        p_imp,
        p_imp_obj,
        p_exp,
        p_soc,
        cycle_cost_per_kwh,
        charge_loss,
        discharge_loss,
        time_discount_rate,
        replacement_price_per_kwh,
        fuse_active,
        usable_kwh=usable_kwh,
        max_charge_per_slot=max_charge_per_slot,
    )

    if export_mode_off is not None:
        c_obj[export_mode_off : export_mode_off + m] = _EXPORT_MODE_TIEBREAK_COST

    if secondary_active:
        from custom_components.hsem.planner.milp._secondary_storage import (
            _add_secondary_objective,
        )

        assert secondary_layout is not None
        assert secondary_storage is not None
        _add_secondary_objective(
            c_obj,
            layout=secondary_layout,
            config=secondary_storage,
            slots=slots,
            future_idx=future_idx,
            p_imp_obj=p_imp_obj,
            p_exp=p_exp,
            time_discount_rate=time_discount_rate,
            now=now,
        )

    constraints = _build_constraints(
        m,
        base_n_vars,
        ec_off,
        ed_off,
        gi_off,
        ge_off,
        pv_off,
        m_off,
        curt_off,
        gi_pen_off,
        s_max_off,
        s_min_off,
        ev_var_offsets,
        ev_pen_offsets,
        active_evs,
        pv_avail,
        base_load,
        ev_accounted,
        charge_eff,
        discharge_eff,
        current_kwh,
        usable_kwh,
        max_charge_per_slot,
        max_dis,
        max_grid_import_per_slot_kwh,
        fuse_active,
        no_export,
        session_slots_set,
        session_ev_indices,
        SESSION_SLOTS,
        slot_hours,
        _has_session_demand,
        max_grid_export_per_slot_kwh=max_grid_export_per_slot_kwh,
        export_limit_active=export_limit_active,
        battery_export_blocked=battery_export_blocked,
    )

    if export_mode_off is not None:
        from custom_components.hsem.planner.milp._export_reserve import (
            _add_battery_export_reserve_constraints,
        )

        assert export_reserve_checkpoints is not None
        constraints = _add_battery_export_reserve_constraints(
            constraints,
            n_vars=base_n_vars,
            m=m,
            ec_off=ec_off,
            ed_off=ed_off,
            export_mode_off=export_mode_off,
            current_kwh=current_kwh,
            usable_kwh=usable_kwh,
            discharge_eff=discharge_eff,
            max_discharge_kwh=max_dis,
            # base_load is already net of forecast PV, so daytime discharge
            # that would displace solar also activates the export reserve.
            residual_house_load=base_load,
            checkpoints=export_reserve_checkpoints,
            reserve_kwh=export_reserve_kwh,
        )

    integrality = None
    if secondary_active:
        from custom_components.hsem.planner.milp._secondary_storage import (
            _extend_secondary_constraints,
            _secondary_integrality,
        )

        assert secondary_layout is not None
        assert secondary_storage is not None
        constraints = _extend_secondary_constraints(
            constraints,
            n_vars=n_vars,
            m=m,
            layout=secondary_layout,
            config=secondary_storage,
            slots=slots,
            future_idx=future_idx,
            primary_discharge_off=ed_off,
            primary_max_discharge_kwh=max_dis,
        )
        integrality = _secondary_integrality(n_vars, m, secondary_layout)

    if export_mode_off is not None:
        if integrality is None:
            integrality = np.zeros(n_vars, dtype=int)
        integrality[export_mode_off : export_mode_off + m] = 1

    from custom_components.hsem.planner.milp._phase_fuse import (
        _add_phase_fuse_constraints,
        _phase_fuse_enabled,
    )

    phase_fuse_active = _phase_fuse_enabled(
        main_fuse_amps=main_fuse_amps,
        main_fuse_phases=main_fuse_phases,
        phase_power_imbalance_w=phase_power_imbalance_w,
    )
    if phase_fuse_active:
        assert main_fuse_amps is not None
        assert phase_power_imbalance_w is not None
        constraints = _add_phase_fuse_constraints(
            constraints,
            n_vars=n_vars,
            m=m,
            slots=slots,
            future_idx=future_idx,
            base_load=base_load,
            pv_avail=pv_avail,
            gi_off=gi_off,
            ge_off=ge_off,
            main_fuse_amps=main_fuse_amps,
            phase_power_imbalance_w=phase_power_imbalance_w,
            secondary_layout=secondary_layout,
            secondary_storage=secondary_storage,
        )
        log_planner(
            "debug",
            "[milp] Hard per-phase fuse constraints active: offsets=%s W",
            tuple(round(value, 1) for value in phase_power_imbalance_w),
        )

    A_eq = constraints["A_eq"]
    b_eq = constraints["b_eq"]
    A_ub = constraints["A_ub"]
    b_ub = constraints["b_ub"]
    bounds = constraints["bounds"]

    # Every per-slot decision block must align with the active horizon before
    # a time-limited incumbent may be decoded. Single scalar penalty variables
    # are covered by the full-vector length, bound, and matrix checks.
    variable_blocks: dict[str, tuple[int, int]] = {
        "primary_charge": (ec_off, m),
        "primary_discharge": (ed_off, m),
        "grid_import": (gi_off, m),
        "grid_export": (ge_off, m),
        "pv": (pv_off, m),
        "primary_throughput": (m_off, m),
        "soc_max_penalty": (s_max_off, m),
        "soc_min_penalty": (s_min_off, m),
        "curtailment": (curt_off, m),
    }
    for ev_index, offset in enumerate(ev_var_offsets):
        variable_blocks[f"ev_{ev_index}_charge"] = (offset, m)
    if fuse_active:
        variable_blocks["grid_import_penalty"] = (gi_pen_off, m)
    if export_mode_off is not None:
        variable_blocks["battery_export_mode"] = (export_mode_off, m)
    if secondary_layout is not None:
        for name, offset in secondary_layout.items():
            variable_blocks[f"secondary_{name}"] = (offset, m)

    # ------------------------------------------------------------------
    # Solve using HiGHS
    # ------------------------------------------------------------------
    solver_options: dict[str, float | bool] = {
        "time_limit": solver_time_limit,
        "disp": False,
    }
    if secondary_active or export_reserve_active:
        solver_options["mip_rel_gap"] = 0.005
    solve_started = perf_counter()
    try:
        result = linprog(
            c_obj,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            integrality=integrality,
            method="highs",
            options=solver_options,
        )
    except Exception as exc:
        elapsed = perf_counter() - solve_started
        _sync_attempt_diagnostics(
            attempt,
            attempt_diagnostics,
            solver_status="solver_exception",
            solver_message=str(exc),
            solver_elapsed_seconds=round(elapsed, 3),
            fallback_reason="solver_exception",
        )
        log_planner("warning", "[milp] Solver raised an exception: %s", exc)
        return None

    elapsed = perf_counter() - solve_started
    status_code = int(getattr(result, "status", -1))
    solver_message = str(getattr(result, "message", ""))
    raw_mip_gap = getattr(result, "mip_gap", None)
    try:
        mip_gap = float(raw_mip_gap) if raw_mip_gap is not None else None
    except TypeError, ValueError:
        mip_gap = None
    if mip_gap is not None and not math.isfinite(mip_gap):
        mip_gap = None

    is_time_limit = status_code == 1 and "time limit" in solver_message.casefold()
    if not result.success and not is_time_limit:
        _sync_attempt_diagnostics(
            attempt,
            attempt_diagnostics,
            solver_status="solver_failed",
            solver_status_code=status_code,
            solver_message=solver_message,
            solver_elapsed_seconds=round(elapsed, 3),
            solver_mip_gap=mip_gap,
            fallback_reason=f"solver_status_{status_code}",
        )
        log_planner(
            "warning",
            "[milp] Solver failed status=%s elapsed=%.3fs (%s); using fallback",
            status_code,
            elapsed,
            solver_message,
        )
        return None

    from custom_components.hsem.planner.milp._incumbent import validate_incumbent

    validation = validate_incumbent(
        getattr(result, "x", None),
        n_vars=n_vars,
        slot_count=len(slots),
        future_idx=future_idx,
        m=m,
        variable_blocks=variable_blocks,
        a_eq=A_eq,
        b_eq=b_eq,
        a_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        integrality=integrality,
    )
    if not validation.valid:
        if is_time_limit and validation.reason == "missing_solution_vector":
            solver_status = "time_limit_no_incumbent"
        elif is_time_limit:
            solver_status = "time_limit_invalid_incumbent"
        else:
            solver_status = "solver_invalid_solution"
        _sync_attempt_diagnostics(
            attempt,
            attempt_diagnostics,
            solver_status=solver_status,
            solver_status_code=status_code,
            solver_message=solver_message,
            solver_elapsed_seconds=round(elapsed, 3),
            solver_mip_gap=mip_gap,
            incumbent_validation=validation.reason,
            solution_validation=validation.as_dict(),
            fallback_reason=solver_status,
        )
        log_planner(
            "warning",
            "[milp] Rejected solver solution status=%s elapsed=%.3fs "
            "validation=%s; using fallback",
            status_code,
            elapsed,
            validation.reason,
        )
        return None

    # A time limit means only that optimality was not proven. The candidate
    # name intentionally remains exactly ``milp`` because it is a control-flow
    # key: the selector must trust its pre-populated primary and PowMr flows.
    solver_status = "time_limit_feasible_incumbent" if is_time_limit else "optimal"
    _sync_attempt_diagnostics(
        attempt,
        attempt_diagnostics,
        solver_status=solver_status,
        solver_optimal=not is_time_limit and status_code == 0,
        solver_status_code=status_code,
        solver_message=solver_message,
        solver_elapsed_seconds=round(elapsed, 3),
        solver_mip_gap=mip_gap,
        incumbent_used=is_time_limit,
        incumbent_validation=validation.reason,
        solution_validation=validation.as_dict(),
        fallback_reason="",
    )

    objective = getattr(result, "fun", None)
    try:
        objective_value = float(objective) if objective is not None else math.nan
    except TypeError, ValueError:
        objective_value = math.nan
    if not math.isfinite(objective_value):
        objective_value = float(np.dot(c_obj, result.x))
        result.fun = objective_value

    if is_time_limit:
        log_planner(
            "warning",
            "[milp] Time limit reached after %.3fs; using validated feasible "
            "incumbent gap=%s candidate=%s",
            elapsed,
            f"{mip_gap:.6f}" if mip_gap is not None else "n/a",
            CANDIDATE_MILP,
        )
    else:
        log_planner(
            "debug",
            "[milp] Optimal solution accepted elapsed=%.3fs gap=%s",
            elapsed,
            f"{mip_gap:.6f}" if mip_gap is not None else "n/a",
        )

    # ------------------------------------------------------------------
    # Compute terminal-SoC credit at end-of-horizon (diagnostic).
    # This matches cost_function.py's terminal_soc_value calculation:
    # terminal_soc_value = (initial_kwh - final_kwh) * replacement_price
    #
    # The LP objective now INCLUDES this term (see c_obj construction
    # above), so the solution itself already reflects this valuation.
    # This post-hoc calculation is retained as a diagnostic consistency
    # check and for the diagnostics dict.
    # ------------------------------------------------------------------
    ec_sol = result.x[ec_off : ec_off + m]
    ed_sol = result.x[ed_off : ed_off + m]

    # Compute final SoC from the LP solution
    final_soc_kwh = current_kwh + float(np.sum(ec_sol)) - float(np.sum(ed_sol))
    final_soc_kwh = max(0.0, min(final_soc_kwh, usable_kwh))  # clamp to bounds

    # Terminal-SoC credit: positive when plan ends with less energy (penalty),
    # negative when plan ends with more energy (credit).
    terminal_soc_credit = 0.0
    if replacement_price_per_kwh is not None and abs(replacement_price_per_kwh) > 1e-9:
        terminal_soc_credit = (current_kwh - final_soc_kwh) * replacement_price_per_kwh
        log_planner(
            "debug",
            "[milp] Terminal-SoC credit: initial=%.3f  final=%.3f  repl_price=%.4f  credit=%.4f",
            current_kwh,
            final_soc_kwh,
            replacement_price_per_kwh,
            terminal_soc_credit,
        )

    # Pre-compute curtailment solution (needed by both write-out and diagnostics)
    curt_sol_full = result.x[curt_off : curt_off + m]

    # Import helpers here to avoid circular imports with the milp package __init__
    from custom_components.hsem.planner.milp._diagnostics import (
        _compute_milp_diagnostics,
    )
    from custom_components.hsem.planner.milp._write_results import (
        _write_milp_results_to_slots,
    )

    # Write MILP decision variables into output slots
    out_slots = _write_milp_results_to_slots(
        slots,
        future_idx,
        now,
        ec_sol,
        ed_sol,
        result.x,
        m,
        ge_off,
        active_evs,
        ev_var_offsets,
        pv_avail,
        base_load,
        charge_eff,
        discharge_eff,
        p_exp,
        min_export_price,
        _has_session_demand,
        session_slots_set,
        current_kwh,
        usable_kwh,
        curt_sol_full,
        _min_action_kwh=_MIN_ACTION_KWH,
    )

    # Compute diagnostics
    diagnostics = _compute_milp_diagnostics(
        result,
        out_slots,
        slots,
        future_idx,
        m,
        s_max_off,
        s_min_off,
        curt_off,
        gi_off,
        gi_pen_off,
        replacement_price_per_kwh,
        min_export_price,
        p_imp_obj,
        discharge_loss,
        fuse_active,
        max_grid_import_per_slot_kwh,
        active_evs,
        ev_var_offsets,
        ev_pen_offsets,
        terminal_soc_credit,
        _min_action_kwh=_MIN_ACTION_KWH,
    )
    diagnostics.update(attempt)

    diagnostics["phase_fuse_active"] = phase_fuse_active

    diagnostics["battery_export_reserve_active"] = export_reserve_active
    diagnostics["battery_export_reserve_kwh"] = round(export_reserve_kwh, 6)
    if export_mode_off is not None:
        assert export_reserve_checkpoints is not None
        export_mode_sol = result.x[export_mode_off : export_mode_off + m]
        export_slots = [t for t, value in enumerate(export_mode_sol) if value > 0.5]
        soc_after = current_kwh + np.cumsum(ec_sol - ed_sol)
        checkpoint_soc = [
            float(soc_after[int(export_reserve_checkpoints[t])]) for t in export_slots
        ]
        min_checkpoint_soc = min(checkpoint_soc) if checkpoint_soc else None
        diagnostics["battery_export_reserve_slots"] = len(export_slots)
        diagnostics["battery_export_reserve_min_checkpoint_soc_kwh"] = (
            round(min_checkpoint_soc, 6) if min_checkpoint_soc is not None else None
        )
        log_planner(
            "debug",
            "[milp] battery_export_reserve  buffer=%.3f  export_slots=%d  "
            "min_checkpoint_soc=%s",
            export_reserve_kwh,
            len(export_slots),
            (f"{min_checkpoint_soc:.3f}" if min_checkpoint_soc is not None else "n/a"),
        )
    if phase_fuse_active:
        from custom_components.hsem.planner.milp._phase_fuse import (
            _phase_imports_from_solution_kwh,
        )

        assert phase_power_imbalance_w is not None
        phase_imports = _phase_imports_from_solution_kwh(
            result_x=result.x,
            m=m,
            slots=slots,
            future_idx=future_idx,
            gi_off=gi_off,
            ge_off=ge_off,
            phase_power_imbalance_w=phase_power_imbalance_w,
            secondary_layout=secondary_layout,
            secondary_storage=secondary_storage,
        )
        diagnostics["max_phase_import_kwh"] = round(
            max(value for phases in phase_imports for value in phases),
            6,
        )

    if secondary_active:
        from dataclasses import asdict

        from custom_components.hsem.planner.milp._secondary_diagnostics import (
            build_secondary_result_summary,
            log_secondary_result,
        )
        from custom_components.hsem.planner.milp._secondary_storage import (
            _write_secondary_results,
        )

        assert secondary_layout is not None
        assert secondary_storage is not None
        secondary_diagnostics = _write_secondary_results(
            out_slots,
            result_x=result.x,
            layout=secondary_layout,
            config=secondary_storage,
            future_idx=future_idx,
            minimum_action_kwh=_MIN_ACTION_KWH,
        )
        diagnostics.update(secondary_diagnostics)
        secondary_result = build_secondary_result_summary(
            out_slots,
            result_x=result.x,
            layout=secondary_layout,
            config=secondary_storage,
            future_idx=future_idx,
            min_export_price=min_export_price,
        )
        diagnostics["secondary_result"] = asdict(secondary_result)
        log_secondary_result(secondary_result)

    return out_slots, diagnostics


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
