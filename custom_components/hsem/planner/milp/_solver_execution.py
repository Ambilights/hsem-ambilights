"""Build, solve, validate, and publish a prepared MILP model."""

from __future__ import annotations

import math
from operator import itemgetter
from time import perf_counter
from typing import TYPE_CHECKING, Any

from custom_components.hsem.utils.logger import log_planner

if TYPE_CHECKING:
    from custom_components.hsem.models.planned_slot import PlannedSlot


_POSTWRITE_INVENTORY_TOLERANCE_KWH = 1e-6
_POSTWRITE_PHASE_TOLERANCE_KWH = 1e-3


def _validate_primary_postwrite_inventory(
    slots: list[PlannedSlot],
    future_idx: list[int],
    *,
    current_kwh: float,
    usable_kwh: float,
) -> dict[str, object]:
    """Validate cumulative primary inventory from executable slot energies.

    The solver constraints apply to raw floating-point variables, while the
    coordinator executes the three-decimal charge/discharge fields written to
    ``PlannedSlot``.  Validate that published trajectory independently so a
    future writeback change cannot manufacture energy or silently rely on the
    downstream SoC display clamp.

    An already-invalid initial reading remains admissible only up to its
    initial violation: over-capacity state may not exceed ``current_kwh`` and
    below-floor state may not fall below it.
    """
    initial = float(current_kwh)
    capacity = float(usable_kwh)
    if not math.isfinite(initial) or not math.isfinite(capacity) or capacity < 0.0:
        return {
            "valid": False,
            "reason": "invalid_inventory_bounds",
            "initial_kwh": initial,
            "usable_kwh": capacity,
        }

    lower_bound = min(initial, 0.0)
    upper_bound = max(initial, capacity)
    running = initial
    observed_min = initial
    observed_max = initial

    for sequence, slot_i in enumerate(future_idx):
        slot = slots[slot_i]
        try:
            charge = float(slot.batteries_charged_kwh)
            discharge = float(slot.batteries_discharged_kwh)
        except TypeError, ValueError:
            return {
                "valid": False,
                "reason": "non_numeric_primary_energy",
                "slot": sequence,
                "slot_index": slot_i,
            }
        if (
            not math.isfinite(charge)
            or not math.isfinite(discharge)
            or charge < 0.0
            or discharge < 0.0
        ):
            return {
                "valid": False,
                "reason": "invalid_primary_energy",
                "slot": sequence,
                "slot_index": slot_i,
                "charge_kwh": charge,
                "discharge_kwh": discharge,
            }

        running += charge - discharge
        observed_min = min(observed_min, running)
        observed_max = max(observed_max, running)
        if running < lower_bound - _POSTWRITE_INVENTORY_TOLERANCE_KWH:
            return {
                "valid": False,
                "reason": "primary_inventory_below_floor",
                "slot": sequence,
                "slot_index": slot_i,
                "inventory_kwh": round(running, 6),
                "lower_bound_kwh": round(lower_bound, 6),
            }
        if running > upper_bound + _POSTWRITE_INVENTORY_TOLERANCE_KWH:
            return {
                "valid": False,
                "reason": "primary_inventory_above_ceiling",
                "slot": sequence,
                "slot_index": slot_i,
                "inventory_kwh": round(running, 6),
                "upper_bound_kwh": round(upper_bound, 6),
            }

    return {
        "valid": True,
        "reason": "ok",
        "initial_kwh": round(initial, 6),
        "final_kwh": round(running, 6),
        "minimum_kwh": round(observed_min, 6),
        "maximum_kwh": round(observed_max, 6),
        "lower_bound_kwh": round(lower_bound, 6),
        "upper_bound_kwh": round(upper_bound, 6),
    }


def _build_solve_and_finalize(
    scope: dict[str, Any],
) -> tuple[list[PlannedSlot], dict[str, Any]] | None:
    """Complete a MILP solve from the prepared solve-context snapshot."""
    (
        np,
        slots,
        future_idx,
        now,
        m,
        n_vars,
        base_n_vars,
        column_layout,
        ec_off,
        ed_off,
        gi_off,
        ge_off,
        pv_off,
        m_off,
        primary_export_off,
        pv_export_off,
        export_source_mode_off,
        primary_action_mode_off,
        grid_flow_mode_off,
        s_max_off,
        s_min_off,
        curt_off,
        gi_pen_off,
        ev_var_offsets,
        ev_pen_offsets,
        active_evs,
        p_imp,
        p_imp_obj,
        p_exp,
        cycle_cost_per_kwh,
        time_discount_rate,
        replacement_price_per_kwh,
        terminal_cost_to_go,
        terminal_tiers,
        terminal_value_off,
        fuse_active,
        usable_kwh,
        max_charge_per_slot,
        export_mode_off,
        export_mode_tiebreak_cost,
        secondary_active,
        secondary_layout,
        secondary_storage,
        price_actionable,
        pv_avail,
        pv_export_ub_per_slot,
        grid_import_ub_per_slot,
        grid_export_ub_per_slot,
        base_load,
        ev_accounted,
        charge_eff,
        discharge_eff,
        current_kwh,
        max_dis,
        max_grid_import_per_slot_kwh,
        no_export,
        session_slots_set,
        session_ev_indices,
        session_slots,
        slot_hours,
        has_session_demand,
        max_grid_export_per_slot_kwh,
        export_limit_active,
        primary_export_blocked,
        export_reserve_checkpoints,
        export_reserve_kwh,
        export_reserve_active,
        main_fuse_amps,
        main_fuse_phases,
        phase_power_imbalance_w,
        solver_time_limit,
        linprog,
        attempt,
        attempt_diagnostics,
        sync_attempt_diagnostics,
        candidate_milp,
        min_action_kwh,
        min_export_price,
        battery_export_min_price,
    ) = itemgetter(
        "np",
        "slots",
        "future_idx",
        "now",
        "m",
        "n_vars",
        "base_n_vars",
        "column_layout",
        "ec_off",
        "ed_off",
        "gi_off",
        "ge_off",
        "pv_off",
        "m_off",
        "primary_export_off",
        "pv_export_off",
        "export_source_mode_off",
        "primary_action_mode_off",
        "grid_flow_mode_off",
        "s_max_off",
        "s_min_off",
        "curt_off",
        "gi_pen_off",
        "ev_var_offsets",
        "ev_pen_offsets",
        "active_evs",
        "p_imp",
        "p_imp_obj",
        "p_exp",
        "cycle_cost_per_kwh",
        "time_discount_rate",
        "replacement_price_per_kwh",
        "terminal_cost_to_go",
        "terminal_tiers",
        "terminal_value_off",
        "fuse_active",
        "usable_kwh",
        "max_charge_per_slot",
        "export_mode_off",
        "export_mode_tiebreak_cost",
        "secondary_active",
        "secondary_layout",
        "secondary_storage",
        "price_actionable",
        "pv_avail",
        "pv_export_ub_per_slot",
        "grid_import_ub_per_slot",
        "grid_export_ub_per_slot",
        "base_load",
        "ev_accounted",
        "charge_eff",
        "discharge_eff",
        "current_kwh",
        "max_dis",
        "max_grid_import_per_slot_kwh",
        "no_export",
        "session_slots_set",
        "session_ev_indices",
        "session_slots",
        "slot_hours",
        "has_session_demand",
        "max_grid_export_per_slot_kwh",
        "export_limit_active",
        "primary_export_blocked",
        "export_reserve_checkpoints",
        "export_reserve_kwh",
        "export_reserve_active",
        "main_fuse_amps",
        "main_fuse_phases",
        "phase_power_imbalance_w",
        "solver_time_limit",
        "linprog",
        "attempt",
        "attempt_diagnostics",
        "sync_attempt_diagnostics",
        "candidate_milp",
        "min_action_kwh",
        "min_export_price",
        "battery_export_min_price",
    )(scope)

    # ------------------------------------------------------------------
    # Build objective vector and constraint matrices
    # ------------------------------------------------------------------
    p_imp_max = float(np.max(p_imp)) if m > 0 else 0.1
    p_soc = max(p_imp_max, 0.1) * 100.0

    from custom_components.hsem.planner.milp._constraints import _build_constraints
    from custom_components.hsem.planner.milp._layout import MilpBoundsBuilder
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
        primary_export_off,
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
        time_discount_rate,
        (replacement_price_per_kwh if terminal_cost_to_go is None else None),
        fuse_active,
        usable_kwh=usable_kwh,
        max_charge_per_slot=max_charge_per_slot,
    )

    if terminal_value_off is not None:
        for tier_i, tier in enumerate(terminal_tiers):
            c_obj[terminal_value_off + tier_i] = -tier.value_per_kwh

    if export_mode_off is not None:
        c_obj[export_mode_off : export_mode_off + m] = export_mode_tiebreak_cost

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
            time_discount_rate=time_discount_rate,
            now=now,
        )

    from custom_components.hsem.utils.datetime_utils import slot_contains
    from custom_components.hsem.utils.units import slot_duration_hours

    available_slot_hours = np.asarray(
        [
            (
                slot_duration_hours(now, slots[slot_i].end)
                if slot_contains(slots[slot_i].start, slots[slot_i].end, now)
                else slot_duration_hours(slots[slot_i].start, slots[slot_i].end)
            )
            for slot_i in future_idx
        ],
        dtype=float,
    )
    session_slot_hours = np.zeros(m)
    session_hours_remaining = 2.0 if session_ev_indices else 0.0
    for t, available_hours in enumerate(available_slot_hours):
        if session_hours_remaining <= 1e-9:
            break
        # Commands are slot-constant. Include the whole slot that crosses the
        # two-hour certainty boundary so fixed energy still maps back to the
        # observed session power instead of a diluted partial-slot command.
        allocated_hours = float(available_hours)
        session_slot_hours[t] = allocated_hours
        session_hours_remaining -= allocated_hours
    session_slots_set = {
        t for t, hours in enumerate(session_slot_hours) if hours > 1e-9
    }
    session_slots = len(session_slots_set)
    has_session_demand = bool(session_slots_set)

    bounds_builder = MilpBoundsBuilder(column_layout)
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
        price_actionable,
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
        session_slots,
        slot_hours,
        available_slot_hours,
        session_slot_hours,
        has_session_demand,
        bounds_builder,
        max_grid_export_per_slot_kwh=max_grid_export_per_slot_kwh,
        export_limit_active=export_limit_active,
        battery_export_blocked=primary_export_blocked,
        primary_action_mode_off=primary_action_mode_off,
        pv_export_ub_per_slot=pv_export_ub_per_slot,
        grid_flow_mode_off=grid_flow_mode_off,
        grid_import_ub_per_slot=grid_import_ub_per_slot,
        grid_export_ub_per_slot=grid_export_ub_per_slot,
    )

    if export_mode_off is not None:
        from custom_components.hsem.planner.milp._export_reserve import (
            _add_battery_export_reserve_constraints,
        )

        assert export_reserve_checkpoints is not None
        constraints = _add_battery_export_reserve_constraints(
            constraints,
            bounds_builder=bounds_builder,
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
            primary_export_off=primary_export_off,
        )

    if terminal_value_off is not None:
        from custom_components.hsem.planner.milp._terminal_cost_to_go import (
            add_terminal_cost_to_go_constraints,
        )

        constraints = add_terminal_cost_to_go_constraints(
            constraints,
            bounds_builder=bounds_builder,
            n_vars=base_n_vars,
            m=m,
            ec_off=ec_off,
            ed_off=ed_off,
            terminal_value_off=terminal_value_off,
            tiers=terminal_tiers,
            current_kwh=current_kwh,
        )

    # Explicit destination rows supersede the legacy static site-load cap.
    # Retain the helper mask for direct-call compatibility, but do not add its
    # narrower rows to production models.
    primary_site_discharge_limited = np.zeros(m, dtype=bool)
    primary_site_discharge_cap_kwh = base_load.copy()
    if not active_evs:
        primary_site_discharge_cap_kwh = np.where(
            ev_accounted > 1e-9,
            np.maximum(base_load - ev_accounted, 0.0),
            primary_site_discharge_cap_kwh,
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
            bounds_builder=bounds_builder,
            n_vars=n_vars,
            m=m,
            layout=secondary_layout,
            config=secondary_storage,
            slots=slots,
            future_idx=future_idx,
            primary_discharge_off=ed_off,
            primary_charge_off=ec_off,
            primary_max_discharge_kwh=max_dis,
            primary_discharge_efficiency_fraction=discharge_eff,
            primary_max_charge_kwh=max_charge_per_slot,
            primary_site_discharge_limited=primary_site_discharge_limited,
            primary_site_discharge_cap_kwh=primary_site_discharge_cap_kwh,
            price_actionable=price_actionable,
            now=now,
        )
        integrality = _secondary_integrality(n_vars, m, secondary_layout)
    from custom_components.hsem.planner.milp._export_sources import (
        _add_export_source_constraints,
    )

    constraints = _add_export_source_constraints(
        constraints,
        n_vars=n_vars,
        m=m,
        slots=slots,
        future_idx=future_idx,
        ed_off=ed_off,
        gi_off=gi_off,
        ge_off=ge_off,
        primary_export_off=primary_export_off,
        pv_export_off=pv_export_off,
        export_source_mode_off=export_source_mode_off,
        grid_flow_mode_off=grid_flow_mode_off,
        discharge_eff=discharge_eff,
        primary_site_discharge_cap_kwh=primary_site_discharge_cap_kwh,
        primary_discharge_ub_per_slot=constraints["ed_ub_per_slot"],
        pv_export_ub_per_slot=pv_export_ub_per_slot,
        grid_import_ub_per_slot=grid_import_ub_per_slot,
        grid_export_ub_per_slot=grid_export_ub_per_slot,
        secondary_layout=secondary_layout,
        secondary_storage=secondary_storage,
    )

    if fuse_active:
        from custom_components.hsem.planner.milp._constraints import (
            _add_hard_aggregate_fuse_constraints,
        )

        constraints = _add_hard_aggregate_fuse_constraints(
            constraints,
            n_vars=n_vars,
            m=m,
            slots=slots,
            future_idx=future_idx,
            base_load=base_load,
            pv_avail=pv_avail,
            gi_off=gi_off,
            max_grid_import_per_slot_kwh=max_grid_import_per_slot_kwh,
            active_evs=active_evs,
            session_ev_indices=session_ev_indices,
            slot_hours=slot_hours,
            available_slot_hours=available_slot_hours,
            session_slot_hours=session_slot_hours,
            ev_var_offsets=ev_var_offsets,
            secondary_layout=secondary_layout,
            secondary_storage=secondary_storage,
        )

    if integrality is None:
        integrality = np.zeros(n_vars, dtype=int)
    integrality[export_source_mode_off : export_source_mode_off + m] = 1
    integrality[primary_action_mode_off : primary_action_mode_off + m] = 1
    integrality[grid_flow_mode_off : grid_flow_mode_off + m] = 1

    if export_mode_off is not None:
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
            active_evs=active_evs,
            ev_var_offsets=ev_var_offsets,
            session_slots_set=session_slots_set,
            secondary_layout=secondary_layout,
            secondary_storage=secondary_storage,
            now=now,
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
    bounds = bounds_builder.finalize()
    constraints["bounds"] = bounds

    # One declaration owns every block and validates all hand-built consumers.
    if n_vars != column_layout.column_count:
        raise ValueError(
            f"MILP variable count {n_vars} != declared {column_layout.column_count}"
        )
    column_layout.assert_model_width(
        objective=c_obj,
        a_eq=A_eq,
        a_ub=A_ub,
        bounds=bounds,
    )
    if integrality is not None and len(integrality) != n_vars:
        raise ValueError(
            f"MILP integrality width {len(integrality)} != declared {n_vars}"
        )
    variable_blocks = column_layout.variable_blocks()
    integral_blocks = [
        name
        for name, block in column_layout.blocks.items()
        if integrality is not None
        and block.width > 0
        and bool(
            np.all(
                integrality[block.offset : block.offset + block.width] != 0,
            )
        )
    ]

    # ------------------------------------------------------------------
    # Solve using HiGHS
    # ------------------------------------------------------------------
    solver_options: dict[str, float | bool] = {
        "time_limit": solver_time_limit,
        "disp": False,
    }
    if integrality is not None:
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
        sync_attempt_diagnostics(
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
        sync_attempt_diagnostics(
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
        sync_attempt_diagnostics(
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
    sync_attempt_diagnostics(
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
            candidate_milp,
        )
    else:
        log_planner(
            "debug",
            "[milp] Optimal solution accepted elapsed=%.3fs gap=%s",
            elapsed,
            f"{mip_gap:.6f}" if mip_gap is not None else "n/a",
        )

    # Path-independent final-inventory value: bounded and piecewise in
    # production, uniform only for compatible legacy callers. Non-actionable
    # primary blocks are fixed at zero, so the solved sum is the published-price
    # prefix.
    ec_sol = result.x[ec_off : ec_off + m]
    ed_sol = result.x[ed_off : ed_off + m]
    primary_export_sol = result.x[primary_export_off : primary_export_off + m]

    final_soc_kwh = current_kwh + float(np.sum(ec_sol)) - float(np.sum(ed_sol))
    final_soc_kwh = max(0.0, min(final_soc_kwh, usable_kwh))

    replacement_price = max(replacement_price_per_kwh or 0.0, 0.0)
    if not math.isfinite(replacement_price):
        replacement_price = 0.0
    if terminal_cost_to_go is None:
        terminal_inventory_value = replacement_price * (
            float(np.sum(ed_sol)) - float(np.sum(ec_sol))
        )
    else:
        terminal_inventory_value = terminal_cost_to_go.inventory_value(
            current_kwh
        ) - terminal_cost_to_go.inventory_value(final_soc_kwh)
    from custom_components.hsem.planner.cost_helpers import (
        PRIMARY_ACTION_TIEBREAK_COST,
    )

    primary_action_tiebreak = PRIMARY_ACTION_TIEBREAK_COST * (
        float(np.sum(ec_sol))
        + float(np.sum(ed_sol))
        - 1.5 * float(np.sum(ed_sol - primary_export_sol))
    )

    # Compatibility alias retained for diagnostics consumers.
    terminal_soc_credit = terminal_inventory_value
    log_planner(
        "debug",
        "[milp] Terminal inventory: initial=%.3f final=%.3f R=%.4f "
        "inventory=%.4f action_tie=%.6f",
        current_kwh,
        final_soc_kwh,
        replacement_price,
        terminal_inventory_value,
        primary_action_tiebreak,
    )
    # Pre-compute curtailment solution (needed by both write-out and diagnostics)
    curt_sol_full = result.x[curt_off : curt_off + m]

    # Import helpers here to avoid circular imports with the milp package __init__
    from custom_components.hsem.planner.milp._diagnostics import (
        _compute_milp_diagnostics,
    )
    from custom_components.hsem.planner.milp._write_results import (
        _reconcile_export_sources,
        _write_milp_results_to_slots,
    )

    # Write MILP decision variables into output slots
    hard_import_cap = constraints.get("hard_grid_import_cap_per_slot_kwh")
    if hard_import_cap is None:
        writeback_import_cap = grid_import_ub_per_slot
    else:
        writeback_import_cap = np.minimum(
            np.asarray(hard_import_cap, dtype=float),
            grid_import_ub_per_slot,
        )
    secondary_site_consumption_ac_per_slot = np.zeros(m)
    if secondary_active:
        from custom_components.hsem.utils.misc import clamp_efficiency

        assert secondary_layout is not None
        assert secondary_storage is not None
        secondary_charge_eff = clamp_efficiency(secondary_storage.charge_efficiency_pct)
        secondary_charge_off = secondary_layout["charge"]
        secondary_site_consumption_ac_per_slot = (
            result.x[secondary_charge_off : secondary_charge_off + m]
            / secondary_charge_eff
        )
        if not secondary_storage.base_load_includes_dedicated_load:
            secondary_sbu_off = secondary_layout["sbu_mode"]
            for t, slot_i in enumerate(future_idx):
                if float(result.x[secondary_sbu_off + t]) <= 0.5:
                    # Utility supplies a dedicated load that is absent from
                    # base_load, so it consumes the same PV pool as EV and
                    # PowMr charging. SBU supplies that load locally instead.
                    secondary_site_consumption_ac_per_slot[t] += max(
                        slots[slot_i].secondary_storage_load_kwh,
                        0.0,
                    )

    raw_phase_imports: list[tuple[float, float, float]] | None = None
    phase_import_limits: list[tuple[float, float, float]] | None = None
    phase_extra_ac_per_slot_kwh = None
    if phase_fuse_active:
        from custom_components.hsem.planner.milp._phase_fuse import (
            _full_slot_power_scale,
            _phase_import_limits_kwh,
            _phase_imports_from_solution_kwh,
        )

        assert main_fuse_amps is not None
        assert phase_power_imbalance_w is not None
        raw_phase_imports = _phase_imports_from_solution_kwh(
            result_x=result.x,
            m=m,
            slots=slots,
            future_idx=future_idx,
            gi_off=gi_off,
            ge_off=ge_off,
            phase_power_imbalance_w=phase_power_imbalance_w,
            active_evs=active_evs,
            ev_var_offsets=ev_var_offsets,
            secondary_layout=secondary_layout,
            secondary_storage=secondary_storage,
            now=now,
        )
        phase_import_limits = _phase_import_limits_kwh(
            slots=slots,
            future_idx=future_idx,
            base_load=base_load,
            pv_avail=pv_avail,
            main_fuse_amps=main_fuse_amps,
            phase_power_imbalance_w=phase_power_imbalance_w,
            active_evs=active_evs,
            session_slots_set=session_slots_set,
            secondary_storage=secondary_storage,
            now=now,
        )
        # Charger phase topology is not configured.  Every phase row therefore
        # treats the full EV command as if it landed on that phase.  The
        # tightest raw slack is already aggregate EV AC energy; do not multiply
        # it by three when bounding minimum-power redistribution.
        phase_extra_ac_per_slot_kwh = np.asarray(
            [
                max(
                    min(
                        limit - raw
                        for raw, limit in zip(raw_phases, limit_phases, strict=True)
                    )
                    / max(
                        _full_slot_power_scale(slots[future_idx[t]], now),
                        1.0,
                    ),
                    0.0,
                )
                for t, (raw_phases, limit_phases) in enumerate(
                    zip(raw_phase_imports, phase_import_limits, strict=True)
                )
            ],
            dtype=float,
        )

    ev_writeback_diagnostics: dict[str, dict[str, object]] = {}
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
        has_session_demand,
        session_slots_set,
        current_kwh,
        usable_kwh,
        curt_sol_full,
        gi_off=gi_off,
        grid_import_cap_per_slot_kwh=writeback_import_cap,
        phase_extra_ac_per_slot_kwh=phase_extra_ac_per_slot_kwh,
        secondary_site_consumption_ac_per_slot=(secondary_site_consumption_ac_per_slot),
        ev_writeback_diagnostics=ev_writeback_diagnostics,
        session_ev_indices=session_ev_indices,
        _min_action_kwh=min_action_kwh,
    )

    diagnostics = dict(attempt)

    primary_inventory_validation = _validate_primary_postwrite_inventory(
        out_slots,
        future_idx,
        current_kwh=current_kwh,
        usable_kwh=usable_kwh,
    )
    if not bool(primary_inventory_validation["valid"]):
        sync_attempt_diagnostics(
            attempt,
            attempt_diagnostics,
            solver_status="postwrite_invariant_failed",
            result_available=False,
            fallback_reason="primary_postwrite_inventory_failed",
            primary_postwrite_inventory_validation=primary_inventory_validation,
        )
        log_planner(
            "warning",
            "[milp] Rejected published primary inventory trajectory: %s",
            primary_inventory_validation,
        )
        return None
    diagnostics["primary_postwrite_inventory_validation"] = primary_inventory_validation

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
        assert raw_phase_imports is not None
        diagnostics["max_phase_import_kwh"] = round(
            max(value for phases in raw_phase_imports for value in phases),
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
            minimum_action_kwh=min_action_kwh,
            now=now,
            export_min_price=min_export_price,
            battery_export_min_price=max(
                min_export_price,
                battery_export_min_price,
            ),
            primary_site_discharge_limited=primary_site_discharge_limited,
        )
        if secondary_diagnostics is None:
            sync_attempt_diagnostics(
                attempt,
                attempt_diagnostics,
                solver_status="postwrite_invariant_failed",
                result_available=False,
                fallback_reason="secondary_postwrite_invariant_failed",
            )
            return None
        diagnostics.update(secondary_diagnostics)
        secondary_result = build_secondary_result_summary(
            out_slots,
            result_x=result.x,
            layout=secondary_layout,
            config=secondary_storage,
            future_idx=future_idx,
            min_export_price=min_export_price,
            now=now,
        )
        diagnostics["secondary_result"] = asdict(secondary_result)
        log_secondary_result(secondary_result)

    export_source_error = _reconcile_export_sources(
        out_slots,
        future_idx=future_idx,
        primary_export_dc=primary_export_sol,
        discharge_eff=discharge_eff,
    )

    if phase_fuse_active:
        from custom_components.hsem.planner.milp._phase_fuse import (
            _phase_imports_from_published_slots_kwh,
        )

        assert phase_power_imbalance_w is not None
        assert phase_import_limits is not None
        published_phase_imports = _phase_imports_from_published_slots_kwh(
            slots=out_slots,
            future_idx=future_idx,
            phase_power_imbalance_w=phase_power_imbalance_w,
            active_evs=active_evs,
            session_slots_set=session_slots_set,
            secondary_storage=secondary_storage,
            now=now,
        )
        phase_violations = [
            max(published - limit, 0.0)
            for published_phases, limit_phases in zip(
                published_phase_imports,
                phase_import_limits,
                strict=True,
            )
            for published, limit in zip(
                published_phases,
                limit_phases,
                strict=True,
            )
        ]
        max_phase_violation = max(phase_violations, default=0.0)
        published_max_phase_import = max(
            value for phases in published_phase_imports for value in phases
        )
        phase_postwrite_validation = {
            "valid": max_phase_violation <= _POSTWRITE_PHASE_TOLERANCE_KWH,
            "raw_max_phase_import_kwh": diagnostics["max_phase_import_kwh"],
            "published_max_phase_import_kwh": round(
                published_max_phase_import,
                6,
            ),
            "max_violation_kwh": round(max_phase_violation, 6),
            "tolerance_kwh": _POSTWRITE_PHASE_TOLERANCE_KWH,
        }
        diagnostics["published_max_phase_import_kwh"] = round(
            published_max_phase_import,
            6,
        )
        diagnostics["phase_postwrite_max_violation_kwh"] = round(
            max_phase_violation,
            6,
        )
        diagnostics["phase_postwrite_validation"] = phase_postwrite_validation
        if not bool(phase_postwrite_validation["valid"]):
            sync_attempt_diagnostics(
                attempt,
                attempt_diagnostics,
                solver_status="postwrite_invariant_failed",
                result_available=False,
                fallback_reason="phase_postwrite_invariant_failed",
                phase_postwrite_validation=phase_postwrite_validation,
            )
            log_planner(
                "warning",
                "[milp] Rejected published phase trajectory: %s",
                phase_postwrite_validation,
            )
            return None

    # Publish the objective terms from the same three-decimal executable
    # flows consumed by score_plan. Exact action binaries ensure this is only
    # publication-rounding reconciliation, never concealment of a raw cycle.
    reconciled_inventory_change_kwh = sum(
        out_slots[i].batteries_charged_kwh - out_slots[i].batteries_discharged_kwh
        for i in future_idx
    )
    final_soc_kwh = max(
        0.0,
        min(current_kwh + reconciled_inventory_change_kwh, usable_kwh),
    )
    if terminal_cost_to_go is None:
        terminal_inventory_value = -replacement_price * reconciled_inventory_change_kwh
    else:
        terminal_inventory_value = terminal_cost_to_go.inventory_value(
            current_kwh
        ) - terminal_cost_to_go.inventory_value(final_soc_kwh)
    primary_action_tiebreak = 0.0
    for slot_i in future_idx:
        slot = out_slots[slot_i]
        primary_export_dc = min(
            max(slot.primary_battery_export_kwh, 0.0) / discharge_eff,
            max(slot.batteries_discharged_kwh, 0.0),
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
    terminal_soc_credit = terminal_inventory_value

    diagnostics.update(
        _compute_milp_diagnostics(
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
            fuse_active,
            max_grid_import_per_slot_kwh,
            active_evs,
            ev_var_offsets,
            ev_pen_offsets,
            terminal_soc_credit,
            _min_action_kwh=min_action_kwh,
        )
    )
    if ev_writeback_diagnostics:
        diagnostics["ev"] = ev_writeback_diagnostics
    if terminal_cost_to_go is None:
        terminal_source = "legacy_scalar"
        terminal_boundary = None
        terminal_tier_records: list[dict[str, object]] = []
        terminal_total_quantity = max(usable_kwh, 0.0)
        terminal_highest_value = replacement_price
        terminal_lowest_value = replacement_price
        terminal_initial_valued_quantity = max(
            0.0,
            min(current_kwh, usable_kwh),
        )
        terminal_final_valued_quantity = max(
            0.0,
            min(final_soc_kwh, usable_kwh),
        )
        terminal_initial_value = replacement_price * terminal_initial_valued_quantity
        terminal_final_value = replacement_price * terminal_final_valued_quantity
    else:
        terminal_source = terminal_cost_to_go.source
        terminal_boundary = (
            terminal_cost_to_go.boundary.isoformat()
            if terminal_cost_to_go.boundary is not None
            else None
        )
        terminal_tier_records = [
            {
                "start": tier.start.isoformat(),
                "quantity_kwh": round(tier.quantity_kwh, 6),
                "value_per_kwh": round(tier.value_per_kwh, 6),
                "forecast_price_per_kwh": round(
                    tier.forecast_price_per_kwh,
                    6,
                ),
            }
            for tier in terminal_cost_to_go.tiers
        ]
        terminal_total_quantity = terminal_cost_to_go.total_quantity_kwh
        terminal_values = [tier.value_per_kwh for tier in terminal_cost_to_go.tiers]
        terminal_highest_value = max(terminal_values, default=0.0)
        terminal_lowest_value = min(terminal_values, default=0.0)
        terminal_initial_valued_quantity = (
            terminal_cost_to_go.inventory_valued_quantity(current_kwh)
        )
        terminal_final_valued_quantity = terminal_cost_to_go.inventory_valued_quantity(
            final_soc_kwh
        )
        terminal_initial_value = terminal_cost_to_go.inventory_value(current_kwh)
        terminal_final_value = terminal_cost_to_go.inventory_value(final_soc_kwh)
    terminal_inventory_value = terminal_initial_value - terminal_final_value
    diagnostics["terminal_inventory_value"] = round(terminal_inventory_value, 6)
    diagnostics["terminal_soc_credit"] = diagnostics["terminal_inventory_value"]

    diagnostics.update(
        {
            "terminal_cost_to_go_source": terminal_source,
            "terminal_cost_to_go_boundary": terminal_boundary,
            "terminal_cost_to_go_tier_count": len(terminal_tier_records),
            "terminal_cost_to_go_total_quantity_kwh": round(
                terminal_total_quantity,
                6,
            ),
            "terminal_cost_to_go_highest_value_per_kwh": round(
                terminal_highest_value,
                6,
            ),
            "terminal_cost_to_go_lowest_value_per_kwh": round(
                terminal_lowest_value,
                6,
            ),
            "terminal_cost_to_go_tiers": terminal_tier_records,
            "terminal_cost_to_go_initial_inventory_kwh": round(current_kwh, 6),
            "terminal_cost_to_go_final_inventory_kwh": round(final_soc_kwh, 6),
            "terminal_cost_to_go_initial_valued_quantity_kwh": round(
                terminal_initial_valued_quantity,
                6,
            ),
            "terminal_cost_to_go_final_valued_quantity_kwh": round(
                terminal_final_valued_quantity,
                6,
            ),
            "terminal_cost_to_go_initial_value": round(terminal_initial_value, 6),
            "terminal_cost_to_go_final_value": round(terminal_final_value, 6),
        }
    )
    diagnostics["primary_action_tiebreak"] = round(
        primary_action_tiebreak,
        6,
    )
    diagnostics["primary_battery_export_kwh"] = round(
        sum(out_slots[i].primary_battery_export_kwh for i in future_idx),
        6,
    )
    diagnostics["pv_export_kwh"] = round(
        sum(out_slots[i].pv_export_kwh for i in future_idx),
        6,
    )
    diagnostics["export_source_balance_max_error_kwh"] = round(
        export_source_error,
        6,
    )
    raw_grid_import = np.maximum(result.x[gi_off : gi_off + m], 0.0)
    raw_grid_export = np.maximum(result.x[ge_off : ge_off + m], 0.0)
    diagnostics["grid_import_export_overlap_max_kwh"] = round(
        float(np.max(np.minimum(raw_grid_import, raw_grid_export))),
        9,
    )
    diagnostics["model_variable_blocks"] = column_layout.as_dict()
    diagnostics["model_integral_blocks"] = integral_blocks
    diagnostics["model_integrality_count"] = int(np.count_nonzero(integrality))
    diagnostics["model_column_count"] = column_layout.column_count
    diagnostics["model_bounds_count"] = len(bounds)
    diagnostics["model_objective_column_count"] = len(c_obj)
    diagnostics["model_equality_column_count"] = int(A_eq.shape[1])
    diagnostics["model_inequality_column_count"] = int(A_ub.shape[1])
    return out_slots, diagnostics
