"""Core planning flow for the HSEM planner.

Orchestrates the planning pipeline and returns a :class:`PlannerOutput`.

**No Home Assistant types are imported here.**  Makes the engine
directly testable with plain ``pytest`` without a running HA instance.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from custom_components.hsem.models.ev_config import EVConfig
from custom_components.hsem.models.planner_input import PlannerInput
from custom_components.hsem.models.planner_output import PlannerOutput
from custom_components.hsem.models.terminal_cost_to_go import TerminalCostToGo
from custom_components.hsem.planner.candidate_generator import (
    CANDIDATE_MILP,
    CANDIDATE_PASSIVE,
    generate_candidates,
)
from custom_components.hsem.planner.candidate_selector import (
    select_best_candidate,
)
from custom_components.hsem.planner.cost_function import CostWeights, score_plan
from custom_components.hsem.planner.cost_helpers import slot_grid_cash_flow_cost
from custom_components.hsem.planner.engine_ev import (
    _build_and_inject_for_ev,
    _compute_ev_charger_power,
)
from custom_components.hsem.planner.engine_ev_milp import (
    _build_ev_configs_for_milp,
)
from custom_components.hsem.planner.engine_explanation import (
    _build_explanation,
    _derive_windows,
)
from custom_components.hsem.planner.engine_population import (
    _inject_live_data_into_current_slot,
    _parse_now,
    _populate_slots,
)
from custom_components.hsem.planner.engine_scheduling import _schedule_slots
from custom_components.hsem.planner.ev_planner import (
    EVChargingPlan,
    rebuild_ev_plan_from_slots,
)
from custom_components.hsem.planner.future_value import build_terminal_cost_to_go
from custom_components.hsem.planner.secondary_storage import (
    SecondaryTerminalPriceResolution,
    populate_secondary_storage_load,
    resolve_secondary_terminal_price_details,
)
from custom_components.hsem.planner.slot_population import (
    build_slots,
    build_time_series_index,
    populate_estimated_cost,
    populate_net_consumption,
    usable_capacity,
)
from custom_components.hsem.planner.solver_diagnostics import (
    populate_solver_explanation as _populate_solver_explanation,
    publish_selected_candidate_warnings as _publish_selected_candidate_warnings,
)
from custom_components.hsem.utils.datetime_utils import slot_contains, utc_key
from custom_components.hsem.utils.logger import log_planner
from custom_components.hsem.utils.misc import (
    calculate_recommended_threshold,
    resolve_cycle_cost,
)
from custom_components.hsem.utils.recommendations import Recommendations
from custom_components.hsem.utils.units import slot_duration_hours


def _restore_uncontrollable_sessions_to_passive(
    slots: list,
    ev_configs: list[EVConfig] | None,
    now: datetime,
) -> None:
    """Keep observed session-only demand visible in the fail-closed fallback."""
    fixed_sessions = [
        ev
        for ev in (ev_configs or [])
        if ev.fixed_session_only
        and ev.session_charge_kw is not None
        and ev.session_charge_kw > 1e-9
    ]
    for ev in fixed_sessions:
        session_charge_kw = ev.session_charge_kw
        assert session_charge_kw is not None
        hours_left = 2.0
        for slot in slots:
            if hours_left <= 1e-9:
                break
            if utc_key(slot.end) <= utc_key(now):
                continue
            available_hours = (
                slot_duration_hours(max(now, slot.start), slot.end)
                if slot_contains(slot.start, slot.end, now)
                else slot_duration_hours(slot.start, slot.end)
            )
            if available_hours <= 1e-9:
                continue
            # Commands are slot-constant, so include the whole slot that
            # crosses the two-hour certainty boundary, matching the MILP.
            session_ac_kwh = float(session_charge_kw) * available_hours
            slot.ev_total_planned_load_kwh = round(
                slot.ev_total_planned_load_kwh + session_ac_kwh,
                3,
            )
            current_was_removed = (
                ev.current_session_removed_from_base
                and slot_contains(slot.start, slot.end, now)
            )
            if ev.base_load_includes_ev and not current_was_removed:
                slot.ev_accounted_load_kwh = round(
                    slot.ev_accounted_load_kwh + session_ac_kwh,
                    3,
                )
            else:
                slot.ev_planned_load_kwh = round(
                    slot.ev_planned_load_kwh + session_ac_kwh,
                    3,
                )
            slot.batteries_charged_kwh = 0.0
            slot.batteries_discharged_kwh = 0.0
            slot.primary_battery_hold = True
            slot.recommendation = Recommendations.EVSmartCharging.value
            slot.estimated_net_consumption_kwh = (
                slot.avg_house_consumption_kwh
                + slot.ev_planned_load_kwh
                - slot.solcast_pv_estimate_kwh
            )
            hours_left -= available_hours


def _select_candidate(
    slots: list,
    inp: PlannerInput,
    now: datetime,
    current_kwh: float,
    usable_kwh: float,
    mcps: float,
    mdps: float | None,
    max_soc_kwh: float,
    terminal_cost_to_go: TerminalCostToGo,
    cw: CostWeights,
    sdh: float,
    rc: float,
    ev_configs: list[EVConfig] | None = None,
) -> tuple:
    """Generate and select best candidate plan."""
    candidates = generate_candidates(
        slots,
        inp,
        now,
        mcps,
        current_kwh=current_kwh,
        usable_kwh=usable_kwh,
        max_discharge_per_slot=mdps,
        terminal_cost_to_go=terminal_cost_to_go,
        ev_configs=ev_configs,
    )
    # Passive is the only executable fallback when the MILP is unavailable or
    # later rejected. Its baseline EV schedule has no hard fuse/source model,
    # so keep it EV-empty unconditionally before simulation and scoring.
    for candidate in candidates:
        if candidate.name != CANDIDATE_PASSIVE:
            continue
        for slot in candidate.slots:
            if utc_key(slot.end) <= utc_key(now):
                continue
            slot.ev_planned_load_kwh = 0.0
            slot.ev_accounted_load_kwh = 0.0
            slot.ev_total_planned_load_kwh = 0.0
            slot.ev_charger_calculated_power = 0.0
            slot.ev_second_charger_calculated_power = 0.0
            slot.estimated_net_consumption_kwh = (
                slot.avg_house_consumption_kwh - slot.solcast_pv_estimate_kwh
            )
            if slot.recommendation == Recommendations.EVSmartCharging.value:
                slot.recommendation = Recommendations.BatteriesWaitMode.value
        _restore_uncontrollable_sessions_to_passive(candidate.slots, ev_configs, now)

    winner, rejected, hyst = select_best_candidate(
        candidates,
        now=now,
        current_kwh=current_kwh,
        usable_kwh=usable_kwh,
        max_soc_capacity_kwh=max_soc_kwh,
        max_charge_per_slot=mcps,
        max_discharge_per_slot=mdps,
        rated_kwh=inp.battery_rated_capacity_kwh,
        end_of_discharge_soc_pct=inp.battery_end_of_discharge_soc_pct,
        cost_weights=cw,
        slot_duration_hours=sdh,
        charge_efficiency_pct=inp.battery_charge_efficiency_pct,
        discharge_efficiency_pct=inp.battery_discharge_efficiency_pct,
        terminal_cost_to_go=terminal_cost_to_go,
        required_capacity=rc,
        months_winter=inp.months_winter,
        export_min_price=inp.export_min_price,
        seasonal_fill_mode=inp.seasonal_fill_mode,
        wait_mode_behavior=inp.batteries_wait_mode_behavior,
        hysteresis_enabled=inp.planner_hysteresis_enabled,
        hysteresis_absolute=inp.planner_hysteresis_absolute,
        hysteresis_percentage=inp.planner_hysteresis_percentage,
        previous_winner_name=inp.previous_winner_name,
        previous_winner_score=inp.previous_winner_score,
        secondary_storage=inp.secondary_storage,
    )
    log_planner(
        "debug",
        "[core] _select_candidate DONE  candidates=%d  winner=%s  rejected=%d  hyst=%s",
        len(candidates),
        winner.name,
        len(rejected),
        f"applied={hyst.applied}" if hyst.applied else "inactive",
    )
    return candidates, winner, rejected, hyst


def run_planner(inp: PlannerInput) -> PlannerOutput:
    """Execute the HSEM planner and return a :class:`PlannerOutput`."""
    warnings: list[str] = []
    missing_inputs: list[str] = []
    now = _parse_now(inp.now_iso, inp.timezone_name)
    log_planner(
        "debug",
        "==== HSEM PLANNER RUN START ==== now=%s interval=%dmin horizon=%dh",
        inp.now_iso,
        inp.interval_minutes,
        inp.interval_length_hours,
    )
    # Dynamic discharge floor (issue #600): when enabled and higher than the
    # configured minimum, use it as the effective discharge floor.  This
    # reduces usable capacity and current capacity above the floor, which
    # naturally limits export and preserves reserve energy.
    _effective_eod_soc = inp.battery_end_of_discharge_soc_pct
    if (
        inp.dynamic_discharge_floor_pct is not None
        and inp.dynamic_discharge_floor_pct > _effective_eod_soc
    ):
        _effective_eod_soc = inp.dynamic_discharge_floor_pct
        log_planner(
            "debug",
            "[core] Dynamic discharge floor active: %.1f%% (configured min: %.1f%%)",
            _effective_eod_soc,
            inp.battery_end_of_discharge_soc_pct,
        )
    usable_kwh, current_kwh = usable_capacity(
        inp.battery_rated_capacity_kwh,
        inp.battery_soc_pct,
        _effective_eod_soc,
        inp.battery_max_soc_pct,
    )
    if inp.battery_rated_capacity_kwh <= 0:
        warnings.append(
            "battery_rated_capacity_kwh is zero or negative; battery simulation disabled."
        )
        usable_kwh = 0.0
        current_kwh = 0.0
    ws = inp.weight_1d + inp.weight_3d + inp.weight_7d + inp.weight_14d
    if ws != 100:
        warnings.append(
            f"Consumption weights sum to {ws}, not 100. Results may not be meaningful."
        )
    tsi = build_time_series_index(inp, now)
    slots = build_slots(inp, now)
    if not slots:
        log_planner(
            "warning",
            "[core] run_planner ABORTED — no slots generated",
        )
        warnings.append(
            "No slots generated; check interval_minutes and interval_length_hours."
        )
        return PlannerOutput(missing_inputs=missing_inputs, warnings=warnings)
    # Step 1 — populate time-series data
    data_quality, warnings, missing_inputs = _populate_slots(
        slots, inp, tsi, now, warnings, missing_inputs
    )
    log_planner(
        "debug",
        "[core] run_planner  step=1_populate_slots COMPLETE  "
        "data_quality=horizon_has_tomorrow=%s,horizon_days=%d  "
        "warnings=%d  missing=%d",
        data_quality.horizon_has_tomorrow,
        data_quality.horizon_days,
        len(warnings),
        len(missing_inputs),
    )
    # Step 1b — inject live solar and consumption into the current slot
    _inject_live_data_into_current_slot(slots, inp, now)

    secondary_terminal_price = SecondaryTerminalPriceResolution(None, "none")
    if inp.secondary_storage.valid:
        populate_secondary_storage_load(slots, inp.secondary_storage, now)
        secondary_terminal_price = resolve_secondary_terminal_price_details(
            slots,
            inp.secondary_storage,
            now,
            forecast=inp.price_forecast,
        )
        inp = replace(
            inp,
            secondary_storage=replace(
                inp.secondary_storage,
                replacement_price_per_kwh=secondary_terminal_price.price_per_kwh,
            ),
        )
        log_planner(
            "debug",
            "[core] Secondary storage enabled: capacity=%.3f kWh soc=%.1f%% "
            "load=%.0f W terminal_value=%s terminal_source=%s",
            inp.secondary_storage.capacity_kwh,
            inp.secondary_storage.current_soc_pct,
            inp.secondary_storage.load_power_w,
            (
                f"{secondary_terminal_price.price_per_kwh:.6f}"
                if secondary_terminal_price.price_per_kwh is not None
                else "None"
            ),
            secondary_terminal_price.source,
        )

    # Step 2 — EV planned load injection
    ev_cp: EVChargingPlan | None = None
    ev2_cp: EVChargingPlan | None = None
    combined_ev_raw = [0.0] * len(slots)
    combined_ev_inj = [0.0] * len(slots)
    populate_net_consumption(slots)
    sns = [max(-s.estimated_net_consumption_kwh, 0.0) for s in slots]
    ss = [s.start for s in slots]
    se = [s.end for s in slots]
    sp = [s.price.import_price for s in slots]
    if inp.ev_planned_load_enabled:
        ev_cp = _build_and_inject_for_ev(
            enabled=True,
            connected=inp.ev_planned_load_connected,
            smart=inp.ev_planned_load_smart_charging_enabled,
            soc=inp.ev_planned_load_current_soc_pct,
            target=inp.ev_planned_load_target_soc_pct,
            cap_kwh=inp.ev_planned_load_battery_capacity_kwh,
            pwr_kw=inp.ev_planned_load_charger_power_kw,
            eff=inp.ev_planned_load_charger_efficiency_pct,
            min_pwr_w=inp.ev_planned_load_charger_min_power_w,
            deadline=inp.ev_planned_load_deadline,
            base_includes=inp.ev_planned_load_base_load_includes_ev,
            allow_past_target=inp.ev_planned_allow_charge_past_target_soc,
            label="primary",
            now=now,
            slots=slots,
            slot_starts=ss,
            slot_ends=se,
            slot_prices=sp,
            slot_net_surplus=sns,
            combined_ev_raw_load=combined_ev_raw,
            combined_ev_injected_load=combined_ev_inj,
            warnings=warnings,
        )
    if inp.ev_second_planned_load_enabled:
        ev2_cp = _build_and_inject_for_ev(
            enabled=True,
            connected=inp.ev_second_planned_load_connected,
            smart=inp.ev_second_planned_load_smart_charging_enabled,
            soc=inp.ev_second_planned_load_current_soc_pct,
            target=inp.ev_second_planned_load_target_soc_pct,
            cap_kwh=inp.ev_second_planned_load_battery_capacity_kwh,
            pwr_kw=inp.ev_second_planned_load_charger_power_kw,
            eff=inp.ev_second_planned_load_charger_efficiency_pct,
            min_pwr_w=inp.ev_second_planned_load_charger_min_power_w,
            deadline=inp.ev_second_planned_load_deadline,
            base_includes=inp.ev_second_planned_load_base_load_includes_ev,
            allow_past_target=inp.ev_second_allow_charge_past_target_soc,
            label="second",
            now=now,
            slots=slots,
            slot_starts=ss,
            slot_ends=se,
            slot_prices=sp,
            slot_net_surplus=sns,
            combined_ev_raw_load=combined_ev_raw,
            combined_ev_injected_load=combined_ev_inj,
            warnings=warnings,
        )
    for i, s in enumerate(slots):
        s.ev_planned_load_kwh = combined_ev_inj[i]
        s.ev_accounted_load_kwh = round(combined_ev_raw[i] - combined_ev_inj[i], 3)
        s.ev_total_planned_load_kwh = round(combined_ev_raw[i], 3)

    # Compute per-slot EV charger target power (W) from the planner's
    # per-slot energy targets.  The EVChargingSlot.estimated_charged_kwh is
    # battery-side (DC) kWh delivered to the EV.  The AC power the charger
    # must draw is larger by 1/eff to account for charger/cable losses.
    _compute_ev_charger_power(slots, ss, ev_cp, inp.interval_minutes, now)
    _compute_ev_charger_power(slots, ss, ev2_cp, inp.interval_minutes, now, second=True)
    populate_net_consumption(slots)
    populate_estimated_cost(slots, export_min_price=inp.export_min_price)
    rt = calculate_recommended_threshold(
        purchase_price=inp.battery_purchase_price,
        expected_cycles=inp.battery_expected_cycles,
        usable_capacity=usable_kwh,
        capacity_loss_pct=inp.battery_capacity_loss_pct,
    )
    if rt > 0:
        warnings.append(
            f"Recommended price threshold: {rt:.4f} (battery depreciation)."
        )

    # Resolve the effective cycle cost once so the MILP, cost function, and
    # heuristic charge passes all use the same value.
    # Uses resolve_cycle_cost() — the single source of truth.
    effective_cycle_cost = resolve_cycle_cost(
        purchase_price=inp.battery_purchase_price,
        usable_kwh=usable_kwh,
        expected_cycles=inp.battery_expected_cycles,
        capacity_loss_pct=inp.battery_capacity_loss_pct,
        user_margin=inp.battery_cycle_cost_per_kwh,
    )

    # Step 3 — charge/discharge scheduling
    log_planner(
        "debug",
        "[core] run_planner  step=3_schedule_slots START  "
        "current=%.3f  usable=%.3f  rt=%.4f  cycle_cost=%.6f",
        current_kwh,
        usable_kwh,
        rt,
        effective_cycle_cost,
    )
    baseline_warnings: list[str] = []
    mcps, mdps, max_soc_kwh, rc, baseline_warnings = _schedule_slots(
        slots,
        inp,
        now,
        current_kwh,
        usable_kwh,
        rt,
        effective_cycle_cost,
        baseline_warnings,
    )
    log_planner(
        "debug",
        "[core] run_planner  step=3_schedule_slots COMPLETE",
    )
    # Step 4 — candidate plan generation and selection
    cw = CostWeights(
        min_soc_pct=_effective_eod_soc,
        max_soc_pct=inp.battery_max_soc_pct,
        cycle_cost_per_kwh=effective_cycle_cost,
        battery_purchase_price=inp.battery_purchase_price,
        battery_rated_capacity_kwh=inp.battery_rated_capacity_kwh,
        battery_expected_cycles=inp.battery_expected_cycles,
        battery_capacity_loss_pct=inp.battery_capacity_loss_pct,
        charge_efficiency_pct=inp.battery_charge_efficiency_pct,
        discharge_efficiency_pct=inp.battery_discharge_efficiency_pct,
        export_min_price=inp.export_min_price,
        battery_export_min_price=inp.battery_export_min_price,
        time_discount_rate=inp.time_discount_rate,
        battery_usable_capacity_kwh=usable_kwh,
        max_charge_per_slot_kwh=mcps,
        secondary_storage_enabled=inp.secondary_storage.valid,
        secondary_storage_charge_efficiency_pct=(
            inp.secondary_storage.charge_efficiency_pct
        ),
        secondary_storage_discharge_efficiency_pct=(
            inp.secondary_storage.discharge_efficiency_pct
        ),
        secondary_storage_cycle_cost_per_kwh=(inp.secondary_storage.cycle_cost_per_kwh),
        secondary_storage_replacement_price_per_kwh=(
            inp.secondary_storage.replacement_price_per_kwh
        ),
    )
    sdh = inp.interval_minutes / 60.0
    terminal_cost_to_go = build_terminal_cost_to_go(
        slots,
        now,
        forecast=inp.price_forecast,
        usable_kwh=usable_kwh,
        max_discharge_per_slot=mdps,
        discharge_efficiency_pct=inp.battery_discharge_efficiency_pct,
        cycle_cost_per_kwh=effective_cycle_cost,
    )
    terminal_boundary = (
        terminal_cost_to_go.boundary.isoformat()
        if terminal_cost_to_go.boundary is not None
        else "none"
    )
    log_planner(
        "debug",
        "[core] run_planner  step=4_candidate_selection START  source=%s "
        "boundary=%s tiers=%d quantity=%.3f",
        terminal_cost_to_go.source,
        terminal_boundary,
        len(terminal_cost_to_go.tiers),
        terminal_cost_to_go.total_quantity_kwh,
    )
    # Note: concentrate_discharge_on_expensive_slots() is now applied per-candidate
    # in the selector before scoring, so we don't run it on the baseline here.

    # Build EV configs for MILP co-optimisation (when EVs are active)
    ev_configs = _build_ev_configs_for_milp(inp, slots, now)
    candidates, winner, candidate_rejected, hysteresis_result = _select_candidate(
        slots,
        inp,
        now,
        current_kwh,
        usable_kwh,
        mcps,
        mdps,
        max_soc_kwh,
        terminal_cost_to_go,
        cw,
        sdh,
        rc,
        ev_configs=ev_configs,
    )
    _publish_selected_candidate_warnings(
        warnings,
        baseline_warnings,
        winner.name,
    )
    # Surface MILP penalty violations in warnings if the winner used penalties
    if (
        winner.name == CANDIDATE_MILP
        and winner.diagnostics is not None
        and winner.diagnostics.get("has_violations", False)
    ):
        diag = winner.diagnostics
        total = diag.get("total_violation_kwh", 0.0)
        fuse_total = diag.get("total_fuse_violation_kwh", 0.0)
        parts: list[str] = []
        if total > 1e-9:
            parts.append(f"SoC penalty={total:.4f} kWh")
        if fuse_total > 1e-9:
            parts.append(f"fuse excess={fuse_total:.4f} kWh")
        if parts:
            warnings.append(
                f"MILP: Penalty violations detected ({', '.join(parts)}). "
                f"The plan may have been forced due to out-of-bounds initial SoC "
                f"or main fuse limit."
            )
    # Step 5 — finalize plan from winner
    # Note: apply_optimization_strategy() and simulate_soc() are now applied
    # in the selector before scoring, so the winner's slots are already fully
    # populated. We do NOT re-run simulate_soc() here to avoid double-simulation
    # drift and to ensure the final score matches the selector's score.
    slots = winner.slots

    # Spec (planner-spec.md, Layer 2): slots with ev_total_planned_load_kwh > 0
    # are relabelled ev_smart_charging UNLESS the recommendation is one of the
    # protected set below.  batteries_charge_solar and batteries_wait_mode are
    # intentionally NOT protected — they are overridden so dashboards reflect
    # the EV activity rather than a solar-charge label during an EV session.
    _EV_KEEP = frozenset(
        {
            Recommendations.BatteriesChargeGrid.value,
            Recommendations.ForceBatteriesDischarge.value,
            Recommendations.ForceExport.value,
            Recommendations.TimePassed.value,
            Recommendations.MissingInputEntities.value,
        }
    )
    for s in slots:
        if abs(s.ev_total_planned_load_kwh) > 1e-9 and s.recommendation not in _EV_KEEP:
            s.recommendation = Recommendations.EVSmartCharging.value
        s.estimated_cost_currency = round(
            slot_grid_cash_flow_cost(
                s,
                export_min_price=inp.export_min_price,
            ),
            4,
        )
    cur_rec: str | None = None
    for s in slots:
        if slot_contains(s.start, s.end, now):
            cur_rec = s.recommendation
            break
    fut = [s for s in slots if utc_key(s.end) > utc_key(now)]
    bsoc_end = fut[-1].estimated_battery_soc_pct if fut else 0.0
    secondary_soc_end = (
        fut[-1].secondary_storage_estimated_soc_pct
        if fut and inp.secondary_storage.valid
        else 0.0
    )
    cw_out, dw_out = _derive_windows(slots)
    expl = _build_explanation(inp, slots, bsoc_end, now)
    expl.winner_name = winner.name
    expl.hysteresis_active = hysteresis_result.applied
    expl.hysteresis_reason = hysteresis_result.reason
    expl.previous_plan_name = hysteresis_result.previous_plan_name
    _populate_solver_explanation(
        expl,
        candidates,
        winner.name,
        inp.milp_solver_timeout_seconds,
    )
    selected_final_inventory_kwh = max(
        current_kwh
        + sum(
            slot.batteries_charged_kwh - slot.batteries_discharged_kwh
            for slot in slots
            if utc_key(slot.end) > utc_key(now) and slot.price_actionable
        ),
        0.0,
    )
    expl.terminal_cost_to_go_source = terminal_cost_to_go.source
    expl.terminal_cost_to_go_boundary = (
        terminal_cost_to_go.boundary.isoformat()
        if terminal_cost_to_go.boundary is not None
        else None
    )
    expl.terminal_cost_to_go_tier_count = len(terminal_cost_to_go.tiers)
    expl.terminal_cost_to_go_total_quantity_kwh = terminal_cost_to_go.total_quantity_kwh
    expl.terminal_cost_to_go_highest_value_per_kwh = max(
        (tier.value_per_kwh for tier in terminal_cost_to_go.tiers),
        default=0.0,
    )
    expl.terminal_cost_to_go_lowest_value_per_kwh = min(
        (tier.value_per_kwh for tier in terminal_cost_to_go.tiers),
        default=0.0,
    )
    expl.terminal_cost_to_go_initial_valued_quantity_kwh = (
        terminal_cost_to_go.inventory_valued_quantity(current_kwh)
    )
    expl.terminal_cost_to_go_final_valued_quantity_kwh = (
        terminal_cost_to_go.inventory_valued_quantity(selected_final_inventory_kwh)
    )
    expl.terminal_cost_to_go_initial_value = terminal_cost_to_go.inventory_value(
        current_kwh
    )
    expl.terminal_cost_to_go_final_value = terminal_cost_to_go.inventory_value(
        selected_final_inventory_kwh
    )
    expl.secondary_terminal_price_source = secondary_terminal_price.source
    expl.secondary_terminal_price_per_kwh = secondary_terminal_price.price_per_kwh
    pc = score_plan(
        slots,
        cw,
        slot_duration_hours=sdh,
        now=now,
        initial_battery_kwh=current_kwh,
        terminal_cost_to_go=terminal_cost_to_go,
    )
    for rp in candidate_rejected:
        expl.rejected_plans.append(rp)

    log_planner(
        "debug",
        "[core] run_planner DONE  winner=%s  cost=%.4f  score=%.4f  "
        "cur_rec=%s  bsoc_end=%.1f%%  rc=%.3f  warnings=%d  missing=%d  "
        "cw=%d  dw=%d",
        winner.name,
        pc.total_cost,
        pc.score,
        cur_rec if cur_rec is not None else "(none)",
        bsoc_end,
        rc,
        len(warnings),
        len(missing_inputs),
        len(cw_out),
        len(dw_out),
    )

    # Rebuild the EV sensor plans from the selected executable slots. This is
    # required both for MILP writeback and for the fail-closed passive fallback.
    if ev_cp is not None:
        ev_cp = rebuild_ev_plan_from_slots(
            ev_cp,
            slots,
            now,
            charger_efficiency_pct=inp.ev_planned_load_charger_efficiency_pct,
            is_second=False,
        )
    if ev2_cp is not None:
        ev2_cp = rebuild_ev_plan_from_slots(
            ev2_cp,
            slots,
            now,
            charger_efficiency_pct=inp.ev_second_planned_load_charger_efficiency_pct,
            is_second=True,
        )

    return PlannerOutput(
        slots=slots,
        charge_windows=cw_out,
        discharge_windows=dw_out,
        current_recommendation=cur_rec,
        battery_soc_at_end=bsoc_end,
        secondary_storage_soc_at_end=secondary_soc_end,
        required_capacity_kwh=rc,
        missing_inputs=missing_inputs,
        warnings=warnings,
        time_series_index=tsi,
        data_quality=data_quality,
        explanation=expl,
        plan_cost=pc,
        candidates=candidates,
        winner_name=winner.name,
        ev_charging_plan=ev_cp,
        ev_second_charging_plan=ev2_cp,
    )
