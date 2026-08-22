"""Write MILP decision variables back into output slots.

Extracted from ``solve_milp`` so the orchestrator remains under 30 KB.
"""

from __future__ import annotations

import copy
import math
from datetime import datetime

import numpy as np

from custom_components.hsem.models.ev_config import EVConfig
from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.planner.cost_helpers import slot_grid_cash_flow_cost
from custom_components.hsem.utils.datetime_utils import slot_contains
from custom_components.hsem.utils.logger import log_planner
from custom_components.hsem.utils.units import (
    ev_dc_to_ac_kwh,
    is_material_planned_energy_kwh,
    slot_duration_hours,
)


def _reconcile_export_sources(
    out_slots: list[PlannedSlot],
    *,
    future_idx: list[int],
    primary_export_dc: np.ndarray | None,
    discharge_eff: float,
) -> float:
    """Publish AC source fields whose rounded sum is aggregate grid export."""
    max_error = 0.0
    for t, slot_i in enumerate(future_idx):
        slot = out_slots[slot_i]
        aggregate = round(max(slot.grid_export_kwh, 0.0), 3)
        solved_dc = (
            max(float(primary_export_dc[t]), 0.0)
            if primary_export_dc is not None
            else 0.0
        )
        battery_ac_limit = min(
            solved_dc * discharge_eff,
            max(slot.batteries_discharged_kwh, 0.0) * discharge_eff,
        )
        battery_ac = round(min(battery_ac_limit, aggregate), 3)
        pv_ac = round(max(aggregate - battery_ac, 0.0), 3)
        # Derive one source from the other after publication rounding so the
        # invariant is exact, rather than merely within solver tolerance.
        battery_ac = round(max(aggregate - pv_ac, 0.0), 3)
        slot.grid_export_kwh = aggregate
        slot.primary_battery_export_kwh = battery_ac
        slot.pv_export_kwh = pv_ac
        max_error = max(
            max_error,
            abs(aggregate - battery_ac - pv_ac),
        )
    return max_error


def _redistribute_below_minimum_power(
    dc_by_slot: dict[int, float],
    *,
    slot_hours: dict[int, float],
    charger_efficiency: float,
    charger_min_power_w: float,
    rated_ac_power_w: float,
    max_extra_dc: dict[int, float] | None = None,
) -> tuple[dict[int, float], float]:
    """Concentrate an EV allocation into slots the charger can actually run.

    The MILP models EV charge as a continuous variable, so it may spread a
    small amount of energy thinly across many slots.  A charger cannot run
    below its minimum operating power, so a slot whose implied AC power falls
    short of that minimum delivers nothing at all.  Zeroing only the power
    command leaves that energy in the plan's own accounting while no charger
    is ever told to deliver it, and the schedule silently falls short of
    ``total_kwh_needed``.

    Energy is first carried from later slots into earlier ones.  If the
    earliest fragments still combine to less than the minimum, a bounded
    recovery pass fills unused headroom in slots the solver already selected.
    That keeps every recipient inside the deadline the MILP respected while
    avoiding a plan/command mismatch when a commandable slot has room for the
    otherwise-unplaceable residue.

    Args:
        dc_by_slot: Battery-side kWh per output-slot index, as solved.
        slot_hours: Charging hours actually available in each of those slots.
        charger_efficiency: Charger efficiency as a fraction (0-1).
        charger_min_power_w: Minimum AC power the charger needs to start.
        rated_ac_power_w: Charger nameplate AC power, the per-slot ceiling.
        max_extra_dc: Per-slot ceiling on how much may be *added* to a slot
            beyond what the solver put there.  The solved plan satisfies the
            fuse, surplus-only and import constraints as solved; raising a
            slot's load past what unused PV or already-accepted import can
            fund would break them, so the ceiling keeps redistribution inside
            the feasible region rather than trusting the nameplate alone.

    Returns:
        ``(dc_by_slot, unplaceable_dc_kwh)`` — the concentrated allocation, and
        any energy no remaining slot could absorb.
    """
    if charger_min_power_w <= 1e-9 or not dc_by_slot:
        return dict(dc_by_slot), 0.0

    placed: dict[int, float] = {}
    deficit = 0.0
    ceilings = max_extra_dc or {}
    for slot_i in sorted(dc_by_slot, reverse=True):
        hours = slot_hours[slot_i]
        solved = dc_by_slot[slot_i]
        min_dc = charger_min_power_w * hours * charger_efficiency / 1000.0
        max_dc = rated_ac_power_w * hours * charger_efficiency / 1000.0
        # Never push a slot past what its own constraints can fund.
        max_dc = min(max_dc, solved + max(ceilings.get(slot_i, math.inf), 0.0))
        amount = solved + deficit
        accepted = min(amount, max_dc)
        deficit = amount - accepted
        if accepted >= min_dc - 1e-9:
            placed[slot_i] = accepted
        else:
            # Still below the charger's minimum even after absorbing later
            # slots; carry the whole amount further back rather than command
            # a power the charger would ignore.
            deficit += accepted

    # A reverse-only pass can still strand energy when the earliest solved
    # fragments add up to less than the charger minimum, even though a later
    # commandable slot has spare nameplate/constraint headroom.  For example,
    # [0.45, 0.45, 3.0, 2.1] kWh at a 1.242 kWh minimum used to publish only
    # 5.1 of the 6.0 kWh the solver and diagnostics promised.  Fill only slots
    # that the solver already selected, and reuse the same per-slot ceiling as
    # the first pass, so this cannot push energy beyond the solved deadline or
    # through a surplus-only constraint.
    if deficit > 1e-12:
        for slot_i in sorted(placed):
            hours = slot_hours[slot_i]
            solved = dc_by_slot[slot_i]
            max_dc = rated_ac_power_w * hours * charger_efficiency / 1000.0
            max_dc = min(
                max_dc,
                solved + max(ceilings.get(slot_i, math.inf), 0.0),
            )
            extra = min(max(max_dc - placed[slot_i], 0.0), deficit)
            placed[slot_i] += extra
            deficit -= extra
            if deficit <= 1e-12:
                deficit = 0.0
                break
    return placed, deficit


def _write_milp_results_to_slots(
    slots: list[PlannedSlot],
    future_idx: list[int],
    now: datetime,
    ec_sol: np.ndarray,  # type: ignore[name-defined]
    ed_sol: np.ndarray,  # type: ignore[name-defined]
    result_x: np.ndarray,  # type: ignore[name-defined]
    m: int,
    ge_off: int,
    active_evs: list[EVConfig],
    ev_var_offsets: list[int],
    pv_avail: np.ndarray,  # type: ignore[name-defined]
    base_load: np.ndarray,  # type: ignore[name-defined]
    charge_eff: float,
    discharge_eff: float,
    p_exp: np.ndarray,  # type: ignore[name-defined]
    min_export_price: float,
    _has_session_demand: bool,
    session_slots_set: set[int],
    current_kwh: float,
    usable_kwh: float,
    curt_sol_full: np.ndarray,  # type: ignore[name-defined]
    *,
    gi_off: int | None = None,
    grid_import_cap_per_slot_kwh: np.ndarray | None = None,  # type: ignore[name-defined]
    phase_extra_ac_per_slot_kwh: np.ndarray | None = None,  # type: ignore[name-defined]
    secondary_site_consumption_ac_per_slot: np.ndarray | None = None,  # type: ignore[name-defined]
    ev_writeback_diagnostics: dict[str, dict[str, object]] | None = None,
    session_ev_indices: list[int] | None = None,
    _min_action_kwh: float = 1e-4,
) -> list[PlannedSlot]:
    """Write MILP solution into a deep-copied slot list.

    Args:
        slots: Original slot list (will be deep-copied).
        future_idx: Indices of future (LP-variable) slots.
        now: Current datetime for time-aware power calculation.
        ec_sol: Solved charge energy per LP slot (kWh).
        ed_sol: Solved discharge energy per LP slot (kWh).
        result_x: Full LP solution vector.
        m: Number of active LP slots (``len(future_idx)``).
        ge_off: Offset of ``ge[t]`` variables in *result_x*.
        active_evs: List of active EV configs for EV write-out.
        ev_var_offsets: Start offset of each EV's ``ev_c[t]`` block.
        pv_avail: Per-slot PV surplus (positive kWh).
        base_load: Per-slot house demand after PV (positive kWh).
        charge_eff: Charge-side efficiency fraction (0-1).
        discharge_eff: Discharge-side efficiency fraction (0-1).
        p_exp: Per-slot export price array.
        min_export_price: Minimum export price threshold.
        _has_session_demand: Whether any EV has active session demand.
        session_slots_set: Session slot indices where grid-charge is blocked.
        session_ev_indices: Indices into *active_evs* of the EVs that actually
            have a live session.  A slot in ``session_slots_set`` is a fixed
            LP equality only for those EVs; another EV's allocation in the same
            slot stays flexible.
        current_kwh: Battery energy at horizon start (above floor, kWh).
        usable_kwh: Maximum usable energy (kWh).
        curt_sol_full: Solved curtailment per LP slot (kWh).
        gi_off: Offset of ``gi[t]`` variables in *result_x*. Direct helper
            callers may omit it; production supplies it so redistribution can
            use the exact solved import as its headroom baseline.
        grid_import_cap_per_slot_kwh: Finite physical/fuse import cap for each
            LP slot. EV concentration may absorb unused PV and only this much
            additional grid headroom.
        phase_extra_ac_per_slot_kwh: Additional EV AC load each LP slot can
            accept under the conservative unknown-phase envelope without
            exceeding any hard phase-import row. When the phase model is
            active, both PV and grid-funded concentration are capped by this
            stricter headroom.
        secondary_site_consumption_ac_per_slot: Incremental PowMr AC demand
            not already present in ``base_load``: charging plus its dedicated
            load while Utility supplies that load. It consumes the same PV
            headroom as EV charging.
        ev_writeback_diagnostics: Optional mutable mapping populated from the
            executable, post-redistribution EV allocation.
        _min_action_kwh: Minimum kWh threshold for action slots.
        Recommendations: The canonical Recommendations enum.

    Returns:
        A list of ``PlannedSlot`` copies with MILP-derived fields populated.
    """

    from custom_components.hsem.utils.recommendations import Recommendations

    out_slots: list[PlannedSlot] = [copy.copy(s) for s in slots]

    # Reset charge/discharge, energy-flow, and EV fields on all future slots;
    # past slots keep TimePassed.
    for i in future_idx:
        out_slots[i].recommendation = None
        out_slots[i].batteries_charged_kwh = 0.0
        out_slots[i].batteries_discharged_kwh = 0.0
        out_slots[i].grid_import_kwh = 0.0
        out_slots[i].grid_export_kwh = 0.0
        out_slots[i].primary_battery_export_kwh = 0.0
        out_slots[i].pv_export_kwh = 0.0
        out_slots[i].primary_battery_hold = False
        out_slots[i].ev_planned_load_kwh = 0.0
        out_slots[i].ev_accounted_load_kwh = 0.0
        out_slots[i].ev_total_planned_load_kwh = 0.0
        out_slots[i].ev_charger_calculated_power = 0.0
        out_slots[i].ev_second_charger_calculated_power = 0.0

    # Pre-compute per-slot total EV AC load from the LP solution.
    # Needed for deriving grid import/export from the energy balance
    # equation when mutex resolution alters ec/ed (issue #659):
    #   gi + pv + ed·η_dis = base_load + ec/η_chg + ge + curt + Σ ev_c/eff
    ev_ac_load_by_slot: dict[int, float] = {}
    # Battery-side kWh actually scheduled per EV, keyed by output-slot index,
    # after concentrating allocations the charger could not physically run.
    placed_dc_by_ev: list[dict[int, float]] = []
    session_dc_by_ev: list[dict[int, float]] = []
    hours_by_slot: dict[int, float] = {}
    if active_evs:
        first_future_slot = out_slots[future_idx[0]]
        full_slot_hours = slot_duration_hours(
            first_future_slot.start, first_future_slot.end
        )
        for slot_i in future_idx:
            slot_end = out_slots[slot_i].end
            if slot_contains(out_slots[slot_i].start, slot_end, now):
                # Only the remaining minutes of the current slot are usable,
                # which changes both the power an energy implies and how much
                # the slot can absorb.
                hours_by_slot[slot_i] = max(
                    slot_duration_hours(now, slot_end),
                    1.0 / 3600.0,  # 1 s minimum guard
                )
            else:
                hours_by_slot[slot_i] = full_slot_hours

        # Per-slot ceiling used when an EV charges past target on surplus
        # only.  The LP carries explicit surplus-only rows for that mode, so
        # concentrating energy there could manufacture grid import in a slot
        # the solver restricted to PV.  A slot the plan already imports in
        # has accepted grid energy at that price and may take more; a slot
        # funded purely by PV may absorb only the surplus still unused.
        ev_ac_solved = np.zeros(m)
        for ev_idx, ev in enumerate(active_evs):
            ev_off = ev_var_offsets[ev_idx]
            sol = result_x[ev_off : ev_off + m]
            for lp_t in range(m):
                dc = float(sol[lp_t])
                if dc >= _min_action_kwh:
                    ev_ac_solved[lp_t] += ev_dc_to_ac_kwh(dc, ev.charger_efficiency)

        unused_pv_ac_by_slot: dict[int, float] = {}
        total_extra_ac_by_slot: dict[int, float] = {}
        for lp_t, slot_i in enumerate(future_idx):
            partial_current_slot = hours_by_slot[slot_i] < full_slot_hours - 1e-9
            if phase_extra_ac_per_slot_kwh is None:
                phase_extra_ac = math.inf
            else:
                phase_extra_ac = float(phase_extra_ac_per_slot_kwh[lp_t])
                if not math.isfinite(phase_extra_ac):
                    phase_extra_ac = 0.0
                phase_extra_ac = max(phase_extra_ac, 0.0)
            # Only unused PV is source-safe headroom. Aggregate export is not:
            # part of it may be battery-origin, and moving EV energy into that
            # slot would silently make the house battery feed the car.
            unused_pv_ac = (
                0.0
                if partial_current_slot
                else max(
                    float(pv_avail[lp_t])
                    - float(ec_sol[lp_t]) / charge_eff
                    - (
                        float(secondary_site_consumption_ac_per_slot[lp_t])
                        if secondary_site_consumption_ac_per_slot is not None
                        else 0.0
                    )
                    - float(curt_sol_full[lp_t])
                    - ev_ac_solved[lp_t],
                    0.0,
                )
            )
            unused_pv_ac = min(unused_pv_ac, phase_extra_ac)
            unused_pv_ac_by_slot[slot_i] = unused_pv_ac

            # Grid headroom is safe only when the recipient is not already a
            # discharge/export slot.  Otherwise an added EV load consumes the
            # solved battery-origin flow before it creates any new import,
            # silently turning a grid/house-battery export into battery-to-EV
            # transfer.  Genuinely unused PV remains available through the
            # separate source-safe pool above.
            storage_or_export_active = (
                float(ed_sol[lp_t]) > _min_action_kwh
                or float(result_x[ge_off + lp_t]) > _min_action_kwh
            )
            if storage_or_export_active or partial_current_slot:
                grid_headroom_ac = 0.0
            elif grid_import_cap_per_slot_kwh is None:
                grid_headroom_ac = math.inf
            else:
                solved_grid_import = (
                    max(float(result_x[gi_off + lp_t]), 0.0)
                    if gi_off is not None
                    else max(
                        float(base_load[lp_t])
                        + float(ec_sol[lp_t]) / charge_eff
                        - float(ed_sol[lp_t]) * discharge_eff
                        + float(curt_sol_full[lp_t])
                        + ev_ac_solved[lp_t]
                        - float(pv_avail[lp_t]),
                        0.0,
                    )
                )
                grid_headroom_ac = max(
                    float(grid_import_cap_per_slot_kwh[lp_t]) - solved_grid_import,
                    0.0,
                )
            total_extra_ac_by_slot[slot_i] = min(
                unused_pv_ac + grid_headroom_ac,
                phase_extra_ac,
            )

        session_evs = set(session_ev_indices or [])
        for ev_idx, ev in enumerate(active_evs):
            ev_off = ev_var_offsets[ev_idx]
            ev_c_sol = result_x[ev_off : ev_off + m]
            # Only this EV's own session fixes its energy.  Keying on the slot
            # alone made one EV's live session freeze a second EV's flexible
            # allocation, which then kept the minimum-power zeroing and was
            # commanded 0 W while still counted as charged in the plan.
            ev_has_session = _has_session_demand and ev_idx in session_evs
            rated_ac_power_w = round(
                (
                    ev_dc_to_ac_kwh(ev.max_charge_per_slot, ev.charger_efficiency)
                    / full_slot_hours
                )
                * 1000
            )
            solved_dc: dict[int, float] = {}
            session_dc: dict[int, float] = {}
            for lp_t, slot_i in enumerate(future_idx):
                ev_dc = float(ev_c_sol[lp_t])
                if ev_dc < _min_action_kwh:
                    continue
                if ev_has_session and lp_t in session_slots_set:
                    # A live session is a fixed LP equality — the car is
                    # already drawing this power.  Observed demand, not a
                    # schedulable allocation, so it must never be moved.
                    session_dc[slot_i] = ev_dc
                else:
                    solved_dc[slot_i] = ev_dc

            placed_dc, unplaceable_dc = _redistribute_below_minimum_power(
                solved_dc,
                slot_hours=hours_by_slot,
                charger_efficiency=ev.charger_efficiency,
                charger_min_power_w=ev.charger_min_power_w,
                rated_ac_power_w=rated_ac_power_w,
                # The surplus-only rows the LP carries apply solely to an EV
                # charging past its target, so only that EV needs the ceiling.
                # Target-driven charging may legitimately import, and capping
                # it would drop the charge rather than concentrate it.
                max_extra_dc=(
                    {
                        slot_i: unused_pv_ac_by_slot.get(slot_i, 0.0)
                        * ev.charger_efficiency
                        for slot_i in solved_dc
                    }
                    if ev.charge_past_target
                    else {
                        slot_i: total_extra_ac_by_slot.get(slot_i, 0.0)
                        * ev.charger_efficiency
                        for slot_i in solved_dc
                    }
                ),
            )
            if unplaceable_dc > 1e-6:
                log_planner(
                    "debug",
                    "[milp] EV%s: %.3f kWh could not be placed at or above the "
                    "charger minimum of %.0f W before the deadline",
                    "2" if ev.is_second else "1",
                    unplaceable_dc,
                    ev.charger_min_power_w,
                )
            # PV and import headroom are shared across EVs. Spend only the
            # amount added beyond this EV's solved slot allocation; energy
            # removed from a donor slot is deliberately not reused, keeping
            # the writeback conservative and independent of EV iteration.
            for slot_i, dc in placed_dc.items():
                claimed_ac = max((dc - solved_dc.get(slot_i, 0.0)), 0.0) / max(
                    ev.charger_efficiency, 1e-9
                )
                pv_claim = min(
                    claimed_ac,
                    unused_pv_ac_by_slot.get(slot_i, 0.0),
                )
                unused_pv_ac_by_slot[slot_i] = max(
                    unused_pv_ac_by_slot.get(slot_i, 0.0) - pv_claim,
                    0.0,
                )
                total_extra_ac_by_slot[slot_i] = max(
                    total_extra_ac_by_slot.get(slot_i, 0.0) - claimed_ac,
                    0.0,
                )

            placed_dc_by_ev.append(placed_dc)
            session_dc_by_ev.append(session_dc)

            if ev_writeback_diagnostics is not None:
                published_dc = sum(placed_dc.values()) + sum(session_dc.values())
                deadline_penalty = 0.0
                if (
                    ev.deadline_slot is not None
                    and ev.target_kwh > ev.initial_soc_kwh + 1e-9
                ):
                    deadline_lp = max(0, min(ev.deadline_slot, m - 1))
                    lp_by_slot = {
                        output_slot: lp_index
                        for lp_index, output_slot in enumerate(future_idx)
                    }
                    delivered_by_deadline = sum(
                        dc
                        for output_slot, dc in {**session_dc, **placed_dc}.items()
                        if lp_by_slot[output_slot] <= deadline_lp
                    )
                    deadline_penalty = max(
                        ev.target_kwh - ev.initial_soc_kwh - delivered_by_deadline,
                        0.0,
                    )
                    # Executable slot energies publish at millikWh precision.
                    # Treat sub-5 Wh reconciliation residue as met rather than
                    # exposing a false deadline failure after safe writeback.
                    if deadline_penalty < 0.005:
                        deadline_penalty = 0.0
                ev_writeback_diagnostics[f"ev{ev_idx}"] = {
                    "total_dc_kwh": round(published_dc, 4),
                    "deadline_penalty_kwh": round(deadline_penalty, 4),
                    "deadline_met": deadline_penalty < 1e-6,
                    "unplaceable_dc_kwh": round(unplaceable_dc, 4),
                }

            # Build the AC load map from the *placed* allocation so grid
            # import/export, PV attribution and cost all derive from the
            # schedule that will actually be commanded.  Deriving them from
            # the raw LP solution instead would break the energy balance for
            # every slot the redistribution moved energy into or out of.
            lp_by_slot = {slot_i: lp_t for lp_t, slot_i in enumerate(future_idx)}
            for slot_i, dc in {**session_dc, **placed_dc}.items():
                lp_t = lp_by_slot[slot_i]
                ev_ac_load_by_slot[lp_t] = ev_ac_load_by_slot.get(
                    lp_t, 0.0
                ) + ev_dc_to_ac_kwh(dc, ev.charger_efficiency)

    # ------------------------------------------------------------------
    # Single merged energy-flow write-out pass (issue #659).
    #
    # Resolves degenerate LP vertices (simultaneous charge+discharge),
    # sets recommendation, and populates ALL per-slot energy-flow fields
    # (charge, discharge, grid import, grid export) consistently from
    # the SAME resolved ec/ed decision.  Grid import/export are derived
    # from the slot's energy balance equation rather than read from the
    # raw LP arrays, so they remain correct even when ec/ed are adjusted
    # by the mutex resolution.
    #
    # The SoC simulation (simulate_soc) must use these verbatim when
    # milp_prepopulated=True — never re-derive a different (greedy)
    # value from the recommendation label and net_demand.
    # ------------------------------------------------------------------
    running_soc = current_kwh
    published_soc_floor = min(current_kwh, 0.0)
    published_soc_ceiling = max(current_kwh, usable_kwh)
    for lp_t, slot_i in enumerate(future_idx):
        ec_kwh = float(ec_sol[lp_t])
        ed_kwh = float(ed_sol[lp_t])
        if ec_kwh > _min_action_kwh and ed_kwh > _min_action_kwh:
            # Degenerate LP vertex (simultaneous charge+discharge).
            # The LP is indifferent among cost-equivalent ec/ed
            # combinations.  Check actual SoC headroom at this point
            # in the resolved trajectory to distinguish a genuine
            # economic signal from solver noise near a SoC bound
            # (issue #662).
            #
            # net_charge_profit = p_imp·(η_dis − 1/η_chg) − 2·cycle_cost
            # is structurally always ≤ 0 for realistic efficiencies
            # and costs, so it cannot discriminate.  The LP's
            # s_max_pen/s_min_pen variables are a per-slot
            # hard-bound-violation signal, not a horizon-wide
            # degeneracy signal — they miss degenerate vertices
            # where SoC is merely near (not at) a bound.
            # Use actual resolved SoC headroom instead.
            net = ec_kwh - ed_kwh
            if net > _min_action_kwh:
                # Net charge candidate: clamp to remaining ceiling
                # headroom.  usable_kwh is the energy available
                # between end_of_discharge_soc and max_soc;
                # running_soc is measured from the same floor.
                headroom = usable_kwh - running_soc
                if headroom <= _min_action_kwh:
                    ec_kwh = 0.0
                    ed_kwh = 0.0
                else:
                    chosen = min(net, headroom)
                    ec_kwh = chosen
                    ed_kwh = 0.0
            elif net < -_min_action_kwh:
                # Net discharge candidate: clamp to remaining floor
                # headroom.  The discharge floor is already baked
                # into current_kwh/usable_kwh (see usable_capacity),
                # so 0.0 is the floor reference for running_soc.
                floor_headroom = running_soc
                if floor_headroom <= _min_action_kwh:
                    ec_kwh = 0.0
                    ed_kwh = 0.0
                else:
                    chosen = min(-net, floor_headroom)
                    ec_kwh = 0.0
                    ed_kwh = chosen
            else:
                # Net ~0 (both within _min_action_kwh of each other):
                # pure wash vertex — zero both.
                ec_kwh = 0.0
                ed_kwh = 0.0

        # Store the same rounded energy that diagnostics, simulation, and the
        # applier consume.  Do not create an action for sub-display solver
        # residue that rounds to zero.
        resolved_charge = round(max(ec_kwh, 0.0), 3)
        resolved_discharge = round(max(ed_kwh, 0.0), 3)

        # Reconcile publication rounding against the cumulative executable
        # inventory, not only the raw solver trajectory. Independent 3-decimal
        # rounding can otherwise accumulate across a long horizon and publish
        # a few Wh below the floor or above the ceiling even though the raw
        # variables satisfy every SoC row. Round the final admissible fragment
        # down to millikWh precision so writeback itself remains physical.
        if resolved_charge > 0.0:
            charge_headroom = max(published_soc_ceiling - running_soc, 0.0)
            if resolved_charge > charge_headroom:
                resolved_charge = max(
                    math.floor((charge_headroom + 1e-12) * 1000.0) / 1000.0,
                    0.0,
                )
        if resolved_discharge > 0.0:
            discharge_headroom = max(running_soc - published_soc_floor, 0.0)
            if resolved_discharge > discharge_headroom:
                resolved_discharge = max(
                    math.floor((discharge_headroom + 1e-12) * 1000.0) / 1000.0,
                    0.0,
                )

        if resolved_charge > 0.0:
            # Primary charge is battery-side DC energy; PV surplus, EV load,
            # and incremental PowMr consumption are AC-side. Subtract both
            # co-optimised consumers (fixed EV demand is already included in
            # pv_avail), then require the remaining PV to
            # cover the complete rounded battery allocation after losses.
            # Raw EV AC is deliberately conservative if a later minimum-power
            # guard suppresses a tiny EV allocation: it may choose grid mode,
            # but can never falsely claim a grid-funded charge is solar.
            available_primary_pv_ac_kwh = max(
                float(pv_avail[lp_t])
                - ev_ac_load_by_slot.get(lp_t, 0.0)
                - (
                    float(secondary_site_consumption_ac_per_slot[lp_t])
                    if secondary_site_consumption_ac_per_slot is not None
                    else 0.0
                )
                - max(float(curt_sol_full[lp_t]), 0.0),
                0.0,
            )
            required_primary_charge_ac_kwh = resolved_charge / charge_eff
            # Classify against the same three-decimal publication contract used
            # by execution.  A shortfall that publishes as exactly 0.001 kWh is
            # rounding residue and must not enable forced TOU grid charging;
            # anything larger remains a real grid-funded charge.
            published_source_shortfall_kwh = round(
                max(
                    required_primary_charge_ac_kwh - available_primary_pv_ac_kwh,
                    0.0,
                ),
                3,
            )
            solar_covers_charge = not is_material_planned_energy_kwh(
                published_source_shortfall_kwh
            )

            if solar_covers_charge:
                out_slots[
                    slot_i
                ].recommendation = Recommendations.BatteriesChargeSolar.value
            else:
                # Session-slot guard: do NOT assign BatteriesChargeGrid
                # during session EV demand slots (issue #615).  The LP
                # constraints already prevent ec[t] > 0 here, but this
                # guard protects against any edge case.
                is_session_slot = _has_session_demand and lp_t in session_slots_set
                if not is_session_slot:
                    out_slots[
                        slot_i
                    ].recommendation = Recommendations.BatteriesChargeGrid.value
                else:
                    # A validated incumbent should never reach this branch:
                    # the session constraint caps primary charging at residual
                    # PV.  Fail closed if solver tolerance or a future model
                    # change nevertheless returns a grid-funded session charge.
                    resolved_charge = 0.0
                    ec_kwh = 0.0
        # Write resolved charge/discharge kWh fields consistently.
        out_slots[slot_i].batteries_charged_kwh = resolved_charge
        out_slots[slot_i].batteries_discharged_kwh = resolved_discharge

        # Derive grid import/export from the slot's energy balance using
        # the SAME resolved (rounded) charge/discharge values stored in
        # the slot fields.  This guarantees the equality
        #   gi + pv + ed·η_dis = house_load + ec/η_chg + ge
        # holds exactly at 3-decimal precision.
        #
        # LP energy balance:  gi + pv + ed·η_dis
        #     = base_load + ec/η_chg + ge + curt + Σ ev_c/eff
        # ⇒ gi − ge = base_load + ec/η_chg − ed·η_dis + curt + Σ ev_c/eff − pv
        curt_kwh = float(curt_sol_full[lp_t])
        ev_ac_kwh = ev_ac_load_by_slot.get(lp_t, 0.0)
        net_flow = (
            base_load[lp_t]
            + resolved_charge / charge_eff
            - resolved_discharge * discharge_eff
            + curt_kwh
            + ev_ac_kwh
            - pv_avail[lp_t]
        )
        if net_flow > 0:
            out_slots[slot_i].grid_import_kwh = round(net_flow, 3)
            out_slots[slot_i].grid_export_kwh = 0.0
        else:
            out_slots[slot_i].grid_import_kwh = 0.0
            out_slots[slot_i].grid_export_kwh = round(-net_flow, 3)

        if resolved_discharge > 0.0:
            # Classify from the final rounded export field, not raw LP ge.
            # This keeps the executable mode tied to the same authoritative
            # energy balance that is exposed and scored downstream.
            if (
                is_material_planned_energy_kwh(out_slots[slot_i].grid_export_kwh)
                and p_exp[lp_t] >= min_export_price
            ):
                out_slots[
                    slot_i
                ].recommendation = Recommendations.ForceBatteriesDischarge.value
            else:
                out_slots[
                    slot_i
                ].recommendation = Recommendations.BatteriesDischargeMode.value

        # Complete a genuinely idle solved slot without changing its energy
        # allocation. The generic seasonal-fill pass is not label-only and
        # must never run after a validated MILP solve: it could consume PV that
        # this energy balance already assigned to export. The explicit hold
        # intent survives later EV display relabelling and tells the hardware
        # applier to execute the solved zero-charge/zero-discharge decision.
        if (
            out_slots[slot_i].recommendation is None
            and resolved_charge <= 1e-9
            and resolved_discharge <= 1e-9
        ):
            out_slots[slot_i].recommendation = Recommendations.BatteriesWaitMode.value
            out_slots[slot_i].primary_battery_hold = True

        # Advance resolved SoC for headroom-based degenerate-vertex
        # resolution in subsequent slots (issue #662).
        running_soc += resolved_charge - resolved_discharge

    # ------------------------------------------------------------------
    # Write MILP-derived EV charging decisions to output slots
    # ------------------------------------------------------------------
    if active_evs:
        for ev_idx, ev in enumerate(active_evs):
            # Charger nameplate: the MILP treats all slots as full-width, so
            # it may allocate max_charge_per_slot to a slot with only minutes
            # left.  The charger cannot exceed its rating either way.
            rated_ac_power_w = round(
                (
                    ev_dc_to_ac_kwh(ev.max_charge_per_slot, ev.charger_efficiency)
                    / full_slot_hours
                )
                * 1000
            )
            placed_dc = placed_dc_by_ev[ev_idx]
            session_dc = session_dc_by_ev[ev_idx]

            for slot_i in sorted({**session_dc, **placed_dc}):
                ev_dc_kwh = placed_dc.get(slot_i, session_dc.get(slot_i, 0.0))
                # AC load = DC / charger_eff (grid/PV draw)
                ac_load = round(ev_dc_to_ac_kwh(ev_dc_kwh, ev.charger_efficiency), 3)
                # Accumulate into slot EV fields (additive for multiple EVs).
                # Live current-slot injection leaves a pure-house baseline when
                # this measured session was removed, so its AC demand must be
                # added back as planned load even if historical house data
                # normally includes the EV. Future forecast slots remain
                # accounted in the usual base-includes topology.
                current_pure_house = (
                    ev.current_session_removed_from_base
                    and slot_contains(
                        out_slots[slot_i].start,
                        out_slots[slot_i].end,
                        now,
                    )
                )
                if ev.base_load_includes_ev and not current_pure_house:
                    out_slots[slot_i].ev_accounted_load_kwh += ac_load
                else:
                    out_slots[slot_i].ev_planned_load_kwh += ac_load
                out_slots[slot_i].ev_total_planned_load_kwh += ac_load

                ac_power_w = round(
                    (
                        ev_dc_to_ac_kwh(ev_dc_kwh, ev.charger_efficiency)
                        / hours_by_slot[slot_i]
                    )
                    * 1000
                )
                ac_power_w = min(ac_power_w, rated_ac_power_w)

                # Flexible allocations have already been concentrated above
                # the configured minimum. A live session is observed demand
                # from a charger that is already running, so its measured power
                # remains authoritative even when below the startup minimum.

                if ev.fixed_session_only:
                    # Observed external demand is modeled for energy/fuse safety,
                    # but disabled/ineligible smart planning must not emit an
                    # HSEM charger command.
                    continue

                # Write to the correct charger power field by EV identity
                # (is_second), NOT by list position (ev_idx).  When the
                # primary EV is disabled, active_evs[0] IS the second EV,
                # and ev_idx==0 would incorrectly route its power to
                # ev_charger_calculated_power instead of
                # ev_second_charger_calculated_power (issue #646).
                if ev.is_second:
                    out_slots[slot_i].ev_second_charger_calculated_power = max(
                        ac_power_w,
                        out_slots[slot_i].ev_second_charger_calculated_power,
                    )
                else:
                    out_slots[slot_i].ev_charger_calculated_power = max(
                        ac_power_w, out_slots[slot_i].ev_charger_calculated_power
                    )
        # Recompute the display net-load forecast to reflect new EV loads.
        for i in future_idx:
            s = out_slots[i]
            s.estimated_net_consumption_kwh = (
                s.avg_house_consumption_kwh
                + s.ev_planned_load_kwh
                - s.solcast_pv_estimate_kwh
            )

    # Meter cash flow must follow the final published grid flows, including
    # signed actionable prices. This also refreshes no-EV direct solves whose
    # baseline slot cost may otherwise be stale.
    for i in future_idx:
        s = out_slots[i]
        s.estimated_cost_currency = round(
            slot_grid_cash_flow_cost(s, export_min_price=min_export_price),
            4,
        )

    return out_slots
