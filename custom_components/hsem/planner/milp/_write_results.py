"""Write MILP decision variables back into output slots.

Extracted from ``solve_milp`` so the orchestrator remains under 30 KB.
"""

from __future__ import annotations

import copy
from datetime import datetime

import numpy as np

from custom_components.hsem.models.ev_config import EVConfig
from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.utils.datetime_utils import slot_contains
from custom_components.hsem.utils.units import (
    ev_dc_to_ac_kwh,
    is_material_planned_energy_kwh,
    slot_duration_hours,
)


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
        current_kwh: Battery energy at horizon start (above floor, kWh).
        usable_kwh: Maximum usable energy (kWh).
        curt_sol_full: Solved curtailment per LP slot (kWh).
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
    if active_evs:
        for ev_idx, ev in enumerate(active_evs):
            ev_off = ev_var_offsets[ev_idx]
            ev_c_sol = result_x[ev_off : ev_off + m]
            for lp_t in range(m):
                ev_dc = float(ev_c_sol[lp_t])
                if ev_dc >= _min_action_kwh:
                    ev_ac_load_by_slot[lp_t] = ev_ac_load_by_slot.get(
                        lp_t, 0.0
                    ) + ev_dc_to_ac_kwh(ev_dc, ev.charger_efficiency)

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

        if resolved_charge > 0.0:
            # Primary charge is battery-side DC energy; PV surplus and EV load
            # are AC-side.  Subtract co-optimised EV demand (fixed EV demand is
            # already included in pv_avail), then require the remaining PV to
            # cover the complete rounded battery allocation after losses.
            # Raw EV AC is deliberately conservative if a later minimum-power
            # guard suppresses a tiny EV allocation: it may choose grid mode,
            # but can never falsely claim a grid-funded charge is solar.
            available_primary_pv_ac_kwh = max(
                float(pv_avail[lp_t])
                - ev_ac_load_by_slot.get(lp_t, 0.0)
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
        # Pre-compute full slot hours for power calculation (same for all slots
        # when interval is uniform).
        first_future_slot = out_slots[future_idx[0]]
        full_slot_hours = slot_duration_hours(
            first_future_slot.start, first_future_slot.end
        )

        for ev_idx, ev in enumerate(active_evs):
            ev_off = ev_var_offsets[ev_idx]
            ev_c_sol = result_x[ev_off : ev_off + m]
            for lp_t, slot_i in enumerate(future_idx):
                ev_dc_kwh = float(ev_c_sol[lp_t])
                if ev_dc_kwh < _min_action_kwh:
                    continue
                # AC load = DC / charger_eff (grid/PV draw)
                ac_load = round(ev_dc_to_ac_kwh(ev_dc_kwh, ev.charger_efficiency), 3)
                # Accumulate into slot EV fields (additive for multiple EVs)
                if ev.base_load_includes_ev:
                    out_slots[slot_i].ev_accounted_load_kwh += ac_load
                else:
                    out_slots[slot_i].ev_planned_load_kwh += ac_load
                out_slots[slot_i].ev_total_planned_load_kwh += ac_load

                # Compute AC charger target power (W) for this EV in this slot.
                # For the current (partially elapsed) slot, use remaining time
                # instead of the full slot width so the charger ramps to meet
                # the MILP's energy target within the available minutes.
                #
                # Cap at the charger's rated AC power — the MILP treats all
                # slots as full-width, so it may allocate max_charge_per_slot
                # to a slot with only a few minutes remaining.  The charger
                # physically cannot exceed its nameplate rating.
                max_ac_power_w = round(
                    (
                        ev_dc_to_ac_kwh(ev.max_charge_per_slot, ev.charger_efficiency)
                        / full_slot_hours
                    )
                    * 1000
                )
                slot_start = out_slots[slot_i].start
                slot_end = out_slots[slot_i].end
                if slot_contains(slot_start, slot_end, now):
                    remaining_hours = max(
                        slot_duration_hours(now, slot_end),
                        1.0 / 3600.0,  # 1 s minimum guard
                    )
                    ac_power_w = round(
                        (
                            ev_dc_to_ac_kwh(ev_dc_kwh, ev.charger_efficiency)
                            / remaining_hours
                        )
                        * 1000
                    )
                else:
                    ac_power_w = round(
                        (
                            ev_dc_to_ac_kwh(ev_dc_kwh, ev.charger_efficiency)
                            / full_slot_hours
                        )
                        * 1000
                    )
                ac_power_w = min(ac_power_w, max_ac_power_w)

                # Floor at the charger's minimum operating power — if the
                # target power is below the minimum the charger needs to
                # start, it will never deliver any energy.  Zero out the
                # field so the applier does not attempt to throttle below
                # the minimum.
                if (
                    ev.charger_min_power_w > 1e-9
                    and ac_power_w < ev.charger_min_power_w
                ):
                    ac_power_w = 0

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
        # Recompute estimated_net_consumption_kwh and estimated_cost_currency
        # to reflect new EV loads
        for i in future_idx:
            s = out_slots[i]
            s.estimated_net_consumption_kwh = (
                s.avg_house_consumption_kwh
                + s.ev_planned_load_kwh
                - s.solcast_pv_estimate_kwh
            )
            net = s.estimated_net_consumption_kwh
            if not s.price_actionable:
                s.estimated_cost_currency = 0.0
            elif net > 0:
                s.estimated_cost_currency = round(net * s.price.import_price, 4)
            else:
                s.estimated_cost_currency = round(net * s.price.export_price, 4)

    return out_slots
