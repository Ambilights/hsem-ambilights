"""Build MILP constraints and variable bounds.

Extracted from ``solve_milp`` so the orchestrator remains under 30 KB.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from custom_components.hsem.planner.milp._layout import MilpBoundsBuilder
from custom_components.hsem.planner.secondary_storage import (
    secondary_site_load_offset_kwh,
)
from custom_components.hsem.utils.misc import clamp_efficiency

if TYPE_CHECKING:
    from custom_components.hsem.models.ev_config import EVConfig
    from custom_components.hsem.models.planned_slot import PlannedSlot
    from custom_components.hsem.models.secondary_storage_config import (
        SecondaryStorageConfig,
    )


def _add_hard_aggregate_fuse_constraints(
    constraints: dict[str, Any],
    *,
    n_vars: int,
    m: int,
    slots: list[PlannedSlot],
    future_idx: list[int],
    base_load: np.ndarray,  # type: ignore[name-defined]
    pv_avail: np.ndarray,  # type: ignore[name-defined]
    gi_off: int,
    max_grid_import_per_slot_kwh: float,
    active_evs: list[EVConfig],
    session_ev_indices: list[int],
    slot_hours: float,
    available_slot_hours: np.ndarray,  # type: ignore[name-defined]
    session_slot_hours: np.ndarray,  # type: ignore[name-defined]
    ev_var_offsets: list[int],
    secondary_layout: dict[str, int] | None,
    secondary_storage: SecondaryStorageConfig | None,
) -> dict[str, Any]:
    """Prevent controllable charging from worsening an aggregate fuse overload.

    The existing aggregate fuse row is intentionally soft so an unavoidable
    household, live-session, or dedicated-load overload cannot make the whole
    model infeasible.  Its penalty alone is not a physical guard, however: a
    larger EV deadline or terminal-inventory benefit can rationally pay that
    penalty and schedule still more controllable import.

    These companion hard rows cap grid import at the larger of the configured
    fuse limit and the fixed no-action site demand.  Fixed demand already above
    the fuse therefore remains feasible and visible through ``gi_pen``, while
    Huawei, flexible EV, and PowMr charging can never make it worse.
    """
    old_a_ub = constraints["A_ub"]
    old_b_ub = constraints["b_ub"]
    old_rows = old_a_ub.shape[0]
    a_ub = np.zeros((old_rows + m, n_vars))
    b_ub = np.zeros(old_rows + m)
    a_ub[:old_rows, : old_a_ub.shape[1]] = old_a_ub
    b_ub[:old_rows] = old_b_ub

    session_evs = set(session_ev_indices)
    hard_caps = np.zeros(m)
    secondary_charge_eff = (
        clamp_efficiency(secondary_storage.charge_efficiency_pct)
        if secondary_storage is not None and secondary_storage.valid
        else 1.0
    )
    for t, slot_i in enumerate(future_idx):
        available_hours = max(float(available_slot_hours[t]), 1e-9)
        full_slot_scale = max(slot_hours / available_hours, 1.0)
        fixed_session_ac_kwh = sum(
            max(ev.session_charge_kw or 0.0, 0.0)
            * float(session_slot_hours[t])
            * full_slot_scale
            for ev_idx, ev in enumerate(active_evs)
            if ev_idx in session_evs
        )
        fixed_secondary_load_kwh = 0.0
        if (
            secondary_storage is not None
            and secondary_storage.valid
            and not secondary_storage.base_load_includes_dedicated_load
        ):
            fixed_secondary_load_kwh = max(
                slots[slot_i].secondary_storage_load_kwh,
                0.0,
            )

        fixed_site_import_kwh = max(
            float(base_load[t] - pv_avail[t])
            + fixed_session_ac_kwh
            + fixed_secondary_load_kwh * full_slot_scale,
            0.0,
        )
        allowed_import_kwh = max(
            max_grid_import_per_slot_kwh,
            fixed_site_import_kwh,
        )
        hard_caps[t] = allowed_import_kwh
        row = old_rows + t
        a_ub[row, gi_off + t] = 1.0

        # gi mixes full-slot primary/house projections with current-slot EV
        # and PowMr energy. Normalize only those partial-duration terms back to
        # their power-equivalent full-slot frame for the fuse comparison.
        duration_correction = full_slot_scale - 1.0
        if duration_correction > 1e-9:
            for ev_idx, ev in enumerate(active_evs):
                a_ub[row, ev_var_offsets[ev_idx] + t] += (
                    duration_correction / ev.charger_efficiency
                )
            if secondary_layout is not None and secondary_storage is not None:
                a_ub[row, secondary_layout["charge"] + t] += (
                    duration_correction / secondary_charge_eff
                )
                a_ub[row, secondary_layout["sbu_mode"] + t] -= (
                    duration_correction
                    * secondary_site_load_offset_kwh(slots[slot_i], secondary_storage)
                )
        # A dedicated load excluded from the house baseline is a constant term
        # already present once in gi; move only its scale correction to RHS.
        b_ub[row] = allowed_import_kwh - duration_correction * fixed_secondary_load_kwh

    constraints["A_ub"] = a_ub
    constraints["b_ub"] = b_ub
    constraints["hard_grid_import_cap_per_slot_kwh"] = hard_caps
    return constraints


def _build_constraints(
    m: int,
    n_vars: int,
    ec_off: int,
    ed_off: int,
    gi_off: int,
    ge_off: int,
    pv_off: int,
    m_off: int,
    curt_off: int,
    gi_pen_off: int,
    s_max_off: int,
    s_min_off: int,
    ev_var_offsets: list[int],
    ev_pen_offsets: list[int],
    active_evs: list[EVConfig],
    price_actionable: np.ndarray,  # type: ignore[name-defined]
    pv_avail: np.ndarray,  # type: ignore[name-defined]
    base_load: np.ndarray,  # type: ignore[name-defined]
    ev_accounted: np.ndarray,  # type: ignore[name-defined]
    charge_eff: float,
    discharge_eff: float,
    current_kwh: float,
    usable_kwh: float,
    max_charge_per_slot: float,
    max_dis: float,
    max_grid_import_per_slot_kwh: float,
    fuse_active: bool,
    no_export: bool,
    session_slots_set: set[int],
    session_ev_indices: list[int],
    session_slots: int,
    slot_hours: float,
    available_slot_hours: np.ndarray,  # type: ignore[name-defined]
    session_slot_hours: np.ndarray,  # type: ignore[name-defined]
    _has_session_demand: bool,
    bounds_builder: MilpBoundsBuilder,
    max_grid_export_per_slot_kwh: float = 0.0,
    export_limit_active: bool = False,
    battery_export_blocked: np.ndarray | None = None,  # type: ignore[name-defined]
    primary_action_mode_off: int | None = None,
    pv_export_ub_per_slot: np.ndarray | None = None,
    grid_flow_mode_off: int | None = None,
    grid_import_ub_per_slot: np.ndarray | None = None,
    grid_export_ub_per_slot: np.ndarray | None = None,
) -> dict:
    """Build all LP constraint matrices and variable bounds.

    Returns a dict with keys ``A_eq``, ``b_eq``, ``A_ub``, ``b_ub``,
    ``ev_discharge_guard_active``, and ``ed_ub_per_slot``. Variable bounds
    are written into *bounds_builder* by declared block name.
    """
    import numpy as np

    # ------------------------------------------------------------------
    # Equality constraints: energy balance per slot
    # gi[t] + pv[t] + ed[t]*discharge_eff
    #     = base_load[t] + ec[t]/charge_eff + ge[t] + curt[t] + Σ ev_c/eff
    # ->  gi - ec/η_chg + ed·η_dis + pv - ge - curt - Σ ev_c/eff = base_load
    #
    # EV charge energy ev_c[t] is DC-side (delivered to EV battery).
    # The AC grid/PV draw is ev_c[t] / charger_efficiency — that is the
    # load the house must supply.  base_load already EXCLUDES EV load
    # when ev_configs is active (net_load was rebuilt without EV).
    #
    # curt[t] allows the LP to explicitly curtail PV when battery is full
    # and export prices are low/negative.
    # ------------------------------------------------------------------
    A_eq = np.zeros((m, n_vars))  # NOSONAR
    for t in range(m):
        A_eq[t, ec_off + t] = -1.0 / charge_eff  # -ec[t]/charge_eff
        A_eq[t, ed_off + t] = 1.0 * discharge_eff  # +ed[t]*discharge_eff
        A_eq[t, gi_off + t] = 1.0  # +gi[t]
        A_eq[t, ge_off + t] = -1.0  # -ge[t]
        A_eq[t, pv_off + t] = 1.0  # +pv[t] (fixed to pv_avail[t])
        A_eq[t, curt_off + t] = -1.0  # -curt[t] (curtailment reduces available PV)
        # EV AC load: -ev_c[t] / charger_eff per active EV
        for ev_idx, ev in enumerate(active_evs):
            A_eq[t, ev_var_offsets[ev_idx] + t] = -1.0 / ev.charger_efficiency
    b_eq = base_load.copy()  # always non-negative — pv[t] covers surplus

    # ------------------------------------------------------------------
    # Inequality constraints:
    #   1. SoC recurrence: soc[t] = soc[0] + Σ_{k≤t} (ec[k] − ed[k])
    #      Upper (soft): Σ_{k≤t}(ec[k]−ed[k]) − s_max_pen[t] ≤ usable−soc0
    #      Lower (soft): −Σ_{k≤t}(ec[k]−ed[k]) − s_min_pen[t] ≤ soc0
    #      Penalty variables s_max_pen[t] and s_min_pen[t] absorb violations
    #      at high cost, preventing infeasibility from out-of-bounds initial SoC.
    #   2. Mutual exclusion: ec[t]/max_charge + ed[t]/max_dis ≤ 1
    #   3. ec[t] ≤ max_charge_per_slot  (via bounds)
    #   4. ed[t] ≤ max_dis              (via bounds)
    # ------------------------------------------------------------------
    # We encode SoC bounds as inequality rows:
    #   upper: cumsum(ec−ed)[t] − s_max_pen[t] ≤ (usable_kwh − current_kwh)
    #   lower: −cumsum(ec−ed)[t] − s_min_pen[t] ≤ current_kwh
    soc_rows = 2 * m
    exact_action_mode = primary_action_mode_off is not None
    action_rows = 2 * m if exact_action_mode else m
    # Cycle cost auxiliary rows: m[t] >= ec[t] and m[t] >= ed[t]
    #   → -m[t] + ec[t] <= 0  and  -m[t] + ed[t] <= 0
    cycle_rows = 2 * m
    A_ub = np.zeros((soc_rows + action_rows + cycle_rows, n_vars))  # NOSONAR
    b_ub = np.zeros(soc_rows + action_rows + cycle_rows)

    for t in range(m):
        for k in range(t + 1):
            # Upper SoC bound row (soft)
            A_ub[t, ec_off + k] = 1.0
            A_ub[t, ed_off + k] = -1.0
            # Lower SoC bound row (soft)
            A_ub[m + t, ec_off + k] = -1.0
            A_ub[m + t, ed_off + k] = 1.0
        # Penalty variable absorbs violation in upper bound
        A_ub[t, s_max_off + t] = -1.0
        # Penalty variable absorbs violation in lower bound
        A_ub[m + t, s_min_off + t] = -1.0
        b_ub[t] = usable_kwh - current_kwh  # upper SoC headroom
        b_ub[m + t] = current_kwh  # lower SoC headroom

        if exact_action_mode:
            assert primary_action_mode_off is not None
            charge_row = 2 * m + t
            discharge_row = 3 * m + t
            A_ub[charge_row, ec_off + t] = 1.0
            A_ub[charge_row, primary_action_mode_off + t] = -max_charge_per_slot
            A_ub[discharge_row, ed_off + t] = 1.0
            A_ub[discharge_row, primary_action_mode_off + t] = max_dis
            b_ub[discharge_row] = max_dis
        else:
            A_ub[2 * m + t, ec_off + t] = 1.0 / max_charge_per_slot
            A_ub[2 * m + t, ed_off + t] = 1.0 / max_dis
            b_ub[2 * m + t] = 1.0

    # Cycle cost auxiliary: m[t] >= ec[t]  →  -m[t] + ec[t] <= 0
    #                     m[t] >= ed[t]  →  -m[t] + ed[t] <= 0
    cycle_row_start = soc_rows + action_rows
    for t in range(m):
        A_ub[cycle_row_start + t, ec_off + t] = 1.0
        A_ub[cycle_row_start + t, m_off + t] = -1.0
        b_ub[cycle_row_start + t] = 0.0
        A_ub[cycle_row_start + m + t, ed_off + t] = 1.0
        A_ub[cycle_row_start + m + t, m_off + t] = -1.0
        b_ub[cycle_row_start + m + t] = 0.0

    # ------------------------------------------------------------------
    # EV discharge guard: when base_load_includes_ev=True and EV
    # co-optimisation is NOT active, the EV load is already captured in
    # avg_house_consumption_kwh via ev_accounted_load_kwh.  The battery
    # must not discharge to cover this portion of base_load — the EV is
    # served by grid (or PV).
    #
    # When co-optimisation IS active, the EV has its own decision
    # variables and base_load already excludes EV load, so the guard is
    # skipped.
    #
    # Without this cap, the live-injected current-slot house consumption
    # (which includes EV power when the CT clamp is upstream of the
    # charger) causes the MILP to discharge the home battery into the EV
    # (issue #592).
    #
    # Per-slot upper bound on ed:
    #   ed[t] ≤ max(0, base_load[t] − ev_accounted[t]) / η_dis
    # Only slots where ev_accounted > 0 are capped; uncapped slots use max_dis.
    #
    # Note on PV interaction: although base_load is net of PV and
    # ev_accounted is gross EV load, the formula is exact — not
    # over-conservative.  With H = gross house consumption (incl. EV) and
    # P = PV production, base_load = max(H−P, 0), and the non-EV unmet
    # demand the battery may serve is max(H − ev − P, 0).  When
    # base_load > 0: base_load − ev = H − P − ev, identical.  When
    # base_load = 0 (PV surplus): H − P ≤ 0 so both sides are 0.  Hence
    # max(base_load − ev, 0) == max(H − ev − P, 0) in all cases.
    #
    # no_export and battery_export_blocked constrain the explicit battery-
    # export source block below, not total discharge. Local discharge may
    # still serve every eligible sink represented by the model without being
    # misclassified as export.
    #
    # ------------------------------------------------------------------
    ev_discharge_guard_active = (not active_evs) and bool(np.any(ev_accounted > 1e-9))
    if battery_export_blocked is None:
        battery_export_blocked = np.zeros(len(base_load), dtype=bool)
    ed_ub_per_slot: list[float] = []
    for t in range(m):
        if not bool(price_actionable[t]):
            ed_ub_per_slot.append(0.0)
        elif ev_discharge_guard_active and ev_accounted[t] > 1e-9:
            house_only_ac = max(
                float(base_load[t]) - float(ev_accounted[t]),
                0.0,
            )
            ed_ub_per_slot.append(min(house_only_ac / discharge_eff, max_dis))
        else:
            ed_ub_per_slot.append(max_dis)

    # ------------------------------------------------------------------
    # EV constraints (only when active_evs is non-empty)
    # ------------------------------------------------------------------
    # Row counts for EV constraints
    session_evs = set(session_ev_indices)
    num_evs = len(active_evs)
    ev_soc_rows = num_evs * m  # cumulative SOC upper bound per EV
    ev_deadline_rows = sum(
        1
        for ev in active_evs
        if ev.deadline_slot is not None and ev.target_kwh > ev.initial_soc_kwh + 1e-9
    )
    # Post-deadline zero-charge rows: for EVs with a deadline and no
    # charge-past-target, ev_c[t] = 0 for all t > deadline_slot.
    ev_post_deadline_rows = sum(
        m - 1 - max(0, min(ev.deadline_slot, m - 1))
        for ev in active_evs
        if ev.deadline_slot is not None
        and ev.target_kwh > ev.initial_soc_kwh + 1e-9
        and not ev.charge_past_target
    )
    # Target-cap rows: for EVs with a deadline and no charge-past-target,
    # Σ_{k≤D} ev_c[k] ≤ target_kwh - initial_soc_kwh
    # Caps EV charging at the economic target for pre-deadline slots,
    # preventing overcharge to full capacity_kwh.
    ev_target_rows = sum(
        1
        for ev in active_evs
        if ev.deadline_slot is not None
        and ev.target_kwh > ev.initial_soc_kwh + 1e-9
        and not ev.charge_past_target
    )
    # Surplus-only rows: for charge-past-target EVs, ev_c[t]/eff ≤ max(0, pv[t] - base_load[t])
    ev_surplus_rows = sum(1 for ev in active_evs if ev.charge_past_target) * m
    ev_total_rows = (
        ev_soc_rows
        + ev_deadline_rows
        + ev_target_rows
        + ev_post_deadline_rows
        + ev_surplus_rows
    )

    if ev_total_rows > 0:
        # Extend A_ub and b_ub to accommodate EV rows
        existing_rows = soc_rows + action_rows + cycle_rows
        A_ub_old = A_ub
        b_ub_old = b_ub
        A_ub = np.zeros((existing_rows + ev_total_rows, n_vars))
        b_ub = np.zeros(existing_rows + ev_total_rows)
        A_ub[:existing_rows, :] = A_ub_old
        b_ub[:existing_rows] = b_ub_old

        ev_row = existing_rows
        for ev_idx, ev in enumerate(active_evs):
            ev_off = ev_var_offsets[ev_idx]
            is_session_ev = ev_idx in session_evs
            # EV SOC upper bound per slot: Σ_{k≤t} ev_c[k] ≤ cap − init
            #   For each t in 0..m-1:
            #   Σ_{k=0..t} ev_c[k] ≤ ev.capacity_kwh - ev.initial_soc_kwh
            headroom = max(ev.capacity_kwh - ev.initial_soc_kwh, 0.0)
            for t in range(m):
                fixed_session_dc = 0.0
                for k in range(t + 1):
                    if is_session_ev and k in session_slots_set:
                        fixed_session_dc += (
                            max(ev.session_charge_kw or 0.0, 0.0)
                            * float(session_slot_hours[k])
                            * ev.charger_efficiency
                        )
                    else:
                        A_ub[ev_row + t, ev_off + k] = 1.0
                b_ub[ev_row + t] = max(headroom - fixed_session_dc, 0.0)
            ev_row += m

            # EV deadline soft constraint:
            # initial_soc + Σ_{k≤D} ev_c[k] + penalty ≥ target
            # → -Σ_{k≤D} ev_c[k] - penalty ≤ initial_soc - target
            if (
                ev.deadline_slot is not None
                and ev.target_kwh > ev.initial_soc_kwh + 1e-9
            ):
                d = ev.deadline_slot
                # Clamp deadline to valid range
                d = max(0, min(d, m - 1))
                for k in range(d + 1):
                    A_ub[ev_row, ev_off + k] = -1.0
                A_ub[ev_row, ev_pen_offsets[ev_idx]] = -1.0
                b_ub[ev_row] = ev.initial_soc_kwh - ev.target_kwh
                ev_row += 1

            # EV target-cap constraint:
            # Σ_{k≤D} ev_c[k] ≤ target_kwh - initial_soc_kwh
            # Caps EV charging at the economic target for pre-deadline
            # slots.  Without this, the benefit coefficient on ev_c[t]
            # would drive charging all the way to capacity_kwh
            # regardless of the actual shortfall.
            # Does NOT apply when charge_past_target is enabled — that
            # mode intentionally allows charging beyond target_kwh via
            # a separate surplus-only mechanism.
            if (
                ev.deadline_slot is not None
                and ev.target_kwh > ev.initial_soc_kwh + 1e-9
                and not ev.charge_past_target
            ):
                shortfall = ev.target_kwh - ev.initial_soc_kwh
                d = ev.deadline_slot
                d = max(0, min(d, m - 1))
                fixed_session_dc = 0.0
                for k in range(d + 1):
                    if is_session_ev and k in session_slots_set:
                        fixed_session_dc += (
                            max(ev.session_charge_kw or 0.0, 0.0)
                            * float(session_slot_hours[k])
                            * ev.charger_efficiency
                        )
                    else:
                        A_ub[ev_row, ev_off + k] = 1.0
                b_ub[ev_row] = max(shortfall - fixed_session_dc, 0.0)
                ev_row += 1

            # Post-deadline zero-charge constraint:
            # For EVs with a deadline and no charge-past-target,
            # ev_c[t] = 0 for all t > deadline_slot.
            # This prevents the MILP from charging after the deadline
            # unless charge_past_target is enabled (which uses surplus PV).
            if (
                ev.deadline_slot is not None
                and ev.target_kwh > ev.initial_soc_kwh + 1e-9
                and not ev.charge_past_target
            ):
                d = ev.deadline_slot
                d = max(0, min(d, m - 1))
                for t in range(d + 1, m):
                    if not (is_session_ev and t in session_slots_set):
                        A_ub[ev_row, ev_off + t] = 1.0
                    b_ub[ev_row] = 0.0
                    ev_row += 1

            # Surplus-only constraint for charge-past-target EVs:
            # ev_c[t] / charger_eff ≤ max(0, pv[t] - base_load[t])
            # This ensures past-target charging ONLY uses genuine PV
            # surplus — never battery discharge or grid import.
            if ev.charge_past_target:
                for t in range(m):
                    if not (is_session_ev and t in session_slots_set):
                        surplus_kwh = max(pv_avail[t] - base_load[t], 0.0)
                        A_ub[ev_row + t, ev_off + t] = 1.0 / ev.charger_efficiency
                        b_ub[ev_row + t] = surplus_kwh
                ev_row += m

    # ------------------------------------------------------------------
    # Session EV grid-charge prevention (issue #615).
    # For session slots, battery grid-charging is blocked: the battery
    # may only charge from PV surplus remaining after the fixed EV
    # session load is met.
    #   ec[t] / charge_eff  ≤ max(0, pv_avail[t] - total_session_ac[t])
    # ------------------------------------------------------------------
    session_rows = len(session_slots_set) if _has_session_demand else 0
    if session_rows > 0:
        # Compute per-slot total AC-side session EV load
        session_ac_by_slot: dict[int, float] = {}
        for ev_idx in session_ev_indices:
            ev = active_evs[ev_idx]
            skw = ev.session_charge_kw
            assert skw is not None
            # AC-side session load per slot (kW × hours).  The DC/AC
            # efficiency conversion cancels out by definition, so this is
            # simply the AC power multiplied by the slot duration.
            for t in session_slots_set:
                session_ac = skw * float(session_slot_hours[t])
                session_ac_by_slot[t] = session_ac_by_slot.get(t, 0.0) + session_ac

        session_t_list = sorted(session_slots_set)
        existing_rows = soc_rows + action_rows + cycle_rows + ev_total_rows
        A_ub_old = A_ub
        b_ub_old = b_ub
        A_ub = np.zeros((existing_rows + session_rows, n_vars))
        b_ub = np.zeros(existing_rows + session_rows)
        A_ub[:existing_rows, :] = A_ub_old
        b_ub[:existing_rows] = b_ub_old
        for row, t in enumerate(session_t_list):
            A_ub[existing_rows + row, ec_off + t] = 1.0 / charge_eff
            b_ub[existing_rows + row] = max(
                pv_avail[t] - session_ac_by_slot.get(t, 0.0), 0.0
            )
        ev_total_rows += session_rows

    # ------------------------------------------------------------------
    # Fuse constraint (soft): gi[t] - gi_pen[t] ≤ max_grid_import_per_slot_kwh
    # The penalty variable gi_pen[t] absorbs any excess at high cost,
    # preventing infeasibility when house base load alone exceeds the fuse.
    # ------------------------------------------------------------------
    fuse_rows = m if fuse_active else 0
    if fuse_active:
        existing_rows = soc_rows + action_rows + cycle_rows + ev_total_rows
        A_ub_old = A_ub
        b_ub_old = b_ub
        A_ub = np.zeros((existing_rows + fuse_rows, n_vars))
        b_ub = np.zeros(existing_rows + fuse_rows)
        A_ub[:existing_rows, :] = A_ub_old
        b_ub[:existing_rows] = b_ub_old
        for t in range(m):
            A_ub[existing_rows + t, gi_off + t] = 1.0
            A_ub[existing_rows + t, gi_pen_off + t] = -1.0
            b_ub[existing_rows + t] = max_grid_import_per_slot_kwh

    # ------------------------------------------------------------------
    # Variable bounds: all ≥ 0, charge/discharge capped by power limits.
    # SoC slacks may preserve only an already-invalid initial reading.  A
    # valid initial SoC therefore has hard physical bounds, while an
    # out-of-range reading can recover but can never move farther out of
    # range.  Other penalty variables remain unbounded above.
    # ------------------------------------------------------------------
    unbounded: tuple[float, float | None] = (0.0, None)
    if grid_import_ub_per_slot is None:
        grid_import_bounds = [unbounded] * m
    else:
        grid_import_bounds = [
            (0.0, max(float(grid_import_ub_per_slot[t]), 0.0)) for t in range(m)
        ]
    if grid_export_ub_per_slot is None:
        grid_export_bounds = [
            ((0.0, max_grid_export_per_slot_kwh) if export_limit_active else unbounded)
            for _t in range(m)
        ]
    else:
        grid_export_bounds = [
            (
                0.0,
                min(
                    max(float(grid_export_ub_per_slot[t]), 0.0),
                    max_grid_export_per_slot_kwh,
                )
                if export_limit_active
                else max(float(grid_export_ub_per_slot[t]), 0.0),
            )
            for t in range(m)
        ]
    bounds_builder.set(
        "primary_charge",
        [
            ((0.0, max_charge_per_slot) if bool(price_actionable[t]) else (0.0, 0.0))
            for t in range(m)
        ],
    )
    bounds_builder.set(
        "primary_discharge",
        [(0.0, float(ed_ub_per_slot[t])) for t in range(m)],
    )
    bounds_builder.set("grid_import", grid_import_bounds)
    bounds_builder.set("grid_export", grid_export_bounds)
    bounds_builder.set(
        "pv",
        [(pv_avail[t], pv_avail[t]) for t in range(m)],
    )
    bounds_builder.fill("primary_throughput", unbounded)
    bounds_builder.fill(
        "soc_max_penalty",
        (0.0, max(float(current_kwh - usable_kwh), 0.0)),
    )
    bounds_builder.fill(
        "soc_min_penalty",
        (0.0, max(float(-current_kwh), 0.0)),
    )
    bounds_builder.set(
        "curtailment",
        [(0.0, float(pv_avail[t])) for t in range(m)],
    )
    # --- EV bounds ---
    for ev_idx, ev in enumerate(active_evs):
        is_session_ev = ev_idx in session_ev_indices
        ev_bounds: list[tuple[float, float | None]] = []
        for t in range(m):
            if is_session_ev and t < session_slots and ev.session_charge_kw is not None:
                # Fixed bound: session demand (DC-side kWh per slot)
                session_dc = (
                    ev.session_charge_kw
                    * float(session_slot_hours[t])
                    * ev.charger_efficiency
                )
                session_dc = min(session_dc, ev.max_charge_per_slot)
                ev_bounds.append((session_dc, session_dc))
            elif ev.fixed_session_only:
                ev_bounds.append((0.0, 0.0))
            elif not bool(price_actionable[t]):
                # Optional smart charging must not treat an unpublished price
                # as free. A live session remains fixed by the branch above.
                ev_bounds.append((0.0, 0.0))
            else:
                duration_scale = min(
                    max(float(available_slot_hours[t]) / max(slot_hours, 1e-9), 0.0),
                    1.0,
                )
                ev_bounds.append((0.0, ev.max_charge_per_slot * duration_scale))
        bounds_builder.set(f"ev_{ev_idx}_charge", ev_bounds)
        # ev deadline penalty: [0, unbounded)
        bounds_builder.fill(f"ev_{ev_idx}_target_penalty", unbounded)
    # --- Fuse penalty bounds ---
    if fuse_active:
        bounds_builder.fill("grid_import_penalty", unbounded)
    if pv_export_ub_per_slot is not None:
        if (
            len(pv_export_ub_per_slot) != m
            or primary_action_mode_off is None
            or grid_flow_mode_off is None
            or grid_import_ub_per_slot is None
            or grid_export_ub_per_slot is None
        ):
            raise ValueError("incomplete explicit export-source layout")
        # Explicit source and binary-mode blocks follow primary/EV/fuse blocks.
        bounds_builder.set(
            "primary_battery_export",
            [
                (
                    (0.0, 0.0)
                    if (
                        no_export
                        or bool(battery_export_blocked[t])
                        or not bool(price_actionable[t])
                    )
                    else (0.0, float(ed_ub_per_slot[t]))
                )
                for t in range(m)
            ],
        )
        bounds_builder.set(
            "pv_export",
            [(0.0, max(float(pv_export_ub_per_slot[t]), 0.0)) for t in range(m)],
        )
        bounds_builder.fill("export_source_mode", (0.0, 1.0))
        bounds_builder.fill("primary_action_mode", (0.0, 1.0))
        bounds_builder.fill("grid_flow_mode", (0.0, 1.0))

    return {
        "A_eq": A_eq,
        "b_eq": b_eq,
        "A_ub": A_ub,
        "b_ub": b_ub,
        "ev_discharge_guard_active": ev_discharge_guard_active,
        "ed_ub_per_slot": ed_ub_per_slot,
    }
