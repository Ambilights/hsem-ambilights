"""Hard per-phase import constraints for controllable site charging."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from custom_components.hsem.planner.secondary_storage import (
    SECONDARY_MODE_SBU,
    secondary_site_load_offset_kwh,
    secondary_slot_duration_hours,
)
from custom_components.hsem.utils.misc import clamp_efficiency
from custom_components.hsem.utils.phase_power import (
    NOMINAL_PHASE_VOLTAGE_V,
    PHASE_COUNT,
    PhasePowers,
)
from custom_components.hsem.utils.units import slot_duration_hours

if TYPE_CHECKING:
    from custom_components.hsem.models.ev_config import EVConfig
    from custom_components.hsem.models.planned_slot import PlannedSlot
    from custom_components.hsem.models.secondary_storage_config import (
        SecondaryStorageConfig,
    )
    from custom_components.hsem.planner.milp._secondary_storage import (
        SecondaryLayout,
    )


def _phase_fuse_enabled(
    *,
    main_fuse_amps: float | None,
    main_fuse_phases: int,
    phase_power_imbalance_w: PhasePowers | None,
) -> bool:
    """Return whether a valid three-phase hard-limit model can be built."""
    return bool(
        main_fuse_amps is not None
        and main_fuse_amps > 1e-9
        and main_fuse_phases == PHASE_COUNT
        and phase_power_imbalance_w is not None
        and all(np.isfinite(value) for value in phase_power_imbalance_w)
    )


def _secondary_phase_terms(
    *,
    slot: PlannedSlot,
    config: SecondaryStorageConfig | None,
) -> tuple[float, float]:
    """Return fixed utility load and SBU-removable load for one slot."""
    if config is None or not config.valid:
        return 0.0, 0.0
    fixed_utility_load_kwh = (
        slot.secondary_storage_load_kwh
        if not config.base_load_includes_dedicated_load
        else 0.0
    )
    return fixed_utility_load_kwh, secondary_site_load_offset_kwh(slot, config)


def _full_slot_power_scale(slot: PlannedSlot, now: datetime) -> float:
    """Return the factor expressing remaining-slot energy as full-slot power."""
    full_hours = slot_duration_hours(slot.start, slot.end)
    effective_hours = secondary_slot_duration_hours(slot, now)
    if full_hours <= 1e-9 or effective_hours <= 1e-9:
        return 0.0
    return full_hours / effective_hours


def _fixed_session_ac_kwh(
    *,
    active_evs: list[EVConfig],
    lp_t: int,
    session_slots_set: set[int],
    hours: float,
    unmanaged_only: bool = False,
) -> float:
    """Return full-slot AC energy for measured EV sessions.

    Session variables are fixed to the observed charger power during the
    certainty window. ``fixed_session_only`` sessions deliberately publish no
    actuator command, so final-flow reconstruction must add their measured
    demand separately from the executable EV command fields.
    """
    if lp_t not in session_slots_set:
        return 0.0
    return sum(
        max(float(ev.session_charge_kw or 0.0), 0.0) * hours
        for ev in active_evs
        if not unmanaged_only or ev.fixed_session_only
    )


def _phase_import_limits_kwh(
    *,
    slots: list[PlannedSlot],
    future_idx: list[int],
    base_load: np.ndarray,  # type: ignore[name-defined]
    pv_avail: np.ndarray,  # type: ignore[name-defined]
    main_fuse_amps: float,
    phase_power_imbalance_w: PhasePowers,
    active_evs: list[EVConfig],
    session_slots_set: set[int],
    secondary_storage: SecondaryStorageConfig | None,
    now: datetime,
) -> list[PhasePowers]:
    """Return the exact per-phase ceilings used by the hard MILP rows.

    The rated fuse limit is relaxed only to an already-unavoidable forecast
    baseline. Keeping this calculation in one helper lets post-solve EV
    redistribution and final published-flow validation use the same physical
    ceiling as the optimisation model.
    """
    phase_limit_w = max(main_fuse_amps, 0.0) * NOMINAL_PHASE_VOLTAGE_V
    secondary_phase_index = (
        min(max(secondary_storage.grid_phase, 1), PHASE_COUNT) - 1
        if secondary_storage is not None
        else PHASE_COUNT - 1
    )
    limits: list[PhasePowers] = []

    for t, slot_i in enumerate(future_idx):
        slot = slots[slot_i]
        hours = slot_duration_hours(slot.start, slot.end)
        full_slot_scale = _full_slot_power_scale(slot, now)
        phase_limit_kwh = phase_limit_w * hours / 1000.0
        base_net_kwh = float(base_load[t] - pv_avail[t])
        fixed_load_kwh, _sbu_offset_kwh = _secondary_phase_terms(
            slot=slot,
            config=secondary_storage,
        )
        fixed_session_kwh = _fixed_session_ac_kwh(
            active_evs=active_evs,
            lp_t=t,
            session_slots_set=session_slots_set,
            hours=hours,
        )
        values = tuple(
            max(
                phase_limit_kwh,
                base_net_kwh / PHASE_COUNT
                # Charger phase topology is not configured.  Represent an
                # unavoidable live session on every phase so the relaxed
                # baseline is safe whichever phase it actually occupies.
                + fixed_session_kwh
                + phase_power_imbalance_w[phase_index] * hours / 1000.0
                + (
                    fixed_load_kwh * full_slot_scale
                    if phase_index == secondary_phase_index
                    else 0.0
                ),
            )
            for phase_index in range(PHASE_COUNT)
        )
        limits.append((values[0], values[1], values[2]))
    return limits


def _add_phase_fuse_constraints(
    constraints: dict[str, Any],
    *,
    n_vars: int,
    m: int,
    slots: list[PlannedSlot],
    future_idx: list[int],
    base_load: np.ndarray,  # type: ignore[name-defined]
    pv_avail: np.ndarray,  # type: ignore[name-defined]
    gi_off: int,
    ge_off: int,
    main_fuse_amps: float,
    phase_power_imbalance_w: PhasePowers,
    active_evs: list[EVConfig],
    ev_var_offsets: list[int],
    session_slots_set: set[int],
    secondary_layout: SecondaryLayout | None,
    secondary_storage: SecondaryStorageConfig | None,
    now: datetime,
) -> dict[str, Any]:
    """Append hard phase-import rows while preserving baseline feasibility.

    The Huawei inverter is balanced across all three phases.  ``gi-ge`` is
    therefore split equally, then corrected by the measured fixed imbalance.
    The PowMr site delta is moved from that equal split onto its configured
    physical phase.  If uncontrollable forecast load already exceeds a fuse,
    that exact baseline is allowed but controllable charging may not worsen it.
    """
    old_a_ub = constraints["A_ub"]
    old_b_ub = constraints["b_ub"]
    old_rows = old_a_ub.shape[0]
    added_rows = PHASE_COUNT * m
    a_ub = np.zeros((old_rows + added_rows, n_vars))
    b_ub = np.zeros(old_rows + added_rows)
    a_ub[:old_rows, : old_a_ub.shape[1]] = old_a_ub
    b_ub[:old_rows] = old_b_ub

    secondary_phase_index = (
        min(max(secondary_storage.grid_phase, 1), PHASE_COUNT) - 1
        if secondary_storage is not None
        else PHASE_COUNT - 1
    )
    secondary_charge_eff = (
        clamp_efficiency(secondary_storage.charge_efficiency_pct)
        if secondary_storage is not None and secondary_storage.valid
        else 1.0
    )
    phase_limits = _phase_import_limits_kwh(
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

    for t, slot_i in enumerate(future_idx):
        slot = slots[slot_i]
        hours = slot_duration_hours(slot.start, slot.end)
        full_slot_scale = _full_slot_power_scale(slot, now)
        fixed_load_kwh, sbu_offset_kwh = _secondary_phase_terms(
            slot=slot,
            config=secondary_storage,
        )

        for phase_index in range(PHASE_COUNT):
            row = old_rows + phase_index * m + t
            on_secondary_phase = phase_index == secondary_phase_index
            # ``gi-ge`` carries the actual partial-slot PowMr energy, while the
            # live primary load and phase imbalance remain represented in the
            # established full-slot power frame. Remove the actual PowMr delta
            # from its balanced G/3 share, then add its full-slot-equivalent
            # power only on the configured phase.
            topology_factor = (full_slot_scale if on_secondary_phase else 0.0) - (
                1.0 / PHASE_COUNT
            )
            imbalance_kwh = phase_power_imbalance_w[phase_index] * hours / 1000.0

            # Signed phase flow from total site-grid flow.
            a_ub[row, gi_off + t] = 1.0 / PHASE_COUNT
            a_ub[row, ge_off + t] = -1.0 / PHASE_COUNT

            # ``gi-ge`` already assigns one third of every EV variable to this
            # phase.  Charger topology is not configured, so each hard row
            # must conservatively assume that the whole EV load can land on
            # that phase.  The full-slot scale also preserves instantaneous
            # power for a partially elapsed current slot.
            ev_topology_correction = full_slot_scale - 1.0 / PHASE_COUNT
            if ev_topology_correction > 1e-9:
                for ev_idx, ev in enumerate(active_evs):
                    a_ub[row, ev_var_offsets[ev_idx] + t] += (
                        ev_topology_correction / max(ev.charger_efficiency, 0.01)
                    )

            if secondary_layout is not None and secondary_storage is not None:
                a_ub[row, secondary_layout["charge"] + t] = (
                    topology_factor / secondary_charge_eff
                )
                a_ub[row, secondary_layout["sbu_mode"] + t] = (
                    -topology_factor * sbu_offset_kwh
                )

            allowed_phase_kwh = phase_limits[t][phase_index]
            b_ub[row] = (
                allowed_phase_kwh - imbalance_kwh - topology_factor * fixed_load_kwh
            )

    constraints["A_ub"] = a_ub
    constraints["b_ub"] = b_ub
    return constraints


def _phase_imports_from_solution_kwh(
    *,
    result_x: np.ndarray,  # type: ignore[name-defined]
    m: int,
    slots: list[PlannedSlot],
    future_idx: list[int],
    gi_off: int,
    ge_off: int,
    phase_power_imbalance_w: PhasePowers,
    active_evs: list[EVConfig],
    ev_var_offsets: list[int],
    secondary_layout: SecondaryLayout | None,
    secondary_storage: SecondaryStorageConfig | None,
    now: datetime,
) -> list[PhasePowers]:
    """Reconstruct signed phase energy from a solved decision vector."""
    secondary_phase_index = (
        min(max(secondary_storage.grid_phase, 1), PHASE_COUNT) - 1
        if secondary_storage is not None
        else PHASE_COUNT - 1
    )
    secondary_charge_eff = (
        clamp_efficiency(secondary_storage.charge_efficiency_pct)
        if secondary_storage is not None and secondary_storage.valid
        else 1.0
    )
    phase_imports: list[PhasePowers] = []

    for t, slot_i in enumerate(future_idx):
        slot = slots[slot_i]
        hours = slot_duration_hours(slot.start, slot.end)
        full_slot_scale = _full_slot_power_scale(slot, now)
        total_grid_kwh = float(result_x[gi_off + t] - result_x[ge_off + t])
        fixed_load_kwh, sbu_offset_kwh = _secondary_phase_terms(
            slot=slot,
            config=secondary_storage,
        )
        secondary_delta_kwh = fixed_load_kwh
        if secondary_layout is not None and secondary_storage is not None:
            secondary_delta_kwh += (
                float(result_x[secondary_layout["charge"] + t]) / secondary_charge_eff
            )
            secondary_delta_kwh -= (
                float(result_x[secondary_layout["sbu_mode"] + t]) * sbu_offset_kwh
            )
        ev_delta_kwh = sum(
            float(result_x[ev_var_offsets[ev_idx] + t])
            / max(ev.charger_efficiency, 0.01)
            for ev_idx, ev in enumerate(active_evs)
        )
        balanced_kwh = (
            total_grid_kwh - secondary_delta_kwh - ev_delta_kwh
        ) / PHASE_COUNT
        normalized_ev_delta_kwh = ev_delta_kwh * full_slot_scale
        normalized_secondary_delta_kwh = secondary_delta_kwh * full_slot_scale
        values = tuple(
            balanced_kwh
            # This is a conservative envelope, not a physical phase sum: with
            # unknown charger topology, every phase is checked as though it
            # carries the entire EV load.
            + normalized_ev_delta_kwh
            + phase_power_imbalance_w[phase_index] * hours / 1000.0
            + (
                normalized_secondary_delta_kwh
                if phase_index == secondary_phase_index
                else 0.0
            )
            for phase_index in range(PHASE_COUNT)
        )
        phase_imports.append((values[0], values[1], values[2]))
    return phase_imports


def _phase_imports_from_published_slots_kwh(
    *,
    slots: list[PlannedSlot],
    future_idx: list[int],
    phase_power_imbalance_w: PhasePowers,
    active_evs: list[EVConfig],
    session_slots_set: set[int],
    secondary_storage: SecondaryStorageConfig | None,
    now: datetime,
) -> list[PhasePowers]:
    """Reconstruct signed phase energy from final executable slot fields."""
    secondary_phase_index = (
        min(max(secondary_storage.grid_phase, 1), PHASE_COUNT) - 1
        if secondary_storage is not None
        else PHASE_COUNT - 1
    )
    secondary_charge_eff = (
        clamp_efficiency(secondary_storage.charge_efficiency_pct)
        if secondary_storage is not None and secondary_storage.valid
        else 1.0
    )
    phase_imports: list[PhasePowers] = []

    for lp_t, slot_i in enumerate(future_idx):
        slot = slots[slot_i]
        hours = slot_duration_hours(slot.start, slot.end)
        full_slot_scale = _full_slot_power_scale(slot, now)
        total_grid_kwh = float(slot.grid_import_kwh - slot.grid_export_kwh)
        fixed_load_kwh, sbu_offset_kwh = _secondary_phase_terms(
            slot=slot,
            config=secondary_storage,
        )
        secondary_delta_kwh = fixed_load_kwh
        if secondary_storage is not None and secondary_storage.valid:
            effective_hours = secondary_slot_duration_hours(slot, now)
            executable_charge_dc_kwh = (
                max(float(slot.secondary_storage_charge_current_a), 0.0)
                * max(secondary_storage.nominal_voltage_v, 0.0)
                * effective_hours
                / 1000.0
            )
            secondary_delta_kwh += executable_charge_dc_kwh / secondary_charge_eff
            if slot.secondary_storage_mode == SECONDARY_MODE_SBU:
                secondary_delta_kwh -= sbu_offset_kwh
        planned_ev_delta_kwh = max(float(slot.ev_total_planned_load_kwh), 0.0)
        executable_ev_power_w = max(float(slot.ev_charger_calculated_power), 0.0) + max(
            float(slot.ev_second_charger_calculated_power),
            0.0,
        )
        executable_ev_full_slot_kwh = executable_ev_power_w * hours / 1000.0
        unmanaged_session_full_slot_kwh = _fixed_session_ac_kwh(
            active_evs=active_evs,
            lp_t=lp_t,
            session_slots_set=session_slots_set,
            hours=hours,
            unmanaged_only=True,
        )
        balanced_kwh = (
            total_grid_kwh - secondary_delta_kwh - planned_ev_delta_kwh
        ) / PHASE_COUNT
        normalized_secondary_delta_kwh = secondary_delta_kwh * full_slot_scale
        values = tuple(
            balanced_kwh
            # Unknown charger topology: validate every phase against the
            # worst case where it carries the entire executable EV load.
            + executable_ev_full_slot_kwh
            + unmanaged_session_full_slot_kwh
            + phase_power_imbalance_w[phase_index] * hours / 1000.0
            + (
                normalized_secondary_delta_kwh
                if phase_index == secondary_phase_index
                else 0.0
            )
            for phase_index in range(PHASE_COUNT)
        )
        phase_imports.append((values[0], values[1], values[2]))
    return phase_imports
