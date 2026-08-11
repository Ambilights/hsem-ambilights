"""Hard per-phase import constraints for three-phase battery charging."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from custom_components.hsem.planner.secondary_storage import (
    secondary_site_load_offset_kwh,
)
from custom_components.hsem.utils.misc import clamp_efficiency
from custom_components.hsem.utils.phase_power import (
    NOMINAL_PHASE_VOLTAGE_V,
    PHASE_COUNT,
    PhasePowers,
)
from custom_components.hsem.utils.units import slot_duration_hours

if TYPE_CHECKING:
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
    secondary_layout: SecondaryLayout | None,
    secondary_storage: SecondaryStorageConfig | None,
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

    phase_limit_w = max(main_fuse_amps, 0.0) * NOMINAL_PHASE_VOLTAGE_V
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

    for t, slot_i in enumerate(future_idx):
        slot = slots[slot_i]
        hours = slot_duration_hours(slot.start, slot.end)
        phase_limit_kwh = phase_limit_w * hours / 1000.0
        base_net_kwh = float(base_load[t] - pv_avail[t])
        fixed_load_kwh, sbu_offset_kwh = _secondary_phase_terms(
            slot=slot,
            config=secondary_storage,
        )

        for phase_index in range(PHASE_COUNT):
            row = old_rows + phase_index * m + t
            on_secondary_phase = phase_index == secondary_phase_index
            topology_factor = (1.0 if on_secondary_phase else 0.0) - (1.0 / PHASE_COUNT)
            imbalance_kwh = phase_power_imbalance_w[phase_index] * hours / 1000.0

            # Signed phase flow from total site-grid flow.
            a_ub[row, gi_off + t] = 1.0 / PHASE_COUNT
            a_ub[row, ge_off + t] = -1.0 / PHASE_COUNT

            if secondary_layout is not None and secondary_storage is not None:
                a_ub[row, secondary_layout["charge"] + t] = (
                    topology_factor / secondary_charge_eff
                )
                a_ub[row, secondary_layout["sbu_mode"] + t] = (
                    -topology_factor * sbu_offset_kwh
                )

            # Preserve feasibility when fixed household demand is already
            # above the fuse: allow that baseline, but no controllable charge
            # may increase it.  Normally this resolves to the rated limit.
            baseline_phase_kwh = (
                base_net_kwh / PHASE_COUNT
                + imbalance_kwh
                + (fixed_load_kwh if on_secondary_phase else 0.0)
            )
            allowed_phase_kwh = max(phase_limit_kwh, baseline_phase_kwh)
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
    secondary_layout: SecondaryLayout | None,
    secondary_storage: SecondaryStorageConfig | None,
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
        balanced_kwh = (total_grid_kwh - secondary_delta_kwh) / PHASE_COUNT
        values = tuple(
            balanced_kwh
            + phase_power_imbalance_w[phase_index] * hours / 1000.0
            + (secondary_delta_kwh if phase_index == secondary_phase_index else 0.0)
            for phase_index in range(PHASE_COUNT)
        )
        phase_imports.append((values[0], values[1], values[2]))
    return phase_imports
