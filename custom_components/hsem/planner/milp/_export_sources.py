"""Explicit attribution of aggregate grid export to physical sources."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from custom_components.hsem.planner.secondary_storage import (
    secondary_site_load_offset_kwh,
)
from custom_components.hsem.utils.misc import clamp_efficiency

if TYPE_CHECKING:
    from custom_components.hsem.models.planned_slot import PlannedSlot
    from custom_components.hsem.models.secondary_storage_config import (
        SecondaryStorageConfig,
    )
    from custom_components.hsem.planner.milp._secondary_storage import SecondaryLayout


def _add_export_source_constraints(
    constraints: dict[str, Any],
    *,
    n_vars: int,
    m: int,
    slots: list[PlannedSlot],
    future_idx: list[int],
    ed_off: int,
    gi_off: int,
    ge_off: int,
    primary_export_off: int,
    pv_export_off: int,
    export_source_mode_off: int,
    grid_flow_mode_off: int,
    discharge_eff: float,
    primary_site_discharge_cap_kwh: np.ndarray,
    primary_discharge_ub_per_slot: list[float],
    pv_export_ub_per_slot: np.ndarray,
    grid_import_ub_per_slot: np.ndarray,
    grid_export_ub_per_slot: np.ndarray,
    secondary_layout: SecondaryLayout | None,
    secondary_storage: SecondaryStorageConfig | None,
) -> dict[str, Any]:
    """Append exact export-source balance and local-destination rows.

    Primary export is battery-side DC kWh; both other export flows are AC.
    Binary rows make battery-origin export exactly the lesser of aggregate
    export and delivered battery discharge, so concurrent PV cannot make
    battery-origin export appear to be local primary-battery use.
    """
    old_a_eq = constraints["A_eq"]
    old_b_eq = constraints["b_eq"]
    old_eq_rows = old_a_eq.shape[0]
    a_eq = np.zeros((old_eq_rows + m, n_vars))
    b_eq = np.zeros(old_eq_rows + m)
    a_eq[:old_eq_rows, : old_a_eq.shape[1]] = old_a_eq
    b_eq[:old_eq_rows] = old_b_eq

    old_a_ub = constraints["A_ub"]
    old_b_ub = constraints["b_ub"]
    old_ub_rows = old_a_ub.shape[0]
    a_ub = np.zeros((old_ub_rows + 6 * m, n_vars))
    b_ub = np.zeros(old_ub_rows + 6 * m)
    a_ub[:old_ub_rows, : old_a_ub.shape[1]] = old_a_ub
    b_ub[:old_ub_rows] = old_b_ub

    eta = min(max(discharge_eff, 0.01), 1.0)
    secondary_charge_eff = (
        clamp_efficiency(secondary_storage.charge_efficiency_pct)
        if secondary_storage is not None
        else 1.0
    )

    for t, slot_i in enumerate(future_idx):
        # Aggregate AC export is exactly battery-origin AC plus non-battery
        # (normally direct PV) AC export.
        a_eq[old_eq_rows + t, ge_off + t] = 1.0
        a_eq[old_eq_rows + t, primary_export_off + t] = -eta
        a_eq[old_eq_rows + t, pv_export_off + t] = -1.0

        # A battery-origin share cannot exceed total primary discharge.
        a_ub[old_ub_rows + t, primary_export_off + t] = 1.0
        a_ub[old_ub_rows + t, ed_off + t] = -1.0

        # The non-exported battery discharge must fit the complete set of AC
        # sinks that the model physically permits Huawei to serve.
        local_row = old_ub_rows + m + t
        a_ub[local_row, ed_off + t] = eta
        a_ub[local_row, primary_export_off + t] = -eta
        eligible_load = max(float(primary_site_discharge_cap_kwh[t]), 0.0)

        if secondary_layout is not None and secondary_storage is not None:
            load_kwh = max(slots[slot_i].secondary_storage_load_kwh, 0.0)
            site_offset_kwh = secondary_site_load_offset_kwh(
                slots[slot_i],
                secondary_storage,
            )
            if not secondary_storage.base_load_includes_dedicated_load:
                eligible_load += load_kwh
            # SBU can reveal PV export even with no Huawei discharge. Only
            # subtract the residual AC sink that Huawei could have served;
            # the remainder was already supplied by PV, not by the battery.
            a_ub[local_row, secondary_layout["sbu_mode"] + t] = min(
                site_offset_kwh,
                eligible_load,
            )
            if secondary_storage.allow_primary_battery_transfer:
                a_ub[local_row, secondary_layout["charge"] + t] = (
                    -1.0 / secondary_charge_eff
                )

        b_ub[local_row] = eligible_load + 1e-9

        discharge_big_m = max(float(primary_discharge_ub_per_slot[t]), 0.0)
        pv_export_big_m = max(float(pv_export_ub_per_slot[t]), 0.0)

        # z=1 fixes bx=ed (all discharge is exported); z=0 is relaxed.
        discharge_min_row = old_ub_rows + 2 * m + t
        a_ub[discharge_min_row, ed_off + t] = 1.0
        a_ub[discharge_min_row, primary_export_off + t] = -1.0
        a_ub[discharge_min_row, export_source_mode_off + t] = discharge_big_m
        b_ub[discharge_min_row] = discharge_big_m

        # z=0 fixes ge=eta*bx (all export is battery-origin); z=1 is
        # relaxed only by the finite non-battery export capacity.
        export_min_row = old_ub_rows + 3 * m + t
        a_ub[export_min_row, ge_off + t] = 1.0
        a_ub[export_min_row, primary_export_off + t] = -eta
        a_ub[export_min_row, export_source_mode_off + t] = -pv_export_big_m
        b_ub[export_min_row] = 0.0

        # Grid import and export are opposite physical directions through the
        # same meter. y=1 selects import and y=0 selects export. Finite
        # physical bounds make both big-M rows exact without an arbitrary
        # numerical constant.
        grid_import_big_m = max(float(grid_import_ub_per_slot[t]), 0.0)
        import_mode_row = old_ub_rows + 4 * m + t
        a_ub[import_mode_row, gi_off + t] = 1.0
        a_ub[import_mode_row, grid_flow_mode_off + t] = -grid_import_big_m

        grid_export_big_m = max(float(grid_export_ub_per_slot[t]), 0.0)
        export_mode_row = old_ub_rows + 5 * m + t
        a_ub[export_mode_row, ge_off + t] = 1.0
        a_ub[export_mode_row, grid_flow_mode_off + t] = grid_export_big_m
        b_ub[export_mode_row] = grid_export_big_m

    constraints["A_eq"] = a_eq
    constraints["b_eq"] = b_eq
    constraints["A_ub"] = a_ub
    constraints["b_ub"] = b_ub
    return constraints
