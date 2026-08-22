"""Mixed-integer MILP extension for dedicated-load secondary storage."""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.secondary_storage_config import (
    SecondaryStorageConfig,
)
from custom_components.hsem.planner.cost_helpers import slot_grid_cash_flow_cost
from custom_components.hsem.planner.secondary_storage import (
    SECONDARY_MODE_CHARGE,
    SECONDARY_MODE_SBU,
    SECONDARY_MODE_UTILITY,
    secondary_charge_limits_kwh,
    secondary_site_load_offset_kwh,
    secondary_slot_duration_hours,
)
from custom_components.hsem.utils.datetime_utils import slot_contains, utc_key
from custom_components.hsem.utils.logger import log_planner
from custom_components.hsem.utils.misc import clamp_efficiency
from custom_components.hsem.utils.recommendations import Recommendations
from custom_components.hsem.utils.units import (
    hours_ahead,
    is_material_planned_energy_kwh,
)

SecondaryLayout = dict[str, int]
if TYPE_CHECKING:
    from custom_components.hsem.planner.milp._layout import (
        MilpBoundsBuilder,
        MilpColumnLayout,
    )


def _allocate_secondary_variables(
    base_n_vars: int,
    m: int,
    *,
    column_layout: MilpColumnLayout | None = None,
) -> tuple[SecondaryLayout, int]:
    """Append secondary energy, mode, and integer-current-step blocks."""
    if column_layout is None:
        layout = {
            "charge": base_n_vars,
            "discharge": base_n_vars + m,
            "throughput": base_n_vars + 2 * m,
            "charge_mode": base_n_vars + 3 * m,
            "sbu_mode": base_n_vars + 4 * m,
            "charge_steps": base_n_vars + 5 * m,
        }
        return layout, base_n_vars + 6 * m

    if column_layout.column_count != base_n_vars:
        raise ValueError(
            f"secondary MILP layout starts at {column_layout.column_count}, "
            f"expected {base_n_vars}"
        )

    layout = {
        "charge": column_layout.add("secondary_charge", m),
        "discharge": column_layout.add("secondary_discharge", m),
        "throughput": column_layout.add("secondary_throughput", m),
        "charge_mode": column_layout.add("secondary_charge_mode", m),
        "sbu_mode": column_layout.add("secondary_sbu_mode", m),
        "charge_steps": column_layout.add("secondary_charge_steps", m),
    }
    return layout, column_layout.column_count


def _extend_secondary_constraints(
    constraints: dict[str, Any],
    *,
    bounds_builder: MilpBoundsBuilder,
    n_vars: int,
    m: int,
    layout: SecondaryLayout,
    config: SecondaryStorageConfig,
    slots: list[PlannedSlot],
    future_idx: list[int],
    primary_charge_off: int,
    primary_discharge_off: int,
    primary_max_charge_kwh: float,
    primary_max_discharge_kwh: float,
    primary_discharge_efficiency_fraction: float,
    primary_site_discharge_limited: np.ndarray,  # type: ignore[name-defined]
    primary_site_discharge_cap_kwh: np.ndarray,  # type: ignore[name-defined]
    price_actionable: np.ndarray,  # type: ignore[name-defined]
    now: datetime,
) -> dict[str, Any]:
    """Add site balance, secondary SoC, discrete modes, and no-transfer rows."""
    charge_eff = clamp_efficiency(config.charge_efficiency_pct)
    discharge_eff = clamp_efficiency(config.discharge_efficiency_pct)
    charge_off = layout["charge"]
    discharge_off = layout["discharge"]
    throughput_off = layout["throughput"]
    charge_mode_off = layout["charge_mode"]
    sbu_mode_off = layout["sbu_mode"]
    charge_steps_off = layout["charge_steps"]

    old_a_eq = constraints["A_eq"]
    old_b_eq = constraints["b_eq"]
    old_eq_rows = old_a_eq.shape[0]
    a_eq = np.zeros((old_eq_rows + 2 * m, n_vars))
    b_eq = np.zeros(old_eq_rows + 2 * m)
    a_eq[:old_eq_rows, : old_a_eq.shape[1]] = old_a_eq
    b_eq[:old_eq_rows] = old_b_eq

    for t, slot_i in enumerate(future_idx):
        load_kwh = slots[slot_i].secondary_storage_load_kwh
        site_load_offset_kwh = secondary_site_load_offset_kwh(slots[slot_i], config)
        hours = secondary_slot_duration_hours(slots[slot_i], now)
        standby_kwh = max(config.inverter_standby_power_w, 0.0) * hours / 1000.0
        battery_draw_kwh = load_kwh / discharge_eff + standby_kwh

        # Site bus: charging always draws AC. SBU removes the dedicated load
        # from the site bus because the secondary output cannot backfeed it.
        a_eq[t, charge_off + t] = -1.0 / charge_eff
        a_eq[t, sbu_mode_off + t] = site_load_offset_kwh
        if not config.base_load_includes_dedicated_load:
            b_eq[t] += load_kwh

        # Dedicated-load node: SBU must supply exactly its load and overhead.
        a_eq[old_eq_rows + t, discharge_off + t] = 1.0
        a_eq[old_eq_rows + t, sbu_mode_off + t] = -battery_draw_kwh

        # PowMr exposes a 10 A-step number. Tie stored charge energy to an
        # integer count of those physical current steps so planned energy and
        # the adapter command remain identical.
        step_energy_kwh = (
            config.nominal_voltage_v * config.charge_current_step_a * hours / 1000.0
        )
        a_eq[old_eq_rows + m + t, charge_off + t] = 1.0
        a_eq[old_eq_rows + m + t, charge_steps_off + t] = -step_energy_kwh

    old_a_ub = constraints["A_ub"]
    old_b_ub = constraints["b_ub"]
    old_ub_rows = old_a_ub.shape[0]
    transfer_rows = 0 if config.allow_primary_battery_transfer else 2 * m
    site_discharge_rows = int(np.count_nonzero(primary_site_discharge_limited))
    added_rows = 7 * m + transfer_rows + site_discharge_rows
    a_ub = np.zeros((old_ub_rows + added_rows, n_vars))
    b_ub = np.zeros(old_ub_rows + added_rows)
    a_ub[:old_ub_rows, : old_a_ub.shape[1]] = old_a_ub
    b_ub[:old_ub_rows] = old_b_ub

    current_kwh = min(max(config.current_usable_kwh, 0.0), config.usable_kwh)
    usable_kwh = config.usable_kwh
    charge_limits = [
        secondary_charge_limits_kwh(
            config,
            secondary_slot_duration_hours(slots[slot_i], now),
        )
        for slot_i in future_idx
    ]
    maximum_steps = int(
        max(config.max_charge_current_a, 0.0) // max(config.charge_current_step_a, 1e-9)
    )
    current_lock = config.current_slot_mode_lock
    if current_lock not in {
        None,
        SECONDARY_MODE_CHARGE,
        SECONDARY_MODE_SBU,
        SECONDARY_MODE_UTILITY,
    }:
        log_planner(
            "warning",
            "[milp] Ignoring invalid secondary current-slot mode lock: %s",
            current_lock,
        )
        current_lock = None
    lock_t = (
        0
        if current_lock is not None
        and future_idx
        and slot_contains(slots[future_idx[0]].start, slots[future_idx[0]].end, now)
        else None
    )
    row = old_ub_rows
    for t in range(m):
        for k in range(t + 1):
            a_ub[row + t, charge_off + k] = 1.0
            a_ub[row + t, discharge_off + k] = -1.0
            a_ub[row + m + t, charge_off + k] = -1.0
            a_ub[row + m + t, discharge_off + k] = 1.0
        b_ub[row + t] = usable_kwh - current_kwh
        b_ub[row + m + t] = current_kwh
    row += 2 * m

    for t in range(m):
        minimum_charge, maximum_charge = charge_limits[t]
        # Charge current can be non-zero only in charge mode.
        a_ub[row + t, charge_off + t] = 1.0
        a_ub[row + t, charge_mode_off + t] = -maximum_charge
        a_ub[row + m + t, charge_off + t] = -1.0
        a_ub[row + m + t, charge_mode_off + t] = minimum_charge

        # Charge and SBU are mutually exclusive ternary states.
        a_ub[row + 2 * m + t, charge_mode_off + t] = 1.0
        a_ub[row + 2 * m + t, sbu_mode_off + t] = 1.0
        b_ub[row + 2 * m + t] = 1.0

        # Throughput auxiliary equals max(charge, discharge) at optimum.
        a_ub[row + 3 * m + t, charge_off + t] = 1.0
        a_ub[row + 3 * m + t, throughput_off + t] = -1.0
        a_ub[row + 4 * m + t, discharge_off + t] = 1.0
        a_ub[row + 4 * m + t, throughput_off + t] = -1.0
    row += 5 * m

    if not config.allow_primary_battery_transfer:
        for t in range(m):
            # Conservative source guard: never discharge Huawei while PowMr
            # grid charging is enabled, avoiding DC→AC→DC battery transfer.
            a_ub[row + t, primary_discharge_off + t] = 1.0
            a_ub[row + t, charge_mode_off + t] = primary_max_discharge_kwh
            b_ub[row + t] = primary_max_discharge_kwh

            # Symmetric destination guard: PowMr SBU must not free site-bus
            # energy for Huawei charging unless cross-battery transfer is an
            # explicit opt-in. With z_sbu=1 this fixes primary charge to zero.
            a_ub[row + m + t, primary_charge_off + t] = 1.0
            a_ub[row + m + t, sbu_mode_off + t] = primary_max_charge_kwh
            b_ub[row + m + t] = primary_max_charge_kwh
        row += 2 * m

    # The base primary cap is computed before PowMr mode variables exist. In a
    # site-limited slot, SBU removes the included dedicated load from the same
    # AC demand Huawei may serve. Couple both decisions so the MILP cannot
    # solve a primary discharge that becomes forbidden export after write-out:
    #   primary_discharge * eta + included_load * sbu <= primary_site_cap.
    discharge_eff = min(max(primary_discharge_efficiency_fraction, 0.01), 1.0)
    for t, limited in enumerate(primary_site_discharge_limited):
        if not bool(limited):
            continue
        site_cap_kwh = max(float(primary_site_discharge_cap_kwh[t]), 0.0)
        included_load_kwh = (
            min(
                secondary_site_load_offset_kwh(slots[future_idx[t]], config),
                site_cap_kwh,
            )
            if config.base_load_includes_dedicated_load
            else 0.0
        )
        a_ub[row, primary_discharge_off + t] = discharge_eff
        a_ub[row, sbu_mode_off + t] = included_load_kwh
        b_ub[row] = site_cap_kwh + 1e-6
        row += 1

    constraints["A_eq"] = a_eq
    constraints["b_eq"] = b_eq
    constraints["A_ub"] = a_ub
    constraints["b_ub"] = b_ub
    charge_bounds: list[tuple[float, float | None]] = []
    discharge_bounds: list[tuple[float, float | None]] = []
    charge_mode_bounds: list[tuple[float, float | None]] = []
    sbu_mode_bounds: list[tuple[float, float | None]] = []
    charge_step_bounds: list[tuple[float, float | None]] = []
    for t in range(m):
        actionable = bool(price_actionable[t])
        mode_lock = current_lock if t == lock_t else None
        if not actionable or mode_lock == SECONDARY_MODE_UTILITY:
            charge_bounds.append((0.0, 0.0))
            discharge_bounds.append((0.0, 0.0))
            charge_mode_bounds.append((0.0, 0.0))
            sbu_mode_bounds.append((0.0, 0.0))
            charge_step_bounds.append((0.0, 0.0))
            continue

        charge_bounds.append((0.0, charge_limits[t][1]))
        discharge_bounds.append((0.0, config.usable_kwh))
        charge_mode_bounds.append(
            (1.0, 1.0) if mode_lock == SECONDARY_MODE_CHARGE else (0.0, 1.0)
        )
        sbu_mode_bounds.append(
            (1.0, 1.0) if mode_lock == SECONDARY_MODE_SBU else (0.0, 1.0)
        )
        charge_step_bounds.append((0.0, float(maximum_steps)))
        if mode_lock == SECONDARY_MODE_CHARGE:
            sbu_mode_bounds[-1] = (0.0, 0.0)
        elif mode_lock == SECONDARY_MODE_SBU:
            charge_bounds[-1] = (0.0, 0.0)
            charge_mode_bounds[-1] = (0.0, 0.0)
            charge_step_bounds[-1] = (0.0, 0.0)

    bounds_builder.set("secondary_charge", charge_bounds)
    bounds_builder.set("secondary_discharge", discharge_bounds)
    bounds_builder.fill("secondary_throughput", (0.0, None))
    bounds_builder.set("secondary_charge_mode", charge_mode_bounds)
    bounds_builder.set("secondary_sbu_mode", sbu_mode_bounds)
    bounds_builder.set("secondary_charge_steps", charge_step_bounds)
    return constraints


def _add_secondary_objective(
    objective: np.ndarray,  # type: ignore[name-defined]
    *,
    layout: SecondaryLayout,
    config: SecondaryStorageConfig,
    slots: list[PlannedSlot],
    future_idx: list[int],
    time_discount_rate: float,
    now: datetime,
) -> None:
    """Add secondary wear and terminal-value coefficients.

    Secondary conversion losses already affect the physical site balance and
    final battery inventory. Pricing them here as well would charge the same
    loss twice and bias the optimizer away from SBU operation.
    """
    use_discount = time_discount_rate < 1.0 - 1e-9
    replacement = max(config.replacement_price_per_kwh or 0.0, 0.0)
    if not math.isfinite(replacement):
        replacement = 0.0

    for t, slot_i in enumerate(future_idx):
        if not slots[slot_i].price_actionable:
            continue
        discount = 1.0
        if use_discount:
            slot = slots[slot_i]
            start_utc = utc_key(slot.start)
            midpoint = start_utc + (utc_key(slot.end) - start_utc) / 2
            discount = time_discount_rate ** hours_ahead(now, midpoint)

        objective[layout["throughput"] + t] += (
            max(config.cycle_cost_per_kwh, 0.0) * discount
        )

        # Uniform, undiscounted final-inventory value makes an equal secondary
        # discharge and refill cancel exactly; physical slot economics decide.
        if replacement > 1e-9:
            objective[layout["charge"] + t] -= replacement
            objective[layout["discharge"] + t] += replacement


def _secondary_integrality(
    n_vars: int,
    m: int,
    layout: SecondaryLayout,
) -> np.ndarray:  # type: ignore[name-defined]
    """Return a HiGHS integrality vector for charge and SBU mode blocks."""
    integrality = np.zeros(n_vars, dtype=int)
    integrality[layout["charge_mode"] : layout["charge_mode"] + m] = 1
    integrality[layout["sbu_mode"] : layout["sbu_mode"] + m] = 1
    integrality[layout["charge_steps"] : layout["charge_steps"] + m] = 1
    return integrality


def _write_secondary_results(
    out_slots: list[PlannedSlot],
    *,
    result_x: np.ndarray,  # type: ignore[name-defined]
    layout: SecondaryLayout,
    config: SecondaryStorageConfig,
    future_idx: list[int],
    minimum_action_kwh: float,
    now: datetime,
    export_min_price: float = 0.0,
    battery_export_min_price: float = 0.0,
    primary_site_discharge_limited: np.ndarray | None = None,  # type: ignore[name-defined]
) -> dict[str, float | int] | None:
    """Write solved secondary flows, modes, current targets, SoC, and labels.

    The primary writer classifies Huawei actions before the PowMr site-bus
    delta is applied. Reconcile only those Huawei action labels whose source
    or destination changes after that final balance adjustment; solved primary
    charge/discharge energy remains authoritative and is never mutated here.
    """
    charge_eff = clamp_efficiency(config.charge_efficiency_pct)
    running_capacity = config.current_usable_kwh
    charge_slots = 0
    sbu_slots = 0
    total_charge = 0.0
    total_discharge = 0.0
    if primary_site_discharge_limited is None:
        primary_site_discharge_limited = np.zeros(len(future_idx), dtype=bool)

    for t, slot_i in enumerate(future_idx):
        slot = out_slots[slot_i]
        charge_kwh = max(float(result_x[layout["charge"] + t]), 0.0)
        discharge_kwh = max(float(result_x[layout["discharge"] + t]), 0.0)
        charge_mode = float(result_x[layout["charge_mode"] + t]) > 0.5
        sbu_mode = float(result_x[layout["sbu_mode"] + t]) > 0.5
        load_kwh = slot.secondary_storage_load_kwh
        site_load_offset_kwh = secondary_site_load_offset_kwh(slot, config)

        if sbu_mode:
            mode = SECONDARY_MODE_SBU
            sbu_slots += 1
        elif charge_mode and charge_kwh > minimum_action_kwh:
            mode = SECONDARY_MODE_CHARGE
            charge_slots += 1
        else:
            mode = SECONDARY_MODE_UTILITY

        running_capacity += charge_kwh - discharge_kwh
        running_capacity = min(max(running_capacity, 0.0), config.usable_kwh)
        reserve_kwh = config.capacity_kwh * config.min_soc_pct / 100.0
        absolute_soc = (running_capacity + reserve_kwh) / config.capacity_kwh * 100.0

        effective_hours = secondary_slot_duration_hours(slot, now)
        current_a = 0.0
        if charge_kwh > minimum_action_kwh and effective_hours > 1e-9:
            # Use the same effective duration as the integer current-step
            # constraint so stored energy and the physical 10 A command stay
            # identical when the active slot is already partly elapsed.
            current_a = (
                charge_kwh * 1000.0 / (config.nominal_voltage_v * effective_hours)
            )
            current_a = min(
                max(current_a, config.min_charge_current_a), config.max_charge_current_a
            )

        powmr_grid_import = (0.0 if sbu_mode else load_kwh) + charge_kwh / charge_eff
        if config.base_load_includes_dedicated_load:
            site_delta = charge_kwh / charge_eff - (
                site_load_offset_kwh if sbu_mode else 0.0
            )
        else:
            site_delta = powmr_grid_import
        net_grid = slot.grid_import_kwh - slot.grid_export_kwh + site_delta

        slot.secondary_storage_charged_kwh = round(charge_kwh, 3)
        slot.secondary_storage_discharged_kwh = round(discharge_kwh, 3)
        slot.secondary_storage_grid_import_kwh = round(powmr_grid_import, 3)
        slot.secondary_storage_estimated_capacity_kwh = round(running_capacity, 3)
        slot.secondary_storage_estimated_soc_pct = round(absolute_soc, 2)
        slot.secondary_storage_charge_current_a = round(current_a, 1)
        slot.secondary_storage_mode = mode
        # The base writer deliberately reconstructs grid flow from Huawei/EV
        # fields instead of raw gi/ge (it may resolve degenerate ec/ed values).
        # Apply the secondary branch once to that reconstructed base flow.
        slot.grid_import_kwh = round(max(net_grid, 0.0), 3)
        slot.grid_export_kwh = round(max(-net_grid, 0.0), 3)
        slot.estimated_cost_currency = round(
            slot_grid_cash_flow_cost(
                slot,
                export_min_price=export_min_price,
            ),
            4,
        )

        if (
            sbu_mode
            and not config.allow_primary_battery_transfer
            and is_material_planned_energy_kwh(slot.batteries_charged_kwh)
        ):
            log_planner(
                "warning",
                "[milp] secondary SBU transfer invariant failed  slot=%s  "
                "primary_charge=%.3f",
                slot.start.isoformat(),
                slot.batteries_charged_kwh,
            )
            return None

        # Huawei is physically constrained to self-consumption in these slots:
        # export is disabled, its price is below a configured floor, or the
        # non-co-optimised EV guard applies. The MILP coupling above makes
        # Huawei-derived post-SBU export impossible, while independent PV
        # export remains valid. Do not hide a broken battery-energy allocation
        # behind a BDM label: reject the candidate so the caller can retain or
        # fall back to a plan whose flows are executable.
        if (
            bool(primary_site_discharge_limited[t])
            and is_material_planned_energy_kwh(slot.grid_export_kwh)
            and is_material_planned_energy_kwh(slot.batteries_discharged_kwh)
        ):
            log_planner(
                "warning",
                "[milp] secondary write-out invariant failed  slot=%s  "
                "grid_export=%.3f  export_price=%.4f  export_floor=%.4f",
                slot.start.isoformat(),
                slot.grid_export_kwh,
                slot.price.export_price,
                battery_export_min_price,
            )
            return None

        # Recommendation labels drive materially different Huawei modes. A
        # primary charge that needed grid energy before SBU removed the
        # dedicated load may be fully PV-backed in the final site balance, so
        # it must use MSC rather than forced TOU grid charge. Likewise, MSC
        # cannot execute a solved primary discharge that now exports after the
        # SBU adjustment; use fully-fed-to-grid only when the final export also
        # clears the configured battery-export price floor.
        if (
            slot.recommendation == Recommendations.BatteriesChargeGrid.value
            and slot.batteries_charged_kwh > minimum_action_kwh
            and not is_material_planned_energy_kwh(slot.grid_import_kwh)
        ):
            slot.recommendation = Recommendations.BatteriesChargeSolar.value
        elif (
            slot.recommendation
            in {
                Recommendations.BatteriesDischargeMode.value,
                Recommendations.ForceBatteriesDischarge.value,
            }
            and slot.batteries_discharged_kwh > minimum_action_kwh
        ):
            if (
                is_material_planned_energy_kwh(slot.grid_export_kwh)
                and slot.price.export_price >= battery_export_min_price
            ):
                slot.recommendation = Recommendations.ForceBatteriesDischarge.value
            else:
                slot.recommendation = Recommendations.BatteriesDischargeMode.value

        total_charge += charge_kwh
        total_discharge += discharge_kwh

    return {
        "secondary_charge_slots": charge_slots,
        "secondary_final_usable_kwh": round(running_capacity, 6),
        "secondary_sbu_slots": sbu_slots,
        "secondary_total_charged_kwh": round(total_charge, 6),
        "secondary_total_discharged_kwh": round(total_discharge, 6),
    }
