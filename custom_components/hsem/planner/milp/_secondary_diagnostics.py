"""Aggregate decision diagnostics for the secondary-storage MILP."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.secondary_storage_config import (
    SecondaryStorageConfig,
)
from custom_components.hsem.planner.cost_types import CostWeights
from custom_components.hsem.planner.secondary_cost import (
    SecondaryCostAccumulator,
)
from custom_components.hsem.planner.secondary_storage import (
    SECONDARY_MODE_CHARGE,
    SECONDARY_MODE_SBU,
    secondary_charge_limits_kwh,
    secondary_site_load_offset_kwh,
)
from custom_components.hsem.utils.logger import log_planner
from custom_components.hsem.utils.misc import clamp_efficiency
from custom_components.hsem.utils.units import slot_duration_hours

_EPSILON = 1e-9


@dataclass(frozen=True)
class SecondaryResultSummary:
    """One solved secondary-storage decision and cost summary."""

    sbu_slots: int
    charge_slots: int
    utility_slots: int
    sbu_energy_kwh: float
    charge_energy_kwh: float
    sbu_saving: float
    charge_cost: float
    cycle_cost: float
    conversion_loss: float
    terminal_credit: float
    net: float
    soc_start_pct: float
    soc_end_pct: float
    reason: str


def _sanitised_import_price(slot: PlannedSlot) -> float:
    """Return the non-negative import price used by the scorer."""
    price = slot.price.import_price
    return 0.0 if math.isnan(price) else max(price, 0.0)


def _sanitised_prices(slot: PlannedSlot) -> tuple[float, float]:
    """Return the NaN-safe raw prices passed to secondary cost scoring."""
    import_price = slot.price.import_price
    export_price = slot.price.export_price
    return (
        0.0 if math.isnan(import_price) else import_price,
        0.0 if math.isnan(export_price) else export_price,
    )


def _secondary_cost_weights(
    config: SecondaryStorageConfig,
    min_export_price: float,
) -> CostWeights:
    """Build the same secondary cost inputs used by ``score_plan``."""
    return CostWeights(
        export_min_price=min_export_price,
        secondary_storage_enabled=True,
        secondary_storage_charge_efficiency_pct=config.charge_efficiency_pct,
        secondary_storage_discharge_efficiency_pct=(config.discharge_efficiency_pct),
        secondary_storage_cycle_cost_per_kwh=config.cycle_cost_per_kwh,
        secondary_storage_replacement_price_per_kwh=(config.replacement_price_per_kwh),
    )


def _actual_import_values(
    slots: list[PlannedSlot],
    config: SecondaryStorageConfig,
    future_idx: list[int],
) -> tuple[float, float]:
    """Return isolated avoided-import value and grid-charge import cost."""
    charge_eff = clamp_efficiency(config.charge_efficiency_pct)
    sbu_saving = 0.0
    charge_cost = 0.0

    for slot_i in future_idx:
        slot = slots[slot_i]
        price = _sanitised_import_price(slot)
        net_grid_kwh = slot.grid_import_kwh - slot.grid_export_kwh

        if slot.secondary_storage_mode == SECONDARY_MODE_SBU:
            # Add the dedicated load back to the solved SBU grid balance.
            # The positive-import delta is what SBU actually avoided; any
            # portion that merely increased export is deliberately excluded.
            load_offset = secondary_site_load_offset_kwh(slot, config)
            utility_import = max(net_grid_kwh + load_offset, 0.0)
            sbu_import = max(net_grid_kwh, 0.0)
            sbu_saving += max(utility_import - sbu_import, 0.0) * price

        if slot.secondary_storage_mode == SECONDARY_MODE_CHARGE:
            charge_ac_kwh = slot.secondary_storage_charged_kwh / charge_eff
            import_with_charge = max(net_grid_kwh, 0.0)
            import_without_charge = max(net_grid_kwh - charge_ac_kwh, 0.0)
            charge_cost += max(import_with_charge - import_without_charge, 0.0) * price

    return sbu_saving, charge_cost


def _minimum_secondary_discharge_kwh(
    slot: PlannedSlot,
    config: SecondaryStorageConfig,
) -> float:
    """Return battery energy required to serve this slot in SBU mode."""
    discharge_eff = clamp_efficiency(config.discharge_efficiency_pct)
    hours = slot_duration_hours(slot.start, slot.end)
    standby_kwh = max(config.inverter_standby_power_w, 0.0) * hours / 1000.0
    return slot.secondary_storage_load_kwh / discharge_eff + standby_kwh


def _potential_sbu_saving(
    slot: PlannedSlot,
    config: SecondaryStorageConfig,
) -> float:
    """Return import value of changing one utility slot to SBU."""
    price = _sanitised_import_price(slot)
    net_grid_kwh = slot.grid_import_kwh - slot.grid_export_kwh
    load_offset = secondary_site_load_offset_kwh(slot, config)
    utility_import = max(net_grid_kwh, 0.0)
    sbu_import = max(net_grid_kwh - load_offset, 0.0)
    return max(utility_import - sbu_import, 0.0) * price


def _parked_reason(
    slots: list[PlannedSlot],
    config: SecondaryStorageConfig,
    future_idx: list[int],
    weights: CostWeights,
) -> str:
    """Return a conservative explanation for a solution with no cycling."""
    future_slots = [slots[slot_i] for slot_i in future_idx]
    if not future_slots:
        return "unknown"

    required_draws = [
        _minimum_secondary_discharge_kwh(slot, config)
        for slot in future_slots
        if slot.secondary_storage_load_kwh > _EPSILON
    ]
    if not required_draws:
        return "no_price_spread"

    current_usable = max(config.current_usable_kwh, 0.0)
    if all(draw > current_usable + _EPSILON for draw in required_draws):
        headroom = max(config.usable_kwh - current_usable, 0.0)
        minimum_charge = min(
            secondary_charge_limits_kwh(
                config,
                slot_duration_hours(slot.start, slot.end),
            )[0]
            for slot in future_slots
        )
        if minimum_charge > headroom + _EPSILON:
            return "below_min_charge"

    best_without_terminal = -math.inf
    best_with_terminal = -math.inf
    terminal_blocks_profitable_slot = False

    for slot in future_slots:
        required_draw = _minimum_secondary_discharge_kwh(slot, config)
        if required_draw > current_usable + _EPSILON:
            continue

        potential = copy.copy(slot)
        potential.secondary_storage_charged_kwh = 0.0
        potential.secondary_storage_discharged_kwh = required_draw
        potential.secondary_storage_mode = SECONDARY_MODE_SBU
        costs = SecondaryCostAccumulator()
        import_price, export_price = _sanitised_prices(slot)
        conversion, cycle, terminal = costs.add_slot(
            potential,
            weights,
            import_price=import_price,
            export_price=export_price,
        )
        saving = _potential_sbu_saving(slot, config)
        without_terminal = saving - conversion - cycle
        with_terminal = without_terminal - terminal
        best_without_terminal = max(best_without_terminal, without_terminal)
        best_with_terminal = max(best_with_terminal, with_terminal)
        if without_terminal > _EPSILON and terminal > _EPSILON:
            terminal_blocks_profitable_slot = True

    if (
        terminal_blocks_profitable_slot
        and best_without_terminal > _EPSILON
        and best_with_terminal <= _EPSILON
    ):
        return "terminal_credit_wins"
    if best_without_terminal <= _EPSILON:
        return "no_price_spread"
    return "unknown"


def build_secondary_result_summary(
    slots: list[PlannedSlot],
    *,
    result_x: Sequence[float],
    layout: Mapping[str, int],
    config: SecondaryStorageConfig,
    future_idx: list[int],
    min_export_price: float,
) -> SecondaryResultSummary:
    """Build a read-only aggregate from one successful secondary MILP solve."""
    slot_count = len(future_idx)
    sbu_slots = sum(
        float(result_x[layout["sbu_mode"] + t]) > 0.5 for t in range(slot_count)
    )
    charge_slots = sum(
        float(result_x[layout["charge_steps"] + t]) > 0.5 for t in range(slot_count)
    )
    utility_slots = max(slot_count - sbu_slots - charge_slots, 0)
    sbu_energy = sum(
        max(float(result_x[layout["discharge"] + t]), 0.0) for t in range(slot_count)
    )
    charge_energy = sum(
        max(float(result_x[layout["charge"] + t]), 0.0) for t in range(slot_count)
    )

    weights = _secondary_cost_weights(config, min_export_price)
    costs = SecondaryCostAccumulator()
    for slot_i in future_idx:
        slot = slots[slot_i]
        import_price, export_price = _sanitised_prices(slot)
        costs.add_slot(
            slot,
            weights,
            import_price=import_price,
            export_price=export_price,
        )

    sbu_saving, charge_cost = _actual_import_values(slots, config, future_idx)
    terminal_credit = -costs.terminal_soc_value
    net = (
        sbu_saving
        - charge_cost
        - costs.cycle_cost
        - costs.conversion_loss_cost
        + terminal_credit
    )
    final_usable_kwh = min(
        max(config.current_usable_kwh + charge_energy - sbu_energy, 0.0),
        config.usable_kwh,
    )
    soc_end_pct = config.min_soc_pct + final_usable_kwh / config.capacity_kwh * 100.0
    reason = (
        "scheduled"
        if sbu_slots > 0 or charge_slots > 0
        else _parked_reason(slots, config, future_idx, weights)
    )

    return SecondaryResultSummary(
        sbu_slots=sbu_slots,
        charge_slots=charge_slots,
        utility_slots=utility_slots,
        sbu_energy_kwh=sbu_energy,
        charge_energy_kwh=charge_energy,
        sbu_saving=sbu_saving,
        charge_cost=charge_cost,
        cycle_cost=costs.cycle_cost,
        conversion_loss=costs.conversion_loss_cost,
        terminal_credit=terminal_credit,
        net=net,
        soc_start_pct=config.current_soc_pct,
        soc_end_pct=soc_end_pct,
        reason=reason,
    )


def log_secondary_result(summary: SecondaryResultSummary) -> None:
    """Emit exactly one compact debug line for one successful enabled solve."""
    log_planner(
        "debug",
        "[milp] secondary_result  sbu_slots=%d  charge_slots=%d  utility_slots=%d  "
        "sbu_energy=%.3f  charge_energy=%.3f  sbu_saving=%.4f  "
        "charge_cost=%.4f  cycle_cost=%.4f  conversion_loss=%.4f  "
        "terminal_credit=%.4f  net=%.4f  soc_start=%.1f  soc_end=%.1f  "
        "reason=%s",
        summary.sbu_slots,
        summary.charge_slots,
        summary.utility_slots,
        summary.sbu_energy_kwh,
        summary.charge_energy_kwh,
        summary.sbu_saving,
        summary.charge_cost,
        summary.cycle_cost,
        summary.conversion_loss,
        summary.terminal_credit,
        summary.net,
        summary.soc_start_pct,
        summary.soc_end_pct,
        summary.reason,
    )
