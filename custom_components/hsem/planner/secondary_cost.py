"""Cost terms for dedicated-load secondary storage."""

from __future__ import annotations

from dataclasses import dataclass

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.planner.cost_helpers import compute_charge_premium
from custom_components.hsem.planner.cost_types import CostWeights
from custom_components.hsem.utils.misc import clamp_efficiency


@dataclass
class SecondaryCostAccumulator:
    """Accumulate authoritative secondary-storage cost terms."""

    conversion_loss_cost: float = 0.0
    cycle_cost: float = 0.0
    terminal_soc_value: float = 0.0

    def add_slot(
        self,
        slot: PlannedSlot,
        weights: CostWeights,
        *,
        import_price: float,
        export_price: float,
    ) -> tuple[float, float, float]:
        """Add and return one slot's conversion, cycle, and terminal terms."""
        conversion, cycle, terminal = secondary_slot_cost(
            slot,
            weights,
            import_price=import_price,
            export_price=export_price,
        )
        self.conversion_loss_cost += conversion
        self.cycle_cost += cycle
        self.terminal_soc_value += terminal
        return conversion, cycle, terminal


def secondary_slot_cost(
    slot: PlannedSlot,
    weights: CostWeights,
    *,
    import_price: float,
    export_price: float,
) -> tuple[float, float, float]:
    """Return conversion, cycle, and terminal costs for one secondary slot."""
    charge = slot.secondary_storage_charged_kwh
    discharge = slot.secondary_storage_discharged_kwh
    cycle = max(charge, discharge) * max(
        weights.secondary_storage_cycle_cost_per_kwh,
        0.0,
    )
    if not slot.price_actionable:
        # Wear remains a physical cost, but unknown prices cannot create
        # conversion valuation or a replacement/terminal incentive.
        return (0.0, cycle, 0.0)

    raw_import_price = import_price
    import_price = max(import_price, 0.0)
    if weights.export_min_price > 1e-9 and export_price < weights.export_min_price:
        export_price = 0.0
    export_price = min(export_price, raw_import_price)
    charge_eff = clamp_efficiency(weights.secondary_storage_charge_efficiency_pct)
    discharge_eff = clamp_efficiency(weights.secondary_storage_discharge_efficiency_pct)
    conversion = (
        charge * (1.0 - charge_eff) + discharge * (1.0 - discharge_eff)
    ) * import_price

    terminal = 0.0
    replacement = weights.secondary_storage_replacement_price_per_kwh
    if replacement is not None and replacement > 1e-9:
        charge_premium = compute_charge_premium(
            replacement_price_per_kwh=replacement,
            imp_price_obj=import_price,
            exp_price=export_price,
            charge_eff=charge_eff,
            deferred_export_price=None,
        )
        discharge_premium = max(0.0, replacement - import_price)
        terminal = -charge * charge_premium + discharge * discharge_premium

    return conversion, cycle, terminal
