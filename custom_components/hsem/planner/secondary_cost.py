"""Cost terms for dedicated-load secondary storage."""

from __future__ import annotations

import math
from dataclasses import dataclass

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.planner.cost_types import CostWeights


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

    # The physical site balance and final battery inventory already include
    # secondary conversion losses. Retain the diagnostic field/API but do not
    # price the same loss a second time here.
    conversion = 0.0
    _ = import_price, export_price
    replacement = max(
        weights.secondary_storage_replacement_price_per_kwh or 0.0,
        0.0,
    )
    if not math.isfinite(replacement):
        replacement = 0.0
    terminal = (discharge - charge) * replacement

    return conversion, cycle, terminal
