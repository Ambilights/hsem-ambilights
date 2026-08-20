"""Bounded primary-battery value beyond the published price boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class TerminalValueTier:
    """One post-boundary opportunity that can consume terminal inventory.

    The quantity and value are battery-side DC terms. The forecast price
    retains the conservative AC import-price input for diagnostics; it is not
    an actionable planner price.
    """

    start: datetime
    quantity_kwh: float
    value_per_kwh: float
    forecast_price_per_kwh: float


@dataclass(frozen=True)
class TerminalCostToGo:
    """Piecewise-linear value of primary inventory at the action boundary.

    Tiers are ordered from highest to lowest marginal value. Stored energy is
    assigned to those opportunities in that order, so inventory above the
    total bounded demand has no synthetic salvage value.
    """

    tiers: tuple[TerminalValueTier, ...] = field(default_factory=tuple)
    source: str = "hardware_floor_only"
    boundary: datetime | None = None

    def __post_init__(self) -> None:
        """Canonicalize finite tiers for deterministic scorer/MILP parity."""
        valid: list[TerminalValueTier] = []
        for tier in self.tiers:
            try:
                quantity = float(tier.quantity_kwh)
                value = float(tier.value_per_kwh)
                forecast_price = float(tier.forecast_price_per_kwh)
                sort_time = tier.start.timestamp()
            except AttributeError, TypeError, ValueError, OverflowError:
                continue
            if not (
                math.isfinite(quantity)
                and quantity > 1e-9
                and math.isfinite(value)
                and value > 1e-9
                and math.isfinite(forecast_price)
                and math.isfinite(sort_time)
            ):
                continue
            valid.append(
                TerminalValueTier(
                    start=tier.start,
                    quantity_kwh=quantity,
                    value_per_kwh=value,
                    forecast_price_per_kwh=forecast_price,
                )
            )
        ordered = tuple(
            sorted(
                valid,
                key=lambda tier: (-tier.value_per_kwh, tier.start.timestamp()),
            )
        )
        object.__setattr__(self, "tiers", ordered)

    @property
    def active(self) -> bool:
        """Whether at least one finite positive-valued tier is available."""
        return any(
            math.isfinite(tier.quantity_kwh)
            and tier.quantity_kwh > 1e-9
            and math.isfinite(tier.value_per_kwh)
            and tier.value_per_kwh > 1e-9
            for tier in self.tiers
        )

    @property
    def total_quantity_kwh(self) -> float:
        """Return the total battery-side demand represented by all tiers."""
        return sum(
            max(tier.quantity_kwh, 0.0)
            for tier in self.tiers
            if math.isfinite(tier.quantity_kwh)
        )

    def inventory_valued_quantity(self, inventory_kwh: float) -> float:
        """Return inventory covered by the bounded economic tiers."""
        try:
            inventory = float(inventory_kwh)
        except TypeError, ValueError, OverflowError:
            return 0.0
        if not math.isfinite(inventory):
            return 0.0
        return min(max(inventory, 0.0), self.total_quantity_kwh)

    def inventory_value(self, inventory_kwh: float) -> float:
        """Return the bounded cost-to-go value of inventory.

        The calculation depends only on final inventory, never on the path
        taken to reach it. Invalid or negative inventory is treated as zero.
        """
        remaining = self.inventory_valued_quantity(inventory_kwh)
        value = 0.0
        for tier in self.tiers:
            if remaining <= 1e-9:
                break
            if not (
                math.isfinite(tier.quantity_kwh)
                and tier.quantity_kwh > 1e-9
                and math.isfinite(tier.value_per_kwh)
                and tier.value_per_kwh > 1e-9
            ):
                continue
            allocated = min(remaining, tier.quantity_kwh)
            value += allocated * tier.value_per_kwh
            remaining -= allocated
        return value
