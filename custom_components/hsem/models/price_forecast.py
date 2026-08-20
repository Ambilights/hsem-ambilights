"""Predicted import prices for the horizon the market has not published yet.

Nord Pool publishes day-ahead prices around 13:00 CET.  A 48 h planning
horizon therefore carries an unpublished tail that ``price_actionable``
deliberately closes: no price-driven control happens there.  That is correct
for actuators, but it also means the planner has no reason to fill a battery
today for an expensive tomorrow — and a slow charger may not be able to catch
up once real prices arrive.

This module carries a *separate* price channel for that gap.  Two properties
make it safe:

- **It never enters** PlannedSlot.price and never extends price_actionable. A
  prediction cannot directly enable charge, discharge, or export in its own
  slot. The primary optimiser sees only derived terminal inventory tiers, not
  a forecast action price.
- **Primary value is conservative and bounded.** After MAE and operator
  margin, a forecast point contributes only when it aligns with residual local
  load in a strictly non-actionable slot at or beyond the published boundary.
  Its battery-side quantity is capped by physical discharge and usable
  capacity, so excess final inventory receives no synthetic value. The legacy
  secondary-storage scalar helper separately retains its charge-only
  max(published-only, forecast-derived) behaviour.

Prices are expected in the same basis as the configured import price sensor
(VAT and fixed costs already applied); HSEM applies no currency or tax
transform anywhere, so matching the basis is the source's responsibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class ForecastPricePoint:
    """One predicted import price.

    Attributes:
        start: Timezone-aware start of the slot the prediction covers.
        value: Predicted import price in local currency/kWh, already in the
            configured import basis.
    """

    start: datetime
    value: float


@dataclass(frozen=True)
class PriceForecast:
    """Predicted import prices plus the confidence haircut to apply.

    Attributes:
        points: Predicted prices, in no guaranteed order.
        mae: The source's own published mean absolute error for this horizon,
            in the same units as ``value``.  Subtracted from every point so a
            decision is only taken on margin that survives the source's
            measured error.
        margin: Extra operator-configured haircut on top of ``mae``.  Defaults
            to zero — the published error is normally the honest number, and a
            second arbitrary knob mostly hides it.
        enabled: Whether the operator switched the feature on.  Kept explicit
            rather than inferred from ``points`` so an empty feed and a
            disabled feature stay distinguishable in diagnostics.
    """

    points: tuple[ForecastPricePoint, ...] = field(default_factory=tuple)
    mae: float = 0.0
    margin: float = 0.0
    enabled: bool = False

    @property
    def usable(self) -> bool:
        """Whether points and confidence inputs may contribute to valuation."""
        return (
            self.enabled
            and bool(self.points)
            and math.isfinite(self.mae)
            and math.isfinite(self.margin)
        )

    @property
    def haircut(self) -> float:
        """Total amount subtracted from each predicted price."""
        return max(self.mae, 0.0) + max(self.margin, 0.0)
