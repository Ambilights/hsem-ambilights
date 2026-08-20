"""Build a :class:`PriceForecast` from a Home Assistant sensor's attributes.

Pure — no Home Assistant imports — so the parsing rules are unit-testable
without a running instance.

Expected attribute shape on the configured sensor::

    forecast: [{"start": "<ISO-8601 with offset>", "value": <float>}, ...]
    mae:      <float>          # the source's own published error, optional

``low`` and ``high`` may also be present and are ignored: on the reference
source their 80 % interval is far too wide to carry any signal (the lower
bound sat within 0.02 of zero across a whole day whose point estimates spanned
0.2-0.8), so the published mean absolute error is the usable haircut instead.

Prices must already be in the same basis as the configured import price sensor
— VAT, grid fees and markup applied. HSEM performs no tax or currency
transform anywhere, so a mismatch here would silently bias every comparison.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from custom_components.hsem.models.price_forecast import (
    ForecastPricePoint,
    PriceForecast,
)
from custom_components.hsem.utils.conversion import convert_to_float


def _nonnegative_finite(value: Any) -> float | None:
    """Return a finite non-negative number, or None when invalid."""
    number = convert_to_float(value)
    if number is None or not math.isfinite(number):
        return None
    return max(number, 0.0)


def _parse_start(raw: Any) -> datetime | None:
    """Return a timezone-aware start, or None when unusable."""
    if isinstance(raw, datetime):
        parsed = raw
    elif isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
    else:
        return None
    # A naive timestamp cannot be placed on the horizon without guessing a
    # zone, and guessing wrong shifts a whole day of prices.
    return parsed if parsed.tzinfo is not None else None


def build_price_forecast(
    attributes: dict[str, Any] | None,
    *,
    enabled: bool,
    margin: float = 0.0,
) -> PriceForecast:
    """Parse forecast points and the source's error from sensor attributes.

    Args:
        attributes: The sensor's ``.attributes`` dict, or None when the entity
            is missing or unavailable.
        enabled: Whether the operator switched the feature on. A disabled
            feature returns an empty forecast without reading anything, so
            turning it off is a hard stop rather than a soft preference.
        margin: Extra haircut on top of the source's published error.

    Returns:
        A :class:`PriceForecast`. Unparseable points are skipped rather than
        failing the whole feed; a feed that yields no usable point returns
        ``usable == False``, which callers treat as "no contribution".
    """
    parsed_margin = _nonnegative_finite(margin)
    margin_valid = parsed_margin is not None
    safe_margin = parsed_margin if parsed_margin is not None else 0.0
    if not enabled or not attributes:
        return PriceForecast(enabled=enabled, margin=safe_margin)

    raw_points = attributes.get("forecast")
    points: list[ForecastPricePoint] = []
    if isinstance(raw_points, list):
        for entry in raw_points:
            if not isinstance(entry, dict):
                continue
            start = _parse_start(entry.get("start"))
            value = convert_to_float(entry.get("value"))
            if start is None or value is None or not math.isfinite(value):
                continue
            points.append(ForecastPricePoint(start=start, value=value))

    # A genuinely absent MAE remains zero. Explicit malformed or nonfinite
    # confidence disables the whole valuation channel: retaining points while
    # replacing its safety haircut would overvalue inventory. Keep confidence
    # fields finite and remove points so coordinator signatures and diagnostics
    # remain stable and JSON-safe while ``usable`` still fails closed.
    if "mae" in attributes:
        parsed_mae = _nonnegative_finite(attributes["mae"])
        mae_valid = parsed_mae is not None
        safe_mae = parsed_mae if parsed_mae is not None else 0.0
    else:
        mae_valid = True
        safe_mae = 0.0

    if not margin_valid or not mae_valid:
        points.clear()

    return PriceForecast(
        points=tuple(points),
        mae=safe_mae,
        margin=safe_margin,
        enabled=True,
    )
