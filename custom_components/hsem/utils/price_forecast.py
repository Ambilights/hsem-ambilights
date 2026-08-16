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
    if not enabled or not attributes:
        return PriceForecast(enabled=enabled, margin=max(margin, 0.0))

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

    # An absent error is treated as zero rather than as a reason to refuse the
    # feed: the operator margin is then the only haircut, which is visible in
    # the config rather than hidden in a silently-dropped channel.
    mae = convert_to_float(attributes.get("mae")) or 0.0
    if not math.isfinite(mae):
        mae = 0.0

    return PriceForecast(
        points=tuple(points),
        mae=max(mae, 0.0),
        margin=max(margin, 0.0),
        enabled=True,
    )
