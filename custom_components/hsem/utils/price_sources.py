"""Validation helpers for timestamped electricity-price source attributes.

The helpers are Home Assistant agnostic: callers pass an attribute mapping and
receive UTC-keyed prices plus a stable reason when the data is unusable. Price
values are preserved exactly apart from numeric parsing; no currency, VAT,
tariff, markup, or grid-fee transformation is performed here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

from custom_components.hsem.utils.conversion import convert_to_float

ENTSOE_PRICE_ATTRIBUTES = ("prices", "prices_today", "prices_tomorrow")
_PRICE_EPSILON = 1e-9


def normalize_price_unit(attributes: Mapping[str, Any] | None) -> str:
    """Return the trimmed unit_of_measurement attribute, or an empty string."""
    if not isinstance(attributes, Mapping):
        return ""
    unit = attributes.get("unit_of_measurement")
    return unit.strip() if isinstance(unit, str) else ""


def parse_entsoe_price_attributes(
    attributes: Mapping[str, Any] | None,
) -> tuple[dict[datetime, float], str | None]:
    """Parse supported ENTSO-E price arrays into a UTC-keyed mapping.

    At least one of prices, prices_today, or prices_tomorrow must be a
    non-empty list. Every entry must be a mapping containing a finite price and
    a timezone-aware time or start value. Repeated timestamps across the
    aggregate and day-specific arrays are accepted only when their prices
    agree.

    Returns:
        A price mapping and None on success. Invalid input returns an empty
        mapping and a stable reason string.
    """
    if not isinstance(attributes, Mapping):
        return {}, "attributes_missing"

    found_prices = False
    prices: dict[datetime, float] = {}
    for attribute in ENTSOE_PRICE_ATTRIBUTES:
        raw_points = attributes.get(attribute)
        if raw_points is None or raw_points == []:
            continue
        if not isinstance(raw_points, list):
            return {}, "price_array_invalid"

        found_prices = True
        attribute_timestamps: set[datetime] = set()
        for point in raw_points:
            if not isinstance(point, Mapping):
                return {}, "price_point_invalid"

            timestamp = _parse_price_timestamp(point.get("time") or point.get("start"))
            if timestamp is None:
                return {}, "timestamp_invalid_or_naive"

            raw_price = point.get("price")
            price = convert_to_float(raw_price)
            if isinstance(raw_price, bool) or price is None or not math.isfinite(price):
                return {}, "price_invalid_or_non_finite"
            if timestamp in attribute_timestamps:
                return {}, "duplicate_timestamp"
            attribute_timestamps.add(timestamp)

            if timestamp in prices:
                if abs(prices[timestamp] - price) > _PRICE_EPSILON:
                    return {}, "conflicting_timestamp_price"
                continue
            prices[timestamp] = price

    if not found_prices or not prices:
        return {}, "price_arrays_missing"
    return dict(sorted(prices.items())), None


def validate_price_cadence(
    prices: Mapping[datetime, float], expected_minutes: int
) -> str | None:
    """Return None when every adjacent UTC price point has the expected cadence."""
    if expected_minutes <= 0:
        return "configured_cadence_invalid"

    timestamps = sorted(prices)
    if len(timestamps) < 2:
        return "insufficient_price_points"

    for timestamp in timestamps:
        if (
            timestamp.second
            or timestamp.microsecond
            or timestamp.minute % expected_minutes
        ):
            return "price_timestamp_misaligned"

    expected_seconds = expected_minutes * 60
    for earlier, later in pairwise(timestamps):
        actual_seconds = (later - earlier).total_seconds()
        if abs(actual_seconds - expected_seconds) > _PRICE_EPSILON:
            return "price_cadence_mismatch"
    return None


def _parse_price_timestamp(raw: Any) -> datetime | None:
    """Return a timezone-aware timestamp normalized to UTC."""
    if isinstance(raw, datetime):
        parsed = raw
    elif isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
    else:
        return None

    try:
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(UTC)
    except OSError, OverflowError, ValueError:
        return None
