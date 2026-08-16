"""Future-price valuation helpers for battery and EV planning."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.price_forecast import PriceForecast
from custom_components.hsem.utils.datetime_utils import utc_key
from custom_components.hsem.utils.recommendations import DISCHARGE_RECS


def forecast_effective_prices(
    forecast: PriceForecast | None,
    now: datetime,
) -> list[float]:
    """Return future predicted prices after the confidence haircut.

    Each point is reduced by the source's own published error plus any
    operator margin, floored at zero, so a valuation is only ever taken on
    margin that survives the forecast's measured accuracy.  Points at or
    before ``now`` are dropped: the past cannot be charged for.

    Returns an empty list when the feature is off or the feed is empty, which
    callers treat as "no forecast contribution" rather than "worth nothing".
    """
    if forecast is None or not forecast.usable:
        return []
    now_utc = utc_key(now)
    cut = forecast.haircut
    return [
        max(point.value - cut, 0.0)
        for point in forecast.points
        if math.isfinite(point.value) and utc_key(point.start) > now_utc
    ]


def replacement_price_from_next_discharge(
    slots: Sequence[PlannedSlot],
    now: datetime,
    top_n: int = 4,
    interval_minutes: int = 15,
    forecast: PriceForecast | None = None,
) -> float | None:
    """Value stored energy from the first upcoming priced discharge window.

    The first contiguous block matters: later schedule occurrences in a
    multi-day horizon must not inflate the value assigned at this horizon's
    endpoint. Only slots with published import/export prices are eligible.

    When ``forecast`` is supplied, the predicted unpublished tail is valued the
    same way — the mean of its ``top_n`` dearest haircut prices — and the
    higher of the two wins.  Taking the maximum is what keeps the feature
    charge-only: a cheap forecast collapses to the published-only answer, so a
    prediction can raise the worth of stored energy but never lower it into
    justifying a discharge.
    """
    published = _published_replacement_price(slots, now, top_n, interval_minutes)
    predicted = _top_n_mean(forecast_effective_prices(forecast, now), top_n)
    if published is None:
        return predicted
    if predicted is None:
        return published
    return max(published, predicted)


def _top_n_mean(prices: list[float], top_n: int) -> float | None:
    """Mean of the ``top_n`` dearest prices, or None when there are none."""
    if not prices:
        return None
    dearest = sorted(prices, reverse=True)[: max(top_n, 1)]
    return sum(dearest) / len(dearest)


def _published_replacement_price(
    slots: Sequence[PlannedSlot],
    now: datetime,
    top_n: int,
    interval_minutes: int,
) -> float | None:
    """Original published-price-only valuation, unchanged."""
    future_discharge = sorted(
        (
            slot
            for slot in slots
            if slot.recommendation in DISCHARGE_RECS
            and slot.price_actionable
            and utc_key(slot.start) > utc_key(now)
            and math.isfinite(slot.price.import_price)
        ),
        key=lambda slot: utc_key(slot.start),
    )
    if not future_discharge:
        return None

    gap_threshold = timedelta(minutes=interval_minutes + 5)
    first_block = [future_discharge[0]]
    for slot in future_discharge[1:]:
        previous_end = utc_key(first_block[-1].end)
        slot_start = utc_key(slot.start)
        if slot_start - previous_end > gap_threshold:
            break
        first_block.append(slot)

    prices = sorted((slot.price.import_price for slot in first_block), reverse=True)[
        :top_n
    ]
    return sum(prices) / len(prices) if prices else None


def ev_future_charge_value_per_kwh(
    slots: Sequence[PlannedSlot],
    now: datetime,
    lookahead_hours: float = 24.0,
    confidence_factor: float = 0.9,
) -> float | None:
    """Estimate avoided future import cost for optional EV surplus charging."""
    now_utc = utc_key(now)
    cutoff = now_utc + timedelta(hours=lookahead_hours)
    future_prices = [
        float(slot.price.import_price)
        for slot in slots
        if now_utc < utc_key(slot.start) <= cutoff
        and slot.price_actionable
        and math.isfinite(slot.price.import_price)
    ]
    if not future_prices:
        return None
    return confidence_factor * (sum(future_prices) / len(future_prices))
