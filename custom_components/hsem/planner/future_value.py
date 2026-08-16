"""Future-price valuation helpers for battery and EV planning."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.price_forecast import PriceForecast
from custom_components.hsem.utils.datetime_utils import utc_key
from custom_components.hsem.utils.recommendations import DISCHARGE_RECS


def published_horizon_end(
    slots: Sequence[PlannedSlot],
    now: datetime,
) -> datetime | None:
    """Return the end of the contiguous future prefix that carries real prices.

    ``price_actionable`` is set only across that prefix and is closed
    permanently at its first gap (see ``engine_population``), so the furthest
    actionable end *is* the boundary between published and unpublished.

    ``None`` means nothing ahead is published.
    """
    now_utc = utc_key(now)
    end: datetime | None = None
    for slot in slots:
        if utc_key(slot.end) <= now_utc or not slot.price_actionable:
            continue
        slot_end = utc_key(slot.end)
        if end is None or slot_end > end:
            end = slot_end
    return end


def forecast_effective_prices(
    forecast: PriceForecast | None,
    now: datetime,
    slots: Sequence[PlannedSlot] = (),
) -> list[float]:
    """Return predicted prices for the unpublished tail, after the haircut.

    A prediction is only ever a stand-in for a price the market has not
    published yet.  Once real prices exist for a slot the forecast has nothing
    to add there and must not compete with them, so every point falling inside
    the published prefix is dropped.  Without that filter an over-optimistic
    forecast could beat a published price through the callers' ``max()`` and
    size a plan on fiction while the real answer was already known.

    Each surviving point is reduced by the source's own published error plus
    any operator margin, floored at zero, so a valuation is only taken on
    margin that survives the forecast's measured accuracy.  Points at or before
    ``now`` are dropped: the past cannot be charged for.

    Returns an empty list when the feature is off, the feed is empty, or the
    whole feed is already published — all of which callers treat as "no
    forecast contribution" rather than "worth nothing".
    """
    if forecast is None or not forecast.usable:
        return []
    now_utc = utc_key(now)
    published_until = published_horizon_end(slots, now)
    cut = forecast.haircut
    return [
        max(point.value - cut, 0.0)
        for point in forecast.points
        if math.isfinite(point.value)
        and utc_key(point.start) > now_utc
        and (published_until is None or utc_key(point.start) >= published_until)
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

    When ``forecast`` is supplied, only its points beyond the published prefix
    are eligible — a prediction never competes with a real price — and those
    are valued the same way, as the mean of the ``top_n`` dearest haircut
    prices.  The higher of the two then wins.  Taking the maximum is what keeps
    the feature charge-only: a cheap forecast collapses to the published-only
    answer, so a prediction can raise the worth of stored energy but never
    lower it into justifying a discharge.
    """
    published = _published_replacement_price(slots, now, top_n, interval_minutes)
    predicted = _top_n_mean(forecast_effective_prices(forecast, now, slots), top_n)
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
