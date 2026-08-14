"""Future-price valuation helpers for battery and EV planning."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.utils.datetime_utils import utc_key
from custom_components.hsem.utils.recommendations import DISCHARGE_RECS


def replacement_price_from_next_discharge(
    slots: Sequence[PlannedSlot],
    now: datetime,
    top_n: int = 4,
    interval_minutes: int = 15,
) -> float | None:
    """Value stored energy from the first upcoming priced discharge window.

    The first contiguous block matters: later schedule occurrences in a
    multi-day horizon must not inflate the value assigned at this horizon's
    endpoint. Only slots with published import/export prices are eligible.
    """
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
