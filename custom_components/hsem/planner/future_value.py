"""Future-price valuation helpers for battery and EV planning."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.price_forecast import PriceForecast
from custom_components.hsem.models.terminal_cost_to_go import (
    TerminalCostToGo,
    TerminalValueTier,
)
from custom_components.hsem.utils.datetime_utils import utc_key
from custom_components.hsem.utils.misc import clamp_efficiency
from custom_components.hsem.utils.recommendations import DISCHARGE_RECS


def _finite_float(value: Any) -> float | None:
    """Return a finite float, or None for malformed planner input."""
    try:
        number = float(value)
    except TypeError, ValueError, OverflowError:
        return None
    return number if math.isfinite(number) else None


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
    the published prefix is dropped. Without that filter an over-optimistic
    forecast could leak into the legacy scalar replacement valuation and
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
    mae = _finite_float(forecast.mae)
    margin = _finite_float(forecast.margin)
    if mae is None or margin is None:
        return []

    now_utc = utc_key(now)
    published_until = published_horizon_end(slots, now)
    cut = max(mae, 0.0) + max(margin, 0.0)
    if not math.isfinite(cut):
        return []

    effective: list[float] = []
    for point in forecast.points:
        value = _finite_float(point.value)
        if value is None:
            continue
        point_start = utc_key(point.start)
        if point_start <= now_utc or (
            published_until is not None and point_start < published_until
        ):
            continue
        effective.append(max(value - cut, 0.0))
    return effective


def build_terminal_cost_to_go(
    slots: Sequence[PlannedSlot],
    now: datetime,
    *,
    forecast: PriceForecast | None,
    usable_kwh: float,
    max_discharge_per_slot: float | None,
    discharge_efficiency_pct: float,
    cycle_cost_per_kwh: float,
) -> TerminalCostToGo:
    """Build bounded value tiers from aligned unpublished-price demand.

    Forecast points remain valuation-only. A point contributes only when its
    UTC start exactly matches a future, non-actionable planner slot at or after
    the end of the contiguous published-price prefix. Its quantity is the
    battery-side residual house demand the primary battery could physically
    serve, excluding explicitly accounted EV demand and capped by power.

    Marginal DC value follows the in-horizon scoring convention: avoided AC
    import through discharge efficiency, less discharge loss and one resolved
    cycle-wear charge. With no valid tier, the existing hardware/effective
    dynamic floor remains the only reserve.
    """
    boundary = published_horizon_end(slots, now)
    empty = TerminalCostToGo(boundary=boundary)
    capacity = _finite_float(usable_kwh)
    efficiency_pct = _finite_float(discharge_efficiency_pct)
    wear = _finite_float(cycle_cost_per_kwh)
    if (
        forecast is None
        or not forecast.usable
        or capacity is None
        or capacity <= 1e-9
        or efficiency_pct is None
        or wear is None
    ):
        return empty

    mae = _finite_float(forecast.mae)
    margin = _finite_float(forecast.margin)
    if mae is None or margin is None:
        return empty
    haircut = max(mae, 0.0) + max(margin, 0.0)
    if not math.isfinite(haircut):
        return empty

    discharge_eff = clamp_efficiency(efficiency_pct)
    discharge_loss = 1.0 - discharge_eff
    wear = max(wear, 0.0)
    if max_discharge_per_slot is None:
        per_slot_cap = capacity
    else:
        discharge_cap = _finite_float(max_discharge_per_slot)
        if discharge_cap is None:
            return empty
        per_slot_cap = max(discharge_cap, 0.0)

    now_utc = utc_key(now)
    aligned_slots = {
        utc_key(slot.start): slot
        for slot in slots
        if utc_key(slot.end) > now_utc
        and not slot.price_actionable
        and (boundary is None or utc_key(slot.start) >= boundary)
    }
    if not aligned_slots:
        return empty

    # Conservatively collapse duplicate predictions to the lower effective
    # value for that one physical slot.
    effective_by_start: dict[datetime, float] = {}
    for point in forecast.points:
        point_start = utc_key(point.start)
        point_value = _finite_float(point.value)
        if (
            point_start <= now_utc
            or point_start not in aligned_slots
            or (boundary is not None and point_start < boundary)
            or point_value is None
        ):
            continue
        effective_price = max(point_value - haircut, 0.0)
        previous = effective_by_start.get(point_start)
        if previous is None or effective_price < previous:
            effective_by_start[point_start] = effective_price

    candidates: list[TerminalValueTier] = []
    for start, effective_price in effective_by_start.items():
        slot = aligned_slots[start]
        house_load = _finite_float(slot.avg_house_consumption_kwh)
        pv_estimate = _finite_float(slot.solcast_pv_estimate_kwh)
        ev_accounted = _finite_float(slot.ev_accounted_load_kwh)
        if house_load is None or pv_estimate is None or ev_accounted is None:
            continue
        eligible_house_load_ac = max(
            house_load - pv_estimate - max(ev_accounted, 0.0),
            0.0,
        )
        quantity_kwh = min(
            eligible_house_load_ac / discharge_eff,
            per_slot_cap,
            capacity,
        )
        marginal_value = (
            effective_price * discharge_eff - effective_price * discharge_loss - wear
        )
        if quantity_kwh <= 1e-9 or marginal_value <= 1e-9:
            continue
        candidates.append(
            TerminalValueTier(
                start=start,
                quantity_kwh=quantity_kwh,
                value_per_kwh=marginal_value,
                forecast_price_per_kwh=effective_price,
            )
        )

    # Highest-value demand gets first claim on finite terminal inventory.
    candidates.sort(key=lambda tier: (-tier.value_per_kwh, utc_key(tier.start)))
    remaining_capacity = capacity
    tiers: list[TerminalValueTier] = []
    for tier in candidates:
        if remaining_capacity <= 1e-9:
            break
        quantity = min(tier.quantity_kwh, remaining_capacity)
        tiers.append(
            TerminalValueTier(
                start=tier.start,
                quantity_kwh=quantity,
                value_per_kwh=tier.value_per_kwh,
                forecast_price_per_kwh=tier.forecast_price_per_kwh,
            )
        )
        remaining_capacity -= quantity

    if not tiers:
        return empty
    return TerminalCostToGo(
        tiers=tuple(tiers),
        source="forecast",
        boundary=boundary,
    )


def replacement_price_from_next_discharge(
    slots: Sequence[PlannedSlot],
    now: datetime,
    top_n: int = 4,
    interval_minutes: int = 15,
    forecast: PriceForecast | None = None,
) -> float | None:
    """Value stored energy from the first upcoming priced discharge window.

    This is the legacy scalar API retained for secondary-storage and direct
    backward-compatible callers. Production primary planning uses
    :func:`build_terminal_cost_to_go` instead.

    The first contiguous block matters: later discharge blocks in a multi-day
    horizon must not inflate the value assigned at this horizon's endpoint.
    Only slots with published import/export prices are eligible.

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
