"""A prediction must never compete with a price the market has published.

Bug
---
``forecast_effective_prices`` filtered its points on three things only: the
feature is enabled, the value is finite, and the point is in the future.  It
never checked whether real prices already existed for those slots, so every
forecast point stayed eligible for the whole horizon and the callers'
``max(published, predicted)`` decided between them.

That maximum stops a forecast *lowering* the value of stored energy.  It does
not stop one being used where the real answer is already known — and in that
direction it is actively harmful:

    forecast under-predicts  ->  max() picks published    (harmless)
    forecast over-predicts   ->  max() picks the fiction  (plan sized on a guess)

Seen live on 2026-08-16.  Nord Pool published 08-17 with a 3.854 peak while the
forecast source still marked that day ``kind: forecast`` and kept emitting 96
points for it, peaking at 1.927.  Nothing went wrong only because the error
happened to be one-sided: realised MAE was 1.346 against an advertised 0.4864,
with a bias of -1.288.  Had it leaned the other way by the same margin, the
fiction would have won.

Fix
---
Points inside the published prefix are dropped.  ``price_actionable`` marks
that prefix and is closed permanently at its first gap, so its furthest end is
the published/unpublished boundary.

``test_optimistic_forecast_cannot_beat_a_published_price`` and
``test_secondary_ignores_forecast_inside_published_prefix`` fail against
v6.2.2-powmr.32.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.price_forecast import (
    ForecastPricePoint,
    PriceForecast,
)
from custom_components.hsem.models.secondary_storage_config import (
    SecondaryStorageConfig,
)
from custom_components.hsem.planner.future_value import (
    forecast_effective_prices,
    published_horizon_end,
    replacement_price_from_next_discharge,
)
from custom_components.hsem.planner.secondary_storage import (
    resolve_secondary_terminal_price,
)
from custom_components.hsem.utils.prices import SlotPrice

NOW = datetime(2026, 8, 16, 13, 0, tzinfo=UTC)
PUBLISHED_SLOTS = 8  # two hours of published quarter-hours

# The live numbers this was built from, in import basis.
PUBLISHED_PEAK = 3.854
OPTIMISTIC_FORECAST = 9.0


def _slot(index: int, price: float, *, actionable: bool) -> PlannedSlot:
    start = NOW + timedelta(minutes=15 * (index + 1))
    slot = PlannedSlot(
        start=start,
        end=start + timedelta(minutes=15),
        price=SlotPrice(import_price=price, export_price=0.0),
        recommendation="batteries_discharge_mode",
    )
    slot.price_actionable = actionable
    slot.import_price_available = actionable
    slot.export_price_available = actionable
    return slot


def _horizon(published_price: float = PUBLISHED_PEAK) -> list[PlannedSlot]:
    """Published quarter-hours followed by an unpublished tail."""
    return [
        _slot(i, published_price, actionable=True) for i in range(PUBLISHED_SLOTS)
    ] + [
        _slot(i, 0.0, actionable=False)
        for i in range(PUBLISHED_SLOTS, PUBLISHED_SLOTS + 8)
    ]


def _forecast_over(published: bool, value: float) -> PriceForecast:
    """Forecast points laid either over the published prefix or past it."""
    first = 0 if published else PUBLISHED_SLOTS
    return PriceForecast(
        points=tuple(
            ForecastPricePoint(
                start=NOW + timedelta(minutes=15 * (first + i + 1)), value=value
            )
            for i in range(PUBLISHED_SLOTS)
        ),
        mae=0.0,
        enabled=True,
    )


def _secondary() -> SecondaryStorageConfig:
    return SecondaryStorageConfig(
        enabled=True,
        capacity_kwh=15.0,
        current_soc_pct=55.0,
        min_soc_pct=50.0,
        max_soc_pct=100.0,
        nominal_voltage_v=25.6,
        load_power_w=200.0,
        max_charge_current_a=60.0,
        min_charge_current_a=10.0,
        charge_efficiency_pct=93.0,
        discharge_efficiency_pct=93.0,
    )


class TestPublishedPrecedence:
    """Published prices win over predictions covering the same hours."""

    def test_optimistic_forecast_cannot_beat_a_published_price(self) -> None:
        """The exposure the max() invariant never closed (fails against v32)."""
        slots = _horizon()
        value = replacement_price_from_next_discharge(
            slots,
            NOW,
            top_n=4,
            forecast=_forecast_over(published=True, value=OPTIMISTIC_FORECAST),
        )
        assert value == pytest.approx(PUBLISHED_PEAK)

    def test_forecast_beyond_the_prefix_still_counts(self) -> None:
        """The feature must keep working where it was meant to."""
        slots = _horizon()
        value = replacement_price_from_next_discharge(
            slots,
            NOW,
            top_n=4,
            forecast=_forecast_over(published=False, value=OPTIMISTIC_FORECAST),
        )
        assert value == pytest.approx(OPTIMISTIC_FORECAST)

    def test_pessimistic_forecast_over_published_is_still_ignored(self) -> None:
        """Direction does not matter — a published slot is simply not eligible."""
        slots = _horizon()
        value = replacement_price_from_next_discharge(
            slots, NOW, top_n=4, forecast=_forecast_over(published=True, value=0.10)
        )
        assert value == pytest.approx(PUBLISHED_PEAK)

    def test_nothing_published_keeps_every_point(self) -> None:
        """With no published prefix the forecast is all there is."""
        slots = [_slot(i, 0.0, actionable=False) for i in range(8)]
        prices = forecast_effective_prices(
            _forecast_over(published=True, value=2.0), NOW, slots
        )
        assert len(prices) == PUBLISHED_SLOTS

    def test_no_slots_argument_keeps_every_point(self) -> None:
        """Default empty slots means no published boundary is known."""
        prices = forecast_effective_prices(
            _forecast_over(published=True, value=2.0), NOW
        )
        assert len(prices) == PUBLISHED_SLOTS


class TestSecondaryPublishedPrecedence:
    """The PowMr mean must not average published and predicted hours together."""

    def test_secondary_ignores_forecast_inside_published_prefix(self) -> None:
        """Fails against v32, where the optimistic mean won."""
        slots = _horizon()
        value = resolve_secondary_terminal_price(
            slots,
            _secondary(),
            NOW,
            forecast=_forecast_over(published=True, value=OPTIMISTIC_FORECAST),
        )
        assert value == pytest.approx(PUBLISHED_PEAK * 0.93)

    def test_secondary_still_uses_the_unpublished_tail(self) -> None:
        slots = _horizon()
        value = resolve_secondary_terminal_price(
            slots,
            _secondary(),
            NOW,
            forecast=_forecast_over(published=False, value=OPTIMISTIC_FORECAST),
        )
        assert value == pytest.approx(OPTIMISTIC_FORECAST * 0.93)


class TestPublishedHorizonEnd:
    """The boundary helper itself."""

    def test_returns_end_of_the_published_prefix(self) -> None:
        slots = _horizon()
        end = published_horizon_end(slots, NOW)
        assert end == slots[PUBLISHED_SLOTS - 1].end

    def test_none_when_nothing_ahead_is_published(self) -> None:
        slots = [_slot(i, 0.0, actionable=False) for i in range(4)]
        assert published_horizon_end(slots, NOW) is None

    def test_elapsed_slots_are_ignored(self) -> None:
        """A published slot already in the past sets no boundary."""
        past = _slot(-4, 1.0, actionable=True)
        assert published_horizon_end([past], NOW) is None
