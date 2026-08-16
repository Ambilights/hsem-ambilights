"""Predicted prices may raise the value of stored energy, and nothing else.

Nord Pool publishes day-ahead around 13:00 CET, so a 48 h horizon carries an
unpublished tail where ``price_actionable`` is False and no price-driven
control happens.  That is right for actuators, but it left the planner with no
reason to fill a battery today for an expensive tomorrow — and the PowMr needs
roughly 5 h at full current for its usable 7.5 kWh, far longer once the L3
phase throttle bites, so "catch up after 13:00" is not always available.

The forecast channel closes that gap under two guarantees, both pinned here:

1. **Charge-only.** Callers combine the predicted and published valuations
   with ``max()``, so a cheap forecast collapses to the published-only answer.
   A prediction can never lower the worth of stored energy into justifying a
   discharge.
2. **Never actionable.** The points live in ``PlannerInput.price_forecast``
   and reach exactly two valuation helpers.  They never populate
   ``PlannedSlot.price`` and never extend ``price_actionable``.

``test_expensive_forecast_raises_primary_valuation`` and
``test_expensive_forecast_raises_secondary_valuation`` fail against
v6.2.2-powmr.30, where neither helper accepts a forecast at all.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from custom_components.hsem.models.hourly_consumption_average import (
    HourlyConsumptionAverage,
)
from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.planner_input import PlannerInput
from custom_components.hsem.models.price_forecast import (
    ForecastPricePoint,
    PriceForecast,
)
from custom_components.hsem.models.price_point import PricePoint
from custom_components.hsem.models.secondary_storage_config import (
    SecondaryStorageConfig,
)
from custom_components.hsem.models.solcast_slot import SolcastSlot
from custom_components.hsem.planner import run_planner
from custom_components.hsem.planner.future_value import (
    forecast_effective_prices,
    replacement_price_from_next_discharge,
)
from custom_components.hsem.planner.secondary_storage import (
    resolve_secondary_terminal_price,
)
from custom_components.hsem.utils.price_forecast import build_price_forecast
from custom_components.hsem.utils.prices import SlotPrice

NOW = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)

# Roughly the live SE4 numbers this was built against, in import basis.
CHEAP_TODAY = 0.903
DEAR_TOMORROW = 1.927
MAE = 0.4864


def _slot(
    start: datetime,
    import_price: float,
    *,
    recommendation: str | None = None,
    actionable: bool = True,
) -> PlannedSlot:
    slot = PlannedSlot(
        start=start,
        end=start + timedelta(minutes=15),
        price=SlotPrice(import_price=import_price, export_price=0.0),
    )
    slot.price_actionable = actionable
    slot.import_price_available = actionable
    slot.export_price_available = actionable
    if recommendation is not None:
        slot.recommendation = recommendation
    return slot


def _published_slots(price: float, count: int = 8) -> list[PlannedSlot]:
    """Future slots carrying a published price and a discharge recommendation."""
    return [
        _slot(
            NOW + timedelta(minutes=15 * (i + 1)),
            price,
            recommendation="batteries_discharge_mode",
        )
        for i in range(count)
    ]


def _forecast(value: float, count: int = 8, mae: float = MAE) -> PriceForecast:
    return PriceForecast(
        points=tuple(
            ForecastPricePoint(start=NOW + timedelta(hours=12 + i), value=value)
            for i in range(count)
        ),
        mae=mae,
        enabled=True,
    )


def _secondary() -> SecondaryStorageConfig:
    return SecondaryStorageConfig(
        enabled=True,
        capacity_kwh=15.0,
        current_soc_pct=50.0,
        min_soc_pct=50.0,
        max_soc_pct=100.0,
        nominal_voltage_v=25.6,
        load_power_w=200.0,
        max_charge_current_a=60.0,
        min_charge_current_a=10.0,
        charge_efficiency_pct=93.0,
        discharge_efficiency_pct=93.0,
    )


class TestPrimaryValuation:
    """replacement_price_from_next_discharge — the Huawei side."""

    def test_expensive_forecast_raises_primary_valuation(self) -> None:
        """A dear forecast lifts stored-energy value (fails against v30)."""
        slots = _published_slots(CHEAP_TODAY)
        published = replacement_price_from_next_discharge(slots, NOW, top_n=4)
        with_forecast = replacement_price_from_next_discharge(
            slots, NOW, top_n=4, forecast=_forecast(DEAR_TOMORROW)
        )
        assert published is not None and with_forecast is not None
        assert published == pytest.approx(CHEAP_TODAY)
        assert with_forecast == pytest.approx(DEAR_TOMORROW - MAE)
        assert with_forecast > published

    def test_cheap_forecast_cannot_lower_the_valuation(self) -> None:
        """The charge-only invariant: max(published, predicted)."""
        slots = _published_slots(DEAR_TOMORROW)
        published = replacement_price_from_next_discharge(slots, NOW, top_n=4)
        with_forecast = replacement_price_from_next_discharge(
            slots, NOW, top_n=4, forecast=_forecast(0.10)
        )
        assert with_forecast == pytest.approx(published)

    def test_disabled_forecast_is_inert(self) -> None:
        slots = _published_slots(CHEAP_TODAY)
        disabled = PriceForecast(
            points=(ForecastPricePoint(NOW + timedelta(hours=12), 9.99),),
            mae=0.0,
            enabled=False,
        )
        assert replacement_price_from_next_discharge(
            slots, NOW, top_n=4, forecast=disabled
        ) == pytest.approx(CHEAP_TODAY)

    def test_forecast_alone_values_energy_with_no_published_discharge(self) -> None:
        """With no priced discharge window, the forecast still values energy."""
        assert replacement_price_from_next_discharge(
            [], NOW, top_n=4, forecast=_forecast(DEAR_TOMORROW)
        ) == pytest.approx(DEAR_TOMORROW - MAE)

    def test_no_forecast_and_no_slots_stays_none(self) -> None:
        assert replacement_price_from_next_discharge([], NOW, top_n=4) is None


class TestSecondaryValuation:
    """resolve_secondary_terminal_price — the PowMr side."""

    def test_expensive_forecast_raises_secondary_valuation(self) -> None:
        """A dear forecast lifts the PowMr terminal price (fails against v30)."""
        slots = _published_slots(CHEAP_TODAY)
        cfg = _secondary()
        published = resolve_secondary_terminal_price(slots, cfg, NOW)
        with_forecast = resolve_secondary_terminal_price(
            slots, cfg, NOW, forecast=_forecast(DEAR_TOMORROW)
        )
        assert published is not None and with_forecast is not None
        assert published == pytest.approx(CHEAP_TODAY * 0.93)
        assert with_forecast == pytest.approx((DEAR_TOMORROW - MAE) * 0.93)
        assert with_forecast > published

    def test_cheap_forecast_cannot_lower_the_secondary_valuation(self) -> None:
        slots = _published_slots(DEAR_TOMORROW)
        cfg = _secondary()
        published = resolve_secondary_terminal_price(slots, cfg, NOW)
        with_forecast = resolve_secondary_terminal_price(
            slots, cfg, NOW, forecast=_forecast(0.10)
        )
        assert with_forecast == pytest.approx(published)

    def test_configured_override_still_wins(self) -> None:
        """An explicit replacement price short-circuits both paths."""
        cfg = _secondary()
        cfg.replacement_price_per_kwh = 1.5
        assert resolve_secondary_terminal_price(
            _published_slots(CHEAP_TODAY), cfg, NOW, forecast=_forecast(9.9)
        ) == pytest.approx(1.5)

    def test_invalid_config_returns_none(self) -> None:
        cfg = _secondary()
        cfg.enabled = False
        assert (
            resolve_secondary_terminal_price(
                _published_slots(CHEAP_TODAY), cfg, NOW, forecast=_forecast(9.9)
            )
            is None
        )


class TestHaircut:
    """The MAE haircut is what keeps a wide-interval forecast honest."""

    def test_mae_and_margin_both_subtract(self) -> None:
        fc = PriceForecast(
            points=(ForecastPricePoint(NOW + timedelta(hours=1), 2.0),),
            mae=0.4,
            margin=0.1,
            enabled=True,
        )
        assert forecast_effective_prices(fc, NOW) == pytest.approx([1.5])

    def test_haircut_floors_at_zero(self) -> None:
        fc = PriceForecast(
            points=(ForecastPricePoint(NOW + timedelta(hours=1), 0.2),),
            mae=5.0,
            enabled=True,
        )
        assert forecast_effective_prices(fc, NOW) == pytest.approx([0.0])

    def test_past_points_are_dropped(self) -> None:
        fc = PriceForecast(
            points=(
                ForecastPricePoint(NOW - timedelta(hours=1), 9.0),
                ForecastPricePoint(NOW + timedelta(hours=1), 2.0),
            ),
            mae=0.0,
            enabled=True,
        )
        assert forecast_effective_prices(fc, NOW) == pytest.approx([2.0])

    def test_none_forecast_is_empty(self) -> None:
        assert forecast_effective_prices(None, NOW) == []


class TestAttributeParsing:
    """build_price_forecast — tolerant of a hand-rolled template sensor."""

    def _attrs(self) -> dict:
        return {
            "forecast": [
                {"start": "2026-08-17T00:00:00+02:00", "value": 1.3374},
                {"start": "2026-08-17T00:15:00+02:00", "value": "1.2000"},
            ],
            "mae": 0.4864,
        }

    def test_parses_points_and_mae(self) -> None:
        fc = build_price_forecast(self._attrs(), enabled=True)
        assert fc.usable is True
        assert len(fc.points) == 2
        assert fc.points[1].value == pytest.approx(1.2)
        assert fc.mae == pytest.approx(0.4864)

    def test_disabled_reads_nothing(self) -> None:
        fc = build_price_forecast(self._attrs(), enabled=False)
        assert fc.usable is False
        assert fc.points == ()

    def test_missing_attributes_are_not_fatal(self) -> None:
        assert build_price_forecast(None, enabled=True).usable is False
        assert build_price_forecast({}, enabled=True).usable is False

    def test_naive_timestamps_are_skipped(self) -> None:
        """A naive start cannot be placed on the horizon without guessing."""
        fc = build_price_forecast(
            {"forecast": [{"start": "2026-08-17T00:00:00", "value": 1.0}]},
            enabled=True,
        )
        assert fc.points == ()

    def test_unparseable_points_are_skipped_individually(self) -> None:
        fc = build_price_forecast(
            {
                "forecast": [
                    {"start": "not-a-date", "value": 1.0},
                    {"start": "2026-08-17T00:00:00+02:00", "value": "nonsense"},
                    {"start": "2026-08-17T01:00:00+02:00", "value": 1.5},
                    "not-a-dict",
                ],
                "mae": "bad",
            },
            enabled=True,
        )
        assert len(fc.points) == 1
        assert fc.points[0].value == pytest.approx(1.5)
        assert fc.mae == pytest.approx(0.0)

    def test_margin_is_carried_even_when_disabled(self) -> None:
        fc = build_price_forecast(None, enabled=False, margin=0.25)
        assert fc.margin == pytest.approx(0.25)
        assert fc.usable is False


class TestIsolationThroughRunPlanner:
    """The forecast must not leak into slot prices or the actionable prefix.

    This is the structural guarantee the design rests on: predictions live in
    ``PlannerInput.price_forecast`` and reach only the two valuation helpers,
    so no actuator, MILP bound or export decision can see one.  Asserting it
    end-to-end means a future refactor that wires the forecast into slot
    population fails here rather than in production.
    """

    def _input(self, *, with_forecast: bool) -> PlannerInput:
        # 24 published hours, then a 24 h unpublished tail. Day 1 must be
        # supplied explicitly as unavailable: a caller passing only 24 day-0
        # points gets hour-only matching, which fills both days.
        prices = [
            PricePoint(hour=h, import_price=0.20, export_price=0.05, day_offset=0)
            for h in range(24)
        ] + [
            PricePoint(
                hour=h,
                import_price=0.0,
                export_price=0.0,
                day_offset=1,
                import_price_available=False,
                export_price_available=False,
            )
            for h in range(24)
        ]
        solar = [SolcastSlot(hour=h, pv_estimate=0.0) for h in range(24)]
        consumption = [
            HourlyConsumptionAverage(
                hour=h, avg_1d=0.5, avg_3d=0.5, avg_7d=0.5, avg_14d=0.5
            )
            for h in range(24)
        ]
        forecast = PriceForecast(
            points=tuple(
                ForecastPricePoint(
                    start=datetime(2024, 6, 16, h, 0, tzinfo=UTC), value=9.99
                )
                for h in range(24)
            ),
            mae=0.0,
            enabled=True,
        )
        return PlannerInput(
            now_iso="2024-06-15T00:00:00+00:00",
            interval_minutes=60,
            interval_length_hours=48,
            battery_soc_pct=50.0,
            battery_rated_capacity_kwh=10.0,
            battery_end_of_discharge_soc_pct=10.0,
            battery_max_charge_power_w=5000.0,
            battery_purchase_price=10_000.0,
            battery_expected_cycles=6000,
            weight_1d=25,
            weight_3d=30,
            weight_7d=30,
            weight_14d=15,
            consumption_averages=consumption,
            price_points=prices,
            solcast_slots=solar,
            battery_schedules=[],
            excess_export_enabled=False,
            months_winter=[1, 2, 3, 4, 10, 11, 12],
            house_power_includes_ev=False,
            is_read_only=True,
            price_forecast=forecast if with_forecast else PriceForecast(),
        )

    def test_forecast_never_makes_the_tail_actionable(self) -> None:
        result = run_planner(self._input(with_forecast=True))
        tail = [
            s
            for s in result.slots
            if s.start >= datetime(2024, 6, 16, 0, 0, tzinfo=UTC)
        ]
        assert tail, "expected an unpublished tail in a 48 h horizon"
        assert all(s.price_actionable is False for s in tail)

    def test_forecast_never_writes_a_slot_price(self) -> None:
        """A 9.99 forecast must not appear in any slot's price."""
        result = run_planner(self._input(with_forecast=True))
        for slot in result.slots:
            assert slot.price.import_price != pytest.approx(9.99), (
                f"forecast price leaked into slot {slot.start}"
            )

    def test_tail_slots_are_identical_with_and_without_the_forecast(self) -> None:
        """Only valuation may differ — the published tail plan must not."""
        without = run_planner(self._input(with_forecast=False))
        with_fc = run_planner(self._input(with_forecast=True))
        cut = datetime(2024, 6, 16, 0, 0, tzinfo=UTC)
        pairs = [
            (a, b)
            for a, b in zip(without.slots, with_fc.slots, strict=True)
            if a.start >= cut
        ]
        assert pairs
        for a, b in pairs:
            assert a.price_actionable == b.price_actionable
            assert a.price.import_price == pytest.approx(b.price.import_price)
            assert a.price.export_price == pytest.approx(b.price.export_price)
