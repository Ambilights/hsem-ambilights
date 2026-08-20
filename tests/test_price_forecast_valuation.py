"""Forecast prices may value post-boundary demand, but never become actionable.

Nord Pool publishes day-ahead around 13:00 CET, so a 48 h horizon carries an
unpublished tail where ``price_actionable`` is False and no price-driven
control happens.  That is right for actuators, but it left the planner with no
reason to fill a battery today for an expensive tomorrow — and the PowMr needs
roughly 5 h at full current for its usable 7.5 kWh, far longer once the L3
phase throttle bites, so "catch up after 13:00" is not always available.

The forecast channel closes that gap under two guarantees, both pinned here:

1. **Bounded and conservative.** Production primary valuation uses only exact
   post-boundary slot matches, applies MAE plus margin, and caps value to
   residual demand the battery can serve. The PowMr compatibility path still
   takes the higher of published and forecast-derived scalar valuations.
2. **Never actionable.** Forecast points remain separate from
   ``PlannedSlot.price`` and never extend ``price_actionable``.

The direct scalar Huawei helper tests below are retained for API compatibility;
production primary planning is covered by the bounded cost-to-go regressions.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

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
    build_terminal_cost_to_go,
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


class TestLegacyPrimaryScalarValuation:
    """Direct compatibility coverage for the retired production scalar path."""

    def test_expensive_forecast_raises_primary_valuation(self) -> None:
        """A dear forecast lifts the direct compatibility helper's value."""
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
            },
            enabled=True,
        )
        assert len(fc.points) == 1
        assert fc.points[0].value == pytest.approx(1.5)
        assert fc.mae == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("confidence_field", "confidence_value", "margin", "expected_usable"),
        [
            pytest.param(None, None, 0.0, True, id="absent-mae"),
            pytest.param("mae", "bad", 0.0, False, id="malformed-mae"),
            pytest.param("mae", float("nan"), 0.0, False, id="nan-mae"),
            pytest.param("mae", float("inf"), 0.0, False, id="positive-inf-mae"),
            pytest.param("mae", float("-inf"), 0.0, False, id="negative-inf-mae"),
            pytest.param(None, None, "bad", False, id="malformed-margin"),
            pytest.param(None, None, float("nan"), False, id="nan-margin"),
            pytest.param(None, None, float("inf"), False, id="positive-inf-margin"),
            pytest.param(
                None,
                None,
                float("-inf"),
                False,
                id="negative-inf-margin",
            ),
        ],
    )
    def test_confidence_metadata_is_fail_closed_end_to_end(
        self,
        confidence_field: str | None,
        confidence_value: Any,
        margin: Any,
        expected_usable: bool,
    ) -> None:
        """Only a genuinely absent MAE may use the zero-haircut default."""
        published = _slot(NOW + timedelta(minutes=15), 1.0)
        tail = _slot(NOW + timedelta(minutes=30), 0.0, actionable=False)
        tail.avg_house_consumption_kwh = 1.0
        tail.solcast_pv_estimate_kwh = 0.0
        tail.ev_accounted_load_kwh = 0.0

        attributes: dict[str, Any] = {
            "forecast": [{"start": tail.start.isoformat(), "value": 5.0}],
        }
        if confidence_field is not None:
            attributes[confidence_field] = confidence_value
        forecast = build_price_forecast(
            attributes,
            enabled=True,
            margin=margin,
        )
        cost_to_go = build_terminal_cost_to_go(
            [published, tail],
            NOW,
            forecast=forecast,
            usable_kwh=1.0,
            max_discharge_per_slot=1.0,
            discharge_efficiency_pct=100.0,
            cycle_cost_per_kwh=0.0,
        )

        assert forecast.enabled is True
        assert forecast.usable is expected_usable
        if expected_usable:
            assert forecast.mae == pytest.approx(0.0)
            assert forecast_effective_prices(
                forecast, NOW, [published, tail]
            ) == pytest.approx([5.0])
            assert cost_to_go.source == "forecast"
            assert cost_to_go.total_quantity_kwh == pytest.approx(1.0)
        else:
            assert forecast.points == ()
            assert forecast_effective_prices(forecast, NOW, [published, tail]) == []
            assert cost_to_go.source == "hardware_floor_only"
            assert cost_to_go.tiers == ()

    def test_margin_is_carried_even_when_disabled(self) -> None:
        fc = build_price_forecast(None, enabled=False, margin=0.25)
        assert fc.margin == pytest.approx(0.25)
        assert fc.usable is False


class TestIsolationThroughRunPlanner:
    """The forecast must not leak into slot prices or the actionable prefix.

    Predictions live in ``PlannerInput.price_forecast`` and reach the bounded
    primary cost-to-go builder plus the PowMr scalar compatibility helper; no
    forecast becomes an actionable slot price. Asserting that boundary
    end-to-end means a future refactor that wires a prediction into slot
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
            excess_export_enabled=False,
            months_winter=[1, 2, 3, 4, 10, 11, 12],
            house_power_includes_ev=False,
            is_read_only=True,
            price_forecast=forecast if with_forecast else PriceForecast(),
        )

    def test_engine_receives_bounded_forecast_cost_to_go(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pass a quantity-capped Unagi model through the production engine."""
        from custom_components.hsem.planner import engine_core

        captured_models: list[Any] = []
        captured_scalars: list[float | None] = []
        original_generate_candidates = engine_core.generate_candidates

        def capture_generate_candidates(*args: Any, **kwargs: Any) -> Any:
            captured_models.append(kwargs.get("terminal_cost_to_go"))
            captured_scalars.append(kwargs.get("replacement_price_per_kwh"))
            return original_generate_candidates(*args, **kwargs)

        monkeypatch.setattr(
            engine_core,
            "generate_candidates",
            capture_generate_candidates,
        )
        output = run_planner(self._input(with_forecast=True))

        assert len(captured_models) == 1
        assert captured_scalars == [None]
        model = captured_models[0]
        assert model is not None
        assert model.source == "forecast"
        assert model.boundary == datetime(2024, 6, 16, 0, 0, tzinfo=UTC)
        assert model.total_quantity_kwh == pytest.approx(9.0)
        assert model.inventory_value(100.0) == pytest.approx(
            sum(tier.quantity_kwh * tier.value_per_kwh for tier in model.tiers)
        )
        assert all(tier.start >= model.boundary for tier in model.tiers)
        explanation = output.explanation
        assert explanation.terminal_cost_to_go_source == model.source
        assert explanation.terminal_cost_to_go_boundary == model.boundary.isoformat()
        assert explanation.terminal_cost_to_go_tier_count == len(model.tiers)
        assert explanation.terminal_cost_to_go_total_quantity_kwh == pytest.approx(
            model.total_quantity_kwh
        )
        assert explanation.terminal_cost_to_go_initial_value == pytest.approx(
            model.inventory_value(4.0)
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
