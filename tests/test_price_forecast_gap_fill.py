"""The dedicated forecast sensors must gap-fill, never overwrite.

Bug
---
``cfg.import_electricity_price_forecast_sensor`` and its export counterpart
were described in code as a *fallback*, but both call sites ran an
unconditional second pass after the primary sensor, through the same writer,
into the same field.  The writer's ``setattr`` never consulted the existing
availability flag, so wherever the forecast sensor covered a slot the primary
had already filled, the **forecast value won**.

On a day-ahead market that is backwards: a model output replaced a settled
Nord Pool price.  Amber Electric's rolling live estimate makes the overwrite
harmless for that provider, which is presumably why it went unnoticed.

Fix
---
Both forecast passes are now ``only_if_missing=True``.  Slots whose channel is
already available are left untouched; the forecast only reaches the
unpublished tail.

``test_forecast_does_not_overwrite_published_price`` fails against
v6.2.2-powmr.30, which reports the forecast value for every published slot.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from custom_components.hsem.custom_sensors.hourly_data_populator.prices_solcast import (
    populate_price_and_solcast_from_snapshot,
)
from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.models.state_snapshot import StateSnapshot

PUBLISHED = 2.5
PREDICTED = 0.4


def _make_rec(start: datetime, end: datetime) -> HourlyRecommendation:
    return HourlyRecommendation(
        start=start,
        end=end,
        recommendation="idle",
        avg_house_consumption_kwh=0.0,
        avg_house_consumption_1d_kwh=0.0,
        avg_house_consumption_3d_kwh=0.0,
        avg_house_consumption_7d_kwh=0.0,
        avg_house_consumption_14d_kwh=0.0,
        batteries_charged_kwh=0.0,
        batteries_discharged_kwh=0.0,
        estimated_battery_capacity_kwh=0.0,
        estimated_battery_soc_pct=0.0,
        estimated_cost_currency=0.0,
        estimated_net_consumption_kwh=0.0,
        export_price=0.0,
        grid_export_kwh=0.0,
        grid_import_kwh=0.0,
        import_price=0.0,
        solcast_pv_estimate_kwh=0.0,
    )


class _Cfg(SensorConfig):
    """SensorConfig with only the fields the price populator reads overridden."""

    def __init__(self) -> None:
        super().__init__()
        self.electricity_price_update_interval = 60
        self.recommendation_interval_minutes = 60
        self.import_electricity_price_sensor = "sensor.spot_import"
        self.export_electricity_price_sensor = "sensor.spot_export"
        self.import_electricity_price_forecast_sensor = "sensor.forecast_import"
        self.export_electricity_price_forecast_sensor = "sensor.forecast_export"
        self.solcast_pv_forecast_forecast_today = None
        self.solcast_pv_forecast_forecast_tomorrow = None
        self.solcast_pv_forecast_forecast_likelihood = "pv_estimate"


def _populate(recs: list[HourlyRecommendation], attrs: dict) -> None:
    populate_price_and_solcast_from_snapshot(
        recs,
        StateSnapshot(
            live=LiveState(), energy_average_values={}, sensor_attributes=attrs
        ),
        _Cfg(),
    )


def _points(
    base: datetime,
    count: int,
    price: float,
    first: int = 0,
    value_key: str = "price",
) -> list[dict]:
    """Build price points.

    ``prices``/``prices_today`` use ``{start, price}``; ``raw_today``/
    ``raw_tomorrow`` use ``{start, value}`` (the custom-components/nordpool
    shape), so the value key is caller-selected.
    """
    return [
        {
            "start": (base + timedelta(hours=first + i)).isoformat(),
            value_key: str(price),
        }
        for i in range(count)
    ]


class TestForecastGapFill:
    """The forecast sensor fills gaps and nothing else."""

    def setup_method(self) -> None:
        self.base = datetime(2026, 8, 16, 0, 0, tzinfo=UTC)
        self.recs = [
            _make_rec(
                self.base + timedelta(hours=i), self.base + timedelta(hours=i + 1)
            )
            for i in range(48)
        ]

    def test_forecast_does_not_overwrite_published_price(self) -> None:
        """Published hours keep the published price (fails against v30).

        The forecast source covers the whole 48 h horizon, the primary only
        the first 24 h.  Before the fix every one of those first 24 slots
        reported ``PREDICTED``.
        """
        _populate(
            self.recs,
            {
                "sensor.spot_import": {
                    "prices_today": _points(self.base, 24, PUBLISHED)
                },
                "sensor.spot_export": {
                    "prices_today": _points(self.base, 24, PUBLISHED)
                },
                "sensor.forecast_import": {"prices": _points(self.base, 48, PREDICTED)},
                "sensor.forecast_export": {"prices": _points(self.base, 48, PREDICTED)},
            },
        )

        for i, rec in enumerate(self.recs[:24]):
            assert rec.import_price == pytest.approx(PUBLISHED), (
                f"slot {i} ({rec.start}) took the forecast over the published price"
            )
            assert rec.export_price == pytest.approx(PUBLISHED)

    def test_forecast_still_fills_the_unpublished_tail(self) -> None:
        """Slots the primary never covered do get the forecast."""
        _populate(
            self.recs,
            {
                "sensor.spot_import": {
                    "prices_today": _points(self.base, 24, PUBLISHED)
                },
                "sensor.spot_export": {
                    "prices_today": _points(self.base, 24, PUBLISHED)
                },
                "sensor.forecast_import": {"prices": _points(self.base, 48, PREDICTED)},
                "sensor.forecast_export": {"prices": _points(self.base, 48, PREDICTED)},
            },
        )

        for rec in self.recs[24:]:
            assert rec.import_price == pytest.approx(PREDICTED)
            assert rec.export_price == pytest.approx(PREDICTED)
            assert rec.import_price_available is True
            assert rec.export_price_available is True

    def test_channels_are_independent(self) -> None:
        """A published import must not be displaced because export was absent."""
        _populate(
            self.recs,
            {
                "sensor.spot_import": {
                    "prices_today": _points(self.base, 24, PUBLISHED)
                },
                "sensor.forecast_import": {"prices": _points(self.base, 48, PREDICTED)},
                "sensor.forecast_export": {"prices": _points(self.base, 48, PREDICTED)},
            },
        )

        assert self.recs[0].import_price == pytest.approx(PUBLISHED)
        assert self.recs[0].export_price == pytest.approx(PREDICTED)

    def test_partial_gap_inside_the_published_window(self) -> None:
        """A hole punched in the middle of published data is filled, edges kept."""
        published = _points(self.base, 6, PUBLISHED) + _points(
            self.base, 6, PUBLISHED, first=12
        )
        _populate(
            self.recs,
            {
                "sensor.spot_import": {"prices_today": published},
                "sensor.spot_export": {"prices_today": published},
                "sensor.forecast_import": {"prices": _points(self.base, 24, PREDICTED)},
                "sensor.forecast_export": {"prices": _points(self.base, 24, PREDICTED)},
            },
        )

        assert [r.import_price for r in self.recs[:6]] == pytest.approx([PUBLISHED] * 6)
        assert [r.import_price for r in self.recs[6:12]] == pytest.approx(
            [PREDICTED] * 6
        )
        assert [r.import_price for r in self.recs[12:18]] == pytest.approx(
            [PUBLISHED] * 6
        )

    def test_primary_sensor_is_unaffected_by_the_flag(self) -> None:
        """With no forecast sensor configured the primary still overwrites itself.

        Guards against ``only_if_missing`` leaking onto the primary pass, where
        a later attribute (``raw_tomorrow`` after ``raw_today``) must still be
        able to write.
        """
        cfg = _Cfg()
        cfg.import_electricity_price_forecast_sensor = None
        cfg.export_electricity_price_forecast_sensor = None
        populate_price_and_solcast_from_snapshot(
            self.recs,
            StateSnapshot(
                live=LiveState(),
                energy_average_values={},
                sensor_attributes={
                    "sensor.spot_import": {
                        "raw_today": _points(
                            self.base, 24, PUBLISHED, value_key="value"
                        ),
                        "raw_tomorrow": _points(
                            self.base, 24, PUBLISHED, first=24, value_key="value"
                        ),
                    },
                    "sensor.spot_export": {
                        "raw_today": _points(
                            self.base, 24, PUBLISHED, value_key="value"
                        ),
                        "raw_tomorrow": _points(
                            self.base, 24, PUBLISHED, first=24, value_key="value"
                        ),
                    },
                },
            ),
            cfg,
        )

        assert all(r.import_price_available for r in self.recs)
        assert all(r.import_price == pytest.approx(PUBLISHED) for r in self.recs)
