"""Regression tests for issue #720 — 15-minute prices collapsed to hourly.

Bug
---
``custom_sensors/hourly_data_populator/prices_solcast.py`` normalised every
price timestamp with ``.replace(minute=0, second=0)`` on both the sensor data
points and the recommendation slots before matching.  With 15-minute
recommendation slots and 15-minute price data (e.g. Energi Data Service on
Nord Pool 15-min MTUs), all four quarter-hour price points within an hour
collapsed onto the same key and the *last* one was written to all four
recommendation slots of that hour.

Fix
---
Both sides of the match are now floored to the enclosing recommendation slot
via ``normalize_slot_start(dt, interval_minutes)`` so each sub-hourly price
lands on exactly one slot.

These tests exercise the snapshot-based path
(``populate_price_and_solcast_from_snapshot``), which shares the matching
helper logic with the async path.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from custom_components.hsem.custom_sensors.hourly_data_populator.prices_solcast import (
    _async_update_hourly_field,
    populate_price_and_solcast_from_snapshot,
)
from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.secondary_storage_config import (
    SecondaryStorageConfig,
)
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.models.state_snapshot import StateSnapshot
from custom_components.hsem.planner import run_planner


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

    def __init__(self, price_interval: int, slot_interval: int) -> None:
        super().__init__()
        self.electricity_price_update_interval = price_interval
        self.recommendation_interval_minutes = slot_interval
        self.import_electricity_price_sensor = "sensor.eds_import"
        self.import_electricity_price_forecast_sensor = None
        self.export_electricity_price_sensor = "sensor.eds_export"
        self.export_electricity_price_forecast_sensor = None
        self.solcast_pv_forecast_forecast_today = None
        self.solcast_pv_forecast_forecast_tomorrow = None
        self.solcast_pv_forecast_forecast_likelihood = "pv_estimate"


def _populate(recs: list[HourlyRecommendation], attrs: dict, cfg: _Cfg) -> None:
    snapshot = StateSnapshot(
        live=LiveState(), energy_average_values={}, sensor_attributes=attrs
    )
    populate_price_and_solcast_from_snapshot(recs, snapshot, cfg)


class TestQuarterHourlyPriceMatching:
    """96 distinct 15-min prices must land on 96 distinct 15-min slots."""

    def setup_method(self) -> None:
        self.base = datetime(2026, 8, 7, 0, 0, 0, tzinfo=UTC)

    def _quarter_hour_prices(self, count: int = 96) -> list[dict[str, str]]:
        return [
            {
                "start": (self.base + timedelta(minutes=15 * i)).isoformat(),
                "price": f"{1.0 + i * 0.01:.5f}",
            }
            for i in range(count)
        ]

    def test_15min_prices_land_on_distinct_slots(self) -> None:
        """Each 15-min price point must reach exactly its own slot (issue #720)."""
        cfg = _Cfg(price_interval=15, slot_interval=15)
        recs = [
            _make_rec(
                self.base + timedelta(minutes=15 * i),
                self.base + timedelta(minutes=15 * (i + 1)),
            )
            for i in range(96)
        ]
        raw = self._quarter_hour_prices()
        attrs = {
            "sensor.eds_import": {"prices_today": raw},
            "sensor.eds_export": {"prices_today": raw},
        }
        _populate(recs, attrs, cfg)

        for i, rec in enumerate(recs):
            expected = 1.0 + i * 0.01
            assert rec.import_price == pytest.approx(expected, abs=1e-5), (
                f"Slot {i} ({rec.start}): expected {expected}, got {rec.import_price}"
            )
            assert rec.export_price == pytest.approx(expected, abs=1e-5)
            assert rec.import_price_available is True
            assert rec.export_price_available is True

    def test_fallback_folds_match_distinct_price_and_solcast_windows(self) -> None:
        copenhagen = ZoneInfo("Europe/Copenhagen")
        first_utc = datetime(2026, 10, 25, 0, 0, tzinfo=UTC)
        starts = [
            (first_utc + timedelta(minutes=15 * i)).astimezone(copenhagen)
            for i in range(8)
        ]
        recs = [
            _make_rec(
                start,
                (start.astimezone(UTC) + timedelta(minutes=15)).astimezone(copenhagen),
            )
            for start in starts
        ]
        cfg = _Cfg(price_interval=15, slot_interval=15)
        cfg.solcast_pv_forecast_forecast_today = "sensor.solcast"
        prices = [
            {"start": start.isoformat(), "price": str(1.0 + i)}
            for i, start in enumerate(starts)
        ]
        solcast = [
            {"period_start": starts[0].isoformat(), "pv_estimate": 4.0},
            {"period_start": starts[4].isoformat(), "pv_estimate": 8.0},
        ]

        _populate(
            recs,
            {
                "sensor.eds_import": {"prices_today": prices},
                "sensor.eds_export": {"prices_today": prices},
                "sensor.solcast": {"detailedForecast": solcast},
            },
            cfg,
        )

        assert [rec.start.fold for rec in recs] == [0] * 4 + [1] * 4
        assert [rec.import_price for rec in recs] == pytest.approx(
            [1.0 + i for i in range(8)]
        )
        assert [rec.solcast_pv_estimate_kwh for rec in recs[:4]] == pytest.approx(
            [1.0] * 4
        )
        assert [rec.solcast_pv_estimate_kwh for rec in recs[4:]] == pytest.approx(
            [2.0] * 4
        )
        assert all(rec.solcast_pv_estimate_available for rec in recs)

    @pytest.mark.asyncio
    async def test_async_fallback_folds_match_distinct_price_windows(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        copenhagen = ZoneInfo("Europe/Copenhagen")
        first_utc = datetime(2026, 10, 25, 0, 0, tzinfo=UTC)
        starts = [
            (first_utc + timedelta(minutes=15 * i)).astimezone(copenhagen)
            for i in range(8)
        ]
        recs = [
            _make_rec(
                start,
                (start.astimezone(UTC) + timedelta(minutes=15)).astimezone(copenhagen),
            )
            for start in starts
        ]
        sensor = MagicMock()
        sensor.hass.states.get.return_value = SimpleNamespace(
            attributes={
                "prices_today": [
                    {"start": start.isoformat(), "price": str(1.0 + i)}
                    for i, start in enumerate(starts)
                ]
            }
        )

        matched = await _async_update_hourly_field(
            sensor,
            recs,
            "sensor.eds_import",
            "import_price",
            1.0,
            "pv_estimate",
            15,
        )

        assert matched == 8
        assert [rec.import_price for rec in recs] == pytest.approx(
            [1.0 + i for i in range(8)]
        )

    def test_unpublished_solcast_is_distinct_from_genuine_zero(self) -> None:
        cfg = _Cfg(price_interval=60, slot_interval=60)
        cfg.solcast_pv_forecast_forecast_today = "sensor.solcast"
        recs = [
            _make_rec(self.base, self.base + timedelta(hours=1)),
            _make_rec(
                self.base + timedelta(hours=1),
                self.base + timedelta(hours=2),
            ),
        ]

        _populate(
            recs,
            {
                "sensor.solcast": {
                    "detailedForecast": [
                        {
                            "period_start": self.base.isoformat(),
                            "pv_estimate": 0.0,
                        }
                    ]
                }
            },
            cfg,
        )

        assert recs[0].solcast_pv_estimate_kwh == 0.0
        assert recs[0].solcast_pv_estimate_available is True
        assert recs[1].solcast_pv_estimate_kwh == 0.0
        assert recs[1].solcast_pv_estimate_available is False

    def test_unpublished_price_is_distinct_from_genuine_zero(self) -> None:
        """Availability, not numeric truthiness, identifies published zero prices."""
        cfg = _Cfg(price_interval=15, slot_interval=15)
        recs = [
            _make_rec(self.base, self.base + timedelta(minutes=15)),
            _make_rec(
                self.base + timedelta(minutes=15),
                self.base + timedelta(minutes=30),
            ),
        ]
        zero = [{"start": self.base.isoformat(), "price": "0.0"}]
        attrs = {
            "sensor.eds_import": {"prices_today": zero},
            "sensor.eds_export": {"prices_today": zero},
        }
        _populate(recs, attrs, cfg)

        assert recs[0].import_price == 0.0
        assert recs[0].import_price_available is True
        assert recs[0].export_price_available is True
        assert recs[1].import_price == 0.0
        assert recs[1].import_price_available is False
        assert recs[1].export_price_available is False

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_snapshot_price_is_unpublished(self, value: float) -> None:
        cfg = _Cfg(price_interval=15, slot_interval=15)
        recs = [_make_rec(self.base, self.base + timedelta(minutes=15))]
        bad = [{"start": self.base.isoformat(), "price": value}]
        _populate(
            recs,
            {
                "sensor.eds_import": {"prices_today": bad},
                "sensor.eds_export": {"prices_today": bad},
            },
            cfg,
        )

        assert recs[0].import_price_available is False
        assert recs[0].export_price_available is False
        assert recs[0].import_price == 0.0
        assert recs[0].export_price == 0.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    async def test_nonfinite_async_price_is_unpublished(self, value: float) -> None:
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        recs = [_make_rec(self.base, self.base + timedelta(minutes=15))]
        sensor = MagicMock()
        sensor.hass.states.get.return_value = SimpleNamespace(
            attributes={
                "prices_today": [{"start": self.base.isoformat(), "price": value}]
            }
        )

        matched = await _async_update_hourly_field(
            sensor,
            recs,
            "sensor.eds_import",
            "import_price",
            1.0,
            "pv_estimate",
            15,
        )

        assert matched == 0
        assert recs[0].import_price_available is False
        assert recs[0].import_price == 0.0

    def test_quarter_hour_prices_not_overwritten_within_hour(self) -> None:
        """Prices inside one hour must differ when the source differs.

        Mirrors the exact scenario from issue #720: 20:00-20:45 all showed
        the same price even though the source had 0.097/0.098/0.170/0.188.
        """
        cfg = _Cfg(price_interval=15, slot_interval=15)
        hour20 = self.base.replace(hour=20)
        recs = [
            _make_rec(
                hour20 + timedelta(minutes=15 * i),
                hour20 + timedelta(minutes=15 * (i + 1)),
            )
            for i in range(4)
        ]
        prices = [0.097, 0.098, 0.170, 0.188]
        raw = [
            {
                "start": (hour20 + timedelta(minutes=15 * i)).isoformat(),
                "price": f"{p:.5f}",
            }
            for i, p in enumerate(prices)
        ]
        attrs = {
            "sensor.eds_import": {"prices_today": raw},
            "sensor.eds_export": {"prices_today": raw},
        }
        _populate(recs, attrs, cfg)

        for rec, expected in zip(recs, prices, strict=True):
            assert rec.import_price == pytest.approx(expected, abs=1e-5), (
                f"{rec.start}: expected {expected}, got {rec.import_price}"
            )

    def test_hourly_prices_still_fan_out_to_15min_slots(self) -> None:
        """60-min price config must still replicate the hourly value to all
        four quarter-hour slots (existing behavior preserved)."""
        cfg = _Cfg(price_interval=60, slot_interval=15)
        recs = [
            _make_rec(
                self.base + timedelta(minutes=15 * i),
                self.base + timedelta(minutes=15 * (i + 1)),
            )
            for i in range(8)
        ]
        raw = [
            {
                "start": (self.base + timedelta(hours=h)).isoformat(),
                "price": f"{0.10 + h * 0.05:.5f}",
            }
            for h in range(2)
        ]
        attrs = {
            "sensor.eds_import": {"prices_today": raw},
            "sensor.eds_export": {"prices_today": raw},
        }
        _populate(recs, attrs, cfg)

        share = 60 / 15  # stored value is raw / share
        for i, rec in enumerate(recs):
            hour = i // 4
            expected = (0.10 + hour * 0.05) / share
            assert rec.import_price == pytest.approx(expected, abs=1e-5), (
                f"Slot {i}: expected {expected}, got {rec.import_price}"
            )

    def test_15min_prices_into_60min_slots_use_slot_start(self) -> None:
        """With 60-min slots and 15-min price data, the price at the slot
        start is used (matching the planner's hourly-equivalent resolution)."""
        cfg = _Cfg(price_interval=15, slot_interval=60)
        recs = [
            _make_rec(
                self.base + timedelta(hours=h), self.base + timedelta(hours=h + 1)
            )
            for h in range(2)
        ]
        raw = self._quarter_hour_prices(count=8)
        attrs = {
            "sensor.eds_import": {"prices_today": raw},
            "sensor.eds_export": {"prices_today": raw},
        }
        _populate(recs, attrs, cfg)

        # share = 15/60 = 0.25 → stored = raw / 0.25
        for h, rec in enumerate(recs):
            expected_raw = 1.0 + (h * 4) * 0.01  # price at the hour boundary
            expected = expected_raw / 0.25
            assert rec.import_price == pytest.approx(expected, abs=1e-4), (
                f"Hour {h}: expected {expected}, got {rec.import_price}"
            )


class TestNordpoolRawFormat:
    """Regression tests for issue #750 — nordpool raw_today/raw_tomorrow
    entries use ``start``/``end``/``value`` keys, not ``hour``/``price``.

    Bug
    ---
    ``custom-components/nordpool`` publishes ``raw_today`` and
    ``raw_tomorrow`` attributes as::

        {"start": datetime, "end": datetime, "value": price}

    HSEM's price populator mapped those attributes as ``{"k": "hour",
    "v": "price"}``, so ``data.get("hour")`` returned ``None`` for every
    entry and all prices were silently skipped.  Every planner slot ended
    up with ``import_price = 0.0`` — no error, no warning.

    Fix
    ---
    The ``raw_today`` / ``raw_tomorrow`` mapping now accepts both the
    legacy ``hour``/``price`` format and the nordpool ``start``/``value``
    format.
    """

    def setup_method(self) -> None:
        self.base = datetime(2026, 8, 11, 0, 0, 0, tzinfo=UTC)

    def _nordpool_entries(self, count: int = 96) -> list[dict[str, str]]:
        """Entries in the exact format published by custom-components/nordpool."""
        return [
            {
                "start": (self.base + timedelta(minutes=15 * i)).isoformat(),
                "end": (self.base + timedelta(minutes=15 * (i + 1))).isoformat(),
                "value": f"{0.5 + i * 0.01:.5f}",
            }
            for i in range(count)
        ]

    def test_nordpool_raw_today_prices_ingested(self) -> None:
        """Nordpool-format raw_today entries must land on the correct slots."""
        cfg = _Cfg(price_interval=15, slot_interval=15)
        recs = [
            _make_rec(
                self.base + timedelta(minutes=15 * i),
                self.base + timedelta(minutes=15 * (i + 1)),
            )
            for i in range(96)
        ]
        raw = self._nordpool_entries()
        attrs = {
            "sensor.eds_import": {"raw_today": raw},
            "sensor.eds_export": {"raw_today": raw},
        }
        _populate(recs, attrs, cfg)

        for i, rec in enumerate(recs):
            expected = 0.5 + i * 0.01
            assert rec.import_price == pytest.approx(expected, abs=1e-5), (
                f"Slot {i} ({rec.start}): expected {expected}, got {rec.import_price}"
            )
            assert rec.export_price == pytest.approx(expected, abs=1e-5)

    def test_nordpool_raw_tomorrow_prices_ingested(self) -> None:
        """Nordpool-format raw_tomorrow entries must land on the correct slots."""
        cfg = _Cfg(price_interval=15, slot_interval=15)
        recs = [
            _make_rec(
                self.base + timedelta(minutes=15 * i),
                self.base + timedelta(minutes=15 * (i + 1)),
            )
            for i in range(96)
        ]
        raw = self._nordpool_entries()
        attrs = {
            "sensor.eds_import": {"raw_tomorrow": raw},
            "sensor.eds_export": {"raw_tomorrow": raw},
        }
        _populate(recs, attrs, cfg)

        for i, rec in enumerate(recs):
            expected = 0.5 + i * 0.01
            assert rec.import_price == pytest.approx(expected, abs=1e-5), (
                f"Slot {i} ({rec.start}): expected {expected}, got {rec.import_price}"
            )

    @pytest.mark.parametrize("tomorrow_valid", [False, "false", "off", "0"])
    def test_explicitly_invalid_tomorrow_ignores_stale_nonempty_array(
        self, tomorrow_valid: bool | str
    ) -> None:
        """Retained arrays cannot override an explicit publication withdrawal."""
        cfg = _Cfg(price_interval=15, slot_interval=15)
        recs = [_make_rec(self.base, self.base + timedelta(minutes=15))]
        stale = self._nordpool_entries(count=1)
        attrs = {
            "sensor.eds_import": {
                "tomorrow_valid": tomorrow_valid,
                "raw_tomorrow": stale,
            },
            "sensor.eds_export": {
                "tomorrow_valid": tomorrow_valid,
                "raw_tomorrow": stale,
            },
        }

        _populate(recs, attrs, cfg)

        assert recs[0].import_price == 0.0
        assert recs[0].export_price == 0.0
        assert recs[0].import_price_available is False
        assert recs[0].export_price_available is False

    @pytest.mark.parametrize("tomorrow_valid", [True, "true", "on", "1"])
    def test_valid_tomorrow_preserves_published_zero_price(
        self, tomorrow_valid: bool | str
    ) -> None:
        """A valid, genuinely zero tomorrow price remains authoritative."""
        cfg = _Cfg(price_interval=15, slot_interval=15)
        recs = [_make_rec(self.base, self.base + timedelta(minutes=15))]
        zero = [{"start": self.base.isoformat(), "value": 0.0}]
        attrs = {
            "sensor.eds_import": {
                "tomorrow_valid": tomorrow_valid,
                "raw_tomorrow": zero,
            },
            "sensor.eds_export": {
                "tomorrow_valid": tomorrow_valid,
                "raw_tomorrow": zero,
            },
        }

        _populate(recs, attrs, cfg)

        assert recs[0].import_price_available is True
        assert recs[0].export_price_available is True
        assert recs[0].import_price == 0.0
        assert recs[0].export_price == 0.0

    @pytest.mark.asyncio
    async def test_async_explicitly_invalid_tomorrow_is_ignored(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        recs = [_make_rec(self.base, self.base + timedelta(minutes=15))]
        sensor = MagicMock()
        sensor.hass.states.get.return_value = SimpleNamespace(
            state="0.5",
            attributes={
                "tomorrow_valid": False,
                "raw_tomorrow": self._nordpool_entries(count=1),
            },
        )

        matched = await _async_update_hourly_field(
            sensor,
            recs,
            "sensor.eds_import",
            "import_price",
            1.0,
            "pv_estimate",
            15,
        )

        assert matched == 0
        assert recs[0].import_price_available is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("source_state", ["unknown", "unavailable"])
    async def test_async_unavailable_source_ignores_stale_attributes(
        self, source_state: str
    ) -> None:
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        recs = [_make_rec(self.base, self.base + timedelta(minutes=15))]
        sensor = MagicMock()
        sensor.hass.states.get.return_value = SimpleNamespace(
            state=source_state,
            attributes={"raw_today": self._nordpool_entries(count=1)},
        )

        matched = await _async_update_hourly_field(
            sensor,
            recs,
            "sensor.eds_import_forecast",
            "import_price",
            1.0,
            "pv_estimate",
            15,
        )

        assert matched == 0
        assert recs[0].import_price_available is False

    def test_legacy_hour_price_format_still_works(self) -> None:
        """The legacy ``hour``/``price`` format must continue to work."""
        cfg = _Cfg(price_interval=15, slot_interval=15)
        recs = [
            _make_rec(
                self.base + timedelta(minutes=15 * i),
                self.base + timedelta(minutes=15 * (i + 1)),
            )
            for i in range(4)
        ]
        raw = [
            {
                "hour": (self.base + timedelta(minutes=15 * i)).isoformat(),
                "price": f"{0.10 + i * 0.01:.5f}",
            }
            for i in range(4)
        ]
        attrs = {
            "sensor.eds_import": {"raw_today": raw},
            "sensor.eds_export": {"raw_today": raw},
        }
        _populate(recs, attrs, cfg)

        for i, rec in enumerate(recs):
            expected = 0.10 + i * 0.01
            assert rec.import_price == pytest.approx(expected, abs=1e-5), (
                f"Slot {i}: expected {expected}, got {rec.import_price}"
            )

    def test_raw_today_only_tail_blocks_all_price_driven_assets(self) -> None:
        """Raw-today data must not turn tomorrow into a zero-price market."""
        from custom_components.hsem.coordinator_builder import build_planner_input

        cfg = _Cfg(price_interval=15, slot_interval=15)
        cfg.recommendation_interval_length = 48
        recs = [
            _make_rec(
                self.base + timedelta(minutes=15 * i),
                self.base + timedelta(minutes=15 * (i + 1)),
            )
            for i in range(192)
        ]
        for rec in recs:
            rec.avg_house_consumption_kwh = 0.05
            rec.avg_house_consumption_1d_kwh = 0.05
            rec.avg_house_consumption_3d_kwh = 0.05
            rec.avg_house_consumption_7d_kwh = 0.05
            rec.avg_house_consumption_14d_kwh = 0.05

        raw_today = self._nordpool_entries()
        attrs = {
            "sensor.eds_import": {"raw_today": raw_today},
            "sensor.eds_export": {"raw_today": raw_today},
        }
        _populate(recs, attrs, cfg)
        with patch(
            "custom_components.hsem.coordinator_builder.hsem_now",
            return_value=self.base,
        ):
            inp = build_planner_input(
                cfg=cfg,
                live=LiveState(),
                hourly_recommendations=recs,
                batteries_schedules=[],
                previous_winner_name=None,
                previous_winner_score=0.0,
            )

        inp = replace(
            inp,
            battery_rated_capacity_kwh=30.0,
            battery_soc_pct=50.0,
            battery_end_of_discharge_soc_pct=5.0,
            battery_max_charge_power_w=10_000.0,
            secondary_storage=SecondaryStorageConfig(
                enabled=True,
                capacity_kwh=15.0,
                current_soc_pct=80.0,
                nominal_voltage_v=25.6,
                load_power_w=190.0,
                min_charge_current_a=10.0,
                max_charge_current_a=60.0,
                cycle_cost_per_kwh=0.05,
                replacement_price_per_kwh=2.0,
            ),
            ev_planned_load_enabled=True,
            ev_planned_load_connected=True,
            ev_planned_load_current_soc_pct=20.0,
            ev_planned_load_target_soc_pct=80.0,
            ev_planned_load_battery_capacity_kwh=63.0,
            ev_planned_load_charger_power_kw=3.7,
            ev_planned_load_deadline=self.base + timedelta(hours=30),
        )
        result = run_planner(inp)
        tomorrow = [
            slot for slot in result.slots if slot.start.date() > self.base.date()
        ]

        assert result.data_quality.tomorrow_price_missing_hours == list(range(24))
        assert result.data_quality.price_actionable_slots == 96
        assert all(not slot.price_actionable for slot in tomorrow)
        assert all(slot.recommendation == "batteries_wait_mode" for slot in tomorrow)
        assert all(slot.primary_battery_hold for slot in tomorrow)
        assert all(slot.batteries_charged_kwh == 0.0 for slot in tomorrow)
        assert all(slot.batteries_discharged_kwh == 0.0 for slot in tomorrow)
        assert all(slot.secondary_storage_mode == "utility" for slot in tomorrow)
        assert all(slot.ev_total_planned_load_kwh == 0.0 for slot in tomorrow)
        assert not any(
            slot.recommendation
            in {
                "batteries_charge_grid",
                "force_batteries_discharge",
                "force_export",
            }
            for slot in tomorrow
        )
