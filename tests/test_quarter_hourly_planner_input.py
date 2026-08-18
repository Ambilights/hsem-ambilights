"""Regression tests for issue #720 (stage 2) — planner input must not
collapse quarter-hourly prices to hourly.

Bug
---
``coordinator_builder.build_planner_input`` deduplicated recommendation
slots on ``(day_offset, hour)`` and appended price points *inside* that
guard, so only the first quarter of each hour survived — 192 correct
quarter-hourly slots were reduced to 48 hourly price points.
``planner.slot_population.populate_prices`` then fanned the survivor back
across the hour via ``align_hourly_prices``, so the MILP saw one flat
price per hour even when Nord Pool 15-min MTU data provided 96 distinct
prices per day.

Fix (three parts)
-----------------
1. ``PricePoint`` gained an optional ``slot_in_day`` field (hour-granular
   callers unaffected).
2. ``build_planner_input`` appends price and Solcast points per slot (outside
   the hourly consumption dedup guard) and sets ``slot_in_day``.
3. ``populate_prices`` keys by ``(day_offset, slot_in_day)`` when points
   carry it. Only explicitly hour-granular points fan out; a missing quarter
   must not borrow an adjacent quarter price.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.price_point import PricePoint
from custom_components.hsem.models.time_series import TimeSeriesIndex
from custom_components.hsem.planner.slot_population import (
    populate_prices,
    populate_solcast,
)


def _tsi(now: datetime, interval: int = 15, hours: int = 48) -> TimeSeriesIndex:
    return TimeSeriesIndex.from_now(now, interval_minutes=interval, horizon_hours=hours)


def _slots_from_tsi(tsi: TimeSeriesIndex) -> list[PlannedSlot]:
    return [PlannedSlot(start=m.start, end=m.end) for m in tsi]


class TestSlotInDayPricePoints:
    """96 distinct 15-min prices must reach 96 distinct planner slots."""

    def setup_method(self) -> None:
        self.now = datetime(2026, 8, 9, 12, 0, 0, tzinfo=UTC)

    def _make_points(self, days: int = 2) -> list[PricePoint]:
        points = []
        for d in range(days):
            for i in range(96):
                hour = i // 4
                price = 1.0 + d * 100 + i * 0.01  # unique per slot per day
                points.append(
                    PricePoint(
                        hour=hour,
                        import_price=price,
                        export_price=price - 0.1,
                        day_offset=d,
                        slot_in_day=i,
                    )
                )
        return points

    def test_quarter_hourly_prices_land_on_distinct_slots(self) -> None:
        tsi = _tsi(self.now)
        slots = _slots_from_tsi(tsi)
        points = self._make_points()
        populate_prices(slots, points, tsi=tsi)

        for d in range(2):
            for i in range(96):
                slot = slots[d * 96 + i]
                expected = 1.0 + d * 100 + i * 0.01
                assert slot.price.import_price == pytest.approx(expected), (
                    f"day {d} slot {i}: expected {expected}, "
                    f"got {slot.price.import_price}"
                )
                assert slot.price.export_price == pytest.approx(expected - 0.1)

    def test_intra_hour_variation_preserved(self) -> None:
        """The Nord Pool SE4 example from the issue: hour 17 has four very
        different quarter-hourly prices; all four must survive."""
        tsi = _tsi(self.now, hours=24)
        slots = _slots_from_tsi(tsi)
        points = [
            PricePoint(
                hour=17,
                import_price=p,
                export_price=p,
                day_offset=0,
                slot_in_day=17 * 4 + q,
            )
            for q, p in enumerate([0.179, 0.363, 0.476, 0.603])
        ]
        populate_prices(slots, points, tsi=tsi)

        got = [slots[17 * 4 + q].price.import_price for q in range(4)]
        assert got == pytest.approx([0.179, 0.363, 0.476, 0.603])

    def test_missing_quarter_is_not_filled_from_adjacent_point(self) -> None:
        """A missing quarter stays unavailable instead of borrowing :00."""
        tsi = _tsi(self.now, hours=24)
        slots = _slots_from_tsi(tsi)
        points = [
            PricePoint(
                hour=h,
                import_price=0.10 + h * 0.01,
                export_price=0.05,
                day_offset=0,
                slot_in_day=h * 4,  # only the :00 slot of each hour
            )
            for h in range(24)
        ]
        populate_prices(slots, points, tsi=tsi)

        for hour in range(24):
            exact = slots[hour * 4]
            assert exact.price.import_price == pytest.approx(0.10 + hour * 0.01)
            assert exact.price_actionable is True
            for quarter in range(1, 4):
                missing = slots[hour * 4 + quarter]
                assert missing.price.import_price == 0.0
                assert missing.import_price_available is False
                assert missing.export_price_available is False
                assert missing.price_actionable is False

    def test_explicit_unavailable_zero_is_missing_but_published_zero_is_actionable(
        self,
    ) -> None:
        """Availability preserves the semantic difference between two zeros."""
        tsi = _tsi(self.now, hours=24)
        slots = _slots_from_tsi(tsi)
        points = [
            PricePoint(
                hour=0,
                import_price=0.0,
                export_price=0.0,
                day_offset=0,
                slot_in_day=0,
            ),
            PricePoint(
                hour=0,
                import_price=0.0,
                export_price=0.0,
                day_offset=0,
                slot_in_day=1,
                import_price_available=False,
                export_price_available=False,
            ),
        ]
        populate_prices(slots, points, tsi=tsi)

        assert slots[0].price_actionable is True
        assert slots[1].price_actionable is False
        assert tsi.slots[1].key in tsi.missing_price_slots

    def test_legacy_hourly_points_unchanged(self) -> None:
        """Points without slot_in_day use the existing hourly path."""
        tsi = _tsi(self.now, hours=24)
        slots = _slots_from_tsi(tsi)
        points = [
            PricePoint(hour=h, import_price=0.20 + h * 0.01, export_price=0.1)
            for h in range(24)
        ]
        populate_prices(slots, points, tsi=tsi)

        for i, slot in enumerate(slots):
            hour = i // 4
            assert slot.price.import_price == pytest.approx(0.20 + hour * 0.01)


class TestBuildPlannerInputSlotInDay:
    """coordinator_builder must emit one price point per recommendation slot."""

    def test_price_point_count_matches_slots(self) -> None:
        """48 h horizon at 15-min slots → 192 price points, not 48."""
        from custom_components.hsem.coordinator_builder import build_planner_input
        from custom_components.hsem.models.hourly_recommendation import (
            HourlyRecommendation,
        )
        from custom_components.hsem.models.live_state import LiveState
        from custom_components.hsem.models.sensor_config import SensorConfig

        cfg = SensorConfig()
        cfg.recommendation_interval_minutes = 15
        cfg.recommendation_interval_length = 48
        cfg.electricity_price_update_interval = 15

        base = datetime(2026, 8, 9, 0, 0, 0, tzinfo=UTC)

        def _rec(i: int) -> HourlyRecommendation:
            start = base + timedelta(minutes=15 * i)
            return HourlyRecommendation(
                start=start,
                end=start + timedelta(minutes=15),
                recommendation="idle",
                avg_house_consumption_kwh=0.1,
                avg_house_consumption_1d_kwh=0.1,
                avg_house_consumption_3d_kwh=0.1,
                avg_house_consumption_7d_kwh=0.1,
                avg_house_consumption_14d_kwh=0.1,
                batteries_charged_kwh=0.0,
                batteries_discharged_kwh=0.0,
                estimated_battery_capacity_kwh=0.0,
                estimated_battery_soc_pct=0.0,
                estimated_cost_currency=0.0,
                estimated_net_consumption_kwh=0.0,
                export_price=round(0.05 + i * 0.001, 5),
                grid_export_kwh=0.0,
                grid_import_kwh=0.0,
                import_price=round(0.10 + i * 0.001, 5),
                solcast_pv_estimate_kwh=0.0,
            )

        recs = [_rec(i) for i in range(192)]

        # build_planner_input uses the imported hsem_now alias. Pin it to the
        # recommendation window so this remains deterministic wall-clock time.
        with patch(
            "custom_components.hsem.coordinator_builder.hsem_now",
            return_value=base,
        ):
            inp = build_planner_input(
                cfg=cfg,
                live=LiveState(),
                hourly_recommendations=recs,
                previous_winner_name=None,
                previous_winner_score=0.0,
            )

        assert len(inp.price_points) == 192
        assert len(inp.consumption_averages) == 48  # still hour-deduplicated
        assert len(inp.solcast_slots) == 192

        # Distinct quarter-hourly prices must survive with distinct slot_in_day.
        slot_keys = {(pp.day_offset, pp.slot_in_day) for pp in inp.price_points}
        assert len(slot_keys) == 192

        # First hour of day 0: four distinct prices, none collapsed.
        first_hour = [pp for pp in inp.price_points if pp.day_offset == 0][:4]
        prices = [pp.import_price for pp in first_hour]
        assert len(set(prices)) == 4

    def test_fallback_hour_keeps_both_folds_and_timezone_name(self) -> None:
        from custom_components.hsem.coordinator_builder import build_planner_input
        from custom_components.hsem.models.hourly_recommendation import (
            HourlyRecommendation,
        )
        from custom_components.hsem.models.live_state import LiveState
        from custom_components.hsem.models.sensor_config import SensorConfig

        stockholm = ZoneInfo("Europe/Stockholm")
        midnight = datetime(2026, 10, 25, 0, 0, tzinfo=stockholm)
        midnight_utc = midnight.astimezone(UTC)
        cfg = SensorConfig()
        cfg.recommendation_interval_minutes = 15
        cfg.recommendation_interval_length = 5
        cfg.electricity_price_update_interval = 15

        recs = []
        for i in range(20):
            start = (midnight_utc + timedelta(minutes=15 * i)).astimezone(stockholm)
            end = (midnight_utc + timedelta(minutes=15 * (i + 1))).astimezone(stockholm)
            recs.append(
                HourlyRecommendation(
                    start=start,
                    end=end,
                    recommendation="idle",
                    avg_house_consumption_kwh=0.1,
                    avg_house_consumption_1d_kwh=0.1,
                    avg_house_consumption_3d_kwh=0.1,
                    avg_house_consumption_7d_kwh=0.1,
                    avg_house_consumption_14d_kwh=0.1,
                    batteries_charged_kwh=0.0,
                    batteries_discharged_kwh=0.0,
                    estimated_battery_capacity_kwh=0.0,
                    estimated_battery_soc_pct=0.0,
                    estimated_cost_currency=0.0,
                    estimated_net_consumption_kwh=0.0,
                    export_price=round(0.05 + i * 0.001, 5),
                    grid_export_kwh=0.0,
                    grid_import_kwh=0.0,
                    import_price=round(0.10 + i * 0.001, 5),
                    solcast_pv_estimate_kwh=round(0.20 + i * 0.01, 3),
                    solcast_pv_estimate_available=True,
                )
            )

        with patch(
            "custom_components.hsem.coordinator_builder.hsem_now",
            return_value=midnight,
        ):
            inp = build_planner_input(
                cfg=cfg,
                live=LiveState(),
                hourly_recommendations=recs,
                previous_winner_name=None,
                previous_winner_score=0.0,
            )

        assert inp.timezone_name == "Europe/Stockholm"
        assert len(inp.price_points) == 20
        assert [point.slot_in_day for point in inp.price_points] == list(range(20))
        assert (
            len({(point.day_offset, point.slot_in_day) for point in inp.price_points})
            == 20
        )
        repeated_points = [point for point in inp.price_points if point.hour == 2]
        assert len(repeated_points) == 8
        assert [point.slot_in_day for point in repeated_points] == list(range(8, 16))
        assert len({point.import_price for point in repeated_points}) == 8

        repeated_solcast = [point for point in inp.solcast_slots if point.hour == 2]
        assert len(repeated_solcast) == 8
        assert [point.slot_in_day for point in repeated_solcast] == list(range(8, 16))

        tsi = TimeSeriesIndex.from_now(midnight, interval_minutes=15, horizon_hours=5)
        planned_slots = _slots_from_tsi(tsi)
        populate_solcast(planned_slots, inp.solcast_slots, 15, tsi=tsi)
        repeated_planned = [slot for slot in planned_slots if slot.start.hour == 2]
        assert [slot.start.fold for slot in repeated_planned] == [0] * 4 + [1] * 4
        assert [slot.solcast_pv_estimate_kwh for slot in repeated_planned] == [
            rec.solcast_pv_estimate_kwh for rec in recs[8:16]
        ]

    def test_solcast_availability_survives_builder_and_population(self) -> None:
        from custom_components.hsem.coordinator_builder import build_planner_input
        from custom_components.hsem.models.hourly_recommendation import (
            HourlyRecommendation,
        )
        from custom_components.hsem.models.live_state import LiveState
        from custom_components.hsem.models.sensor_config import SensorConfig

        base = datetime(2026, 8, 9, 0, 0, tzinfo=UTC)
        cfg = SensorConfig()
        cfg.recommendation_interval_minutes = 15
        cfg.recommendation_interval_length = 2
        cfg.electricity_price_update_interval = 15
        recs = []
        for i in range(8):
            start = base + timedelta(minutes=15 * i)
            recs.append(
                HourlyRecommendation(
                    start=start,
                    end=start + timedelta(minutes=15),
                    recommendation="idle",
                    avg_house_consumption_kwh=0.1,
                    avg_house_consumption_1d_kwh=0.1,
                    avg_house_consumption_3d_kwh=0.1,
                    avg_house_consumption_7d_kwh=0.1,
                    avg_house_consumption_14d_kwh=0.1,
                    batteries_charged_kwh=0.0,
                    batteries_discharged_kwh=0.0,
                    estimated_battery_capacity_kwh=0.0,
                    estimated_battery_soc_pct=0.0,
                    estimated_cost_currency=0.0,
                    estimated_net_consumption_kwh=0.0,
                    export_price=0.1,
                    grid_export_kwh=0.0,
                    grid_import_kwh=0.0,
                    import_price=0.2,
                    solcast_pv_estimate_kwh=0.0,
                    solcast_pv_estimate_available=i < 4,
                )
            )

        with patch(
            "custom_components.hsem.coordinator_builder.hsem_now",
            return_value=base,
        ):
            inp = build_planner_input(
                cfg=cfg,
                live=LiveState(),
                hourly_recommendations=recs,
                previous_winner_name=None,
                previous_winner_score=0.0,
            )

        tsi = TimeSeriesIndex.from_now(base, interval_minutes=15, horizon_hours=2)
        planned_slots = _slots_from_tsi(tsi)
        populate_solcast(planned_slots, inp.solcast_slots, 15, tsi=tsi)

        assert [slot.solcast_pv_estimate_kwh for slot in planned_slots] == [0.0] * 8
        assert tsi.missing_future_day_pv_hours(0) == {1}
        assert {key.slot_in_day for key in tsi.missing_pv_slots} == {4, 5, 6, 7}
