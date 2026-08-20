"""Tests for the Daily Plan-vs-Actual tracking model and sensor."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from custom_components.hsem.models.daily_diff import DailyDiff
from custom_components.hsem.models.daily_metrics import DailyMetrics
from custom_components.hsem.models.daily_plan_vs_actual_tracker import (
    ActualPriceInterval,
    DailyPlanVsActualTracker,
)
from custom_components.hsem.models.daily_record import DailyRecord
from custom_components.hsem.models.day_rollover_result import DayRolloverResult

_MAX_GAP_SECONDS = 600.0


class TestDailyMetrics:
    """Tests for :class:`DailyMetrics`."""

    def test_defaults_are_zero(self) -> None:
        """All fields default to 0.0."""
        m = DailyMetrics()
        assert m.grid_import_kwh == pytest.approx(0.0)
        assert m.grid_import_cost == pytest.approx(0.0)
        assert m.grid_export_kwh == pytest.approx(0.0)
        assert m.grid_export_rev == pytest.approx(0.0)
        assert m.battery_cycled_kwh == pytest.approx(0.0)
        assert m.pv_produced_kwh == pytest.approx(0.0)

    def test_as_dict_rounds(self) -> None:
        """as_dict returns rounded values."""
        m = DailyMetrics(
            grid_import_kwh=1.234567,
            grid_import_cost=2.345678,
        )
        d = m.as_dict()
        assert d["grid_import_kwh"] == pytest.approx(1.235)
        assert d["grid_import_cost"] == pytest.approx(2.346)

    def test_from_dict(self) -> None:
        """from_dict restores values correctly."""
        d = {"grid_import_kwh": 5.0, "grid_import_cost": 10.5}
        m = DailyMetrics.from_dict(d)
        assert m.grid_import_kwh == pytest.approx(5.0)
        assert m.grid_import_cost == pytest.approx(10.5)
        assert m.grid_export_kwh == pytest.approx(0.0)  # missing key → default

    def test_roundtrip(self) -> None:
        """Dict roundtrip preserves values."""
        original = DailyMetrics(
            grid_import_kwh=1.1,
            grid_import_cost=2.2,
            grid_export_kwh=3.3,
            grid_export_rev=4.4,
            battery_cycled_kwh=5.5,
            pv_produced_kwh=6.6,
        )
        restored = DailyMetrics.from_dict(original.as_dict())
        assert restored.grid_import_kwh == pytest.approx(1.1)
        assert restored.grid_import_cost == pytest.approx(2.2)
        assert restored.grid_export_kwh == pytest.approx(3.3)
        assert restored.grid_export_rev == pytest.approx(4.4)
        assert restored.battery_cycled_kwh == pytest.approx(5.5)
        assert restored.pv_produced_kwh == pytest.approx(6.6)


class TestDailyDiff:
    """Tests for :class:`DailyDiff`."""

    def test_roundtrip(self) -> None:
        """Dict roundtrip preserves values."""
        original = DailyDiff(
            grid_import_kwh=1.0,
            grid_import_cost=2.0,
            grid_export_kwh=-1.0,
            grid_export_rev=-2.0,
            battery_cycled_kwh=0.5,
            pv_produced_kwh=-3.0,
            net_cost=5.0,
        )
        restored = DailyDiff.from_dict(original.as_dict())
        assert restored.grid_import_kwh == pytest.approx(1.0)
        assert restored.grid_import_cost == pytest.approx(2.0)
        assert restored.grid_export_kwh == pytest.approx(-1.0)
        assert restored.grid_export_rev == pytest.approx(-2.0)
        assert restored.battery_cycled_kwh == pytest.approx(0.5)
        assert restored.pv_produced_kwh == pytest.approx(-3.0)
        assert restored.net_cost == pytest.approx(5.0)


class TestDailyRecord:
    """Tests for :class:`DailyRecord`."""

    def test_net_cost_actual(self) -> None:
        """Net cost actual = import cost - export revenue."""
        record = DailyRecord(
            date="2026-06-01",
            actual=DailyMetrics(grid_import_cost=50.0, grid_export_rev=20.0),
        )
        assert record.net_cost_actual == pytest.approx(30.0)

    def test_net_cost_plan(self) -> None:
        """Net cost plan = import cost - export revenue."""
        record = DailyRecord(
            date="2026-06-01",
            plan=DailyMetrics(grid_import_cost=40.0, grid_export_rev=25.0),
        )
        assert record.net_cost_plan == pytest.approx(15.0)

    def test_compute_diff(self) -> None:
        """compute_diff sets all diff fields correctly."""
        record = DailyRecord(
            date="2026-06-01",
            actual=DailyMetrics(
                grid_import_kwh=10.0,
                grid_import_cost=20.0,
                grid_export_kwh=5.0,
                grid_export_rev=10.0,
                battery_cycled_kwh=3.0,
                pv_produced_kwh=15.0,
            ),
            plan=DailyMetrics(
                grid_import_kwh=8.0,
                grid_import_cost=16.0,
                grid_export_kwh=6.0,
                grid_export_rev=12.0,
                battery_cycled_kwh=2.0,
                pv_produced_kwh=18.0,
            ),
        )
        record.compute_diff()
        assert record.diff.grid_import_kwh == pytest.approx(2.0)
        assert record.diff.grid_import_cost == pytest.approx(4.0)
        assert record.diff.grid_export_kwh == pytest.approx(-1.0)
        assert record.diff.grid_export_rev == pytest.approx(-2.0)
        assert record.diff.battery_cycled_kwh == pytest.approx(1.0)
        assert record.diff.pv_produced_kwh == pytest.approx(-3.0)
        # Net cost actual = 20 - 10 = 10; plan = 16 - 12 = 4; diff = 6
        assert record.diff.net_cost == pytest.approx(6.0)

    def test_as_dict_includes_diff(self) -> None:
        """as_dict includes the computed diff."""
        record = DailyRecord(
            date="2026-06-01",
            actual=DailyMetrics(grid_import_kwh=1.0),
            plan=DailyMetrics(grid_import_kwh=0.5),
        )
        d = record.as_dict()
        assert d["date"] == "2026-06-01"
        assert d["actual"]["grid_import_kwh"] == pytest.approx(1.0)
        assert d["plan"]["grid_import_kwh"] == pytest.approx(0.5)
        assert d["diff"]["grid_import_kwh"] == pytest.approx(0.5)

    def test_roundtrip(self) -> None:
        """Dict roundtrip preserves all values."""
        record = DailyRecord(
            date="2026-06-01",
            actual=DailyMetrics(
                grid_import_kwh=10.0,
                grid_import_cost=20.0,
                grid_export_kwh=5.0,
                grid_export_rev=10.0,
                battery_cycled_kwh=3.0,
                pv_produced_kwh=15.0,
            ),
            plan=DailyMetrics(
                grid_import_kwh=8.0,
                grid_import_cost=16.0,
            ),
        )
        record.compute_diff()
        restored = DailyRecord.from_dict(record.as_dict())
        assert restored.date == "2026-06-01"
        assert restored.actual.grid_import_kwh == pytest.approx(10.0)
        assert restored.actual.grid_import_cost == pytest.approx(20.0)
        assert restored.plan.grid_import_kwh == pytest.approx(8.0)
        assert restored.plan.grid_import_cost == pytest.approx(16.0)
        assert restored.diff.grid_import_kwh == pytest.approx(2.0)
        assert restored.diff.grid_import_cost == pytest.approx(4.0)


class TestDailyPlanVsActualTracker:
    """Tests for :class:`DailyPlanVsActualTracker`."""

    def test_init_waits_for_ha_local_date(self) -> None:
        """Tracker does not infer a process-local calendar date."""
        tracker = DailyPlanVsActualTracker()
        assert tracker.today == ""

    def test_accumulate_plan_adds_values(self) -> None:
        """accumulate_plan correctly sums values."""
        tracker = DailyPlanVsActualTracker()
        tracker.accumulate_plan(
            grid_import_kwh=2.0,
            grid_export_kwh=1.0,
            cycle_kwh=0.5,
            pv_kwh=3.0,
            import_price=0.5,
            export_price=0.3,
        )
        assert tracker.plan.grid_import_kwh == pytest.approx(2.0)
        assert tracker.plan.grid_import_cost == pytest.approx(1.0)  # 2 * 0.5
        assert tracker.plan.grid_export_kwh == pytest.approx(1.0)
        assert tracker.plan.grid_export_rev == pytest.approx(0.3)  # 1 * 0.3
        assert tracker.plan.battery_cycled_kwh == pytest.approx(0.5)
        assert tracker.plan.pv_produced_kwh == pytest.approx(3.0)

    def test_accumulate_plan_multiple_calls(self) -> None:
        """Multiple accumulate_plan calls sum correctly."""
        tracker = DailyPlanVsActualTracker()
        tracker.accumulate_plan(grid_import_kwh=1.0, import_price=0.5)
        tracker.accumulate_plan(grid_import_kwh=2.0, import_price=1.0)
        assert tracker.plan.grid_import_kwh == pytest.approx(3.0)
        assert tracker.plan.grid_import_cost == pytest.approx(2.5)  # 0.5 + 2.0

    def test_unavailable_prices_keep_energy_but_not_fabricated_money(self) -> None:
        """Meter baselines advance while unpublished intervals stay unpriced."""
        zone = ZoneInfo("Europe/Stockholm")
        start = datetime(2026, 8, 20, 10, 0, tzinfo=zone)
        tracker = DailyPlanVsActualTracker()
        tracker.accumulate_actual(
            grid_import_energy_kwh=100.0,
            grid_export_energy_kwh=50.0,
            import_price_available=False,
            export_price_available=False,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start,
        )
        tracker.accumulate_actual(
            grid_import_energy_kwh=105.0,
            grid_export_energy_kwh=52.0,
            import_price=3.0,
            export_price=2.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start + timedelta(minutes=5),
        )
        tracker.accumulate_actual(
            grid_import_energy_kwh=106.0,
            grid_export_energy_kwh=53.0,
            import_price=3.0,
            export_price=2.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start + timedelta(minutes=10),
        )
        tracker.accumulate_plan(
            grid_import_kwh=4.0,
            grid_export_kwh=2.0,
            import_price=999.0,
            export_price=999.0,
            import_price_available=False,
            export_price_available=False,
        )

        assert tracker.actual.grid_import_kwh == pytest.approx(6.0)
        assert tracker.actual.grid_export_kwh == pytest.approx(3.0)
        assert tracker.actual.grid_import_cost == pytest.approx(3.0)
        assert tracker.actual.grid_export_rev == pytest.approx(2.0)
        assert tracker.plan.grid_import_kwh == pytest.approx(4.0)
        assert tracker.plan.grid_export_kwh == pytest.approx(2.0)
        assert tracker.plan.grid_import_cost == 0.0
        assert tracker.plan.grid_export_rev == 0.0

    def test_meter_delta_is_split_across_two_price_slots(self) -> None:
        """A two-minute, 2 kWh delta prices one minute in each slot."""
        zone = ZoneInfo("Europe/Stockholm")
        tracker = DailyPlanVsActualTracker(today="2026-08-20")
        tracker.accumulate_actual(
            grid_import_energy_kwh=100.0,
            import_price=1.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=datetime(2026, 8, 20, 12, 14, tzinfo=zone),
        )
        intervals = (
            ActualPriceInterval(
                datetime(2026, 8, 20, 12, 0, tzinfo=zone),
                datetime(2026, 8, 20, 12, 15, tzinfo=zone),
                1.0,
                0.0,
            ),
            ActualPriceInterval(
                datetime(2026, 8, 20, 12, 15, tzinfo=zone),
                datetime(2026, 8, 20, 12, 30, tzinfo=zone),
                3.0,
                0.0,
            ),
        )

        tracker.accumulate_actual(
            grid_import_energy_kwh=102.0,
            import_price=3.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=datetime(2026, 8, 20, 12, 16, tzinfo=zone),
            price_intervals=intervals,
        )

        assert tracker.actual.grid_import_kwh == pytest.approx(2.0)
        assert tracker.actual.grid_import_cost == pytest.approx(4.0)

    @pytest.mark.asyncio
    async def test_cross_midnight_delta_uses_prior_and_new_prices(self) -> None:
        """Uncovered old-day energy keeps the prior sampled price authority."""
        zone = ZoneInfo("Europe/Stockholm")
        tracker = DailyPlanVsActualTracker(today="2026-08-20")
        tracker.accumulate_actual(
            grid_import_energy_kwh=100.0,
            import_price=1.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=datetime(2026, 8, 20, 23, 55, tzinfo=zone),
        )
        new_day_intervals = (
            ActualPriceInterval(
                datetime(2026, 8, 21, 0, 0, tzinfo=zone),
                datetime(2026, 8, 21, 0, 15, tzinfo=zone),
                3.0,
                0.0,
            ),
        )

        tracker.accumulate_actual(
            grid_import_energy_kwh=110.0,
            import_price=3.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=datetime(2026, 8, 21, 0, 5, tzinfo=zone),
            price_intervals=new_day_intervals,
        )
        rollover = await tracker.check_day_rollover(
            datetime(2026, 8, 21, 0, 5, tzinfo=zone)
        )

        assert rollover is not None
        assert rollover.record.actual.grid_import_kwh == pytest.approx(5.0)
        assert rollover.record.actual.grid_import_cost == pytest.approx(5.0)
        assert tracker.actual.grid_import_kwh == pytest.approx(5.0)
        assert tracker.actual.grid_import_cost == pytest.approx(15.0)

    @pytest.mark.asyncio
    async def test_midnight_meter_reset_keeps_first_new_day_energy(self) -> None:
        """A daily-meter reset seeds midnight at zero instead of losing 2 kWh."""
        zone = ZoneInfo("Europe/Stockholm")
        tracker = DailyPlanVsActualTracker(today="2026-08-20")
        tracker.accumulate_actual(
            grid_import_energy_kwh=100.0,
            import_price=1.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=datetime(2026, 8, 20, 23, 55, tzinfo=zone),
        )
        new_day_intervals = (
            ActualPriceInterval(
                datetime(2026, 8, 21, 0, 0, tzinfo=zone),
                datetime(2026, 8, 21, 0, 15, tzinfo=zone),
                3.0,
                0.0,
            ),
        )

        tracker.accumulate_actual(
            grid_import_energy_kwh=2.0,
            import_price=3.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=datetime(2026, 8, 21, 0, 5, tzinfo=zone),
            price_intervals=new_day_intervals,
        )
        rollover = await tracker.check_day_rollover(
            datetime(2026, 8, 21, 0, 5, tzinfo=zone)
        )

        assert rollover is not None
        assert rollover.record.actual.grid_import_kwh == pytest.approx(0.0)
        assert tracker.actual.grid_import_kwh == pytest.approx(2.0)
        assert tracker.actual.grid_import_cost == pytest.approx(6.0)
        assert tracker._last_import_energy_kwh == pytest.approx(2.0)

    def test_autumn_fold_prices_both_physical_minutes(self) -> None:
        """The repeated 02:00 hour retains distinct UTC price intervals."""
        zone = ZoneInfo("Europe/Stockholm")
        tracker = DailyPlanVsActualTracker(today="2026-10-25")
        tracker.accumulate_actual(
            grid_import_energy_kwh=100.0,
            import_price=1.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=datetime(2026, 10, 25, 2, 59, tzinfo=zone, fold=0),
        )
        fold_intervals = (
            ActualPriceInterval(
                datetime(2026, 10, 25, 2, 45, tzinfo=zone, fold=0),
                datetime(2026, 10, 25, 2, 0, tzinfo=zone, fold=1),
                1.0,
                0.0,
            ),
            ActualPriceInterval(
                datetime(2026, 10, 25, 2, 0, tzinfo=zone, fold=1),
                datetime(2026, 10, 25, 2, 15, tzinfo=zone, fold=1),
                3.0,
                0.0,
            ),
        )

        tracker.accumulate_actual(
            grid_import_energy_kwh=102.0,
            import_price=3.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=datetime(2026, 10, 25, 2, 1, tzinfo=zone, fold=1),
            price_intervals=fold_intervals,
        )

        assert tracker.actual.grid_import_kwh == pytest.approx(2.0)
        assert tracker.actual.grid_import_cost == pytest.approx(4.0)

    def test_long_gap_rejected_then_short_interval_recovers(self) -> None:
        """A stale interval is skipped and the next short sample resumes."""
        zone = ZoneInfo("Europe/Stockholm")
        start = datetime(2026, 8, 20, 10, 0, tzinfo=zone)
        tracker = DailyPlanVsActualTracker(today="2026-08-20")
        tracker.accumulate_actual(
            grid_import_energy_kwh=100.0,
            grid_export_energy_kwh=50.0,
            pv_energy_kwh=20.0,
            soc_pct=50.0,
            rated_capacity_kwh=10.0,
            import_price=1.0,
            export_price=0.5,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start,
        )

        tracker.accumulate_actual(
            grid_import_energy_kwh=110.0,
            grid_export_energy_kwh=55.0,
            pv_energy_kwh=25.0,
            soc_pct=60.0,
            rated_capacity_kwh=10.0,
            import_price=2.0,
            export_price=1.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start + timedelta(hours=1),
        )

        assert tracker.actual == DailyMetrics()

        tracker.accumulate_actual(
            grid_import_energy_kwh=111.0,
            grid_export_energy_kwh=56.0,
            pv_energy_kwh=26.0,
            soc_pct=61.0,
            rated_capacity_kwh=10.0,
            import_price=2.0,
            export_price=1.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start + timedelta(hours=1, minutes=5),
        )

        assert tracker.actual.grid_import_kwh == pytest.approx(1.0)
        assert tracker.actual.grid_import_cost == pytest.approx(2.0)
        assert tracker.actual.grid_export_kwh == pytest.approx(1.0)
        assert tracker.actual.grid_export_rev == pytest.approx(1.0)
        assert tracker.actual.pv_produced_kwh == pytest.approx(1.0)
        assert tracker.actual.battery_cycled_kwh == pytest.approx(0.1)

    def test_nonpositive_intervals_reject_but_advance_baselines(self) -> None:
        """Equal and reversed instants cannot bridge into recovery."""
        zone = ZoneInfo("Europe/Stockholm")
        start = datetime(2026, 8, 20, 10, 0, tzinfo=zone)
        tracker = DailyPlanVsActualTracker(today="2026-08-20")
        tracker.accumulate_actual(
            grid_import_energy_kwh=100.0,
            soc_pct=50.0,
            rated_capacity_kwh=10.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start,
        )
        tracker.accumulate_actual(
            grid_import_energy_kwh=101.0,
            soc_pct=51.0,
            rated_capacity_kwh=10.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start,
        )
        tracker.accumulate_actual(
            grid_import_energy_kwh=102.0,
            soc_pct=52.0,
            rated_capacity_kwh=10.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start - timedelta(minutes=5),
        )

        assert tracker.actual == DailyMetrics()

        tracker.accumulate_actual(
            grid_import_energy_kwh=103.0,
            soc_pct=53.0,
            rated_capacity_kwh=10.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start,
        )

        assert tracker.actual.grid_import_kwh == pytest.approx(1.0)
        assert tracker.actual.battery_cycled_kwh == pytest.approx(0.1)

    @pytest.mark.parametrize(
        "invalid",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "positive_inf", "negative_inf"],
    )
    def test_nonfinite_meter_sample_resets_then_recovers(self, invalid: float) -> None:
        """Invalid meter telemetry cannot poison totals or bridge recovery."""
        zone = ZoneInfo("Europe/Stockholm")
        start = datetime(2026, 8, 20, 10, 0, tzinfo=zone)
        tracker = DailyPlanVsActualTracker(today="2026-08-20")
        tracker.accumulate_actual(
            grid_import_energy_kwh=100.0,
            grid_export_energy_kwh=50.0,
            pv_energy_kwh=20.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start,
        )
        tracker.accumulate_actual(
            grid_import_energy_kwh=invalid,
            grid_export_energy_kwh=invalid,
            pv_energy_kwh=invalid,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start + timedelta(minutes=5),
        )

        assert tracker.actual == DailyMetrics()
        assert tracker._last_import_energy_kwh is None
        assert tracker._last_export_energy_kwh is None
        assert tracker._last_pv_energy_kwh is None

        tracker.accumulate_actual(
            grid_import_energy_kwh=102.0,
            grid_export_energy_kwh=52.0,
            pv_energy_kwh=22.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start + timedelta(minutes=10),
        )
        assert tracker.actual == DailyMetrics()

        tracker.accumulate_actual(
            grid_import_energy_kwh=103.0,
            grid_export_energy_kwh=53.0,
            pv_energy_kwh=23.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start + timedelta(minutes=15),
        )
        assert tracker.actual.grid_import_kwh == pytest.approx(1.0)
        assert tracker.actual.grid_export_kwh == pytest.approx(1.0)
        assert tracker.actual.pv_produced_kwh == pytest.approx(1.0)

    def test_nonfinite_prior_meter_sample_is_rejected(self) -> None:
        """Legacy invalid baselines are replaced before deltas resume."""
        zone = ZoneInfo("Europe/Stockholm")
        start = datetime(2026, 8, 20, 10, 0, tzinfo=zone)
        tracker = DailyPlanVsActualTracker(today="2026-08-20")
        tracker._last_import_energy_kwh = float("nan")
        tracker._last_export_energy_kwh = float("inf")
        tracker._last_pv_energy_kwh = float("-inf")
        tracker._last_import_sample_at = start
        tracker._last_export_sample_at = start
        tracker._last_pv_sample_at = start

        tracker.accumulate_actual(
            grid_import_energy_kwh=100.0,
            grid_export_energy_kwh=50.0,
            pv_energy_kwh=20.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start + timedelta(minutes=5),
        )
        assert tracker.actual == DailyMetrics()

        tracker.accumulate_actual(
            grid_import_energy_kwh=101.0,
            grid_export_energy_kwh=51.0,
            pv_energy_kwh=21.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start + timedelta(minutes=10),
        )
        assert tracker.actual.grid_import_kwh == pytest.approx(1.0)
        assert tracker.actual.grid_export_kwh == pytest.approx(1.0)
        assert tracker.actual.pv_produced_kwh == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "invalid",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "positive_inf", "negative_inf"],
    )
    def test_nonfinite_soc_resets_then_recovers(self, invalid: float) -> None:
        """Invalid SoC requires a fresh finite pair before cycling resumes."""
        zone = ZoneInfo("Europe/Stockholm")
        start = datetime(2026, 8, 20, 10, 0, tzinfo=zone)
        tracker = DailyPlanVsActualTracker(today="2026-08-20")
        tracker.accumulate_actual(
            soc_pct=50.0,
            rated_capacity_kwh=10.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start,
        )
        tracker.accumulate_actual(
            soc_pct=invalid,
            rated_capacity_kwh=10.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start + timedelta(minutes=5),
        )
        assert tracker.last_soc_pct is None
        assert tracker.actual.battery_cycled_kwh == 0.0

        tracker.accumulate_actual(
            soc_pct=55.0,
            rated_capacity_kwh=10.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start + timedelta(minutes=10),
        )
        assert tracker.actual.battery_cycled_kwh == 0.0

        tracker.accumulate_actual(
            soc_pct=56.0,
            rated_capacity_kwh=10.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start + timedelta(minutes=15),
        )
        assert tracker.actual.battery_cycled_kwh == pytest.approx(0.1)

    @pytest.mark.parametrize(
        "invalid_capacity",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "positive_inf", "negative_inf"],
    )
    def test_nonfinite_capacity_skips_only_its_interval(
        self,
        invalid_capacity: float,
    ) -> None:
        """Invalid capacity skips cycling while retaining the current SoC seed."""
        zone = ZoneInfo("Europe/Stockholm")
        start = datetime(2026, 8, 20, 10, 0, tzinfo=zone)
        tracker = DailyPlanVsActualTracker(today="2026-08-20")
        tracker.accumulate_actual(
            soc_pct=50.0,
            rated_capacity_kwh=10.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start,
        )
        tracker.accumulate_actual(
            soc_pct=55.0,
            rated_capacity_kwh=invalid_capacity,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start + timedelta(minutes=5),
        )
        assert tracker.actual.battery_cycled_kwh == 0.0

        tracker.accumulate_actual(
            soc_pct=56.0,
            rated_capacity_kwh=10.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start + timedelta(minutes=10),
        )
        assert tracker.actual.battery_cycled_kwh == pytest.approx(0.1)

    def test_nonfinite_prior_soc_is_rejected(self) -> None:
        """A legacy invalid SoC baseline is replaced before cycling resumes."""
        zone = ZoneInfo("Europe/Stockholm")
        start = datetime(2026, 8, 20, 10, 0, tzinfo=zone)
        tracker = DailyPlanVsActualTracker(today="2026-08-20")
        tracker.last_soc_pct = float("inf")
        tracker._last_soc_sample_at = start

        tracker.accumulate_actual(
            soc_pct=50.0,
            rated_capacity_kwh=10.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start + timedelta(minutes=5),
        )
        assert tracker.actual.battery_cycled_kwh == 0.0

        tracker.accumulate_actual(
            soc_pct=51.0,
            rated_capacity_kwh=10.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start + timedelta(minutes=10),
        )
        assert tracker.actual.battery_cycled_kwh == pytest.approx(0.1)

    def test_accumulate_actual_soc_tracking(self) -> None:
        """Battery cycle tracking uses SoC delta converted to kWh."""
        zone = ZoneInfo("Europe/Stockholm")
        start = datetime(2026, 8, 20, 10, 0, tzinfo=zone)
        tracker = DailyPlanVsActualTracker()
        tracker.accumulate_actual(
            soc_pct=50.0,
            rated_capacity_kwh=10.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start,
        )
        assert tracker.last_soc_pct == pytest.approx(50.0)
        assert tracker.actual.battery_cycled_kwh == pytest.approx(0.0)

        tracker.accumulate_actual(
            soc_pct=55.0,
            rated_capacity_kwh=10.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start + timedelta(minutes=5),
        )
        assert tracker.last_soc_pct == pytest.approx(55.0)
        assert tracker.actual.battery_cycled_kwh == pytest.approx(0.5)

    def test_accumulate_actual_soc_discharge(self) -> None:
        """SoC decrease is tracked as positive cycle kWh."""
        zone = ZoneInfo("Europe/Stockholm")
        start = datetime(2026, 8, 20, 10, 0, tzinfo=zone)
        tracker = DailyPlanVsActualTracker()
        tracker.accumulate_actual(
            soc_pct=50.0,
            rated_capacity_kwh=10.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start,
        )
        tracker.accumulate_actual(
            soc_pct=45.0,
            rated_capacity_kwh=10.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=start + timedelta(minutes=5),
        )
        assert tracker.actual.battery_cycled_kwh == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_check_day_rollover_no_change(self) -> None:
        """No rollover when day hasn't changed."""
        tracker = DailyPlanVsActualTracker(today="2026-06-01")
        result = await tracker.check_day_rollover(datetime(2026, 6, 1, 12, 0, 0))
        assert result is None

    @pytest.mark.asyncio
    async def test_check_day_rollover_changes(self) -> None:
        """Day rollover returns a result and resets counters."""
        tracker = DailyPlanVsActualTracker(today="2026-06-01")
        tracker.accumulate_plan(grid_import_kwh=5.0, import_price=1.0)
        tracker.accumulate_actual(
            soc_pct=50.0,
            max_gap_seconds=_MAX_GAP_SECONDS,
            now=datetime(2026, 6, 1, 23, 55, tzinfo=ZoneInfo("Europe/Stockholm")),
        )

        result = await tracker.check_day_rollover(datetime(2026, 6, 2, 0, 5, 0))
        assert result is not None
        assert isinstance(result, DayRolloverResult)
        assert result.record.date == "2026-06-01"

        # Counters should be reset.
        assert tracker.today == "2026-06-02"
        assert tracker.plan.grid_import_kwh == pytest.approx(0.0)
        assert tracker.actual.battery_cycled_kwh == pytest.approx(0.0)
        assert tracker.last_soc_pct == pytest.approx(50.0)

        # History should contain the saved record.
        assert len(tracker.history) == 1
        assert tracker.history[0].date == "2026-06-01"

    def test_get_today_record(self) -> None:
        """get_today_record uses the injected HA-local date."""
        tracker = DailyPlanVsActualTracker(today="2026-06-02")
        tracker.accumulate_plan(grid_import_kwh=3.0, import_price=2.0)
        record = tracker.get_today_record()
        assert record.date == "2026-06-02"
        assert record.plan.grid_import_kwh == pytest.approx(3.0)
        assert record.plan.grid_import_cost == pytest.approx(6.0)

    def test_get_yesterday_record(self) -> None:
        """get_yesterday_record uses the injected HA-local date."""
        tracker = DailyPlanVsActualTracker(today="2026-06-02")
        record = DailyRecord(
            date="2026-06-01",
            actual=DailyMetrics(grid_import_kwh=10.0),
        )
        tracker.history = [record]
        result = tracker.get_yesterday_record()
        assert result is not None
        assert result.date == "2026-06-01"

    def test_get_yesterday_record_none_when_empty(self) -> None:
        """get_yesterday_record returns None when history is empty."""
        tracker = DailyPlanVsActualTracker()
        record = tracker.get_yesterday_record()
        assert record is None

    def test_get_yesterday_record_none_when_only_today(self) -> None:
        """get_yesterday_record returns None when history only has today."""
        tracker = DailyPlanVsActualTracker(today="2026-06-02")
        tracker.history = [DailyRecord(date="2026-06-02")]
        record = tracker.get_yesterday_record()
        assert record is None

    @pytest.mark.asyncio
    async def test_history_pruning(self) -> None:
        """History is pruned to max_history_days."""
        tracker = DailyPlanVsActualTracker(max_history_days=3)
        for i in range(5):
            await tracker._save_record_to_history(
                DailyRecord(date=f"2026-06-{i + 1:02d}")
            )
        assert len(tracker.history) == 3
        # Should keep the 3 most recent (June 3, 4, 5).
        assert tracker.history[-1].date == "2026-06-05"

    def test_as_sensor_attributes(self) -> None:
        """as_sensor_attributes returns expected structure."""
        tracker = DailyPlanVsActualTracker()
        tracker.accumulate_plan(grid_import_kwh=1.0, import_price=0.5)

        attrs = tracker.as_sensor_attributes()
        assert "today" in attrs
        assert "yesterday" in attrs
        assert "history" in attrs
        assert "history_file" in attrs
        assert "history_days" in attrs
        assert "history_total_days" in attrs
        assert attrs["history_days"] == 90
        assert attrs["history_total_days"] == 0

    @pytest.mark.asyncio
    async def test_json_persistence_roundtrip(self) -> None:
        """Save and load history through a temp JSON file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create tracker, add records, save.
            tracker = DailyPlanVsActualTracker(
                history_file=tmp_path, max_history_days=90
            )
            tracker.accumulate_plan(
                grid_import_kwh=5.0,
                grid_export_kwh=2.0,
                import_price=1.0,
                export_price=0.5,
            )
            tracker.accumulate_actual(
                soc_pct=60.0,
                max_gap_seconds=_MAX_GAP_SECONDS,
                now=datetime(2026, 6, 1, 23, 55, tzinfo=ZoneInfo("Europe/Stockholm")),
            )

            # Simulate day rollover to save.
            tracker.today = "2026-06-01"
            await tracker.check_day_rollover(datetime(2026, 6, 2, 0, 5, 0))

            # Load from the file with a new tracker.
            tracker2 = DailyPlanVsActualTracker(
                history_file=tmp_path, max_history_days=90
            )
            await tracker2.load_history()
            assert len(tracker2.history) == 1
            assert tracker2.history[0].date == "2026-06-01"
            assert tracker2.history[0].plan.grid_import_kwh == pytest.approx(5.0)
            assert tracker2.history[0].plan.grid_export_kwh == pytest.approx(2.0)
            assert tracker2.history[0].plan.grid_import_cost == pytest.approx(5.0)
            assert tracker2.history[0].plan.grid_export_rev == pytest.approx(1.0)

            # Verify file is valid JSON.
            with open(tmp_path, encoding="utf-8") as f:
                data = json.load(f)
            assert "updated" in data
            assert "days" in data
            assert len(data["days"]) == 1
        finally:
            os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_corrupted_file_handling(self) -> None:
        """Tracker loads gracefully from a corrupted file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
            tmp.write("this is not valid json")
            tmp_path = tmp.name

        try:
            tracker = DailyPlanVsActualTracker(
                history_file=tmp_path, max_history_days=90
            )
            await tracker.load_history()
            # Should have loaded empty history despite corruption.
            assert tracker.history == []
        finally:
            os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_load_missing_file(self) -> None:
        """Tracker handles missing history file gracefully."""
        import tempfile as tf

        nonexistent = str(Path(tf.gettempdir()) / "nonexistent_hsem_history_test.json")
        tracker = DailyPlanVsActualTracker(
            history_file=nonexistent,
            max_history_days=90,
        )
        await tracker.load_history()
        assert tracker.history == []
