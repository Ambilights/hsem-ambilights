"""Tests for the SavingsTracker model (issue #604)."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from custom_components.hsem.coordinator import HSEMDataUpdateCoordinator
from custom_components.hsem.models.daily_plan_vs_actual_tracker import (
    DailyPlanVsActualTracker,
)
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.planner_output import PlannerOutput
from custom_components.hsem.models.savings_day import SavingsDay
from custom_components.hsem.models.savings_tracker import SavingsTracker
from custom_components.hsem.utils.prices import SlotPrice
from custom_components.hsem.utils.recommendations import Recommendations


class TestSavingsDay:
    """Tests for the SavingsDay dataclass."""

    def test_defaults(self) -> None:
        """Default values should be zero."""
        day = SavingsDay(date="2026-06-26")
        assert day.date == "2026-06-26"
        assert day.actual_savings == 0.0
        assert day.missed_savings == 0.0
        assert day.baseline_cost == 0.0

    def test_as_dict(self) -> None:
        """as_dict should round to 4 decimal places."""
        day = SavingsDay(
            date="2026-06-26",
            actual_savings=1.23456,
            missed_savings=0.0,
            baseline_cost=5.0,
        )
        result = day.as_dict()
        assert result["date"] == "2026-06-26"
        assert result["actual_savings"] == 1.2346  # rounded to 4
        assert result["missed_savings"] == 0.0
        assert result["baseline_cost"] == 5.0

    def test_from_dict(self) -> None:
        """from_dict should reconstruct a SavingsDay."""
        data = {
            "date": "2026-06-26",
            "actual_savings": 3.5,
            "missed_savings": 1.2,
            "baseline_cost": 10.0,
        }
        day = SavingsDay.from_dict(data)
        assert day.date == "2026-06-26"
        assert day.actual_savings == 3.5
        assert day.missed_savings == 1.2
        assert day.baseline_cost == 10.0

    def test_from_dict_missing_keys(self) -> None:
        """from_dict should handle missing keys gracefully."""
        day = SavingsDay.from_dict({})
        assert day.date == ""
        assert day.actual_savings == 0.0
        assert day.missed_savings == 0.0
        assert day.baseline_cost == 0.0


class TestSavingsTracker:
    """Tests for the SavingsTracker."""

    def test_defaults(self) -> None:
        """Default values should start at zero."""
        st = SavingsTracker()
        assert st._today == ""
        assert st.actual_savings == 0.0
        assert st.missed_savings == 0.0
        assert st.baseline_cost == 0.0
        assert st.today_actual == 0.0
        assert st.today_missed == 0.0
        assert st.today_baseline == 0.0

    def test_charge_sample_uses_prior_power_and_boundary_overlap(self) -> None:
        """A boundary interval uses prior power and only its cheap-slot share."""
        zone = ZoneInfo("Europe/Stockholm")
        interval_start = datetime(2026, 8, 20, 10, 14, 55, tzinfo=zone)
        interval_end = datetime(2026, 8, 20, 10, 15, 5, tzinfo=zone)
        st = SavingsTracker(_today="2026-08-20")

        assert (
            st.sample_charge_interval(
                interval_start,
                3600.0,
                max_gap_seconds=120.0,
            )
            is None
        )
        sample = st.sample_charge_interval(
            interval_end,
            0.0,
            max_gap_seconds=120.0,
        )
        assert sample is not None
        assert sample[2] == pytest.approx(0.01)

        cheap_slot = PlannedSlot(
            start=datetime(2026, 8, 20, 10, 0, tzinfo=zone),
            end=datetime(2026, 8, 20, 10, 15, tzinfo=zone),
            price=SlotPrice(1.0, 0.0),
            recommendation=Recommendations.BatteriesChargeGrid.value,
            batteries_charged_kwh=8.0,
        )
        expensive_slot = PlannedSlot(
            start=datetime(2026, 8, 20, 10, 15, tzinfo=zone),
            end=datetime(2026, 8, 20, 10, 30, tzinfo=zone),
            price=SlotPrice(3.0, 0.0),
            recommendation=Recommendations.BatteriesWaitMode.value,
        )
        output = PlannerOutput(slots=[cheap_slot, expensive_slot])

        savings = HSEMDataUpdateCoordinator._compute_actual_charge_savings(
            output,
            sample,
        )

        # Only 5 seconds (0.005 kWh) overlap the cheap charge slot. The
        # planner's full 8 kWh charge value must not be credited every cycle.
        assert savings["2026-08-20"] == pytest.approx(0.005)

        repeated = st.sample_charge_interval(
            interval_end,
            0.0,
            max_gap_seconds=120.0,
        )
        assert repeated is None
        assert (
            HSEMDataUpdateCoordinator._compute_actual_charge_savings(
                output,
                repeated,
            )
            == {}
        )

    @pytest.mark.parametrize(
        ("prior_minutes", "current_minutes"),
        [(15, 60), (60, 15)],
    )
    def test_prior_layout_masks_current_overlap_after_cadence_change(
        self,
        prior_minutes: int,
        current_minutes: int,
    ) -> None:
        """An elapsed interval is valued once by its prior plan authority."""
        zone = ZoneInfo("Europe/Stockholm")
        start = datetime(2026, 8, 20, 10, 0, tzinfo=zone)
        prior_slots = (
            PlannedSlot(
                start=start - timedelta(minutes=prior_minutes),
                end=start,
                price=SlotPrice(3.0, 0.0),
                recommendation=Recommendations.BatteriesWaitMode.value,
                price_actionable=False,
            ),
            PlannedSlot(
                start=start,
                end=start + timedelta(minutes=prior_minutes),
                price=SlotPrice(1.0, 0.0),
                recommendation=Recommendations.BatteriesChargeGrid.value,
            ),
        )
        current_output = PlannerOutput(
            slots=[
                PlannedSlot(
                    start=start - timedelta(minutes=current_minutes),
                    end=start,
                    price=SlotPrice(9.0, 0.0),
                    recommendation=Recommendations.BatteriesWaitMode.value,
                    price_actionable=False,
                ),
                PlannedSlot(
                    start=start,
                    end=start + timedelta(minutes=current_minutes),
                    price=SlotPrice(0.0, 0.0),
                    recommendation=Recommendations.BatteriesChargeGrid.value,
                ),
            ]
        )

        savings = HSEMDataUpdateCoordinator._compute_actual_charge_savings(
            current_output,
            (start, start + timedelta(minutes=5), 0.5),
            prior_slots=prior_slots,
        )

        assert savings == {"2026-08-20": pytest.approx(0.5)}

    def test_current_layout_fills_only_uncovered_prior_tail(self) -> None:
        """Current slots value only sample time absent from the prior horizon."""
        zone = ZoneInfo("Europe/Stockholm")
        boundary = datetime(2026, 8, 20, 10, 0, tzinfo=zone)
        prior_slots = (
            PlannedSlot(
                start=boundary - timedelta(minutes=30),
                end=boundary - timedelta(minutes=15),
                price=SlotPrice(3.0, 0.0),
                recommendation=Recommendations.BatteriesWaitMode.value,
                price_actionable=False,
            ),
            PlannedSlot(
                start=boundary - timedelta(minutes=15),
                end=boundary,
                price=SlotPrice(1.0, 0.0),
                recommendation=Recommendations.BatteriesChargeGrid.value,
            ),
        )
        current_output = PlannerOutput(
            slots=[
                PlannedSlot(
                    start=boundary,
                    end=boundary + timedelta(minutes=15),
                    price=SlotPrice(1.0, 0.0),
                    recommendation=Recommendations.BatteriesChargeGrid.value,
                ),
                PlannedSlot(
                    start=boundary + timedelta(minutes=15),
                    end=boundary + timedelta(minutes=30),
                    price=SlotPrice(5.0, 0.0),
                    recommendation=Recommendations.BatteriesWaitMode.value,
                ),
            ]
        )

        savings = HSEMDataUpdateCoordinator._compute_actual_charge_savings(
            current_output,
            (
                boundary - timedelta(minutes=5),
                boundary + timedelta(minutes=5),
                1.0,
            ),
            prior_slots=prior_slots,
        )

        assert savings == {"2026-08-20": pytest.approx(1.5)}

    def test_nonactionable_prior_overlap_masks_current_replan(self) -> None:
        """Current intent cannot retroactively fill covered prior time."""
        zone = ZoneInfo("Europe/Stockholm")
        start = datetime(2026, 8, 20, 10, 0, tzinfo=zone)
        prior_slots = (
            PlannedSlot(
                start=start,
                end=start + timedelta(minutes=15),
                price=SlotPrice(1.0, 0.0),
                recommendation=Recommendations.BatteriesChargeGrid.value,
                price_actionable=False,
            ),
        )
        current_output = PlannerOutput(
            slots=[
                PlannedSlot(
                    start=start,
                    end=start + timedelta(minutes=60),
                    price=SlotPrice(0.0, 0.0),
                    recommendation=Recommendations.BatteriesChargeGrid.value,
                )
            ]
        )

        savings = HSEMDataUpdateCoordinator._compute_actual_charge_savings(
            current_output,
            (start, start + timedelta(minutes=5), 0.5),
            prior_slots=prior_slots,
        )

        assert savings == {}

    @pytest.mark.asyncio
    async def test_midnight_charge_savings_retain_prior_plan_horizon(
        self,
    ) -> None:
        """Old/new charge shares use the plans that governed each interval."""
        zone = ZoneInfo("Europe/Stockholm")
        midnight = datetime(2026, 8, 21, 0, 0, tzinfo=zone)
        old_expensive = PlannedSlot(
            start=midnight - timedelta(minutes=30),
            end=midnight - timedelta(minutes=15),
            price=SlotPrice(3.0, 0.0),
            recommendation=Recommendations.BatteriesWaitMode.value,
            price_actionable=False,
        )
        old_cheap = PlannedSlot(
            start=midnight - timedelta(minutes=15),
            end=midnight,
            price=SlotPrice(1.0, 0.0),
            recommendation=Recommendations.BatteriesChargeGrid.value,
        )
        old_output = PlannerOutput(slots=[old_expensive, old_cheap])
        new_cheap = PlannedSlot(
            start=midnight,
            end=midnight + timedelta(minutes=15),
            price=SlotPrice(1.0, 0.0),
            recommendation=Recommendations.BatteriesChargeGrid.value,
        )
        new_expensive = PlannedSlot(
            start=midnight + timedelta(minutes=15),
            end=midnight + timedelta(minutes=30),
            price=SlotPrice(3.0, 0.0),
            recommendation=Recommendations.BatteriesWaitMode.value,
        )
        new_unpublished = PlannedSlot(
            start=midnight + timedelta(minutes=30),
            end=midnight + timedelta(minutes=45),
            price=SlotPrice(100.0, 0.0),
            recommendation=Recommendations.BatteriesWaitMode.value,
            import_price_available=False,
            price_actionable=False,
        )
        new_output = PlannerOutput(slots=[new_cheap, new_expensive, new_unpublished])

        coordinator = object.__new__(HSEMDataUpdateCoordinator)
        coordinator._savings_tracker = SavingsTracker(_today="2026-08-20")
        coordinator._savings_tracker_initialized = True
        coordinator._daily_tracker = DailyPlanVsActualTracker(today="2026-08-20")
        coordinator._timer_interval = timedelta(minutes=5)
        coordinator._savings_price_slots = ()
        live = LiveState()
        live.force_working_mode_state = "auto"
        live.huawei_batteries_charge_discharge_power_w = 3600.0

        await coordinator._accumulate_savings(
            midnight - timedelta(minutes=5),
            live,
            old_output,
        )
        # The live planner object can be superseded after the cycle; the
        # accounting authority must remain the cached snapshot.
        old_cheap.recommendation = Recommendations.BatteriesWaitMode.value
        await coordinator._accumulate_savings(midnight, live, new_output)

        new_cheap.recommendation = Recommendations.BatteriesWaitMode.value
        live.huawei_batteries_charge_discharge_power_w = 0.0
        await coordinator._accumulate_savings(
            midnight + timedelta(minutes=5),
            live,
            new_output,
        )

        tracker = coordinator._savings_tracker
        assert tracker.daily["2026-08-20"].actual_savings == pytest.approx(0.3)
        assert tracker.daily["2026-08-21"].actual_savings == pytest.approx(0.3)
        assert tracker.actual_savings == pytest.approx(0.6)

    def test_charge_sample_long_gap_advances_without_stale_energy(self) -> None:
        """A valid sample pair beyond the cadence ceiling is not integrated."""
        zone = ZoneInfo("Europe/Stockholm")
        st = SavingsTracker(_today="2026-08-20")
        start = datetime(2026, 8, 20, 10, 0, tzinfo=zone)

        assert (
            st.sample_charge_interval(
                start,
                1000.0,
                max_gap_seconds=600.0,
            )
            is None
        )
        assert (
            st.sample_charge_interval(
                start + timedelta(hours=1),
                1000.0,
                max_gap_seconds=600.0,
            )
            is None
        )

        sample = st.sample_charge_interval(
            start + timedelta(hours=1, minutes=5),
            1000.0,
            max_gap_seconds=600.0,
        )
        assert sample is not None
        assert sample[0] == start + timedelta(hours=1)
        assert sample[2] == pytest.approx(1.0 / 12.0)

    def test_charge_sample_autumn_fold_uses_physical_duration(self) -> None:
        """The Stockholm autumn fold is measured on the UTC timeline."""
        zone = ZoneInfo("Europe/Stockholm")
        st = SavingsTracker(_today="2026-10-25")
        first_fold = datetime(2026, 10, 25, 2, 59, tzinfo=zone, fold=0)
        second_fold = datetime(2026, 10, 25, 2, 1, tzinfo=zone, fold=1)

        assert (
            st.sample_charge_interval(
                first_fold,
                1000.0,
                max_gap_seconds=180.0,
            )
            is None
        )
        sample = st.sample_charge_interval(
            second_fold,
            1000.0,
            max_gap_seconds=180.0,
        )

        assert sample is not None
        assert sample[2] == pytest.approx(1.0 / 30.0)

    def test_accumulate_switch_on(self) -> None:
        """When switch is on, savings accumulate as actual."""
        st = SavingsTracker()
        st.accumulate(
            export_revenue_delta=0.50,
            charge_savings_delta=0.30,
            baseline_cost_delta=2.00,
            switch_on=True,
        )
        assert st.actual_savings == pytest.approx(0.80)  # 0.50 + 0.30
        assert st.missed_savings == 0.0
        assert st.baseline_cost == pytest.approx(2.00)
        assert st.today_actual == pytest.approx(0.80)
        assert st.today_missed == 0.0
        assert st.today_baseline == pytest.approx(2.00)

    def test_accumulate_switch_off(self) -> None:
        """When switch is off, savings accumulate as missed."""
        st = SavingsTracker()
        st.accumulate(
            export_revenue_delta=0.50,
            charge_savings_delta=0.30,
            baseline_cost_delta=2.00,
            switch_on=False,
        )
        assert st.actual_savings == 0.0
        assert st.missed_savings == pytest.approx(0.80)
        assert st.baseline_cost == pytest.approx(2.00)
        assert st.today_actual == 0.0
        assert st.today_missed == pytest.approx(0.80)
        assert st.today_baseline == pytest.approx(2.00)

    def test_accumulate_multiple_cycles(self) -> None:
        """Multiple cycles should accumulate correctly."""
        st = SavingsTracker()
        # Cycle 1: switch on, export 0.50 + charge 0.30
        st.accumulate(0.50, 0.30, 1.00, True)
        # Cycle 2: switch off, export 0.20
        st.accumulate(0.20, 0.00, 0.50, False)
        # Cycle 3: switch on, export 0.10
        st.accumulate(0.10, 0.00, 0.30, True)

        assert st.actual_savings == pytest.approx(0.50 + 0.30 + 0.10)  # 0.90
        assert st.missed_savings == pytest.approx(0.20)
        assert st.baseline_cost == pytest.approx(1.00 + 0.50 + 0.30)  # 1.80

    def test_zero_savings(self) -> None:
        """Zero deltas should not affect totals."""
        st = SavingsTracker()
        st.accumulate(0.0, 0.0, 0.0, True)
        assert st.actual_savings == 0.0
        assert st.missed_savings == 0.0
        assert st.baseline_cost == 0.0

    def test_day_rollover(self) -> None:
        """Day rollover should finalise the previous day and start fresh."""
        st = SavingsTracker()
        st._today = "2026-06-25"
        st.daily["2026-06-25"] = SavingsDay(
            date="2026-06-25",
            actual_savings=5.0,
            missed_savings=0.0,
            baseline_cost=10.0,
        )

        result = st.check_day_rollover("2026-06-26")
        assert result is not None
        assert result.date == "2026-06-25"
        assert result.actual_savings == 5.0

        # New day should be created.
        assert st._today == "2026-06-26"
        assert "2026-06-26" in st.daily
        assert st.daily["2026-06-26"].actual_savings == 0.0

    def test_day_rollover_preserves_staged_new_day_entry(self) -> None:
        """Rollover must not overwrite interval data staged for the new day."""
        st = SavingsTracker(_today="2026-06-25")
        staged = SavingsDay(
            date="2026-06-26",
            actual_savings=1.25,
            baseline_cost=2.5,
        )
        st.daily["2026-06-26"] = staged

        result = st.check_day_rollover("2026-06-26")

        assert result is not None
        assert st.daily["2026-06-26"] is staged
        assert st.today_actual == pytest.approx(1.25)
        assert st.today_baseline == pytest.approx(2.5)

    def test_day_rollover_no_change(self) -> None:
        """When the day hasn't changed, rollover returns None."""
        st = SavingsTracker()
        st._today = "2026-06-26"
        result = st.check_day_rollover("2026-06-26")
        assert result is None

    def test_period_rollups(self) -> None:
        """7-day and 30-day rollups should sum correctly."""
        st = SavingsTracker()
        today = date(2026, 6, 26)
        st._today = today.isoformat()

        # Add entries for last 7 days.
        for i in range(7):
            d = (today - date.resolution * i).isoformat()
            st.daily[d] = SavingsDay(
                date=d, actual_savings=1.0, missed_savings=0.5, baseline_cost=2.0
            )

        assert st.last_7_days_actual == pytest.approx(7.0)
        assert st.last_7_days_missed == pytest.approx(3.5)

        # Add entries for 30 days.
        for i in range(7, 30):
            d = (today - date.resolution * i).isoformat()
            st.daily[d] = SavingsDay(
                date=d, actual_savings=1.0, missed_savings=0.0, baseline_cost=1.0
            )

        assert st.last_30_days_actual == pytest.approx(30.0)
        assert st.last_30_days_missed == pytest.approx(3.5)

    def test_as_dict(self) -> None:
        """as_dict should return a properly structured dictionary."""
        st = SavingsTracker()
        st._today = "2026-06-26"
        st.accumulate(0.50, 0.30, 2.00, True)

        result = st.as_dict()
        assert "today_actual" in result
        assert "today_missed" in result
        assert "today_baseline" in result
        assert "last_7_days_actual" in result
        assert "last_30_days_actual" in result
        assert "total_actual" in result
        assert "total_missed" in result
        assert "total_baseline" in result
        assert "daily" in result
        assert "max_history_days" in result
        assert "history_total_days" in result

        assert result["today_actual"] == pytest.approx(0.80, rel=1e-4)
        assert result["total_actual"] == pytest.approx(0.80, rel=1e-4)
        assert result["today_baseline"] == pytest.approx(2.00, rel=1e-4)

    def test_json_persistence_roundtrip(self, tmp_path: object) -> None:
        """JSON persistence should survive a round-trip."""
        import asyncio
        from pathlib import Path

        st = SavingsTracker()
        history_path = Path(tmp_path) / "test_savings.json"  # type: ignore[arg-type]
        st.history_file = str(history_path)

        # Accumulate some data.
        st._today = "2026-06-26"
        st.accumulate(0.50, 0.30, 2.00, True)
        st.accumulate(0.20, 0.00, 0.50, False)

        # Save.
        result = asyncio.run(st.save_history())
        assert result is True
        assert history_path.exists()

        # Load into a new tracker.
        st2 = SavingsTracker()
        st2.history_file = str(history_path)
        asyncio.run(st2.load_history())

        assert st2.actual_savings == pytest.approx(st.actual_savings)
        assert st2.missed_savings == pytest.approx(st.missed_savings)
        assert st2.baseline_cost == pytest.approx(st.baseline_cost)

        # Daily entries should be preserved.
        assert "2026-06-26" in st2.daily
        entry = st2.daily["2026-06-26"]
        assert entry.actual_savings == pytest.approx(0.80)  # 0.50 + 0.30
        assert entry.missed_savings == pytest.approx(0.20)
        assert entry.baseline_cost == pytest.approx(2.50)  # 2.00 + 0.50

    def test_get_today_entry(self) -> None:
        """get_today_entry should return today's SavingsDay."""
        st = SavingsTracker()
        st._today = "2026-06-26"
        st.accumulate(0.50, 0.30, 2.00, True)

        entry = st.get_today_entry()
        assert entry.date == "2026-06-26"
        assert entry.actual_savings == pytest.approx(0.80)

    def test_get_sorted_daily(self) -> None:
        """get_sorted_daily should return entries sorted by most recent first."""
        st = SavingsTracker()
        st.daily.clear()  # remove auto-created today entry
        st.daily["2026-06-24"] = SavingsDay(date="2026-06-24", actual_savings=1.0)
        st.daily["2026-06-26"] = SavingsDay(date="2026-06-26", actual_savings=3.0)
        st.daily["2026-06-25"] = SavingsDay(date="2026-06-25", actual_savings=2.0)

        sorted_entries = st.get_sorted_daily(3)
        assert len(sorted_entries) == 3
        assert sorted_entries[0].date == "2026-06-26"
        assert sorted_entries[1].date == "2026-06-25"
        assert sorted_entries[2].date == "2026-06-24"

    def test_prune_history(self) -> None:
        """History should be pruned to max_history_days."""
        st = SavingsTracker(max_history_days=3)
        st._today = "2026-06-26"
        # Clear auto-created entries from __post_init__.
        st.daily = {}

        for i in range(5):
            d = f"2026-06-{21 + i}"
            st.daily[d] = SavingsDay(date=d, actual_savings=float(i))

        assert len(st.daily) == 5
        st._prune_history()
        assert len(st.daily) == 3
        # Oldest entries should be removed.
        assert "2026-06-21" not in st.daily
        assert "2026-06-22" not in st.daily
        assert "2026-06-23" in st.daily
        assert "2026-06-24" in st.daily
        assert "2026-06-25" in st.daily

    def test_load_history_corrupted_file(self, tmp_path: object) -> None:
        """Loading a corrupted file should not crash."""
        import asyncio
        from pathlib import Path

        corrupted_path = Path(tmp_path) / "corrupted.json"  # type: ignore[arg-type]
        corrupted_path.write_text("not valid json")

        st = SavingsTracker()
        st.history_file = str(corrupted_path)
        asyncio.run(st.load_history())
        # Should not crash; state remains default.
        assert st.actual_savings == 0.0
