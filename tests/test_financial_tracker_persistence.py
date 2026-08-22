"""Coordinator-level regression tests for financial tracker persistence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from custom_components.hsem.coordinator import HSEMDataUpdateCoordinator
from custom_components.hsem.models.financial_tracker import FinancialTracker
from custom_components.hsem.models.live_state import LiveState


@pytest.mark.asyncio
async def test_loaded_tracker_retains_runtime_history_path(tmp_path: Path) -> None:
    """Replacing a tracker from JSON must not disable all future writes."""
    history_path = tmp_path / "hsem_financial_history.json"
    stored = FinancialTracker(import_cost_total=12.5, export_income_total=4.25)
    assert FinancialTracker._write_history_file(stored.as_dict(), history_path)

    coordinator = object.__new__(HSEMDataUpdateCoordinator)
    coordinator._financial_tracker = FinancialTracker(
        history_file=str(history_path),
    )

    await coordinator._load_financial_tracker()

    assert coordinator._financial_tracker.history_file == str(history_path)
    assert coordinator._financial_tracker.import_cost_total == pytest.approx(12.5)
    assert await coordinator._persist_financial_tracker() is True


@pytest.mark.asyncio
async def test_midnight_sample_is_priced_and_booked_before_rollover() -> None:
    """The interval ending at midnight belongs to the old local calendar day."""
    before_midnight = datetime(2026, 8, 21, 23, 55, tzinfo=UTC)
    midnight = before_midnight + timedelta(minutes=5)
    tracker = FinancialTracker(today="2026-08-21")
    tracker.accumulate(
        100.0,
        50.0,
        import_price=2.0,
        export_price=1.0,
        sample_time=before_midnight,
        max_gap_seconds=600.0,
    )

    coordinator = object.__new__(HSEMDataUpdateCoordinator)
    coordinator._financial_tracker = tracker
    coordinator._financial_tracker_initialized = True
    coordinator._timer_interval = timedelta(minutes=5)
    coordinator._persist_financial_tracker = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )
    live = LiveState()
    live.grid_import_energy_kwh = 101.0
    live.grid_export_energy_kwh = 51.0
    live.import_electricity_price = 5.0
    live.export_electricity_price = 4.0
    live.import_electricity_price_available = True
    live.export_electricity_price_available = True

    await coordinator._accumulate_financials(midnight, live)

    old_day = tracker.daily_log["2026-08-21"]
    assert old_day.import_cost == pytest.approx(2.0)
    assert old_day.export_income == pytest.approx(1.0)
    assert tracker.today == "2026-08-22"
    assert tracker.import_cost_today == pytest.approx(0.0)
    assert tracker.export_income_today == pytest.approx(0.0)

    live.grid_import_energy_kwh = 102.0
    live.grid_export_energy_kwh = 52.0
    live.import_electricity_price = 6.0
    live.export_electricity_price = 3.0
    await coordinator._accumulate_financials(midnight + timedelta(minutes=5), live)

    assert tracker.import_cost_today == pytest.approx(5.0)
    assert tracker.export_income_today == pytest.approx(4.0)
    assert coordinator._persist_financial_tracker.await_count == 2
