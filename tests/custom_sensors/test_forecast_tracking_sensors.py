"""Regressions for forecast-tracking diagnostic sensor persistence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.hsem.custom_sensors.forecast_accuracy_sensor import (
    HSEMForecastAccuracySensor,
)
from custom_components.hsem.custom_sensors.prediction_accuracy_sensor import (
    HSEMPredictionAccuracySensor,
)
from custom_components.hsem.custom_sensors.solar_confidence_sensor import (
    HSEMSolarConfidenceSensor,
)
from custom_components.hsem.entity import HSEMCoordinatorEntity
from custom_components.hsem.utils.forecast_tracker import ForecastTracker
from custom_components.hsem.utils.prediction_tracker import PredictionTracker
from custom_components.hsem.utils.solar_corrector import SolarForecastCorrector


def test_forecast_accuracy_latest_attributes_exclude_legacy_records() -> None:
    """Legacy live-rewritten records cannot appear as the latest eligible slot."""
    tracker = ForecastTracker()
    start = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    end = start + timedelta(minutes=15)
    record = tracker.get_or_create_record(start, end)
    assert tracker.set_forecasts(
        start,
        4.0,
        2.0,
        raw_pv_kwh=5.0,
        forecast_soc_pct=70.0,
        forecast_action="idle",
        observed_at=start - timedelta(minutes=5),
    )
    tracker.freeze_forecasts(start)
    record.actual_pv_kwh = 3.0
    record.actual_load_kwh = 1.5
    record.actual_coverage_seconds = 900.0
    record.finalise()

    payload = tracker.to_dict()
    legacy_start = end
    payload["records"].append(
        {
            "start": legacy_start.isoformat(),
            "end": (legacy_start + timedelta(minutes=15)).isoformat(),
            "forecast_pv_kwh": 99.0,
            "forecast_load_kwh": 99.0,
            "actual_pv_kwh": 0.0,
            "actual_load_kwh": 0.0,
            "finalised": True,
            "mae_pv": 99.0,
            "mae_load": 99.0,
            "bias_pv": 99.0,
            "bias_load": 99.0,
        }
    )
    restored = ForecastTracker()
    restored.load_from_dict(payload)

    sensor = object.__new__(HSEMForecastAccuracySensor)
    sensor.coordinator = SimpleNamespace(  # type: ignore[assignment]
        data=object(), _forecast_tracker=restored
    )

    attrs = sensor.extra_state_attributes
    assert attrs is not None
    assert attrs["finalised_slots"] == 1
    assert attrs["latest_pv_forecast_kwh"] == pytest.approx(4.0)
    assert attrs["latest_pv_actual_kwh"] == pytest.approx(3.0)


def test_prediction_accuracy_drops_restored_scalar_after_live_data() -> None:
    """The v7.1.1 scalar is startup-only and cannot outlive fresh coordinator data."""
    sensor = object.__new__(HSEMPredictionAccuracySensor)
    sensor._restored_state = "12.5"
    coordinator = SimpleNamespace(
        data=None,
        _prediction_tracker=PredictionTracker(),
    )
    sensor.coordinator = coordinator  # type: ignore[assignment]

    assert sensor.native_value == pytest.approx(12.5)

    coordinator.data = object()
    assert sensor.native_value is None


@pytest.mark.asyncio
async def test_forecast_restore_excludes_every_unfinalised_start() -> None:
    """Sensor restore wires unfinished lifecycle records into the SoC exclusion."""
    start = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    source = ForecastTracker()
    record = source.get_or_create_record(start, start + timedelta(minutes=15))
    assert source.set_forecasts(
        start,
        1.0,
        1.0,
        raw_pv_kwh=1.0,
        forecast_soc_pct=75.0,
        forecast_action="idle",
        observed_at=start - timedelta(minutes=1),
    )
    source.freeze_forecasts(start)
    record.actual_coverage_seconds = 899.0
    tracker = ForecastTracker()
    coordinator = SimpleNamespace(
        _forecast_tracker=tracker,
        _prediction_restore_excluded=set(),
    )
    sensor = object.__new__(HSEMForecastAccuracySensor)
    sensor._restored_state = None
    sensor.coordinator = coordinator  # type: ignore[assignment]
    restored = MagicMock()
    restored.state = "0.2"
    restored.attributes = {"_forecast_tracker_data": source.to_dict()}

    with (
        patch.object(
            HSEMCoordinatorEntity,
            "async_added_to_hass",
            new=AsyncMock(),
        ),
        patch.object(
            HSEMForecastAccuracySensor,
            "async_get_last_state",
            new=AsyncMock(return_value=restored),
        ),
    ):
        await sensor.async_added_to_hass()

    assert coordinator._prediction_restore_excluded == {start}


@pytest.mark.asyncio
async def test_solar_confidence_restore_cold_resets_legacy_learned_state() -> None:
    """HA restore discards unversioned factors learned from contaminated slots."""
    corrector = SolarForecastCorrector()
    corrector.update_hour(8, 1.0, 1.4)
    corrector.update_residual(1.0, 1.2)
    sensor = object.__new__(HSEMSolarConfidenceSensor)
    sensor._restored_state = None
    sensor.coordinator = SimpleNamespace(  # type: ignore[assignment]
        _solar_corrector=corrector
    )
    restored = MagicMock()
    restored.state = "0.4"
    restored.attributes = {
        "_solar_corrector_data": {
            "hour_factors": {"12": 0.4},
            "confidence": 0.3,
        }
    }

    with (
        patch.object(
            HSEMCoordinatorEntity,
            "async_added_to_hass",
            new=AsyncMock(),
        ),
        patch.object(
            HSEMSolarConfidenceSensor,
            "async_get_last_state",
            new=AsyncMock(return_value=restored),
        ),
    ):
        await sensor.async_added_to_hass()

    assert sensor._restored_state == "0.4"
    assert corrector.hour_factors == {}
    assert corrector._hour_history == {}
    assert corrector._recent_residuals == []
    assert corrector.confidence == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_solar_confidence_restore_resumes_v3_watermark_and_buffers() -> None:
    """HA restore exposes the exact corrector replay watermark to the coordinator."""
    source = SolarForecastCorrector()
    source.update_hour(8, 1.0, 0.8)
    source.update_residual(1.0, 0.8)
    processed = datetime(2024, 8, 20, 8, 0, tzinfo=UTC)
    source.mark_processed(processed)
    corrector = SolarForecastCorrector()
    coordinator = SimpleNamespace(_solar_corrector=corrector)
    sensor = object.__new__(HSEMSolarConfidenceSensor)
    sensor._restored_state = None
    sensor.coordinator = coordinator  # type: ignore[assignment]
    restored = MagicMock()
    restored.state = "nan"
    restored.attributes = {"_solar_corrector_data": source.to_dict()}

    with (
        patch.object(
            HSEMCoordinatorEntity,
            "async_added_to_hass",
            new=AsyncMock(),
        ),
        patch.object(
            HSEMSolarConfidenceSensor,
            "async_get_last_state",
            new=AsyncMock(return_value=restored),
        ),
    ):
        await sensor.async_added_to_hass()

    assert sensor._restored_state is None
    assert corrector.to_dict() == source.to_dict()
    assert coordinator._solar_corrector_processed_through == processed


@pytest.mark.asyncio
async def test_solar_confidence_restore_malformed_payload_is_safe() -> None:
    """A malformed HA attribute cannot abort entity addition or retain learning."""
    corrector = SolarForecastCorrector()
    corrector.update_hour(8, 1.0, 0.8)
    coordinator = SimpleNamespace(_solar_corrector=corrector)
    sensor = object.__new__(HSEMSolarConfidenceSensor)
    sensor._restored_state = None
    sensor.coordinator = coordinator  # type: ignore[assignment]
    restored = MagicMock()
    restored.state = "inf"
    restored.attributes = {"_solar_corrector_data": ["not", "a", "mapping"]}

    with (
        patch.object(
            HSEMCoordinatorEntity,
            "async_added_to_hass",
            new=AsyncMock(),
        ),
        patch.object(
            HSEMSolarConfidenceSensor,
            "async_get_last_state",
            new=AsyncMock(return_value=restored),
        ),
    ):
        await sensor.async_added_to_hass()

    assert sensor._restored_state is None
    assert corrector.hour_factors == {}
    assert coordinator._solar_corrector_processed_through is None
