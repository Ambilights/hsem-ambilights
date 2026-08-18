"""Focused regressions for price-source provenance handoffs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from custom_components.hsem.coordinator import (
    HSEMDataUpdateCoordinator,
    _price_forecast_signature,
)
from custom_components.hsem.coordinator_builder import build_planner_input
from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.planner_output import PlannerOutput
from custom_components.hsem.models.price_point import PricePoint
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.models.time_series import TimeSeriesIndex
from custom_components.hsem.planner.slot_population import populate_prices
from custom_components.hsem.utils.diagnostics import _slot_to_dict


def _recommendation(start: datetime) -> HourlyRecommendation:
    return HourlyRecommendation(
        start=start,
        end=start + timedelta(minutes=15),
        recommendation=None,
        avg_house_consumption_kwh=0.1,
        avg_house_consumption_1d_kwh=0.1,
        avg_house_consumption_3d_kwh=0.1,
        avg_house_consumption_7d_kwh=0.1,
        avg_house_consumption_14d_kwh=0.1,
        batteries_charged_kwh=0.0,
        batteries_discharged_kwh=0.0,
        estimated_battery_capacity_kwh=5.0,
        estimated_battery_soc_pct=50.0,
        estimated_cost_currency=0.0,
        estimated_net_consumption_kwh=0.1,
        export_price=0.1,
        grid_export_kwh=0.0,
        grid_import_kwh=0.0,
        import_price=0.2,
        solcast_pv_estimate_kwh=0.0,
        import_price_available=True,
        export_price_available=True,
        price_actionable=True,
    )


def test_provenance_survives_builder_population_writeback_and_diagnostics() -> None:
    start = datetime(2026, 8, 17, 0, 0, tzinfo=UTC)
    recommendation = _recommendation(start)
    recommendation.import_price_source = "primary"
    recommendation.export_price_source = "entsoe"

    cfg = SensorConfig()
    cfg.recommendation_interval_minutes = 15
    cfg.recommendation_interval_length = 1
    cfg.electricity_price_update_interval = 15

    with patch(
        "custom_components.hsem.coordinator_builder.hsem_now",
        return_value=start,
    ):
        planner_input = build_planner_input(
            cfg=cfg,
            live=LiveState(),
            hourly_recommendations=[recommendation],
            previous_winner_name=None,
            previous_winner_score=0.0,
        )

    point = planner_input.price_points[0]
    assert point.import_price_source == "primary"
    assert point.export_price_source == "entsoe"

    tsi = TimeSeriesIndex.from_now(start, interval_minutes=15, horizon_hours=1)
    slots = [PlannedSlot(meta.start, meta.end) for meta in tsi]
    populate_prices(slots, planner_input.price_points, tsi=tsi)
    planned = slots[0]
    assert planned.import_price_source == "primary"
    assert planned.export_price_source == "entsoe"

    recommendation.import_price_source = None
    recommendation.export_price_source = None
    coordinator = HSEMDataUpdateCoordinator.__new__(HSEMDataUpdateCoordinator)
    coordinator._hourly_recommendations = [recommendation]
    coordinator._apply_planner_output(PlannerOutput(slots=[planned]))

    assert recommendation.import_price_source == "primary"
    assert recommendation.export_price_source == "entsoe"

    serialized = _slot_to_dict(planned)
    assert serialized["import_price_available"] is True
    assert serialized["export_price_available"] is True
    assert serialized["price_actionable"] is True
    assert serialized["import_price_source"] == "primary"
    assert serialized["export_price_source"] == "entsoe"


def test_hourly_and_legacy_population_preserve_only_available_sources() -> None:
    start = datetime(2026, 8, 17, 0, 0, tzinfo=UTC)
    tsi = TimeSeriesIndex.from_now(start, interval_minutes=15, horizon_hours=1)
    slots = [PlannedSlot(meta.start, meta.end) for meta in tsi]
    hourly = PricePoint(
        hour=0,
        import_price=0.2,
        export_price=0.1,
        import_price_source="entsoe",
        export_price_source="forecast",
    )

    populate_prices(slots, [hourly], tsi=tsi)

    assert all(slot.import_price_source == "entsoe" for slot in slots)
    assert all(slot.export_price_source == "forecast" for slot in slots)

    legacy_slot = PlannedSlot(start, start + timedelta(hours=1))
    unavailable_import = PricePoint(
        hour=0,
        import_price=0.2,
        export_price=0.1,
        import_price_available=False,
        export_price_available=True,
        import_price_source="entsoe",
        export_price_source="primary",
    )
    populate_prices([legacy_slot], [unavailable_import])

    assert legacy_slot.import_price_available is False
    assert legacy_slot.import_price_source is None
    assert legacy_slot.export_price_available is True
    assert legacy_slot.export_price_source == "primary"
    assert legacy_slot.price_actionable is False


def test_same_price_source_change_changes_canonical_signature() -> None:
    start = datetime(2026, 8, 17, 0, 0, tzinfo=UTC)
    recommendation = _recommendation(start)
    recommendation.import_price_source = "primary"
    recommendation.export_price_source = "primary"
    live = LiveState()

    primary_signature = _price_forecast_signature(
        [recommendation],
        live,
        start - timedelta(minutes=1),
    )
    recommendation.import_price_source = "entsoe"
    recommendation.export_price_source = "entsoe"
    entsoe_signature = _price_forecast_signature(
        [recommendation],
        live,
        start - timedelta(minutes=1),
    )

    assert primary_signature != entsoe_signature
    assert primary_signature[2][0][4:] == ("primary", "primary")
    assert entsoe_signature[2][0][4:] == ("entsoe", "entsoe")


def test_diagnostics_defaults_for_pre_provenance_slots() -> None:
    start = datetime(2026, 8, 17, 0, 0, tzinfo=UTC)
    slot_values = vars(PlannedSlot(start, start + timedelta(minutes=15))).copy()
    for field_name in (
        "import_price_available",
        "export_price_available",
        "import_price_source",
        "export_price_source",
        "price_actionable",
    ):
        slot_values.pop(field_name)
    legacy_slot = SimpleNamespace(**slot_values)

    serialized = _slot_to_dict(legacy_slot)

    assert serialized["import_price_available"] is True
    assert serialized["export_price_available"] is True
    assert serialized["price_actionable"] is True
    assert serialized["import_price_source"] is None
    assert serialized["export_price_source"] is None
