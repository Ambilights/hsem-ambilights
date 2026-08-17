"""Regression coverage for the strict ENTSO-E published-price backup.

Nord Pool remains authoritative wherever both price channels are published.
ENTSO-E may fill a missing primary slot only as an import/export pair and only
after both adjusted ENTSO-E sensors prove exact coverage of the entire local
delivery day.  The values consumed here are final local-currency/kWh values;
HSEM must not apply VAT, tariffs, currency conversion, or energy scaling.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from custom_components.hsem.coordinator_builder import build_planner_input
from custom_components.hsem.custom_sensors.hourly_data_populator.prices_solcast import (
    async_populate_price_and_solcast,
    populate_price_and_solcast_from_snapshot,
)
from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.price_source import PriceBackupStatus
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.models.state_snapshot import StateSnapshot
from custom_components.hsem.models.time_series import TimeSeriesIndex
from custom_components.hsem.planner.slot_population import populate_prices

STOCKHOLM = ZoneInfo("Europe/Stockholm")
PRICE_UNIT = "SEK/kWh"
QUARTER_HOUR = timedelta(minutes=15)


class _Cfg(SensorConfig):
    """Minimal price-population configuration for the fallback tests."""

    def __init__(self, *, forecast: bool = False) -> None:
        super().__init__()
        self.electricity_price_update_interval = 15
        self.recommendation_interval_minutes = 15
        self.import_electricity_price_sensor = "sensor.nordpool_import"
        self.export_electricity_price_sensor = "sensor.nordpool_export"
        self.import_electricity_price_entsoe_sensor = "sensor.entsoe_import"
        self.export_electricity_price_entsoe_sensor = "sensor.entsoe_export"
        self.import_electricity_price_forecast_sensor = (
            "sensor.forecast_import" if forecast else None
        )
        self.export_electricity_price_forecast_sensor = (
            "sensor.forecast_export" if forecast else None
        )
        self.solcast_pv_forecast_forecast_today = None
        self.solcast_pv_forecast_forecast_tomorrow = None
        self.solcast_pv_forecast_forecast_likelihood = "pv_estimate"


def _physical_starts(delivery_day: date) -> list[datetime]:
    """Return every physical quarter-hour in one Stockholm delivery day."""
    local_start = datetime.combine(delivery_day, datetime.min.time(), STOCKHOLM)
    local_end = datetime.combine(
        delivery_day + timedelta(days=1),
        datetime.min.time(),
        STOCKHOLM,
    )
    cursor = local_start.astimezone(UTC)
    end = local_end.astimezone(UTC)
    starts: list[datetime] = []
    while cursor < end:
        starts.append(cursor.astimezone(STOCKHOLM))
        cursor += QUARTER_HOUR
    return starts


def _recommendation(start: datetime) -> HourlyRecommendation:
    end = (start.astimezone(UTC) + QUARTER_HOUR).astimezone(STOCKHOLM)
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


def _recommendations(starts: list[datetime]) -> list[HourlyRecommendation]:
    return [_recommendation(start) for start in starts]


def _values(count: int, first: float, step: float = 0.001) -> list[float]:
    return [round(first + step * index, 5) for index in range(count)]


def _points(starts: list[datetime], values: list[float]) -> list[dict[str, object]]:
    assert len(starts) == len(values)
    return [
        {"time": start.isoformat(), "price": value}
        for start, value in zip(starts, values, strict=True)
    ]


def _attrs(
    points: list[dict[str, object]],
    *,
    unit: str | None = PRICE_UNIT,
    attribute: str = "prices_tomorrow",
) -> dict[str, object]:
    attributes: dict[str, object] = {attribute: points}
    if unit is not None:
        attributes["unit_of_measurement"] = unit
    return attributes


def _unit_only(unit: str = PRICE_UNIT) -> dict[str, object]:
    return {"unit_of_measurement": unit}


def _populate(
    recommendations: list[HourlyRecommendation],
    attributes: dict[str, dict[str, object]],
    *,
    cfg: SensorConfig | None = None,
) -> PriceBackupStatus:
    return populate_price_and_solcast_from_snapshot(
        recommendations,
        StateSnapshot(
            live=LiveState(),
            energy_average_values={},
            sensor_attributes=attributes,
        ),
        cfg or _Cfg(),
    )


def _complete_backup_attributes(
    starts: list[datetime],
    *,
    import_values: list[float] | None = None,
    export_values: list[float] | None = None,
) -> tuple[dict[str, object], dict[str, object], list[float], list[float]]:
    imports = import_values or _values(len(starts), 3.548)
    exports = export_values or _values(len(starts), 2.179)
    return (
        _attrs(_points(starts, imports)),
        _attrs(_points(starts, exports)),
        imports,
        exports,
    )


def _sources(
    recommendations: list[HourlyRecommendation],
) -> tuple[list[str | None], list[str | None]]:
    return (
        [rec.import_price_source for rec in recommendations],
        [rec.export_price_source for rec in recommendations],
    )


class TestEntsoeSourceSelection:
    """Primary, ENTSO-E, and legacy forecast have deterministic precedence."""

    def test_complete_primary_pair_wins_including_zero_and_negative_prices(
        self,
    ) -> None:
        starts = _physical_starts(date(2026, 8, 18))
        recs = _recommendations(starts)
        primary_import = _values(len(starts), 1.0)
        primary_export = _values(len(starts), 0.5)
        primary_import[:2] = [0.0, -0.25]
        primary_export[:2] = [-0.1, 0.0]
        backup_import, backup_export, _, _ = _complete_backup_attributes(starts)

        _populate(
            recs,
            {
                "sensor.nordpool_import": _attrs(_points(starts, primary_import)),
                "sensor.nordpool_export": _attrs(_points(starts, primary_export)),
                "sensor.entsoe_import": backup_import,
                "sensor.entsoe_export": backup_export,
            },
        )

        assert [rec.import_price for rec in recs] == pytest.approx(primary_import)
        assert [rec.export_price for rec in recs] == pytest.approx(primary_export)
        import_sources, export_sources = _sources(recs)
        assert import_sources == ["primary"] * len(starts)
        assert export_sources == ["primary"] * len(starts)
        assert all(rec.import_price_available for rec in recs)
        assert all(rec.export_price_available for rec in recs)

    def test_complete_tomorrow_arrays_fill_a_primary_missing_day_unchanged(
        self,
    ) -> None:
        starts = _physical_starts(date(2026, 8, 18))
        assert len(starts) == 96
        recs = _recommendations(starts)
        backup_import, backup_export, imports, exports = _complete_backup_attributes(
            starts
        )

        _populate(
            recs,
            {
                "sensor.nordpool_import": _unit_only(),
                "sensor.nordpool_export": _unit_only(),
                "sensor.entsoe_import": backup_import,
                "sensor.entsoe_export": backup_export,
            },
        )

        assert [rec.import_price for rec in recs] == pytest.approx(imports)
        assert [rec.export_price for rec in recs] == pytest.approx(exports)
        import_sources, export_sources = _sources(recs)
        assert import_sources == ["entsoe"] * 96
        assert export_sources == ["entsoe"] * 96
        assert all(rec.import_price_available for rec in recs)
        assert all(rec.export_price_available for rec in recs)

    def test_pt60m_backup_fans_out_across_15_minute_planning_slots(self) -> None:
        starts = _physical_starts(date(2026, 8, 18))
        hourly_starts = starts[::4]
        assert len(hourly_starts) == 24
        recs = _recommendations(starts)
        imports = _values(24, 3.0, step=0.1)
        exports = _values(24, 2.0, step=0.1)
        cfg = _Cfg()
        cfg.electricity_price_update_interval = 60

        status = _populate(
            recs,
            {
                "sensor.nordpool_import": _unit_only(),
                "sensor.nordpool_export": _unit_only(),
                "sensor.entsoe_import": _attrs(_points(hourly_starts, imports)),
                "sensor.entsoe_export": _attrs(_points(hourly_starts, exports)),
            },
            cfg=cfg,
        )

        expected_imports = [
            round(value / 4.0, 5) for value in imports for _ in range(4)
        ]
        expected_exports = [
            round(value / 4.0, 5) for value in exports for _ in range(4)
        ]
        assert [rec.import_price for rec in recs] == pytest.approx(expected_imports)
        assert [rec.export_price for rec in recs] == pytest.approx(expected_exports)
        assert all(rec.import_price_source == "entsoe" for rec in recs)
        assert all(rec.export_price_source == "entsoe" for rec in recs)
        assert status == PriceBackupStatus(configured=True, matched_slots=96)

    @pytest.mark.parametrize("missing_channel", ["import", "export"])
    def test_one_missing_primary_channel_switches_both_channels_for_that_slot(
        self,
        missing_channel: str,
    ) -> None:
        starts = _physical_starts(date(2026, 8, 18))
        recs = _recommendations(starts)
        gap_index = len(starts) // 2
        primary_import = _values(len(starts), 1.0)
        primary_export = _values(len(starts), 0.5)
        import_points = _points(starts, primary_import)
        export_points = _points(starts, primary_export)
        if missing_channel == "import":
            import_points.pop(gap_index)
        else:
            export_points.pop(gap_index)
        backup_import, backup_export, imports, exports = _complete_backup_attributes(
            starts
        )

        _populate(
            recs,
            {
                "sensor.nordpool_import": _attrs(import_points),
                "sensor.nordpool_export": _attrs(export_points),
                "sensor.entsoe_import": backup_import,
                "sensor.entsoe_export": backup_export,
            },
        )

        for index, rec in enumerate(recs):
            if index == gap_index:
                assert rec.import_price == pytest.approx(imports[index])
                assert rec.export_price == pytest.approx(exports[index])
                assert rec.import_price_source == "entsoe"
                assert rec.export_price_source == "entsoe"
            else:
                assert rec.import_price == pytest.approx(primary_import[index])
                assert rec.export_price == pytest.approx(primary_export[index])
                assert rec.import_price_source == "primary"
                assert rec.export_price_source == "primary"

    def test_precedence_is_primary_then_entsoe_then_legacy_forecast(self) -> None:
        delivery_days = [
            date(2026, 8, 18) + timedelta(days=index) for index in range(3)
        ]
        day_starts = [_physical_starts(day) for day in delivery_days]
        all_starts = [start for starts in day_starts for start in starts]
        recs = _recommendations(all_starts)

        primary_import = _values(96, 1.0)
        primary_export = _values(96, 0.5)
        entsoe_starts = day_starts[0] + day_starts[1]
        entsoe_import = _values(192, 10.0)
        entsoe_export = _values(192, 9.0)
        forecast_import = _values(288, 30.0)
        forecast_export = _values(288, 29.0)

        _populate(
            recs,
            {
                "sensor.nordpool_import": _attrs(
                    _points(day_starts[0], primary_import)
                ),
                "sensor.nordpool_export": _attrs(
                    _points(day_starts[0], primary_export)
                ),
                "sensor.entsoe_import": _attrs(_points(entsoe_starts, entsoe_import)),
                "sensor.entsoe_export": _attrs(_points(entsoe_starts, entsoe_export)),
                "sensor.forecast_import": _attrs(
                    _points(all_starts, forecast_import), attribute="prices"
                ),
                "sensor.forecast_export": _attrs(
                    _points(all_starts, forecast_export), attribute="prices"
                ),
            },
            cfg=_Cfg(forecast=True),
        )

        assert [rec.import_price for rec in recs[:96]] == pytest.approx(primary_import)
        assert [rec.import_price for rec in recs[96:192]] == pytest.approx(
            entsoe_import[96:]
        )
        assert [rec.import_price for rec in recs[192:]] == pytest.approx(
            forecast_import[192:]
        )
        import_sources, export_sources = _sources(recs)
        assert import_sources == ["primary"] * 96 + ["entsoe"] * 96 + ["forecast"] * 96
        assert export_sources == ["primary"] * 96 + ["entsoe"] * 96 + ["forecast"] * 96


class TestEntsoeFailClosedValidation:
    """Any uncertainty in a backup delivery day keeps it non-authoritative."""

    @pytest.mark.parametrize("channel", ["import", "export"])
    @pytest.mark.parametrize("missing_index", [0, 48, 95])
    def test_incomplete_backup_channel_fills_nothing(
        self,
        channel: str,
        missing_index: int,
    ) -> None:
        starts = _physical_starts(date(2026, 8, 18))
        recs = _recommendations(starts)
        import_points = _points(starts, _values(96, 3.0))
        export_points = _points(starts, _values(96, 2.0))
        if channel == "import":
            import_points.pop(missing_index)
        else:
            export_points.pop(missing_index)

        _populate(
            recs,
            {
                "sensor.nordpool_import": _unit_only(),
                "sensor.nordpool_export": _unit_only(),
                "sensor.entsoe_import": _attrs(import_points),
                "sensor.entsoe_export": _attrs(export_points),
            },
        )

        assert not any(rec.import_price_available for rec in recs)
        assert not any(rec.export_price_available for rec in recs)
        assert [rec.import_price for rec in recs] == pytest.approx([0.0] * 96)
        assert [rec.export_price for rec in recs] == pytest.approx([0.0] * 96)
        import_sources, export_sources = _sources(recs)
        assert import_sources == [None] * 96
        assert export_sources == [None] * 96

    @pytest.mark.parametrize(
        ("import_unit", "export_unit"),
        [("EUR/kWh", PRICE_UNIT), (None, PRICE_UNIT), (PRICE_UNIT, None)],
    )
    def test_wrong_or_missing_backup_unit_fills_nothing(
        self,
        import_unit: str | None,
        export_unit: str | None,
    ) -> None:
        starts = _physical_starts(date(2026, 8, 18))
        recs = _recommendations(starts)
        import_points = _points(starts, _values(96, 3.0))
        export_points = _points(starts, _values(96, 2.0))

        _populate(
            recs,
            {
                "sensor.nordpool_import": _unit_only(),
                "sensor.nordpool_export": _unit_only(),
                "sensor.entsoe_import": _attrs(import_points, unit=import_unit),
                "sensor.entsoe_export": _attrs(export_points, unit=export_unit),
            },
        )

        assert not any(rec.import_price_available for rec in recs)
        assert not any(rec.export_price_available for rec in recs)

    @pytest.mark.parametrize(
        "defect",
        ["naive", "misaligned", "nonfinite", "conflicting_duplicate"],
    )
    def test_malformed_backup_point_rejects_the_whole_source(self, defect: str) -> None:
        starts = _physical_starts(date(2026, 8, 18))
        recs = _recommendations(starts)
        import_points = _points(starts, _values(96, 3.0))
        export_points = _points(starts, _values(96, 2.0))
        bad_index = len(starts) // 2
        import_attributes = _attrs(import_points)
        if defect == "naive":
            import_points[bad_index]["time"] = (
                starts[bad_index].replace(tzinfo=None).isoformat()
            )
        elif defect == "misaligned":
            import_points[bad_index]["time"] = (
                starts[bad_index] + timedelta(minutes=1)
            ).isoformat()
        elif defect == "nonfinite":
            import_points[bad_index]["price"] = float("nan")
        else:
            import_attributes["prices"] = [
                {
                    "time": starts[bad_index].isoformat(),
                    "price": 999.0,
                }
            ]

        _populate(
            recs,
            {
                "sensor.nordpool_import": _unit_only(),
                "sensor.nordpool_export": _unit_only(),
                "sensor.entsoe_import": import_attributes,
                "sensor.entsoe_export": _attrs(export_points),
            },
        )

        assert not any(rec.import_price_available for rec in recs)
        assert not any(rec.export_price_available for rec in recs)
        import_sources, export_sources = _sources(recs)
        assert import_sources == [None] * 96
        assert export_sources == [None] * 96

    def test_missing_backup_attributes_are_ignored_before_legacy_forecast(
        self,
    ) -> None:
        starts = _physical_starts(date(2026, 8, 18))
        recs = _recommendations(starts)
        forecast_import = _values(96, 4.0)
        forecast_export = _values(96, 3.0)

        _populate(
            recs,
            {
                "sensor.nordpool_import": _unit_only(),
                "sensor.nordpool_export": _unit_only(),
                "sensor.forecast_import": _attrs(
                    _points(starts, forecast_import), attribute="prices"
                ),
                "sensor.forecast_export": _attrs(
                    _points(starts, forecast_export), attribute="prices"
                ),
            },
            cfg=_Cfg(forecast=True),
        )

        assert [rec.import_price for rec in recs] == pytest.approx(forecast_import)
        assert [rec.export_price for rec in recs] == pytest.approx(forecast_export)
        import_sources, export_sources = _sources(recs)
        assert import_sources == ["forecast"] * 96
        assert export_sources == ["forecast"] * 96

    @pytest.mark.asyncio
    async def test_unavailable_backup_ignores_stale_attributes(self) -> None:
        starts = _physical_starts(date(2026, 8, 18))
        recs = _recommendations(starts)
        stale_import, stale_export, _, _ = _complete_backup_attributes(starts)
        forecast_import = _values(96, 4.0)
        forecast_export = _values(96, 3.0)

        states = {
            "sensor.nordpool_import": SimpleNamespace(
                state="0.0", attributes=_unit_only()
            ),
            "sensor.nordpool_export": SimpleNamespace(
                state="0.0", attributes=_unit_only()
            ),
            "sensor.entsoe_import": SimpleNamespace(
                state="unavailable", attributes=stale_import
            ),
            "sensor.entsoe_export": SimpleNamespace(
                state="unavailable", attributes=stale_export
            ),
            "sensor.forecast_import": SimpleNamespace(
                state="4.0",
                attributes=_attrs(_points(starts, forecast_import), attribute="prices"),
            ),
            "sensor.forecast_export": SimpleNamespace(
                state="3.0",
                attributes=_attrs(_points(starts, forecast_export), attribute="prices"),
            ),
        }
        sensor = SimpleNamespace(
            hass=SimpleNamespace(states=SimpleNamespace(get=states.get))
        )

        status = await async_populate_price_and_solcast(
            sensor, recs, _Cfg(forecast=True)
        )

        assert [rec.import_price for rec in recs] == pytest.approx(forecast_import)
        assert [rec.export_price for rec in recs] == pytest.approx(forecast_export)
        import_sources, export_sources = _sources(recs)
        assert import_sources == ["forecast"] * 96
        assert export_sources == ["forecast"] * 96
        assert status == PriceBackupStatus(
            configured=True,
            rejection_reason="sensor_unavailable",
        )


class TestEntsoeBackupStatus:
    """The runtime exposes stable, safe selection and rejection details."""

    def test_success_reports_exact_number_of_selected_slots(self) -> None:
        starts = _physical_starts(date(2026, 8, 18))
        recs = _recommendations(starts)
        backup_import, backup_export, _, _ = _complete_backup_attributes(starts)

        status = _populate(
            recs,
            {
                "sensor.nordpool_import": _unit_only(),
                "sensor.nordpool_export": _unit_only(),
                "sensor.entsoe_import": backup_import,
                "sensor.entsoe_export": backup_export,
            },
        )

        assert status == PriceBackupStatus(
            configured=True,
            matched_slots=96,
            rejection_reason=None,
        )

    def test_valid_unused_backup_is_not_reported_as_rejected(self) -> None:
        starts = _physical_starts(date(2026, 8, 18))
        recs = _recommendations(starts)
        imports = _values(len(starts), 1.0)
        exports = _values(len(starts), 0.5)
        backup_import, backup_export, _, _ = _complete_backup_attributes(starts)

        status = _populate(
            recs,
            {
                "sensor.nordpool_import": _attrs(_points(starts, imports)),
                "sensor.nordpool_export": _attrs(_points(starts, exports)),
                "sensor.entsoe_import": backup_import,
                "sensor.entsoe_export": backup_export,
            },
        )

        assert status == PriceBackupStatus(configured=True)

    def test_absent_configuration_reports_neutral_status(self) -> None:
        starts = _physical_starts(date(2026, 8, 18))
        cfg = _Cfg()
        cfg.import_electricity_price_entsoe_sensor = None
        cfg.export_electricity_price_entsoe_sensor = None

        status = _populate(
            _recommendations(starts),
            {
                "sensor.nordpool_import": _unit_only(),
                "sensor.nordpool_export": _unit_only(),
            },
            cfg=cfg,
        )

        assert status == PriceBackupStatus()

    def test_half_configured_pair_reports_pair_required(self) -> None:
        starts = _physical_starts(date(2026, 8, 18))
        cfg = _Cfg()
        cfg.export_electricity_price_entsoe_sensor = None

        status = _populate(
            _recommendations(starts),
            {
                "sensor.nordpool_import": _unit_only(),
                "sensor.nordpool_export": _unit_only(),
            },
            cfg=cfg,
        )

        assert status == PriceBackupStatus(
            configured=True,
            rejection_reason="sensor_pair_required",
        )

    def test_missing_snapshot_entities_report_sensor_unavailable(self) -> None:
        starts = _physical_starts(date(2026, 8, 18))

        status = _populate(
            _recommendations(starts),
            {
                "sensor.nordpool_import": _unit_only(),
                "sensor.nordpool_export": _unit_only(),
            },
        )

        assert status == PriceBackupStatus(
            configured=True,
            rejection_reason="sensor_unavailable",
        )

    @pytest.mark.parametrize(
        ("defect", "expected_reason"),
        [
            ("unit_missing", "unit_missing"),
            ("unit_mismatch", "unit_mismatch"),
            ("arrays_missing", "import_price_arrays_missing"),
            ("naive_timestamp", "import_timestamp_invalid_or_naive"),
            ("cadence", "import_price_cadence_mismatch"),
            ("timestamps", "timestamp_mismatch"),
            ("full_day", "delivery_day_incomplete"),
        ],
    )
    def test_validation_failures_have_stable_reasons(
        self,
        defect: str,
        expected_reason: str,
    ) -> None:
        starts = _physical_starts(date(2026, 8, 18))
        import_points = _points(starts, _values(96, 3.0))
        export_points = _points(starts, _values(96, 2.0))
        import_attrs = _attrs(import_points)
        export_attrs = _attrs(export_points)

        if defect == "unit_missing":
            import_attrs.pop("unit_of_measurement")
        elif defect == "unit_mismatch":
            import_attrs["unit_of_measurement"] = "EUR/kWh"
        elif defect == "arrays_missing":
            import_attrs = _unit_only()
        elif defect == "naive_timestamp":
            import_points[48]["time"] = starts[48].replace(tzinfo=None).isoformat()
        elif defect == "cadence":
            import_points.pop(48)
        elif defect == "timestamps":
            next_day = _physical_starts(date(2026, 8, 19))
            export_attrs = _attrs(_points(next_day, _values(96, 2.0)))
        else:
            import_points.pop()
            export_points.pop()

        status = _populate(
            _recommendations(starts),
            {
                "sensor.nordpool_import": _unit_only(),
                "sensor.nordpool_export": _unit_only(),
                "sensor.entsoe_import": import_attrs,
                "sensor.entsoe_export": export_attrs,
            },
        )

        assert status == PriceBackupStatus(
            configured=True,
            rejection_reason=expected_reason,
        )


class TestEntsoeDeliveryDayTimeline:
    """Complete-day validation follows physical time across Stockholm DST."""

    @pytest.mark.parametrize(
        ("delivery_day", "expected_slots"),
        [(date(2026, 3, 29), 92), (date(2026, 10, 25), 100)],
    )
    def test_dst_delivery_days_preserve_every_physical_slot(
        self,
        delivery_day: date,
        expected_slots: int,
    ) -> None:
        starts = _physical_starts(delivery_day)
        assert len(starts) == expected_slots
        recs = _recommendations(starts)
        imports = _values(expected_slots, 3.0)
        exports = _values(expected_slots, 2.0)

        _populate(
            recs,
            {
                "sensor.nordpool_import": _unit_only(),
                "sensor.nordpool_export": _unit_only(),
                "sensor.entsoe_import": _attrs(_points(starts, imports)),
                "sensor.entsoe_export": _attrs(_points(starts, exports)),
            },
        )

        assert [rec.import_price for rec in recs] == pytest.approx(imports)
        assert [rec.export_price for rec in recs] == pytest.approx(exports)
        assert all(rec.import_price_source == "entsoe" for rec in recs)
        assert all(rec.export_price_source == "entsoe" for rec in recs)
        assert len({rec.start.astimezone(UTC) for rec in recs}) == expected_slots

        repeated_hour = [rec for rec in recs if rec.start.hour == 2]
        if expected_slots == 100:
            assert len(repeated_hour) == 8
            assert [rec.start.fold for rec in repeated_hour] == [0] * 4 + [1] * 4
            assert len({rec.import_price for rec in repeated_hour}) == 8
        else:
            assert repeated_hour == []


def test_entsoe_extends_actionable_prices_through_planner_input() -> None:
    """A complete backup second day survives into 192 actionable planner slots."""
    first_starts = _physical_starts(date(2026, 8, 18))
    second_starts = _physical_starts(date(2026, 8, 19))
    all_starts = first_starts + second_starts
    recs = _recommendations(all_starts)
    primary_import = _values(96, 1.0)
    primary_export = _values(96, 0.5)
    backup_import = _values(96, 3.0)
    backup_export = _values(96, 2.0)
    cfg = _Cfg()
    cfg.recommendation_interval_length = 48

    _populate(
        recs,
        {
            "sensor.nordpool_import": _attrs(_points(first_starts, primary_import)),
            "sensor.nordpool_export": _attrs(_points(first_starts, primary_export)),
            "sensor.entsoe_import": _attrs(_points(second_starts, backup_import)),
            "sensor.entsoe_export": _attrs(_points(second_starts, backup_export)),
        },
        cfg=cfg,
    )

    with patch(
        "custom_components.hsem.coordinator_builder.hsem_now",
        return_value=first_starts[0],
    ):
        planner_input = build_planner_input(
            cfg=cfg,
            live=LiveState(),
            hourly_recommendations=recs,
            previous_winner_name=None,
            previous_winner_score=0.0,
        )

    tsi = TimeSeriesIndex.from_now(
        first_starts[0],
        interval_minutes=15,
        horizon_hours=48,
    )
    slots = [PlannedSlot(meta.start, meta.end) for meta in tsi]
    populate_prices(slots, planner_input.price_points, tsi=tsi)

    assert len(planner_input.price_points) == 192
    assert len(slots) == 192
    assert all(slot.price_actionable for slot in slots)
    assert not tsi.missing_price_slots
    assert [slot.import_price_source for slot in slots[:96]] == ["primary"] * 96
    assert [slot.import_price_source for slot in slots[96:]] == ["entsoe"] * 96
    assert [slot.price.import_price for slot in slots[:96]] == pytest.approx(
        primary_import
    )
    assert [slot.price.import_price for slot in slots[96:]] == pytest.approx(
        backup_import
    )
