"""HA bridge and safety tests for dedicated-load secondary storage."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from custom_components.hsem.coordinator_builder import _build_secondary_storage_config
from custom_components.hsem.custom_sensors.config_reader import build_sensor_config
from custom_components.hsem.custom_sensors.secondary_storage_applier import (
    POWMR_CHARGER_SOLAR_ONLY,
    POWMR_CHARGER_UTILITY,
    POWMR_OUTPUT_SBU,
    POWMR_OUTPUT_UTILITY,
    async_apply_secondary_storage,
    build_secondary_write_plan,
)
from custom_components.hsem.custom_sensors.secondary_storage_plan_sensor import (
    _plan_windows,
)
from custom_components.hsem.flows.secondary_storage import (
    get_secondary_storage_step_schema,
    validate_secondary_storage_input,
)
from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.planner.secondary_storage import (
    SECONDARY_MODE_CHARGE,
    SECONDARY_MODE_SBU,
    SECONDARY_MODE_UTILITY,
)
from custom_components.hsem.utils.degraded_mode import DegradedMode
from custom_components.hsem.utils.inverter_verify import ApplyResult, ApplyStatus

_NOW = datetime(2026, 8, 11, tzinfo=ZoneInfo("Europe/Stockholm"))


def _entry(**overrides: object) -> MagicMock:
    """Return a config-entry mock backed by all integration defaults."""
    import voluptuous as vol

    from custom_components.hsem.const import DEFAULT_CONFIG_VALUES

    defaults = {
        key: ("" if value is vol.UNDEFINED else value)
        for key, value in DEFAULT_CONFIG_VALUES.items()
    }
    entry = MagicMock()
    entry.options = {**defaults, **overrides}
    entry.data = {}
    return entry


def _config(*, read_only: bool = False, control: bool = True) -> SensorConfig:
    """Return an enabled PowMr config with the verified HA entity IDs."""
    cfg = build_sensor_config(
        _entry(
            hsem_read_only=read_only,
            hsem_secondary_storage_enabled=True,
            hsem_secondary_storage_control_enabled=control,
        )
    )
    return cfg


def _rec(
    mode: str,
    *,
    current_a: float = 0.0,
    start: datetime = _NOW,
) -> HourlyRecommendation:
    """Build the smallest complete recommendation for adapter tests."""
    return HourlyRecommendation(
        start=start,
        end=start + timedelta(minutes=15),
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
        recommendation=None,
        solcast_pv_estimate_kwh=0.0,
        secondary_storage_charge_current_a=current_a,
        secondary_storage_mode=mode,
    )


def test_config_reader_uses_verified_powmr_defaults() -> None:
    """Enabling the feature should resolve the known entity and hardware defaults."""
    cfg = _config()

    assert cfg.secondary_storage.soc_entity == "sensor.powmr_soc"
    assert (
        cfg.secondary_storage.battery_net_power_entity
        == "sensor.powmr_battery_net_power"
    )
    assert cfg.secondary_storage.load_power_entity == "sensor.powmr_load_power_internal"
    assert cfg.secondary_storage.capacity_kwh == pytest.approx(15.0)
    assert cfg.secondary_storage.nominal_voltage_v == pytest.approx(24.0)
    assert cfg.secondary_storage.min_soc_pct == pytest.approx(20.0)
    assert cfg.secondary_storage.max_charge_current_a == pytest.approx(60.0)
    assert cfg.secondary_storage.grid_phase == 3
    assert cfg.phase_aware_charging_enabled is False
    assert (
        cfg.huawei_solar_batteries_grid_charge_maximum_power
        == "number.batteries_grid_charge_maximum_power"
    )
    assert (
        cfg.huawei_solar_batteries_charge_discharge_power
        == "sensor.batteries_charge_discharge_power"
    )


@pytest.mark.asyncio
async def test_config_schema_uses_verified_powmr_defaults() -> None:
    """The new flow should open with the discovered entities and hardware limits."""
    schema = await get_secondary_storage_step_schema(None)

    values = cast(dict[str, object], schema({}))

    assert values["hsem_secondary_storage_soc_entity"] == "sensor.powmr_soc"
    assert values["hsem_secondary_storage_capacity_kwh"] == pytest.approx(15.0)
    assert values["hsem_secondary_storage_nominal_voltage_v"] == pytest.approx(24.0)
    assert values["hsem_secondary_storage_max_charge_current_a"] == pytest.approx(60.0)
    assert values["hsem_secondary_storage_grid_phase"] == pytest.approx(3.0)
    assert values["hsem_secondary_storage_control_enabled"] is False


@pytest.mark.asyncio
async def test_config_validation_enforces_powmr_current_steps() -> None:
    """PowMr accepts only 10 A increments in its discovered 10–80 A range."""
    user_input = {
        "hsem_secondary_storage_enabled": True,
        "hsem_secondary_storage_control_enabled": False,
        "hsem_secondary_storage_soc_entity": "sensor.powmr_soc",
        "hsem_secondary_storage_load_power_entity": (
            "sensor.powmr_load_power_internal"
        ),
        "hsem_secondary_storage_min_soc_pct": 20.0,
        "hsem_secondary_storage_max_soc_pct": 100.0,
        "hsem_secondary_storage_min_charge_current_a": 10.0,
        "hsem_secondary_storage_max_charge_current_a": 55.0,
    }

    with patch(
        "custom_components.hsem.flows.secondary_storage.async_validate_entity_ids",
        new_callable=AsyncMock,
        return_value={},
    ):
        errors = await validate_secondary_storage_input(MagicMock(), user_input)

    assert errors["hsem_secondary_storage_min_charge_current_a"] == (
        "secondary_storage_invalid_charge_range"
    )


@pytest.mark.asyncio
async def test_config_validation_rejects_invalid_powmr_grid_phase() -> None:
    """The single-phase branch must be assigned to physical phase 1, 2, or 3."""
    user_input = {
        "hsem_secondary_storage_enabled": True,
        "hsem_secondary_storage_control_enabled": False,
        "hsem_secondary_storage_soc_entity": "sensor.powmr_soc",
        "hsem_secondary_storage_load_power_entity": (
            "sensor.powmr_load_power_internal"
        ),
        "hsem_secondary_storage_min_soc_pct": 20.0,
        "hsem_secondary_storage_max_soc_pct": 100.0,
        "hsem_secondary_storage_min_charge_current_a": 10.0,
        "hsem_secondary_storage_max_charge_current_a": 60.0,
        "hsem_secondary_storage_grid_phase": 4,
    }

    with patch(
        "custom_components.hsem.flows.secondary_storage.async_validate_entity_ids",
        new_callable=AsyncMock,
        return_value={},
    ):
        errors = await validate_secondary_storage_input(MagicMock(), user_input)

    assert errors["hsem_secondary_storage_grid_phase"] == (
        "secondary_storage_invalid_grid_phase"
    )


def test_missing_required_telemetry_disables_secondary_planner() -> None:
    """Missing SoC/load may enter degraded mode but must not produce a fake plan."""
    planner_config = _build_secondary_storage_config(_config(), LiveState())

    assert planner_config.enabled is False
    assert planner_config.valid is False


def test_out_of_range_soc_disables_secondary_planner() -> None:
    """Implausible SoC telemetry must never be clamped into an actionable plan."""
    live = LiveState()
    live.secondary_storage.soc_pct = 150.0
    live.secondary_storage.load_power_w = 100.0

    planner_config = _build_secondary_storage_config(_config(), live)

    assert planner_config.enabled is False
    assert planner_config.valid is False


def test_secondary_grid_phase_reaches_planner_model() -> None:
    """The configured physical phase must survive the HA-to-planner bridge."""
    cfg = _config()
    cfg.secondary_storage.grid_phase = 2
    live = LiveState()
    live.secondary_storage.soc_pct = 50.0
    live.secondary_storage.load_power_w = 200.0

    planner_config = _build_secondary_storage_config(cfg, live)

    assert planner_config.grid_phase == 2
    assert planner_config.valid


def test_sbu_transition_disables_grid_charging_before_battery_output() -> None:
    """SBU must select solar-only charging before enabling battery output."""
    cfg = _config()
    live = LiveState()
    live.secondary_storage.soc_pct = 60.0

    operations = build_secondary_write_plan(cfg, live, _rec(SECONDARY_MODE_SBU))

    assert [(op.kind, op.desired) for op in operations] == [
        ("select", POWMR_CHARGER_SOLAR_ONLY),
        ("select", POWMR_OUTPUT_SBU),
    ]


def test_reserve_guard_overrides_sbu_to_utility() -> None:
    """A stale SBU recommendation can never discharge at the 20% reserve."""
    cfg = _config()
    live = LiveState()
    live.secondary_storage.soc_pct = 20.0

    operations = build_secondary_write_plan(cfg, live, _rec(SECONDARY_MODE_SBU))

    assert [op.desired for op in operations] == [
        POWMR_OUTPUT_UTILITY,
        POWMR_CHARGER_SOLAR_ONLY,
    ]


def test_max_soc_guard_stops_charging_before_output_write() -> None:
    """At maximum SoC, stopping utility charge is the first operation."""
    cfg = _config()
    live = LiveState()
    live.secondary_storage.soc_pct = 100.0

    operations = build_secondary_write_plan(
        cfg,
        live,
        _rec(SECONDARY_MODE_CHARGE, current_a=60.0),
    )

    assert [op.desired for op in operations] == [
        POWMR_CHARGER_SOLAR_ONLY,
        POWMR_OUTPUT_UTILITY,
    ]


def test_charge_transition_quantizes_current_and_enables_utility_last() -> None:
    """The 10 A-step number is set only after utility bypass and before charging."""
    cfg = _config()
    live = LiveState()
    live.secondary_storage.soc_pct = 50.0

    operations = build_secondary_write_plan(
        cfg,
        live,
        _rec(SECONDARY_MODE_CHARGE, current_a=37.0),
    )

    assert [(op.kind, op.desired) for op in operations] == [
        ("select", POWMR_OUTPUT_UTILITY),
        ("number", 40.0),
        ("select", POWMR_CHARGER_UTILITY),
    ]


def test_zero_phase_limited_current_disables_grid_charge() -> None:
    """A runtime 0 A limit must not be rounded up to PowMr's 10 A minimum."""
    cfg = _config()
    live = LiveState()
    live.secondary_storage.soc_pct = 50.0

    operations = build_secondary_write_plan(
        cfg,
        live,
        _rec(SECONDARY_MODE_CHARGE, current_a=0.0),
    )

    assert [op.desired for op in operations] == [
        POWMR_OUTPUT_UTILITY,
        POWMR_CHARGER_SOLAR_ONLY,
    ]


@pytest.mark.asyncio
async def test_read_only_blocks_adapter_before_write_and_verify() -> None:
    """The adapter's own gate must remain effective even if called directly."""
    cfg = _config(read_only=True)
    live = LiveState()
    live.secondary_storage.soc_pct = 50.0
    sensor = MagicMock()

    with patch(
        "custom_components.hsem.custom_sensors.secondary_storage_applier.async_write_and_verify",
        new_callable=AsyncMock,
    ) as verifier:
        summary = await async_apply_secondary_storage(
            sensor,
            cfg,
            live,
            _rec(SECONDARY_MODE_CHARGE, current_a=40.0),
        )

    assert summary.results == []
    verifier.assert_not_awaited()


@pytest.mark.asyncio
async def test_feature_control_gate_blocks_adapter() -> None:
    """Shadow planning remains write-free even when global read-only is off."""
    cfg = _config(read_only=False, control=False)
    live = LiveState()
    live.secondary_storage.soc_pct = 50.0
    sensor = MagicMock()

    with patch(
        "custom_components.hsem.custom_sensors.secondary_storage_applier.async_write_and_verify",
        new_callable=AsyncMock,
    ) as verifier:
        summary = await async_apply_secondary_storage(
            sensor,
            cfg,
            live,
            _rec(SECONDARY_MODE_SBU),
        )

    assert summary.results == []
    verifier.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_live_telemetry_blocks_adapter() -> None:
    """The adapter must not trust a stale plan when required telemetry is absent."""
    cfg = _config(read_only=False, control=True)
    live = LiveState()
    sensor = MagicMock()

    with patch(
        "custom_components.hsem.custom_sensors.secondary_storage_applier.async_write_and_verify",
        new_callable=AsyncMock,
    ) as verifier:
        summary = await async_apply_secondary_storage(
            sensor,
            cfg,
            live,
            _rec(SECONDARY_MODE_CHARGE, current_a=40.0),
        )

    assert summary.results == []
    verifier.assert_not_awaited()


@pytest.mark.asyncio
async def test_error_degraded_mode_blocks_adapter() -> None:
    """Critical degraded mode remains an independent PowMr write gate."""
    cfg = _config(read_only=False, control=True)
    live = LiveState()
    live.secondary_storage.soc_pct = 50.0
    live.secondary_storage.load_power_w = 100.0
    live._degraded_mode = DegradedMode.Error
    sensor = MagicMock()

    with patch(
        "custom_components.hsem.custom_sensors.secondary_storage_applier.async_write_and_verify",
        new_callable=AsyncMock,
    ) as verifier:
        summary = await async_apply_secondary_storage(
            sensor,
            cfg,
            live,
            _rec(SECONDARY_MODE_SBU),
        )

    assert summary.results == []
    verifier.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [ApplyStatus.FAILED, ApplyStatus.UNVERIFIED])
async def test_unconfirmed_write_blocks_remaining_powmr_transition(
    status: ApplyStatus,
) -> None:
    """A failed or unreadable first operation must block every later command."""
    cfg = _config(read_only=False, control=True)
    live = LiveState()
    live.secondary_storage.soc_pct = 60.0
    live.secondary_storage.load_power_w = 100.0
    sensor = MagicMock()
    unconfirmed = ApplyResult(
        entity_id=cfg.secondary_storage.charger_source_priority_entity or "",
        desired=POWMR_CHARGER_SOLAR_ONLY,
        actual=None,
        status=status,
        attempts=3,
    )

    with patch(
        "custom_components.hsem.custom_sensors.secondary_storage_applier.async_write_and_verify",
        new_callable=AsyncMock,
        return_value=unconfirmed,
    ) as verifier:
        summary = await async_apply_secondary_storage(
            sensor,
            cfg,
            live,
            _rec(SECONDARY_MODE_SBU),
        )

    assert verifier.await_count == 1
    assert summary.overall_status == status


def test_plan_windows_coalesce_adjacent_modes() -> None:
    """The diagnostic attribute stays compact across 15-minute horizons."""
    first = _rec(SECONDARY_MODE_UTILITY, start=_NOW)
    second = _rec(
        SECONDARY_MODE_UTILITY,
        start=_NOW + timedelta(minutes=15),
    )
    third = _rec(
        SECONDARY_MODE_SBU,
        start=_NOW + timedelta(minutes=30),
    )
    second.secondary_storage_grid_import_kwh = 0.025
    third.secondary_storage_discharged_kwh = 0.04

    windows = _plan_windows([first, second, third])

    assert len(windows) == 2
    assert windows[0]["mode"] == SECONDARY_MODE_UTILITY
    assert windows[0]["end"] == second.end.isoformat()
    assert windows[1]["mode"] == SECONDARY_MODE_SBU
    assert windows[1]["discharged_kwh"] == pytest.approx(0.04)
