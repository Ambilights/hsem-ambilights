"""Tests for battery wait-mode self-consumption feature (issue #742).

Covers:
- Default constant value for wait-mode behaviour in ``const.py``
- Input validation in ``flows/batteries_wait_mode.py``
- Discharge cap helper in ``custom_sensors/applier.py``
- Optimiser-owned battery holds at the hardware-applier boundary
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.hsem.const import (
    DEFAULT_CONFIG_VALUES,
    DEFAULT_HSEM_BATTERIES_WAIT_MODE,
)
from custom_components.hsem.custom_sensors.applier import (
    _wait_mode_self_consumption_cap_w,
    async_apply_battery_settings,
)
from custom_components.hsem.flows.batteries_wait_mode import (
    validate_batteries_wait_mode_input,
)
from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.utils.degraded_mode import DegradedMode
from custom_components.hsem.utils.inverter_verify import ApplyResult, ApplyStatus
from custom_components.hsem.utils.recommendations import Recommendations
from custom_components.hsem.utils.workingmodes import WorkingModes

# ---------------------------------------------------------------------------
# Default constant value tests
# ---------------------------------------------------------------------------


class TestWaitModeDefaults:
    """Verify the wait-mode behaviour default is safe."""

    def test_wait_mode_strict_by_default(self):
        """Wait mode must default to strict to preserve existing behaviour."""
        assert DEFAULT_CONFIG_VALUES["hsem_batteries_wait_mode_behavior"] == "strict"


# ---------------------------------------------------------------------------
# _wait_mode_self_consumption_cap_w tests
# ---------------------------------------------------------------------------


class TestWaitModeSelfConsumptionCapW:
    """Unit tests for the reserve-preserving discharge cap helper."""

    def test_no_surplus_returns_zero(self):
        cap = _wait_mode_self_consumption_cap_w(
            battery_capacity_kwh=2.0,
            required_capacity_kwh=2.0,
            slot_hours=0.25,
            max_discharge_power_w=5000,
        )
        assert cap == 0

    def test_below_reserve_returns_zero(self):
        cap = _wait_mode_self_consumption_cap_w(
            battery_capacity_kwh=1.5,
            required_capacity_kwh=2.0,
            slot_hours=0.25,
            max_discharge_power_w=5000,
        )
        assert cap == 0

    def test_surplus_converted_to_power(self):
        """1 kWh surplus over a 1-hour slot -> 1000 W cap."""
        cap = _wait_mode_self_consumption_cap_w(
            battery_capacity_kwh=3.0,
            required_capacity_kwh=2.0,
            slot_hours=1.0,
            max_discharge_power_w=5000,
        )
        assert cap == 1000

    def test_surplus_over_short_slot(self):
        """1 kWh surplus over a 15-minute slot -> 4000 W cap."""
        cap = _wait_mode_self_consumption_cap_w(
            battery_capacity_kwh=3.0,
            required_capacity_kwh=2.0,
            slot_hours=0.25,
            max_discharge_power_w=5000,
        )
        assert cap == 4000

    def test_cap_limited_by_max_discharge_power(self):
        cap = _wait_mode_self_consumption_cap_w(
            battery_capacity_kwh=10.0,
            required_capacity_kwh=0.0,
            slot_hours=0.25,
            max_discharge_power_w=2500,
        )
        assert cap == 2500

    def test_zero_slot_hours_returns_zero(self):
        cap = _wait_mode_self_consumption_cap_w(
            battery_capacity_kwh=5.0,
            required_capacity_kwh=0.0,
            slot_hours=0.0,
            max_discharge_power_w=5000,
        )
        assert cap == 0


# ---------------------------------------------------------------------------
# Optimiser-owned hold execution
# ---------------------------------------------------------------------------


def _hold_test_config() -> SensorConfig:
    """Return the minimum writable Huawei configuration for hold tests."""
    cfg = SensorConfig()
    cfg.read_only = False
    cfg.batteries_wait_mode_behavior = "self_consumption_with_reserve"
    cfg.huawei_solar_batteries_maximum_discharging_power = (
        "number.batteries_max_discharge"
    )
    cfg.huawei_solar_batteries_working_mode = "select.batteries_working_mode"
    cfg.huawei_solar_batteries_excess_pv_energy_use_in_tou = (
        "select.batteries_excess_pv"
    )
    return cfg


def _hold_test_live() -> LiveState:
    """Return a healthy live snapshot with energy above the reserve."""
    live = LiveState()
    live._degraded_mode = DegradedMode.OK
    live.huawei_batteries_rated_capacity_wh = 30000.0
    live.huawei_batteries_max_discharge_power_w = 10000.0
    live.huawei_batteries_working_mode = WorkingModes.MaximizeSelfConsumption.value
    live.huawei_batteries_excess_pv_use_in_tou = "charge"
    live.huawei_batteries_forcible_charge_state = "Stopped"
    live.battery_current_capacity_kwh = 6.0
    live.tou_periods.periods = list(DEFAULT_HSEM_BATTERIES_WAIT_MODE)
    return live


def _hold_test_recommendation(
    recommendation: str,
    *,
    primary_battery_hold: bool,
) -> HourlyRecommendation:
    """Build one complete 15-minute recommendation for the applier."""
    start = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
    return HourlyRecommendation(
        start=start,
        end=start + timedelta(minutes=15),
        avg_house_consumption_kwh=0.25,
        avg_house_consumption_1d_kwh=0.25,
        avg_house_consumption_3d_kwh=0.25,
        avg_house_consumption_7d_kwh=0.25,
        avg_house_consumption_14d_kwh=0.25,
        batteries_charged_kwh=0.0,
        batteries_discharged_kwh=0.0,
        estimated_battery_capacity_kwh=6.0,
        estimated_battery_soc_pct=25.0,
        estimated_cost_currency=0.0,
        estimated_net_consumption_kwh=0.25,
        export_price=1.0,
        grid_export_kwh=0.0,
        grid_import_kwh=0.25,
        import_price=2.0,
        recommendation=recommendation,
        solcast_pv_estimate_kwh=0.0,
        primary_battery_hold=primary_battery_hold,
    )


def _successful_write(**kwargs: Any) -> ApplyResult:
    """Return a verified result for one mocked hardware write."""
    return ApplyResult(
        entity_id=kwargs["entity_id"],
        desired=kwargs["desired"],
        actual=kwargs["desired"],
        status=ApplyStatus.OK,
        attempts=1,
    )


def _writes(verifier: AsyncMock) -> list[tuple[str, object]]:
    """Return the ordered entity/value pairs sent to the write verifier."""
    return [
        (call.kwargs["entity_id"], call.kwargs["desired"])
        for call in verifier.await_args_list
    ]


class TestPrimaryBatteryHoldExecution:
    """A solved idle MILP slot must execute as a strict battery hold."""

    @pytest.mark.asyncio
    async def test_wait_label_hold_forces_tou_zero_discharge_and_pv_export(self):
        """The marker overrides global self-consumption-with-reserve behavior."""
        sensor = MagicMock()
        sensor.hass = MagicMock()
        cfg = _hold_test_config()
        live = _hold_test_live()
        rec = _hold_test_recommendation(
            Recommendations.BatteriesWaitMode.value,
            primary_battery_hold=True,
        )

        with patch(
            "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
            new_callable=AsyncMock,
            side_effect=_successful_write,
        ) as verifier:
            summary = await async_apply_battery_settings(
                sensor,
                cfg,
                live,
                rec,
                current_required_battery_kwh=5.0,
                now=rec.start,
            )

        assert _writes(verifier) == [
            ("number.batteries_max_discharge", 0),
            ("select.batteries_excess_pv", "fed_to_grid"),
            ("select.batteries_working_mode", WorkingModes.TimeOfUse.value),
        ]
        assert summary.overall_status == ApplyStatus.OK

    @pytest.mark.asyncio
    async def test_ev_display_label_preserves_same_strict_hold(self):
        """EV relabelling must not discard an optimiser-owned primary hold."""
        sensor = MagicMock()
        sensor.hass = MagicMock()
        cfg = _hold_test_config()
        live = _hold_test_live()
        live.ev.is_charging = True
        live.ev.power_w = 7000.0
        live.net_consumption_w = 800.0
        rec = _hold_test_recommendation(
            Recommendations.EVSmartCharging.value,
            primary_battery_hold=True,
        )

        with patch(
            "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
            new_callable=AsyncMock,
            side_effect=_successful_write,
        ) as verifier:
            summary = await async_apply_battery_settings(
                sensor,
                cfg,
                live,
                rec,
                current_required_battery_kwh=5.0,
                now=rec.start,
            )

        assert _writes(verifier) == [
            ("number.batteries_max_discharge", 0),
            ("select.batteries_excess_pv", "fed_to_grid"),
            ("select.batteries_working_mode", WorkingModes.TimeOfUse.value),
        ]
        assert summary.overall_status == ApplyStatus.OK

    @pytest.mark.asyncio
    async def test_non_hold_wait_keeps_self_consumption_with_reserve(self):
        """Without the marker, preserve the configured heuristic behavior."""
        sensor = MagicMock()
        sensor.hass = MagicMock()
        cfg = _hold_test_config()
        live = _hold_test_live()
        live.huawei_batteries_working_mode = WorkingModes.TimeOfUse.value
        live.huawei_batteries_excess_pv_use_in_tou = "fed_to_grid"
        rec = _hold_test_recommendation(
            Recommendations.BatteriesWaitMode.value,
            primary_battery_hold=False,
        )

        with patch(
            "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
            new_callable=AsyncMock,
            side_effect=_successful_write,
        ) as verifier:
            summary = await async_apply_battery_settings(
                sensor,
                cfg,
                live,
                rec,
                current_required_battery_kwh=5.0,
                now=rec.start,
            )

        # One kWh above reserve across a 15-minute slot permits 4 kW.
        assert _writes(verifier) == [
            ("number.batteries_max_discharge", 4000),
            ("select.batteries_excess_pv", "charge"),
            (
                "select.batteries_working_mode",
                WorkingModes.MaximizeSelfConsumption.value,
            ),
        ]
        assert summary.overall_status == ApplyStatus.OK


# ---------------------------------------------------------------------------
# validate_batteries_wait_mode_input tests
# ---------------------------------------------------------------------------


class TestValidateBatteriesWaitModeInput:
    """Unit tests for the wait-mode config-flow input validator."""

    @pytest.mark.asyncio
    async def test_strict_value_is_valid(self):
        errors = await validate_batteries_wait_mode_input(
            {"hsem_batteries_wait_mode_behavior": "strict"}
        )
        assert errors == {}

    @pytest.mark.asyncio
    async def test_self_consumption_value_is_valid(self):
        errors = await validate_batteries_wait_mode_input(
            {"hsem_batteries_wait_mode_behavior": "self_consumption_with_reserve"}
        )
        assert errors == {}

    @pytest.mark.asyncio
    async def test_invalid_value_is_rejected(self):
        errors = await validate_batteries_wait_mode_input(
            {"hsem_batteries_wait_mode_behavior": "something_else"}
        )
        assert "hsem_batteries_wait_mode_behavior" in errors

    @pytest.mark.asyncio
    async def test_missing_field_is_rejected(self):
        errors = await validate_batteries_wait_mode_input({})
        assert "hsem_batteries_wait_mode_behavior" in errors
