"""Tests for TOU schedule verification and fail-closed charge transitions.

Covers:
- ``extract_tou_periods`` reading the ``Period N`` attributes, never the state
- ``_read_tou_periods`` reading HA live rather than the pre-write LiveState
- Wait -> GridCharge and GridCharge -> Wait write ordering
- Pre-disarming grid charge power before the TOU schedule changes
- A failed TOU write blocking every downstream hardware command
- Sub-100 W phase-safe limits flooring to 0 W on GridCharge entry
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.hsem.const import (
    DEFAULT_HSEM_BATTERIES_WAIT_MODE,
    DEFAULT_HSEM_TOU_MODES_FORCE_CHARGE,
)
from custom_components.hsem.custom_sensors.applier import (
    _grid_charging_is_armed,
    _read_tou_periods,
    async_apply_battery_settings,
)
from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.utils.degraded_mode import DegradedMode
from custom_components.hsem.utils.huawei import extract_tou_periods
from custom_components.hsem.utils.inverter_verify import ApplyResult, ApplyStatus
from custom_components.hsem.utils.recommendations import Recommendations
from custom_components.hsem.utils.workingmodes import WorkingModes

TOU_ENTITY = "sensor.batteries_tou"
CHARGE_ENTITY = "number.batteries_grid_charge_max_power"
MODE_ENTITY = "select.batteries_working_mode"
DISCHARGE_ENTITY = "number.batteries_max_discharge"


# ---------------------------------------------------------------------------
# Attribute extraction
# ---------------------------------------------------------------------------


class TestExtractTouPeriods:
    """The schedule lives in attributes; the state is only a period count."""

    def test_reads_period_attributes_in_order(self):
        attrs = {
            "Period 2": "06:00-08:00/1234567/-",
            "Period 1": "00:00-23:59/1234567/+",
            "friendly_name": "TOU",
        }
        assert extract_tou_periods(attrs) == [
            "00:00-23:59/1234567/+",
            "06:00-08:00/1234567/-",
        ]

    def test_no_period_attributes_returns_empty(self):
        assert extract_tou_periods({"friendly_name": "TOU"}) == []

    def test_state_value_is_never_used(self):
        """A count-like state must not leak into the extracted schedule."""
        assert extract_tou_periods({"Period 1": "00:00-00:01/1234567/+"}) != ["1"]


class TestReadTouPeriods:
    """``_read_tou_periods`` must reflect HA *after* a write, not LiveState."""

    def test_reads_live_attributes(self):
        sensor = MagicMock()
        state = MagicMock()
        state.state = "1"
        state.attributes = {"Period 1": "00:00-00:01/1234567/+"}
        sensor.hass.states.get.return_value = state
        assert _read_tou_periods(sensor, TOU_ENTITY) == ["00:00-00:01/1234567/+"]

    def test_missing_entity_returns_none(self):
        sensor = MagicMock()
        sensor.hass.states.get.return_value = None
        assert _read_tou_periods(sensor, TOU_ENTITY) is None

    def test_no_entity_id_returns_none(self):
        assert _read_tou_periods(MagicMock(), None) is None


class TestGridChargingIsArmed:
    """Pre-disarm only when charging can actually happen."""

    def test_armed_in_tou_with_full_day_window(self):
        live = _live(WorkingModes.TimeOfUse.value, DEFAULT_HSEM_TOU_MODES_FORCE_CHARGE)
        assert _grid_charging_is_armed(live) is True

    def test_not_armed_with_placeholder_window(self):
        live = _live(WorkingModes.TimeOfUse.value, DEFAULT_HSEM_BATTERIES_WAIT_MODE)
        assert _grid_charging_is_armed(live) is False

    def test_not_armed_outside_tou_mode(self):
        live = _live(
            WorkingModes.MaximizeSelfConsumption.value,
            DEFAULT_HSEM_TOU_MODES_FORCE_CHARGE,
        )
        assert _grid_charging_is_armed(live) is False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _config() -> SensorConfig:
    cfg = SensorConfig()
    cfg.read_only = False
    cfg.batteries_wait_mode_behavior = "strict"
    cfg.huawei_solar_batteries_maximum_discharging_power = DISCHARGE_ENTITY
    cfg.huawei_solar_batteries_working_mode = MODE_ENTITY
    cfg.huawei_solar_batteries_excess_pv_energy_use_in_tou = (
        "select.batteries_excess_pv"
    )
    cfg.huawei_solar_batteries_tou_charging_and_discharging_periods = TOU_ENTITY
    cfg.huawei_solar_batteries_grid_charge_maximum_power = CHARGE_ENTITY
    cfg.huawei_solar_device_id_batteries = "battery-device-id"
    return cfg


def _live(working_mode: str, periods: list[str]) -> LiveState:
    live = LiveState()
    live._degraded_mode = DegradedMode.OK
    live.huawei_batteries_rated_capacity_wh = 30000.0
    live.huawei_batteries_max_discharge_power_w = 10000.0
    live.huawei_batteries_max_charge_power_w = 10000.0
    live.huawei_batteries_grid_charge_max_power_w = 5000.0
    live.huawei_batteries_working_mode = working_mode
    live.huawei_batteries_excess_pv_use_in_tou = "charge"
    live.huawei_batteries_forcible_charge_state = "Stopped"
    live.battery_current_capacity_kwh = 6.0
    live.tou_periods.periods = list(periods)
    return live


def _recommendation(recommendation: str) -> HourlyRecommendation:
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
    )


def _ok(**kwargs: Any) -> ApplyResult:
    return ApplyResult(
        entity_id=kwargs["entity_id"],
        desired=kwargs["desired"],
        actual=kwargs["desired"],
        status=ApplyStatus.OK,
        attempts=1,
    )


def _fail_tou(**kwargs: Any) -> ApplyResult:
    """Succeed on everything except the TOU entity."""
    if kwargs["entity_id"] == TOU_ENTITY:
        return ApplyResult(
            entity_id=TOU_ENTITY,
            desired=kwargs["desired"],
            actual=None,
            status=ApplyStatus.FAILED,
            attempts=2,
            error_message="simulated service failure",
        )
    return _ok(**kwargs)


def _writes(verifier: AsyncMock) -> list[tuple[str, object]]:
    return [
        (call.kwargs["entity_id"], call.kwargs["desired"])
        for call in verifier.await_args_list
    ]


async def _apply(cfg, live, rec, side_effect, **kwargs):
    sensor = MagicMock()
    sensor.hass = MagicMock()
    with patch(
        "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
        new_callable=AsyncMock,
        side_effect=side_effect,
    ) as verifier:
        summary = await async_apply_battery_settings(
            sensor,
            cfg,
            live,
            rec,
            current_required_battery_kwh=5.0,
            now=rec.start,
            **kwargs,
        )
    return summary, verifier


# ---------------------------------------------------------------------------
# Transition behaviour
# ---------------------------------------------------------------------------


class TestTouVerificationTarget:
    """The TOU write must be verified against the period list."""

    @pytest.mark.asyncio
    async def test_desired_is_the_period_list_not_a_hash_of_state(self):
        cfg = _config()
        live = _live(WorkingModes.TimeOfUse.value, DEFAULT_HSEM_BATTERIES_WAIT_MODE)
        rec = _recommendation(Recommendations.BatteriesChargeGrid.value)

        _, verifier = await _apply(
            cfg, live, rec, _ok, grid_charge_power_limit_w=4000.0
        )

        tou_calls = [
            call
            for call in verifier.await_args_list
            if call.kwargs["entity_id"] == TOU_ENTITY
        ]
        assert len(tou_calls) == 1
        assert tou_calls[0].kwargs["desired"] == list(
            DEFAULT_HSEM_TOU_MODES_FORCE_CHARGE
        )


class TestWaitToGridCharge:
    """Entering grid charge arms power *before* the schedule."""

    @pytest.mark.asyncio
    async def test_charge_limit_written_before_tou(self):
        cfg = _config()
        live = _live(WorkingModes.TimeOfUse.value, DEFAULT_HSEM_BATTERIES_WAIT_MODE)
        rec = _recommendation(Recommendations.BatteriesChargeGrid.value)

        _, verifier = await _apply(
            cfg, live, rec, _ok, grid_charge_power_limit_w=4000.0
        )
        entities = [entity for entity, _ in _writes(verifier)]

        assert entities.index(CHARGE_ENTITY) < entities.index(TOU_ENTITY)

    @pytest.mark.asyncio
    async def test_failed_tou_blocks_downstream_mode_command(self):
        """A failed TOU write must stop the transition, leaving charging off."""
        cfg = _config()
        live = _live(WorkingModes.TimeOfUse.value, DEFAULT_HSEM_BATTERIES_WAIT_MODE)
        rec = _recommendation(Recommendations.BatteriesChargeGrid.value)

        summary, verifier = await _apply(
            cfg, live, rec, _fail_tou, grid_charge_power_limit_w=4000.0
        )
        entities = [entity for entity, _ in _writes(verifier)]

        assert TOU_ENTITY in entities
        assert MODE_ENTITY not in entities
        assert summary.overall_status == ApplyStatus.FAILED

    @pytest.mark.asyncio
    async def test_sub_100w_phase_safe_limit_floors_to_zero(self):
        """Flooring is deliberate: <100 W resolves to 0 W, not a partial arm."""
        cfg = _config()
        live = _live(WorkingModes.TimeOfUse.value, DEFAULT_HSEM_BATTERIES_WAIT_MODE)
        rec = _recommendation(Recommendations.BatteriesChargeGrid.value)

        _, verifier = await _apply(cfg, live, rec, _ok, grid_charge_power_limit_w=99.0)

        charge_writes = [
            desired for entity, desired in _writes(verifier) if entity == CHARGE_ENTITY
        ]
        assert charge_writes == [0.0]


class TestGridChargeToWait:
    """Leaving grid charge must disarm charging before touching the schedule."""

    @pytest.mark.asyncio
    async def test_charge_power_pre_disarmed_before_tou(self):
        cfg = _config()
        live = _live(WorkingModes.TimeOfUse.value, DEFAULT_HSEM_TOU_MODES_FORCE_CHARGE)
        rec = _recommendation(Recommendations.BatteriesWaitMode.value)

        _, verifier = await _apply(cfg, live, rec, _ok)
        writes = _writes(verifier)
        entities = [entity for entity, _ in writes]

        assert (CHARGE_ENTITY, 0.0) in writes
        assert entities.index(CHARGE_ENTITY) < entities.index(TOU_ENTITY)

    @pytest.mark.asyncio
    async def test_failed_tou_leaves_hardware_non_charging(self):
        """The schedule may survive, but its power limit is already zero."""
        cfg = _config()
        live = _live(WorkingModes.TimeOfUse.value, DEFAULT_HSEM_TOU_MODES_FORCE_CHARGE)
        rec = _recommendation(Recommendations.BatteriesWaitMode.value)

        summary, verifier = await _apply(cfg, live, rec, _fail_tou)
        writes = _writes(verifier)
        entities = [entity for entity, _ in writes]

        assert (CHARGE_ENTITY, 0.0) in writes
        assert entities.index(CHARGE_ENTITY) < entities.index(TOU_ENTITY)
        assert MODE_ENTITY not in entities
        assert summary.overall_status == ApplyStatus.FAILED

    @pytest.mark.asyncio
    async def test_no_pre_disarm_when_charging_not_armed(self):
        """Avoid a pointless 0 W write on every non-GridCharge cycle."""
        cfg = _config()
        live = _live(
            WorkingModes.MaximizeSelfConsumption.value,
            DEFAULT_HSEM_BATTERIES_WAIT_MODE,
        )
        rec = _recommendation(Recommendations.BatteriesWaitMode.value)

        _, verifier = await _apply(cfg, live, rec, _ok)

        assert CHARGE_ENTITY not in [entity for entity, _ in _writes(verifier)]

    @pytest.mark.asyncio
    async def test_no_pre_disarm_when_restore_power_unknown(self):
        """Never disarm to 0 W without a known way to re-arm."""
        cfg = _config()
        live = _live(WorkingModes.TimeOfUse.value, DEFAULT_HSEM_TOU_MODES_FORCE_CHARGE)
        live.huawei_batteries_max_charge_power_w = None
        rec = _recommendation(Recommendations.BatteriesWaitMode.value)

        _, verifier = await _apply(cfg, live, rec, _ok)

        assert CHARGE_ENTITY not in [entity for entity, _ in _writes(verifier)]


class TestRoundTripWithoutPhaseAwareCharging:
    """Grid charging must survive a Wait -> GridCharge cycle with no limit.

    Phase-aware charging is disabled by default, so ``grid_charge_power_limit_w``
    is ``None``.  Entry must still re-arm a 0 W limit left by a previous exit,
    otherwise planned grid charging silently never starts.
    """

    @pytest.mark.asyncio
    async def test_entry_rearms_zero_limit_without_phase_aware_limit(self):
        cfg = _config()
        live = _live(WorkingModes.TimeOfUse.value, DEFAULT_HSEM_BATTERIES_WAIT_MODE)
        live.huawei_batteries_grid_charge_max_power_w = 0.0  # left by a prior exit
        rec = _recommendation(Recommendations.BatteriesChargeGrid.value)

        _, verifier = await _apply(cfg, live, rec, _ok)
        writes = _writes(verifier)

        assert (CHARGE_ENTITY, 10000.0) in writes
        entities = [entity for entity, _ in writes]
        assert entities.index(CHARGE_ENTITY) < entities.index(TOU_ENTITY)

    @pytest.mark.asyncio
    async def test_entry_does_not_override_user_chosen_limit(self):
        """A positive limit is the user's; re-arming must not clobber it."""
        cfg = _config()
        live = _live(WorkingModes.TimeOfUse.value, DEFAULT_HSEM_BATTERIES_WAIT_MODE)
        live.huawei_batteries_grid_charge_max_power_w = 3000.0
        rec = _recommendation(Recommendations.BatteriesChargeGrid.value)

        _, verifier = await _apply(cfg, live, rec, _ok)

        assert CHARGE_ENTITY not in [entity for entity, _ in _writes(verifier)]

    @pytest.mark.asyncio
    async def test_full_round_trip_leaves_charging_restorable(self):
        """Exit disarms to 0 W; the next entry re-arms it without phase-aware."""
        cfg = _config()
        exit_live = _live(
            WorkingModes.TimeOfUse.value, DEFAULT_HSEM_TOU_MODES_FORCE_CHARGE
        )
        _, exit_verifier = await _apply(
            cfg,
            exit_live,
            _recommendation(Recommendations.BatteriesWaitMode.value),
            _ok,
        )
        assert (CHARGE_ENTITY, 0.0) in _writes(exit_verifier)

        # Hardware now reports the disarmed limit on the next cycle.
        entry_live = _live(
            WorkingModes.TimeOfUse.value, DEFAULT_HSEM_BATTERIES_WAIT_MODE
        )
        entry_live.huawei_batteries_grid_charge_max_power_w = 0.0
        _, entry_verifier = await _apply(
            cfg,
            entry_live,
            _recommendation(Recommendations.BatteriesChargeGrid.value),
            _ok,
        )

        assert (CHARGE_ENTITY, 10000.0) in _writes(entry_verifier)
