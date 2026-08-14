"""Safety gate tests for inverter/battery hardware write paths (issue P0).

These tests prove that every combination of blocking mode prevents writes from
reaching the Huawei Solar service layer, and that normal mode still allows
them when valid data is present.

Covered scenarios
-----------------
- ``read_only=True`` blocks both :func:`async_apply_inverter_power_control` and
  :func:`async_apply_battery_settings`.
- ``DegradedMode.Error`` blocks both applier functions.
- ``DegradedMode.Degraded`` (non-critical entities missing) still allows writes.
- Normal mode (``read_only=False``, ``DegradedMode.OK``) allows writes.
- The top-level gate in ``_async_apply_hardware_writes`` (working_mode_sensor)
  logs the correct message for each blocking scenario.

All tests are pure-Python and require no running Home Assistant instance.

Note on logging
----------------
``HSEM_LOGGER.debug`` is patched with a no-op ``MagicMock`` in every test so
that planner/applier output never reaches the standard ``custom_components.hsem``
logger during the test run.  This keeps test output clean and decouples the
safety-gate assertions from log-formatting changes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.hsem.custom_sensors.applier import (
    FullyFedDischargeCapState,
    async_apply_battery_settings,
    async_apply_inverter_power_control,
)
from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.utils.degraded_mode import DegradedMode
from custom_components.hsem.utils.inverter_verify import ApplyStatus

# ---------------------------------------------------------------------------
# Module-level patch targets (reused across all test classes)
# ---------------------------------------------------------------------------

# Patch HSEM_LOGGER.debug to suppress log output during tests.
# Use MagicMock because debug() is a synchronous method; AsyncMock would
# return a coroutine that never gets awaited and emits RuntimeWarnings.
_LOGGER_PATCH = "custom_components.hsem.utils.logger.HSEM_LOGGER.debug"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sensor():
    """Return a minimal mock sensor for testing."""
    sensor = MagicMock()
    sensor.hass = MagicMock()
    return sensor


def _make_cfg(*, read_only: bool = False) -> SensorConfig:
    """Return a minimal :class:`SensorConfig` with the given read_only flag."""
    cfg = SensorConfig()
    cfg.read_only = read_only
    cfg.export_electricity_min_price = 0.0
    return cfg


def _make_live(*, degraded_mode: DegradedMode = DegradedMode.OK) -> LiveState:
    """Return a :class:`LiveState` with the chosen degraded mode forced."""
    live = LiveState()
    # Override the lazily-computed cached value directly so no entities need
    # to be set up just to drive the mode.
    live._degraded_mode = degraded_mode
    live.export_electricity_price = 1.0
    return live


def _make_rec(recommendation: str = "batteries_discharge_mode") -> HourlyRecommendation:
    """Return a minimal :class:`HourlyRecommendation` for testing."""
    rec = HourlyRecommendation.__new__(HourlyRecommendation)
    object.__setattr__(rec, "recommendation", recommendation)
    object.__setattr__(rec, "batteries_discharged_kwh", 0.0)
    object.__setattr__(rec, "grid_import_kwh", 0.0)
    return rec


def _make_planned_rec(
    recommendation: str,
    *,
    discharged_kwh: float = 0.45,
    end_capacity_kwh: float = 19.55,
) -> HourlyRecommendation:
    """Return a complete 15-minute recommendation for applier sequencing tests."""
    start = datetime(2026, 8, 12, 15, 45, tzinfo=UTC)
    return HourlyRecommendation(
        start=start,
        end=start + timedelta(minutes=15),
        avg_house_consumption_kwh=0.25,
        avg_house_consumption_1d_kwh=0.25,
        avg_house_consumption_3d_kwh=0.25,
        avg_house_consumption_7d_kwh=0.25,
        avg_house_consumption_14d_kwh=0.25,
        batteries_charged_kwh=0.0,
        batteries_discharged_kwh=discharged_kwh,
        estimated_battery_capacity_kwh=end_capacity_kwh,
        estimated_battery_soc_pct=70.0,
        estimated_cost_currency=0.0,
        estimated_net_consumption_kwh=-2.0,
        export_price=1.0,
        grid_export_kwh=2.0,
        grid_import_kwh=0.0,
        import_price=1.0,
        recommendation=recommendation,
        solcast_pv_estimate_kwh=2.25,
    )


def _ok_apply_result(**kwargs):
    """Build a successful result matching an async_write_and_verify call."""
    from custom_components.hsem.utils.inverter_verify import ApplyResult

    return ApplyResult(
        entity_id=kwargs["entity_id"],
        desired=kwargs["desired"],
        actual=kwargs["desired"],
        status=ApplyStatus.OK,
        attempts=1,
    )


# ---------------------------------------------------------------------------
# async_apply_inverter_power_control — safety gate
# ---------------------------------------------------------------------------


class TestInverterPowerControlSafetyGate:
    """Defense-in-depth gate inside async_apply_inverter_power_control."""

    @pytest.mark.asyncio
    async def test_read_only_blocks_inverter_writes(self):
        """read_only=True must return an empty summary without any service call."""
        sensor = _make_sensor()
        cfg = _make_cfg(read_only=True)
        live = _make_live(degraded_mode=DegradedMode.OK)

        with (
            patch(_LOGGER_PATCH, new_callable=MagicMock),
            patch(
                "custom_components.hsem.custom_sensors.applier.async_set_grid_export_power_pct"
            ) as mock_write,
        ):
            summary = await async_apply_inverter_power_control(sensor, cfg, live)

        mock_write.assert_not_called()
        assert len(summary.results) == 0

    @pytest.mark.asyncio
    async def test_error_mode_blocks_inverter_writes(self):
        """DegradedMode.Error must block all inverter writes."""
        sensor = _make_sensor()
        cfg = _make_cfg(read_only=False)
        live = _make_live(degraded_mode=DegradedMode.Error)

        with (
            patch(_LOGGER_PATCH, new_callable=MagicMock),
            patch(
                "custom_components.hsem.custom_sensors.applier.async_set_grid_export_power_pct"
            ) as mock_write,
        ):
            summary = await async_apply_inverter_power_control(sensor, cfg, live)

        mock_write.assert_not_called()
        assert len(summary.results) == 0

    @pytest.mark.asyncio
    async def test_degraded_mode_allows_inverter_writes(self):
        """DegradedMode.Degraded must NOT block writes (non-critical data only)."""
        sensor = _make_sensor()
        cfg = _make_cfg(read_only=False)
        # Degraded: price entity missing, but battery data present.
        live = _make_live(degraded_mode=DegradedMode.Degraded)
        # Set a numeric export price so the function can compute export_pct.
        live.export_electricity_price = 0.5
        # Set current inverter state to force a write (100 → 0 would write).
        live.huawei_inverter_active_power_control = "Unlimited"

        # Set up an inverter device ID so the write loop has something to call.
        cfg.huawei_solar_device_id_inverter_1 = "device_123"
        cfg.huawei_solar_inverter_active_power_control = (
            "sensor.inverter_active_power_control"
        )
        cfg.export_electricity_min_price = 1.0

        # Make the HA state read return an entity indicating "Unlimited" (100 %).
        mock_state = MagicMock()
        mock_state.state = "Unlimited"
        sensor.hass.states.get.return_value = mock_state

        with (
            patch(_LOGGER_PATCH, new_callable=MagicMock),
            patch(
                "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
                new_callable=AsyncMock,
            ) as mock_wv,
        ):
            from custom_components.hsem.utils.inverter_verify import ApplyResult

            mock_wv.return_value = ApplyResult(
                entity_id="sensor.inverter_active_power_control",
                desired=0,
                actual=0,
                status=ApplyStatus.OK,
                attempts=1,
            )
            _summary = await async_apply_inverter_power_control(sensor, cfg, live)

        # The write-and-verify function should have been reached (not blocked).
        mock_wv.assert_called_once()

    @pytest.mark.asyncio
    async def test_normal_mode_allows_inverter_writes(self):
        """OK mode with read_only=False must reach the write path."""
        sensor = _make_sensor()
        cfg = _make_cfg(read_only=False)
        cfg.huawei_solar_device_id_inverter_1 = "device_abc"
        cfg.huawei_solar_inverter_active_power_control = (
            "sensor.inverter_active_power_control"
        )
        cfg.export_electricity_min_price = 1.0

        live = _make_live(degraded_mode=DegradedMode.OK)
        live.export_electricity_price = 0.5
        live.huawei_inverter_active_power_control = "Unlimited"

        mock_state = MagicMock()
        mock_state.state = "Unlimited"
        sensor.hass.states.get.return_value = mock_state

        with (
            patch(_LOGGER_PATCH, new_callable=MagicMock),
            patch(
                "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
                new_callable=AsyncMock,
            ) as mock_wv,
        ):
            from custom_components.hsem.utils.inverter_verify import ApplyResult

            mock_wv.return_value = ApplyResult(
                entity_id="sensor.inverter_active_power_control",
                desired=0,
                actual=0,
                status=ApplyStatus.OK,
                attempts=1,
            )
            _summary = await async_apply_inverter_power_control(sensor, cfg, live)

        mock_wv.assert_called_once()


# ---------------------------------------------------------------------------
# async_apply_battery_settings — safety gate
# ---------------------------------------------------------------------------


class TestBatterySettingsSafetyGate:
    """Defense-in-depth gate inside async_apply_battery_settings."""

    def _make_rec(self) -> HourlyRecommendation:
        from custom_components.hsem.utils.recommendations import Recommendations

        return _make_rec(Recommendations.BatteriesDischargeMode.value)

    @pytest.mark.asyncio
    async def test_read_only_blocks_battery_writes(self):
        """read_only=True must return an empty summary without any service call."""
        sensor = _make_sensor()
        cfg = _make_cfg(read_only=True)
        live = _make_live(degraded_mode=DegradedMode.OK)
        rec = self._make_rec()

        with (
            patch(_LOGGER_PATCH, new_callable=MagicMock),
            patch(
                "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
                new_callable=AsyncMock,
            ) as mock_wv,
        ):
            summary = await async_apply_battery_settings(sensor, cfg, live, rec, 5.0)

        mock_wv.assert_not_called()
        assert len(summary.results) == 0

    @pytest.mark.asyncio
    async def test_error_mode_blocks_battery_writes(self):
        """DegradedMode.Error must block all battery writes."""
        sensor = _make_sensor()
        cfg = _make_cfg(read_only=False)
        live = _make_live(degraded_mode=DegradedMode.Error)
        rec = self._make_rec()

        with (
            patch(_LOGGER_PATCH, new_callable=MagicMock),
            patch(
                "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
                new_callable=AsyncMock,
            ) as mock_wv,
        ):
            summary = await async_apply_battery_settings(sensor, cfg, live, rec, 5.0)

        mock_wv.assert_not_called()
        assert len(summary.results) == 0

    @pytest.mark.asyncio
    async def test_degraded_mode_allows_battery_writes(self):
        """DegradedMode.Degraded must NOT block battery writes."""
        sensor = _make_sensor()
        cfg = _make_cfg(read_only=False)
        cfg.huawei_solar_batteries_working_mode = "select.batteries_working_mode"
        cfg.huawei_solar_batteries_maximum_discharging_power = (
            "number.batteries_max_discharge"
        )
        cfg.huawei_solar_batteries_excess_pv_energy_use_in_tou = (
            "select.batteries_excess_pv"
        )

        live = _make_live(degraded_mode=DegradedMode.Degraded)
        live.huawei_batteries_max_discharge_power_w = 3000.0
        live.huawei_batteries_rated_capacity_wh = 10000.0
        live.huawei_batteries_working_mode = "TimeOfUse"
        live.huawei_batteries_excess_pv_use_in_tou = "charge"

        from custom_components.hsem.utils.recommendations import Recommendations

        rec = _make_rec(Recommendations.BatteriesDischargeMode.value)

        with (
            patch(_LOGGER_PATCH, new_callable=MagicMock),
            patch(
                "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
                new_callable=AsyncMock,
            ) as mock_wv,
        ):
            from custom_components.hsem.utils.inverter_verify import ApplyResult

            mock_wv.return_value = ApplyResult(
                entity_id="select.batteries_working_mode",
                desired="MaximizeSelfConsumption",
                actual="MaximizeSelfConsumption",
                status=ApplyStatus.OK,
                attempts=1,
            )
            _summary = await async_apply_battery_settings(sensor, cfg, live, rec, 5.0)

        # At least one write was attempted (not blocked).
        mock_wv.assert_called()

    @pytest.mark.asyncio
    async def test_normal_mode_allows_battery_writes(self):
        """OK mode with read_only=False must reach the write path."""
        sensor = _make_sensor()
        cfg = _make_cfg(read_only=False)
        cfg.huawei_solar_batteries_working_mode = "select.batteries_working_mode"
        cfg.huawei_solar_batteries_maximum_discharging_power = (
            "number.batteries_max_discharge"
        )
        cfg.huawei_solar_batteries_excess_pv_energy_use_in_tou = (
            "select.batteries_excess_pv"
        )

        live = _make_live(degraded_mode=DegradedMode.OK)
        live.huawei_batteries_max_discharge_power_w = 3000.0
        live.huawei_batteries_rated_capacity_wh = 10000.0
        live.huawei_batteries_working_mode = "TimeOfUse"
        live.huawei_batteries_excess_pv_use_in_tou = "charge"

        from custom_components.hsem.utils.recommendations import Recommendations

        rec = _make_rec(Recommendations.BatteriesDischargeMode.value)

        with (
            patch(_LOGGER_PATCH, new_callable=MagicMock),
            patch(
                "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
                new_callable=AsyncMock,
            ) as mock_wv,
        ):
            from custom_components.hsem.utils.inverter_verify import ApplyResult

            mock_wv.return_value = ApplyResult(
                entity_id="select.batteries_working_mode",
                desired="MaximizeSelfConsumption",
                actual="MaximizeSelfConsumption",
                status=ApplyStatus.OK,
                attempts=1,
            )
            _summary = await async_apply_battery_settings(sensor, cfg, live, rec, 5.0)

        mock_wv.assert_called()

    @pytest.mark.asyncio
    async def test_phase_charge_cap_is_first_grid_charge_write(self):
        """Huawei power must be capped before TOU forced charging can start."""
        from custom_components.hsem.utils.inverter_verify import ApplyResult
        from custom_components.hsem.utils.recommendations import Recommendations

        sensor = _make_sensor()
        cfg = _make_cfg(read_only=False)
        cfg.huawei_solar_batteries_grid_charge_maximum_power = (
            "number.batteries_grid_charge_maximum_power"
        )
        live = _make_live(degraded_mode=DegradedMode.OK)
        live.huawei_batteries_rated_capacity_wh = 30000.0
        live.huawei_batteries_max_discharge_power_w = 10000.0
        live.huawei_batteries_max_charge_power_w = 10000.0
        live.huawei_batteries_grid_charge_max_power_w = 5000.0
        rec = _make_rec(Recommendations.BatteriesChargeGrid.value)

        with (
            patch(_LOGGER_PATCH, new_callable=MagicMock),
            patch(
                "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
                new_callable=AsyncMock,
                return_value=ApplyResult(
                    entity_id="number.batteries_grid_charge_maximum_power",
                    desired=4100.0,
                    actual=4100.0,
                    status=ApplyStatus.OK,
                    attempts=1,
                ),
            ) as verifier,
        ):
            await async_apply_battery_settings(
                sensor,
                cfg,
                live,
                rec,
                5.0,
                grid_charge_power_limit_w=4175.0,
            )

        first_call = verifier.await_args_list[0]
        assert first_call.kwargs["entity_id"] == (
            "number.batteries_grid_charge_maximum_power"
        )
        assert first_call.kwargs["desired"] == pytest.approx(4100.0)

    @pytest.mark.asyncio
    async def test_failed_phase_charge_cap_blocks_forced_charge(self):
        """An unverified Huawei cap must stop every later battery write."""
        from custom_components.hsem.utils.inverter_verify import ApplyResult
        from custom_components.hsem.utils.recommendations import Recommendations

        sensor = _make_sensor()
        cfg = _make_cfg(read_only=False)
        cfg.huawei_solar_batteries_grid_charge_maximum_power = (
            "number.batteries_grid_charge_maximum_power"
        )
        live = _make_live(degraded_mode=DegradedMode.OK)
        live.huawei_batteries_rated_capacity_wh = 30000.0
        live.huawei_batteries_max_discharge_power_w = 10000.0
        live.huawei_batteries_max_charge_power_w = 10000.0
        live.huawei_batteries_grid_charge_max_power_w = 5000.0
        rec = _make_rec(Recommendations.BatteriesChargeGrid.value)

        with (
            patch(_LOGGER_PATCH, new_callable=MagicMock),
            patch(
                "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
                new_callable=AsyncMock,
                return_value=ApplyResult(
                    entity_id="number.batteries_grid_charge_maximum_power",
                    desired=0.0,
                    actual=5000.0,
                    status=ApplyStatus.FAILED,
                    attempts=3,
                ),
            ) as verifier,
        ):
            summary = await async_apply_battery_settings(
                sensor,
                cfg,
                live,
                rec,
                5.0,
                grid_charge_power_limit_w=0.0,
            )

        assert verifier.await_count == 1
        assert summary.overall_status == ApplyStatus.FAILED


# ---------------------------------------------------------------------------
# Fully Fed battery-export control
# ---------------------------------------------------------------------------


class TestFullyFedBatteryExportControl:
    """Verify bounded power and fail-safe ordering around Fully Fed mode."""

    @staticmethod
    def _cfg() -> SensorConfig:
        cfg = _make_cfg(read_only=False)
        cfg.huawei_solar_device_id_batteries = "battery_device"
        cfg.huawei_solar_batteries_forcible_charge = "sensor.batteries_forcible"
        cfg.huawei_solar_batteries_maximum_discharging_power = (
            "number.batteries_max_discharge"
        )
        cfg.huawei_solar_batteries_working_mode = "select.batteries_working_mode"
        cfg.huawei_solar_batteries_excess_pv_energy_use_in_tou = (
            "select.batteries_excess_pv"
        )
        return cfg

    @staticmethod
    def _live() -> LiveState:
        live = _make_live(degraded_mode=DegradedMode.OK)
        live.huawei_batteries_rated_capacity_wh = 30000.0
        live.huawei_batteries_max_discharge_power_w = 10000.0
        live.huawei_batteries_working_mode = "time_of_use_luna2000"
        live.huawei_batteries_excess_pv_use_in_tou = "fed_to_grid"
        live.huawei_batteries_forcible_charge_state = "Stopped"
        live.battery_current_capacity_kwh = 20.0
        live.battery_usable_capacity_kwh = 25.5
        live.huawei_batteries_soc_pct = 75.0
        return live

    @pytest.mark.asyncio
    async def test_planned_export_caps_power_before_fully_fed_mode(self):
        """0.45 kWh in 15 minutes becomes 1.8 kW before mode selection."""
        sensor = _make_sensor()
        cfg = self._cfg()
        live = self._live()
        rec = _make_planned_rec("force_batteries_discharge")

        with (
            patch(_LOGGER_PATCH, new_callable=MagicMock),
            patch(
                "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
                new_callable=AsyncMock,
                side_effect=_ok_apply_result,
            ) as verifier,
            patch(
                "custom_components.hsem.custom_sensors.applier.async_stop_forcible_discharge",
                new_callable=AsyncMock,
            ) as stop_force,
        ):
            summary = await async_apply_battery_settings(
                sensor, cfg, live, rec, 20.0, now=rec.start
            )

        writes = [
            (call.kwargs["entity_id"], call.kwargs["desired"])
            for call in verifier.await_args_list
        ]
        assert writes == [
            ("number.batteries_max_discharge", 1800),
            ("select.batteries_working_mode", "fully_fed_to_grid"),
        ]
        assert summary.overall_status == ApplyStatus.OK
        stop_force.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_repeated_callback_does_not_ratchet_planned_cap(self):
        """Proportional live energy/time changes keep the same command."""
        sensor = _make_sensor()
        cfg = self._cfg()
        live = self._live()
        live.huawei_batteries_working_mode = "fully_fed_to_grid"
        live.huawei_batteries_max_discharge_power_w = 1800.0
        live.battery_current_capacity_kwh = 19.91
        rec = _make_planned_rec("force_batteries_discharge")

        with patch(
            "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
            new_callable=AsyncMock,
            side_effect=_ok_apply_result,
        ) as verifier:
            summary = await async_apply_battery_settings(
                sensor,
                cfg,
                live,
                rec,
                20.0,
                now=rec.start + timedelta(minutes=3),
            )

        verifier.assert_not_awaited()
        assert summary.overall_status == ApplyStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_coarse_capacity_change_does_not_collapse_latched_cap(self):
        """A one-percent Huawei SoC step is not precise pacing information."""
        sensor = _make_sensor()
        cfg = self._cfg()
        live = self._live()
        live.huawei_batteries_working_mode = "fully_fed_to_grid"
        live.battery_current_capacity_kwh = 18.9
        rec = _make_planned_rec(
            "force_batteries_discharge",
            discharged_kwh=0.458,
            end_capacity_kwh=18.442,
        )
        state = FullyFedDischargeCapState()

        with patch(
            "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
            new_callable=AsyncMock,
            side_effect=_ok_apply_result,
        ) as verifier:
            await async_apply_battery_settings(
                sensor,
                cfg,
                live,
                rec,
                20.0,
                now=rec.end - timedelta(hours=0.139),
                fully_fed_discharge_state=state,
            )
            assert verifier.await_count == 1
            first_call = verifier.await_args
            assert first_call is not None
            first_cap_w = int(first_call.kwargs["desired"])
            assert first_cap_w == 1832

            verifier.reset_mock()
            live.huawei_batteries_max_discharge_power_w = float(first_cap_w)
            live.battery_current_capacity_kwh = 18.6
            await async_apply_battery_settings(
                sensor,
                cfg,
                live,
                rec,
                20.0,
                now=rec.end - timedelta(hours=0.131),
                fully_fed_discharge_state=state,
            )
            verifier.assert_not_awaited()
            assert state.commanded_cap_w == 1832

            # Reaching the rounded endpoint after latching must not turn the
            # same coarse SoC artifact into a complete 0 W collapse.
            live.battery_current_capacity_kwh = rec.estimated_battery_capacity_kwh
            await async_apply_battery_settings(
                sensor,
                cfg,
                live,
                rec,
                20.0,
                now=rec.end - timedelta(minutes=5),
                fully_fed_discharge_state=state,
            )
            verifier.assert_not_awaited()

            # If another actor lowers the entity, normal reconciliation must
            # restore the plan-derived cap without changing the latch.
            live.huawei_batteries_max_discharge_power_w = 1206.0
            await async_apply_battery_settings(
                sensor,
                cfg,
                live,
                rec,
                20.0,
                now=rec.end - timedelta(minutes=4),
                fully_fed_discharge_state=state,
            )

        assert verifier.await_count == 1
        restored = verifier.await_args
        assert restored is not None
        assert restored.kwargs["desired"] == 1832

    @pytest.mark.asyncio
    async def test_changed_plan_recomputes_stable_cap(self):
        """A changed selected discharge allocation invalidates the latch."""
        sensor = _make_sensor()
        cfg = self._cfg()
        live = self._live()
        live.huawei_batteries_working_mode = "fully_fed_to_grid"
        live.battery_current_capacity_kwh = 18.9
        original = _make_planned_rec(
            "force_batteries_discharge",
            discharged_kwh=0.458,
            end_capacity_kwh=18.442,
        )
        changed = _make_planned_rec(
            "force_batteries_discharge",
            discharged_kwh=0.478,
            end_capacity_kwh=18.122,
        )
        state = FullyFedDischargeCapState()

        with patch(
            "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
            new_callable=AsyncMock,
            side_effect=_ok_apply_result,
        ) as verifier:
            await async_apply_battery_settings(
                sensor,
                cfg,
                live,
                original,
                20.0,
                now=original.start + timedelta(seconds=30),
                fully_fed_discharge_state=state,
            )
            original_call = verifier.await_args
            assert original_call is not None
            first_cap_w = int(original_call.kwargs["desired"])
            verifier.reset_mock()
            live.huawei_batteries_max_discharge_power_w = float(first_cap_w)

            await async_apply_battery_settings(
                sensor,
                cfg,
                live,
                changed,
                20.0,
                now=changed.start + timedelta(minutes=1),
                fully_fed_discharge_state=state,
            )

        assert verifier.await_count == 1
        changed_call = verifier.await_args
        assert changed_call is not None
        assert first_cap_w == 1832
        assert changed_call.kwargs["desired"] == 1912

    @pytest.mark.asyncio
    async def test_hardware_limit_change_does_not_rearm_endpoint_guard(self):
        """Changing only the hardware limit must not trust coarse SoC again."""
        sensor = _make_sensor()
        cfg = self._cfg()
        live = self._live()
        live.huawei_batteries_working_mode = "fully_fed_to_grid"
        live.battery_current_capacity_kwh = 18.9
        rec = _make_planned_rec(
            "force_batteries_discharge",
            discharged_kwh=1.0,
            end_capacity_kwh=18.442,
        )
        state = FullyFedDischargeCapState()

        with patch(
            "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
            new_callable=AsyncMock,
            side_effect=_ok_apply_result,
        ) as verifier:
            await async_apply_battery_settings(
                sensor,
                cfg,
                live,
                rec,
                20.0,
                now=rec.start,
                fully_fed_discharge_state=state,
            )
            assert verifier.await_args is not None
            assert verifier.await_args.kwargs["desired"] == 4000

            verifier.reset_mock()
            live.battery_current_capacity_kwh = rec.estimated_battery_capacity_kwh
            live.huawei_batteries_max_discharge_power_w = 4000.0
            live.huawei_batteries_rated_capacity_wh = 5000.0
            await async_apply_battery_settings(
                sensor,
                cfg,
                live,
                rec,
                20.0,
                now=rec.start + timedelta(minutes=1),
                fully_fed_discharge_state=state,
            )
            assert verifier.await_args is not None
            assert verifier.await_args.kwargs["desired"] == 2500

            verifier.reset_mock()
            live.huawei_batteries_max_discharge_power_w = 2500.0
            live.huawei_batteries_rated_capacity_wh = 30000.0
            await async_apply_battery_settings(
                sensor,
                cfg,
                live,
                rec,
                20.0,
                now=rec.start + timedelta(minutes=2),
                fully_fed_discharge_state=state,
            )

        assert verifier.await_args is not None
        assert verifier.await_args.kwargs["desired"] == 4000

    @pytest.mark.asyncio
    async def test_finished_slot_forces_zero_despite_latched_plan(self):
        """Slot completion is an immediate safety stop for a cached plan."""
        sensor = _make_sensor()
        cfg = self._cfg()
        live = self._live()
        live.huawei_batteries_working_mode = "fully_fed_to_grid"
        live.battery_current_capacity_kwh = 20.0
        rec = _make_planned_rec(
            "force_batteries_discharge",
            discharged_kwh=2.5,
            end_capacity_kwh=17.8,
        )
        state = FullyFedDischargeCapState()

        with patch(
            "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
            new_callable=AsyncMock,
            side_effect=_ok_apply_result,
        ) as verifier:
            await async_apply_battery_settings(
                sensor,
                cfg,
                live,
                rec,
                20.0,
                now=rec.start,
                fully_fed_discharge_state=state,
            )
            verifier.reset_mock()
            live.huawei_batteries_max_discharge_power_w = 10000.0

            await async_apply_battery_settings(
                sensor,
                cfg,
                live,
                rec,
                20.0,
                now=rec.end,
                fully_fed_discharge_state=state,
            )

        assert verifier.await_count == 1
        finished_call = verifier.await_args
        assert finished_call is not None
        assert finished_call.kwargs["desired"] == 0

    @pytest.mark.asyncio
    async def test_next_slot_replaces_prior_latched_cap(self):
        """A boundary callback applies the next slot instead of reusing the old cap."""
        sensor = _make_sensor()
        cfg = self._cfg()
        live = self._live()
        live.huawei_batteries_working_mode = "fully_fed_to_grid"
        first = _make_planned_rec(
            "force_batteries_discharge",
            discharged_kwh=0.458,
            end_capacity_kwh=18.442,
        )
        second = _make_planned_rec(
            "force_batteries_discharge",
            discharged_kwh=0.478,
            end_capacity_kwh=18.122,
        )
        second.start = first.end
        second.end = first.end + timedelta(minutes=15)
        state = FullyFedDischargeCapState()

        with patch(
            "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
            new_callable=AsyncMock,
            side_effect=_ok_apply_result,
        ) as verifier:
            await async_apply_battery_settings(
                sensor,
                cfg,
                live,
                first,
                20.0,
                now=first.start,
                fully_fed_discharge_state=state,
            )
            assert verifier.await_args is not None
            assert verifier.await_args.kwargs["desired"] == 1832

            verifier.reset_mock()
            live.huawei_batteries_max_discharge_power_w = 1832.0
            await async_apply_battery_settings(
                sensor,
                cfg,
                live,
                second,
                20.0,
                now=second.start,
                fully_fed_discharge_state=state,
            )

        assert verifier.await_args is not None
        assert verifier.await_args.kwargs["desired"] == 1912
        assert state.slot_start == second.start

    @pytest.mark.asyncio
    async def test_read_only_clears_latched_export_state(self):
        """Re-enabling writes cannot reuse a pre-read-only time sample."""
        sensor = _make_sensor()
        cfg = self._cfg()
        cfg.read_only = True
        live = self._live()
        rec = _make_planned_rec("force_batteries_discharge")
        state = FullyFedDischargeCapState(
            slot_start=rec.start,
            slot_end=rec.end,
            planned_discharge_kwh=rec.batteries_discharged_kwh,
            planned_end_capacity_kwh=rec.estimated_battery_capacity_kwh,
            max_discharge_power_w=10000,
            commanded_cap_w=1800,
        )

        with patch(
            "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
            new_callable=AsyncMock,
        ) as verifier:
            await async_apply_battery_settings(
                sensor,
                cfg,
                live,
                rec,
                20.0,
                fully_fed_discharge_state=state,
            )

        verifier.assert_not_awaited()
        assert state.commanded_cap_w is None

    @pytest.mark.asyncio
    async def test_pv_only_force_export_uses_zero_battery_cap(self):
        sensor = _make_sensor()
        cfg = self._cfg()
        live = self._live()
        rec = _make_planned_rec("force_export")

        with patch(
            "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
            new_callable=AsyncMock,
            side_effect=_ok_apply_result,
        ) as verifier:
            await async_apply_battery_settings(
                sensor, cfg, live, rec, 20.0, now=rec.start
            )

        assert [call.kwargs["desired"] for call in verifier.await_args_list] == [
            0,
            "fully_fed_to_grid",
        ]

    @pytest.mark.asyncio
    async def test_new_plan_at_endpoint_stops_battery_but_keeps_pv_export(self):
        sensor = _make_sensor()
        cfg = self._cfg()
        live = self._live()
        live.battery_current_capacity_kwh = 19.55
        rec = _make_planned_rec("force_batteries_discharge")

        with patch(
            "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
            new_callable=AsyncMock,
            side_effect=_ok_apply_result,
        ) as verifier:
            await async_apply_battery_settings(
                sensor, cfg, live, rec, 20.0, now=rec.start
            )

        assert [call.kwargs["desired"] for call in verifier.await_args_list] == [
            0,
            "fully_fed_to_grid",
        ]

    @pytest.mark.asyncio
    async def test_exit_fails_closed_then_restores_normal_cap(self):
        sensor = _make_sensor()
        cfg = self._cfg()
        live = self._live()
        live.huawei_batteries_working_mode = "fully_fed_to_grid"
        live.huawei_batteries_max_discharge_power_w = 1800.0
        live.huawei_batteries_excess_pv_use_in_tou = "charge"
        rec = _make_planned_rec("batteries_discharge_mode")

        with patch(
            "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
            new_callable=AsyncMock,
            side_effect=_ok_apply_result,
        ) as verifier:
            await async_apply_battery_settings(
                sensor, cfg, live, rec, 20.0, now=rec.start
            )

        assert [call.kwargs["desired"] for call in verifier.await_args_list] == [
            0,
            "maximise_self_consumption",
            10000,
        ]

    @pytest.mark.asyncio
    async def test_partial_msc_restores_plan_cap_after_fully_fed_exit(self):
        """Modeled import keeps MSC bounded to the solved battery allocation."""
        sensor = _make_sensor()
        cfg = self._cfg()
        live = self._live()
        live.huawei_batteries_working_mode = "fully_fed_to_grid"
        live.huawei_batteries_max_discharge_power_w = 1800.0
        live.huawei_batteries_excess_pv_use_in_tou = "charge"
        rec = _make_planned_rec(
            "batteries_discharge_mode",
            discharged_kwh=0.10,
        )
        rec.grid_export_kwh = 0.0
        rec.grid_import_kwh = 0.20

        with patch(
            "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
            new_callable=AsyncMock,
            side_effect=_ok_apply_result,
        ) as verifier:
            await async_apply_battery_settings(
                sensor, cfg, live, rec, 20.0, now=rec.start
            )

        assert [call.kwargs["desired"] for call in verifier.await_args_list] == [
            0,
            "maximise_self_consumption",
            400,
        ]

    @pytest.mark.asyncio
    async def test_rounding_level_import_does_not_throttle_msc(self):
        """A 0.001-kWh import residue is not a deliberate partial allocation."""
        sensor = _make_sensor()
        cfg = self._cfg()
        live = self._live()
        live.huawei_batteries_working_mode = "maximise_self_consumption"
        live.huawei_batteries_max_discharge_power_w = 400.0
        live.huawei_batteries_excess_pv_use_in_tou = "charge"
        rec = _make_planned_rec(
            "batteries_discharge_mode",
            discharged_kwh=0.10,
        )
        rec.grid_export_kwh = 0.0
        rec.grid_import_kwh = 0.001

        with patch(
            "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
            new_callable=AsyncMock,
            side_effect=_ok_apply_result,
        ) as verifier:
            await async_apply_battery_settings(
                sensor, cfg, live, rec, 20.0, now=rec.start
            )

        assert [call.kwargs["desired"] for call in verifier.await_args_list] == [10000]

    @pytest.mark.asyncio
    async def test_partial_msc_plan_cap_respects_hardware_maximum(self):
        """A large partial allocation cannot exceed the physical battery limit."""
        sensor = _make_sensor()
        cfg = self._cfg()
        live = self._live()
        live.huawei_batteries_max_discharge_power_w = 10000.0
        live.huawei_batteries_excess_pv_use_in_tou = "charge"
        rec = _make_planned_rec(
            "batteries_discharge_mode",
            discharged_kwh=3.0,
        )
        rec.grid_export_kwh = 0.0
        rec.grid_import_kwh = 0.20

        with patch(
            "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
            new_callable=AsyncMock,
            side_effect=_ok_apply_result,
        ) as verifier:
            await async_apply_battery_settings(
                sensor, cfg, live, rec, 20.0, now=rec.start
            )

        assert [call.kwargs["desired"] for call in verifier.await_args_list] == [
            "maximise_self_consumption",
        ]

    @pytest.mark.asyncio
    async def test_partial_msc_cap_also_bounds_unplanned_ev_session(self):
        """EV load protection cannot bypass a solved partial-discharge cap."""
        sensor = _make_sensor()
        cfg = self._cfg()
        live = self._live()
        live.huawei_batteries_working_mode = "maximise_self_consumption"
        live.huawei_batteries_max_discharge_power_w = 1800.0
        live.huawei_batteries_excess_pv_use_in_tou = "charge"
        live.ev.is_charging = True
        live.ev.power_w = 7000.0
        live.net_consumption_w = 1000.0
        rec = _make_planned_rec(
            "batteries_discharge_mode",
            discharged_kwh=0.10,
        )
        rec.grid_export_kwh = 0.0
        rec.grid_import_kwh = 0.20

        with patch(
            "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
            new_callable=AsyncMock,
            side_effect=_ok_apply_result,
        ) as verifier:
            await async_apply_battery_settings(
                sensor, cfg, live, rec, 5.0, now=rec.start
            )

        assert [call.kwargs["desired"] for call in verifier.await_args_list] == [400]

    @pytest.mark.asyncio
    async def test_label_only_msc_override_keeps_hardware_maximum(self):
        """A runtime schedule relabel without planned discharge is not capped."""
        sensor = _make_sensor()
        cfg = self._cfg()
        live = self._live()
        live.huawei_batteries_max_discharge_power_w = 10000.0
        live.huawei_batteries_excess_pv_use_in_tou = "charge"
        rec = _make_planned_rec(
            "batteries_discharge_mode",
            discharged_kwh=0.0,
        )
        rec.grid_export_kwh = 0.0
        rec.grid_import_kwh = 0.20

        with patch(
            "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
            new_callable=AsyncMock,
            side_effect=_ok_apply_result,
        ) as verifier:
            await async_apply_battery_settings(
                sensor, cfg, live, rec, 20.0, now=rec.start
            )

        assert [call.kwargs["desired"] for call in verifier.await_args_list] == [
            "maximise_self_consumption",
        ]

    @pytest.mark.asyncio
    async def test_failed_zero_cap_blocks_mode_exit_and_restore(self):
        from custom_components.hsem.utils.inverter_verify import ApplyResult

        sensor = _make_sensor()
        cfg = self._cfg()
        live = self._live()
        live.huawei_batteries_working_mode = "fully_fed_to_grid"
        live.huawei_batteries_max_discharge_power_w = 1800.0
        rec = _make_planned_rec("batteries_discharge_mode")
        failed = ApplyResult(
            entity_id="number.batteries_max_discharge",
            desired=0,
            actual=1800,
            status=ApplyStatus.FAILED,
            attempts=3,
        )

        with patch(
            "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
            new_callable=AsyncMock,
            return_value=failed,
        ) as verifier:
            summary = await async_apply_battery_settings(
                sensor, cfg, live, rec, 20.0, now=rec.start
            )

        assert verifier.await_count == 1
        assert verifier.await_args is not None
        assert verifier.await_args.kwargs["desired"] == 0
        assert summary.overall_status == ApplyStatus.FAILED

    @pytest.mark.asyncio
    async def test_active_legacy_force_is_stopped_once(self):
        sensor = _make_sensor()
        cfg = self._cfg()
        live = self._live()
        live.huawei_batteries_forcible_charge_state = "Discharging at 10000W until 5.0%"
        rec = _make_planned_rec("force_batteries_discharge")

        async def execute_writer(**kwargs):
            await kwargs["writer"]()
            return _ok_apply_result(**kwargs)

        with (
            patch(
                "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
                new_callable=AsyncMock,
                side_effect=execute_writer,
            ) as verifier,
            patch(
                "custom_components.hsem.custom_sensors.applier.async_stop_forcible_discharge",
                new_callable=AsyncMock,
            ) as stop_force,
            patch(
                "custom_components.hsem.custom_sensors.applier.async_set_number_value",
                new_callable=AsyncMock,
            ),
            patch(
                "custom_components.hsem.custom_sensors.applier.async_set_select_option",
                new_callable=AsyncMock,
            ),
        ):
            await async_apply_battery_settings(
                sensor, cfg, live, rec, 20.0, now=rec.start
            )

        stop_force.assert_awaited_once_with(sensor, "battery_device")
        assert verifier.await_args_list[0].kwargs["desired"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Working-mode sensor top-level gate (_async_apply_hardware_writes)
# ---------------------------------------------------------------------------


class TestWorkingModeSensorTopLevelGate:
    """Prove the outer gate in HSEMWorkingModeSensor._async_apply_hardware_writes.

    We import the gate function directly via the applier module to verify
    the plumbing without a full HA setup.
    """

    def _make_coordinator_data(
        self,
        *,
        read_only: bool = False,
        degraded_mode: DegradedMode = DegradedMode.OK,
    ) -> MagicMock:
        """Build a minimal CoordinatorData-like object for gate testing."""
        cfg = _make_cfg(read_only=read_only)
        live = _make_live(degraded_mode=degraded_mode)
        live.energi_data_service_export_price = 1.0  # type: ignore[attr-defined]  # mock attribute set in test

        data = MagicMock()
        data.cfg = cfg
        data.live = live
        data.hourly_recommendation = None
        data.batteries_schedules_remaining_capacity_needed = 0.0
        data.current_required_battery = 0.0
        data.apply_summary = None
        return data

    @pytest.mark.asyncio
    async def test_read_only_skips_both_appliers(self):
        """When read_only=True the applier functions must not be called at all."""
        data = self._make_coordinator_data(read_only=True)

        with (
            patch(
                "custom_components.hsem.custom_sensors.working_mode_sensor._LOGGER",
                new_callable=MagicMock,
            ) as mock_logger,
            patch(
                "custom_components.hsem.custom_sensors.working_mode_sensor.async_apply_inverter_power_control",
                new_callable=AsyncMock,
            ) as mock_inv,
            patch(
                "custom_components.hsem.custom_sensors.working_mode_sensor.async_apply_battery_settings",
                new_callable=AsyncMock,
            ) as mock_bat,
        ):
            # Import here to avoid circular import issues in test collection.
            from custom_components.hsem.custom_sensors.working_mode_sensor import (
                HSEMWorkingModeSensor,
            )

            sensor = MagicMock(spec=HSEMWorkingModeSensor)
            sensor.hass = MagicMock()

            await HSEMWorkingModeSensor._async_apply_hardware_writes(sensor, data)

        mock_inv.assert_not_called()
        mock_bat.assert_not_called()
        mock_logger.debug.assert_called_once_with(
            "Hardware writes SKIPPED — read_only=True"
        )

    @pytest.mark.asyncio
    async def test_error_mode_skips_both_appliers(self):
        """DegradedMode.Error must prevent both applier calls."""
        data = self._make_coordinator_data(
            read_only=False, degraded_mode=DegradedMode.Error
        )

        with (
            patch(
                "custom_components.hsem.custom_sensors.working_mode_sensor._LOGGER",
                new_callable=MagicMock,
            ) as mock_logger,
            patch(
                "custom_components.hsem.custom_sensors.working_mode_sensor.async_apply_inverter_power_control",
                new_callable=AsyncMock,
            ) as mock_inv,
            patch(
                "custom_components.hsem.custom_sensors.working_mode_sensor.async_apply_battery_settings",
                new_callable=AsyncMock,
            ) as mock_bat,
        ):
            from custom_components.hsem.custom_sensors.working_mode_sensor import (
                HSEMWorkingModeSensor,
            )

            sensor = MagicMock(spec=HSEMWorkingModeSensor)
            sensor.hass = MagicMock()

            await HSEMWorkingModeSensor._async_apply_hardware_writes(sensor, data)
            await HSEMWorkingModeSensor._async_apply_hardware_writes(sensor, data)

        mock_inv.assert_not_called()
        mock_bat.assert_not_called()
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_plan_clears_fully_fed_discharge_latch(self):
        """A plan gap invalidates a command based on an older slot time."""
        data = self._make_coordinator_data(
            read_only=False, degraded_mode=DegradedMode.OK
        )

        from custom_components.hsem.utils.inverter_verify import CycleApplySummary

        state = FullyFedDischargeCapState(
            commanded_cap_w=1800,
        )
        with (
            patch(_LOGGER_PATCH, new_callable=MagicMock),
            patch(
                "custom_components.hsem.custom_sensors.working_mode_sensor.async_apply_inverter_power_control",
                new_callable=AsyncMock,
                return_value=CycleApplySummary(),
            ) as mock_inv,
            patch(
                "custom_components.hsem.custom_sensors.working_mode_sensor.async_apply_battery_settings",
                new_callable=AsyncMock,
            ) as mock_bat,
        ):
            from custom_components.hsem.custom_sensors.working_mode_sensor import (
                HSEMWorkingModeSensor,
            )

            sensor = MagicMock(spec=HSEMWorkingModeSensor)
            sensor.hass = MagicMock()
            sensor._fully_fed_discharge_state = state

            await HSEMWorkingModeSensor._async_apply_hardware_writes(sensor, data)

        mock_inv.assert_awaited_once()
        mock_bat.assert_not_awaited()
        assert state.commanded_cap_w is None

    @pytest.mark.asyncio
    async def test_degraded_mode_calls_inverter_applier(self):
        """DegradedMode.Degraded must still call the inverter applier."""
        data = self._make_coordinator_data(
            read_only=False, degraded_mode=DegradedMode.Degraded
        )

        from custom_components.hsem.utils.inverter_verify import CycleApplySummary

        with (
            patch(_LOGGER_PATCH, new_callable=MagicMock),
            patch(
                "custom_components.hsem.custom_sensors.working_mode_sensor.async_apply_inverter_power_control",
                new_callable=AsyncMock,
                return_value=CycleApplySummary(),
            ) as mock_inv,
            patch(
                "custom_components.hsem.custom_sensors.working_mode_sensor.async_apply_battery_settings",
                new_callable=AsyncMock,
                return_value=CycleApplySummary(),
            ) as mock_bat,
        ):
            from custom_components.hsem.custom_sensors.working_mode_sensor import (
                HSEMWorkingModeSensor,
            )

            sensor = MagicMock(spec=HSEMWorkingModeSensor)
            sensor.hass = MagicMock()

            await HSEMWorkingModeSensor._async_apply_hardware_writes(sensor, data)

        mock_inv.assert_called_once()
        # Battery applier not called because hourly_rec is None.
        mock_bat.assert_not_called()

    @pytest.mark.asyncio
    async def test_normal_mode_calls_both_appliers(self):
        """OK mode with read_only=False must call both appliers when a rec exists."""
        data = self._make_coordinator_data(
            read_only=False, degraded_mode=DegradedMode.OK
        )
        # Provide a dummy hourly_recommendation so the battery applier gets called.
        data.hourly_recommendation = MagicMock()

        from custom_components.hsem.utils.inverter_verify import CycleApplySummary

        inv_summary = CycleApplySummary()
        # overall_status of an empty CycleApplySummary is SKIPPED, not FAILED,
        # so the battery gate passes.

        with (
            patch(_LOGGER_PATCH, new_callable=MagicMock),
            patch(
                "custom_components.hsem.custom_sensors.working_mode_sensor.async_apply_inverter_power_control",
                new_callable=AsyncMock,
                return_value=inv_summary,
            ) as mock_inv,
            patch(
                "custom_components.hsem.custom_sensors.working_mode_sensor.async_apply_battery_settings",
                new_callable=AsyncMock,
                return_value=CycleApplySummary(),
            ) as mock_bat,
            patch(
                "custom_components.hsem.custom_sensors.working_mode_sensor.resolve_current_recommendation"
            ),
        ):
            from custom_components.hsem.custom_sensors.working_mode_sensor import (
                HSEMWorkingModeSensor,
            )

            sensor = MagicMock(spec=HSEMWorkingModeSensor)
            sensor.hass = MagicMock()

            await HSEMWorkingModeSensor._async_apply_hardware_writes(sensor, data)

        mock_inv.assert_called_once()
        mock_bat.assert_called_once()

    @pytest.mark.asyncio
    async def test_unverified_huawei_write_blocks_powmr_applier(self):
        """PowMr control must not proceed after an unconfirmed Huawei transition."""
        data = self._make_coordinator_data(
            read_only=False,
            degraded_mode=DegradedMode.OK,
        )
        data.hourly_recommendation = MagicMock()

        from custom_components.hsem.utils.inverter_verify import (
            ApplyResult,
            CycleApplySummary,
        )

        inv_summary = CycleApplySummary(
            results=[
                ApplyResult(
                    entity_id="select.inverter_mode",
                    desired="target",
                    actual=None,
                    status=ApplyStatus.UNVERIFIED,
                )
            ]
        )

        with (
            patch(_LOGGER_PATCH, new_callable=MagicMock),
            patch(
                "custom_components.hsem.custom_sensors.working_mode_sensor.async_apply_inverter_power_control",
                new_callable=AsyncMock,
                return_value=inv_summary,
            ),
            patch(
                "custom_components.hsem.custom_sensors.working_mode_sensor.async_apply_battery_settings",
                new_callable=AsyncMock,
                return_value=CycleApplySummary(),
            ),
            patch(
                "custom_components.hsem.custom_sensors.working_mode_sensor.async_apply_secondary_storage",
                new_callable=AsyncMock,
            ) as mock_secondary,
            patch(
                "custom_components.hsem.custom_sensors.working_mode_sensor.resolve_current_recommendation"
            ),
        ):
            from custom_components.hsem.custom_sensors.working_mode_sensor import (
                HSEMWorkingModeSensor,
            )

            sensor = MagicMock(spec=HSEMWorkingModeSensor)
            sensor.hass = MagicMock()

            await HSEMWorkingModeSensor._async_apply_hardware_writes(sensor, data)

        mock_secondary.assert_not_awaited()


# ---------------------------------------------------------------------------
# hardware_writes_allowed — unit tests (full coverage)
# ---------------------------------------------------------------------------


class TestHardwareWritesAllowedDirectly:
    """Direct unit tests for :func:`hardware_writes_allowed` covering every mode."""

    def test_ok_mode_allows(self):
        from custom_components.hsem.utils.degraded_mode import hardware_writes_allowed

        assert hardware_writes_allowed(DegradedMode.OK) is True

    def test_degraded_mode_allows(self):
        from custom_components.hsem.utils.degraded_mode import hardware_writes_allowed

        assert hardware_writes_allowed(DegradedMode.Degraded) is True

    def test_error_mode_blocks(self):
        from custom_components.hsem.utils.degraded_mode import hardware_writes_allowed

        assert hardware_writes_allowed(DegradedMode.Error) is False
