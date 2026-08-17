"""Tests for async_will_remove_from_hass listener cleanup (Tasks 2 & 5).

Acceptance criteria
-------------------
1. ``HSEMAvgSensor.async_will_remove_from_hass`` cancels all registered
   ``async_track_*`` callbacks.
2. ``HSEMHouseConsumptionPowerSensor.async_will_remove_from_hass`` cancels all
   registered ``async_track_state_change_event`` callbacks.
3. ``HSEMDataUpdateCoordinator.async_teardown`` cancels all state-change listener
   callbacks collected from ``state_collector._register_listeners``.
4. No duplicate callbacks are registered after a simulated reload cycle
   (add → remove → add).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config_entry(**overrides: object) -> MagicMock:
    """Minimal mock ConfigEntry used by HSEMAvgSensor and HSEMHouseConsumption."""
    defaults = {
        "entry_id": "test_entry_id",
        "hsem_house_consumption_power": "sensor.house",
        "hsem_ev_charger_power": "sensor.ev",
        "hsem_house_power_includes_ev_charger_power": False,
    }
    defaults.update(overrides)

    mock = MagicMock()
    mock.entry_id = defaults["entry_id"]
    mock.options = defaults
    mock.data = {}
    return mock


def _make_hass() -> MagicMock:
    hass = MagicMock()
    hass.states.get.return_value = None
    return hass


# ---------------------------------------------------------------------------
# HSEMAvgSensor
# ---------------------------------------------------------------------------


class TestAvgSensorListenerCleanup:
    """HSEMAvgSensor must cancel all async_track_* callbacks on removal."""

    @pytest.mark.asyncio
    async def test_timer_unsub_called_on_removal(self):
        """The interval-timer unsub returned by async_track_time_interval is called."""
        from custom_components.hsem.custom_sensors.avg_sensor import HSEMAvgSensor

        cfg = _make_config_entry()
        sensor = HSEMAvgSensor(
            config_entry=cfg,
            hour_start=10,
            hour_end=11,
            avg=7,
            tracked_entity="sensor.utility",
            name="Test Avg",
            unique_id="test_avg_uid",
            entity_id="sensor.test_avg",
        )
        sensor.hass = _make_hass()

        timer_unsub = MagicMock()
        state_unsub = MagicMock()

        async def _fake_handle_update(event=None):
            """No-op stub replacing the real async handler during listener cleanup tests."""

        sensor._async_handle_update = _fake_handle_update

        with (
            patch(
                "custom_components.hsem.custom_sensors.avg_sensor"
                ".async_track_time_interval",
                return_value=timer_unsub,
            ),
            patch(
                "custom_components.hsem.custom_sensors.avg_sensor"
                ".async_track_state_change_event",
                return_value=state_unsub,
            ),
            patch.object(
                sensor,
                "async_get_last_state",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(sensor, "async_write_ha_state"),
        ):
            await sensor.async_added_to_hass()
            # Simulate entity-tracking registration
            await sensor._async_track_entities()

        # Now remove — both callbacks must be cancelled
        await sensor.async_will_remove_from_hass()

        timer_unsub.assert_called_once()
        state_unsub.assert_called_once()

    @pytest.mark.asyncio
    async def test_unsub_list_cleared_after_removal(self):
        """_unsub_callbacks must be empty after async_will_remove_from_hass."""
        from custom_components.hsem.custom_sensors.avg_sensor import HSEMAvgSensor

        cfg = _make_config_entry()
        sensor = HSEMAvgSensor(
            config_entry=cfg,
            hour_start=12,
            hour_end=13,
            avg=7,
            tracked_entity=None,
            name="Avg Clear Test",
            unique_id="avg_clear_uid",
            entity_id="sensor.avg_clear",
        )
        sensor.hass = _make_hass()

        fake_unsub = MagicMock()
        sensor._unsub_callbacks.append(fake_unsub)

        await sensor.async_will_remove_from_hass()

        assert sensor._unsub_callbacks == []

    @pytest.mark.asyncio
    async def test_no_duplicate_state_listener_after_reload(self) -> None:
        """Calling _async_track_entities twice must not register the entity twice.

        The idempotency guard (_tracked_entities set) must prevent duplicate
        registrations within a single lifecycle.
        """
        from custom_components.hsem.custom_sensors.avg_sensor import HSEMAvgSensor

        cfg = _make_config_entry()
        sensor = HSEMAvgSensor(
            config_entry=cfg,
            hour_start=8,
            hour_end=9,
            avg=7,
            tracked_entity="sensor.utility",
            name="Reload Test",
            unique_id="reload_uid",
            entity_id="sensor.reload_avg",
        )
        sensor.hass = _make_hass()

        register_calls: list[int] = []

        def _track_state(*args: Any, **kwargs: Any) -> MagicMock:
            register_calls.append(1)
            return MagicMock()

        with patch(
            "custom_components.hsem.custom_sensors.avg_sensor"
            ".async_track_state_change_event",
            side_effect=_track_state,
        ):
            # First call registers the entity
            await sensor._async_track_entities()
            # Second call within the same lifecycle — must be a no-op (idempotent)
            await sensor._async_track_entities()

        # Must register exactly once despite two calls
        assert len(register_calls) == 1, (
            f"Expected 1 registration (idempotent), got {len(register_calls)}"
        )


# ---------------------------------------------------------------------------
# HSEMHouseConsumptionPowerSensor
# ---------------------------------------------------------------------------


class TestHouseConsumptionListenerCleanup:
    """HSEMHouseConsumptionPowerSensor must cancel all async_track_* on removal."""

    def _make_sensor(self, cfg=None):
        from custom_components.hsem.custom_sensors.house_consumption_power_sensor import (
            HSEMHouseConsumptionPowerSensor,
        )

        if cfg is None:
            cfg = _make_config_entry()

        with patch(
            "custom_components.hsem.custom_sensors.house_consumption_power_sensor"
            ".get_config_value",
            side_effect=lambda entry, key: entry.options.get(key),
        ):
            sensor = HSEMHouseConsumptionPowerSensor(cfg, 10, 11, MagicMock())
        sensor.hass = _make_hass()
        return sensor

    @pytest.mark.asyncio
    async def test_state_unsubs_called_on_removal(self):
        """All state-change unsubs must be called when entity is removed."""
        sensor = self._make_sensor()

        house_unsub = MagicMock()
        ev_unsub = MagicMock()
        call_order = []

        def _track(hass, entities, callback):
            if "house" in entities[0]:
                call_order.append("house")
                return house_unsub
            call_order.append("ev")
            return ev_unsub

        with patch(
            "custom_components.hsem.custom_sensors.house_consumption_power_sensor"
            ".async_track_state_change_event",
            side_effect=_track,
        ):
            await sensor._async_track_entities()

        await sensor.async_will_remove_from_hass()

        house_unsub.assert_called_once()
        ev_unsub.assert_called_once()

    @pytest.mark.asyncio
    async def test_unsub_list_cleared_after_removal(self):
        """_unsub_callbacks is empty after async_will_remove_from_hass."""
        sensor = self._make_sensor()
        fake_unsub = MagicMock()
        sensor._unsub_callbacks.append(fake_unsub)

        await sensor.async_will_remove_from_hass()

        assert sensor._unsub_callbacks == []

    @pytest.mark.asyncio
    async def test_no_duplicate_listener_after_reload(self):
        """Calling _async_track_entities twice must not register the same entity twice."""
        sensor = self._make_sensor()
        register_count = {"house": 0}

        def _track(hass, entities, callback):
            if entities and "house" in entities[0]:
                register_count["house"] += 1
            return MagicMock()

        with patch(
            "custom_components.hsem.custom_sensors.house_consumption_power_sensor"
            ".async_track_state_change_event",
            side_effect=_track,
        ):
            await sensor._async_track_entities()
            await sensor._async_track_entities()  # second call — already tracked

        # Must have registered exactly once for house (idempotent)
        assert register_count["house"] == 1


# ---------------------------------------------------------------------------
# Coordinator state-change listener cleanup
# ---------------------------------------------------------------------------


class TestCoordinatorListenerCleanup:
    """HSEMDataUpdateCoordinator must cancel state-change unsubs on teardown."""

    def _make_coordinator(self):
        """Return a minimal coordinator with mocked HA internals."""
        from custom_components.hsem.coordinator import HSEMDataUpdateCoordinator

        hass = _make_hass()
        hass.data = {}

        coord = HSEMDataUpdateCoordinator.__new__(HSEMDataUpdateCoordinator)
        coord.hass = hass
        coord._listener_unsubs = []
        coord._hourly_timer_unsub = None
        coord._interval_timer_unsub = None
        coord._force_discharge_monitor_unsub = None
        coord._slot_boundary_timer_unsub = None
        coord._slot_boundary_interval_minutes = None
        coord._window_hysteresis_timer_unsub = None
        coord._window_hysteresis_expiry = None
        coord._window_hysteresis_expiry_replan_pending = False
        coord._tearing_down = False
        coord._secondary_storage_update_debounce_task = None
        coord._price_source_update_debounce_task = None
        coord._price_source_update_pending = False

        return coord

    @pytest.mark.asyncio
    async def test_teardown_cancels_listener_unsubs(self):
        """async_teardown must call every unsub in _listener_unsubs."""
        coord = self._make_coordinator()

        unsub_a = MagicMock()
        unsub_b = MagicMock()
        coord._listener_unsubs = [unsub_a, unsub_b]

        await coord.async_teardown()

        unsub_a.assert_called_once()
        unsub_b.assert_called_once()
        assert coord._listener_unsubs == []

    @pytest.mark.asyncio
    async def test_teardown_also_cancels_timer_unsubs(self):
        """Timers and state listeners are both cancelled during teardown."""
        coord = self._make_coordinator()

        timer_unsub = MagicMock()
        interval_unsub = MagicMock()
        monitor_unsub = MagicMock()
        boundary_unsub = MagicMock()
        hysteresis_unsub = MagicMock()
        listener_unsub = MagicMock()

        coord._hourly_timer_unsub = timer_unsub
        coord._interval_timer_unsub = interval_unsub
        coord._force_discharge_monitor_unsub = monitor_unsub
        coord._slot_boundary_timer_unsub = boundary_unsub
        coord._window_hysteresis_timer_unsub = hysteresis_unsub
        coord._listener_unsubs = [listener_unsub]

        await coord.async_teardown()

        timer_unsub.assert_called_once()
        interval_unsub.assert_called_once()
        monitor_unsub.assert_called_once()
        boundary_unsub.assert_called_once()
        hysteresis_unsub.assert_called_once()
        listener_unsub.assert_called_once()
        assert coord._hourly_timer_unsub is None
        assert coord._interval_timer_unsub is None
        assert coord._force_discharge_monitor_unsub is None
        assert coord._slot_boundary_timer_unsub is None
        assert coord._window_hysteresis_timer_unsub is None
        assert coord._listener_unsubs == []

    @pytest.mark.asyncio
    async def test_teardown_safe_with_empty_lists(self):
        """async_teardown must not raise when no listeners are registered."""
        coord = self._make_coordinator()
        coord._listener_unsubs = []

        # Must not raise
        await coord.async_teardown()

    @pytest.mark.asyncio
    async def test_teardown_cancels_secondary_storage_debounce(self):
        """A pending material-change debounce cannot outlive the coordinator."""
        coord = self._make_coordinator()
        task = MagicMock()
        task.done.return_value = False
        coord._secondary_storage_update_debounce_task = task

        await coord.async_teardown()

        task.cancel.assert_called_once()
        assert coord._secondary_storage_update_debounce_task is None

    @pytest.mark.asyncio
    async def test_teardown_cancels_price_source_debounce(self) -> None:
        """A pending price publication refresh cannot outlive teardown."""
        coord = self._make_coordinator()
        task = MagicMock()
        task.done.return_value = False
        coord._price_source_update_debounce_task = task

        await coord.async_teardown()

        task.cancel.assert_called_once()
        assert coord._price_source_update_debounce_task is None


class TestPriceSourceListenerRouting:
    """Price and PV forecast sources wake the durable refresh path."""

    @pytest.mark.asyncio
    async def test_all_distinct_price_sources_use_price_callback(self) -> None:
        from custom_components.hsem.custom_sensors.state_collector import (
            _register_listeners,
        )
        from custom_components.hsem.models.live_state import LiveState
        from custom_components.hsem.models.sensor_config import SensorConfig

        cfg = SensorConfig()
        cfg.import_electricity_price_sensor = "sensor.import"
        cfg.export_electricity_price_sensor = "sensor.export"
        cfg.import_electricity_price_entsoe_sensor = "sensor.entsoe_import"
        cfg.export_electricity_price_entsoe_sensor = "sensor.entsoe_export"
        cfg.import_electricity_price_forecast_sensor = "sensor.import_forecast"
        cfg.export_electricity_price_forecast_sensor = "sensor.export_forecast"
        cfg.solcast_pv_forecast_forecast_today = "sensor.pv_today"
        cfg.solcast_pv_forecast_forecast_tomorrow = "sensor.pv_tomorrow"
        sensor = MagicMock()
        tracked: set[str] = set()

        with patch(
            "custom_components.hsem.custom_sensors.state_collector"
            ".async_track_state_change_event",
            return_value=MagicMock(),
        ) as register:
            await _register_listeners(sensor, cfg, LiveState(), tracked)

        registered = {call.args[1][0]: call.args[2] for call in register.call_args_list}
        for entity_id in {
            "sensor.import",
            "sensor.export",
            "sensor.entsoe_import",
            "sensor.entsoe_export",
            "sensor.import_forecast",
            "sensor.export_forecast",
            "sensor.pv_today",
            "sensor.pv_tomorrow",
        }:
            assert registered[entity_id] == sensor._async_handle_price_source_change


class TestSecondaryStorageListenerRouting:
    """PowMr listeners use material filtering and omit raw net power."""

    @pytest.mark.asyncio
    async def test_secondary_inputs_use_filtered_callback(self) -> None:
        from custom_components.hsem.custom_sensors.state_collector import (
            _register_listeners,
        )
        from custom_components.hsem.models.live_state import LiveState
        from custom_components.hsem.models.sensor_config import SensorConfig

        cfg = SensorConfig()
        secondary = cfg.secondary_storage
        secondary.enabled = True
        secondary.control_enabled = True
        secondary.soc_entity = "sensor.powmr_soc"
        secondary.load_power_entity = "sensor.powmr_load_avg"
        secondary.battery_net_power_entity = "sensor.powmr_battery_net_power"
        secondary.output_source_priority_entity = "select.powmr_output"
        secondary.charger_source_priority_entity = "select.powmr_charger"
        secondary.max_charge_current_entity = "number.powmr_current"
        sensor = MagicMock()
        tracked: set[str] = set()

        with patch(
            "custom_components.hsem.custom_sensors.state_collector"
            ".async_track_state_change_event",
            return_value=MagicMock(),
        ) as register:
            await _register_listeners(sensor, cfg, LiveState(), tracked)

        registered = {call.args[1][0]: call.args[2] for call in register.call_args_list}
        assert "sensor.powmr_battery_net_power" not in registered
        assert registered["sensor.powmr_soc"] == (
            sensor._async_handle_secondary_storage_change
        )
        assert registered["sensor.powmr_load_avg"] == (
            sensor._async_handle_secondary_storage_change
        )
        assert registered["select.powmr_output"] == (
            sensor._async_handle_secondary_storage_change
        )

    @pytest.mark.asyncio
    async def test_control_entities_are_not_registered_when_control_disabled(
        self,
    ) -> None:
        from custom_components.hsem.custom_sensors.state_collector import (
            _register_listeners,
        )
        from custom_components.hsem.models.live_state import LiveState
        from custom_components.hsem.models.sensor_config import SensorConfig

        cfg = SensorConfig()
        secondary = cfg.secondary_storage
        secondary.enabled = True
        secondary.control_enabled = False
        secondary.soc_entity = "sensor.powmr_soc"
        secondary.load_power_entity = "sensor.powmr_load_avg"
        secondary.output_source_priority_entity = "select.powmr_output"
        secondary.charger_source_priority_entity = "select.powmr_charger"
        secondary.max_charge_current_entity = "number.powmr_current"
        sensor = MagicMock()
        tracked: set[str] = set()

        with patch(
            "custom_components.hsem.custom_sensors.state_collector"
            ".async_track_state_change_event",
            return_value=MagicMock(),
        ) as register:
            await _register_listeners(sensor, cfg, LiveState(), tracked)

        registered = {call.args[1][0] for call in register.call_args_list}
        assert "sensor.powmr_soc" in registered
        assert "sensor.powmr_load_avg" in registered
        assert "select.powmr_output" not in registered
        assert "select.powmr_charger" not in registered
        assert "number.powmr_current" not in registered
