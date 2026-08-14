"""Working-mode sensor for HSEM.

This entity subscribes to :class:`~custom_components.hsem.coordinator.HSEMDataUpdateCoordinator`
and is responsible for:

- Exposing the current working-mode recommendation as HA sensor state.
- Performing hardware writes (inverter + battery commands) after each coordinator
  cycle, gated by ``read_only`` and degraded-mode checks.
- Applying real-time slot overrides via :mod:`recommendation_resolver`.
- Exposing all planning data as ``extra_state_attributes``.

The heavy pipeline work (collect → populate → plan) has moved to the
coordinator.  This entity only reacts to coordinator pushes.
"""

from __future__ import annotations

import asyncio
from math import isfinite
from typing import Any, override

from homeassistant.components.sensor import SensorEntity
from homeassistant.components.sensor.const import SensorDeviceClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL

from custom_components.hsem.coordinator import (
    CoordinatorData,
    HSEMDataUpdateCoordinator,
)
from custom_components.hsem.custom_sensors.applier import (
    FullyFedDischargeCapState,
    async_apply_battery_settings,
    async_apply_inverter_power_control,
    desired_inverter_export_control,
)
from custom_components.hsem.custom_sensors.phase_charge_limiter import (
    build_phase_aware_charge_commands,
)
from custom_components.hsem.custom_sensors.recommendation_resolver import (
    resolve_current_recommendation,
)
from custom_components.hsem.custom_sensors.secondary_storage_applier import (
    async_apply_secondary_storage,
    build_secondary_write_plan,
)
from custom_components.hsem.entity import HSEMCoordinatorEntity, HSEMEntity
from custom_components.hsem.utils.degraded_mode import hardware_writes_allowed
from custom_components.hsem.utils.inverter_verify import (
    ApplyStatus,
    CycleApplySummary,
    WriteFailureBackoff,
)
from custom_components.hsem.utils.logger import HSEM_LOGGER as _LOGGER
from custom_components.hsem.utils.misc import calculate_recommended_threshold
from custom_components.hsem.utils.phase_power import phase_powers_valid
from custom_components.hsem.utils.recommendations import Recommendations
from custom_components.hsem.utils.sensornames.diagnostics import (
    get_working_mode_sensor_entity_id,
    get_working_mode_sensor_name,
    get_working_mode_sensor_unique_id,
)


def _intent_scaled_int(value: Any, scale: float) -> int | None:
    """Return a stable integer identity for a finite numeric command input."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not isfinite(number):
        return None
    return round(number * scale)


def _intent_energy_wh(value: Any) -> int | None:
    """Quantize planner kWh fields to their published whole-Wh precision."""
    return _intent_scaled_int(value, 1000.0)


class HSEMWorkingModeSensor(HSEMCoordinatorEntity, SensorEntity, HSEMEntity):
    """HA sensor entity for the HSEM working-mode recommendation.

    Subscribes to :class:`HSEMDataUpdateCoordinator` for shared state and
    performs hardware writes after each cycle.

    State
    -----
    The ``state`` property reflects the working-mode recommendation string
    for the current planning slot, or a sentinel value such as
    ``"missing_input_entities"`` when required sensors are unavailable.

    Attributes
    ----------
    ``extra_state_attributes`` returns the full planning snapshot including
    battery schedules, price data, EV state, and Solcast estimates.
    """

    _attr_icon = "mdi:chart-timeline-variant"
    _attr_has_entity_name = True
    _attr_translation_key = "working_mode"
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = [r.value for r in Recommendations]
    _unrecorded_attributes = frozenset({MATCH_ALL})

    def __init__(
        self,
        config_entry: ConfigEntry,
        coordinator: HSEMDataUpdateCoordinator,
    ) -> None:
        """Initialise the working-mode sensor.

        Args:
            config_entry: The HSEM config entry.
            coordinator: The shared :class:`HSEMDataUpdateCoordinator`.
        """
        HSEMCoordinatorEntity.__init__(self, coordinator)
        HSEMEntity.__init__(self, config_entry)

        self._config_entry = config_entry

        self._attr_unique_id = get_working_mode_sensor_unique_id(config_entry.entry_id)
        self.entity_id = get_working_mode_sensor_entity_id()
        self._name = get_working_mode_sensor_name()

        # Hardware writes are serialised through one worker. Coordinator
        # updates replace the pending snapshot instead of cancelling a write
        # during its verification delay.
        self._update_task: asyncio.Task | None = None
        self._pending_update_data: CoordinatorData | None = None
        self._active_hardware_intent: tuple[Any, ...] | None = None
        # Number real coordinator listener generations locally, so a refresh
        # can prove it produced a post-transaction snapshot.
        self._coordinator_update_generation = 0
        self._post_write_refresh_needed = False
        self._refresh_in_progress = False
        self._unloading = False
        self._write_failure_backoff = WriteFailureBackoff()
        self._fully_fed_discharge_state = FullyFedDischargeCapState()
        self._last_write_block_signature: tuple[str, tuple[str, ...]] | None = None
        # True only during the synchronous listener refresh that publishes a
        # completed apply summary.  It prevents that diagnostics-only refresh
        # from scheduling the same hardware work again.
        self._publishing_apply_summary = False

    # ------------------------------------------------------------------
    # HA entity properties
    # ------------------------------------------------------------------

    @property
    @override
    def name(self) -> str:
        """Return the display name."""
        return self._name

    @property
    @override
    def unique_id(self) -> str | None:
        """Return the unique ID."""
        return self._attr_unique_id

    @property  # type: ignore[misc]  # HA stub declares state as @final
    @override
    def state(self) -> str | None:
        """Return the working-mode recommendation for the current slot."""
        if self.coordinator.data is None:
            return None
        return self.coordinator.data.state

    @property
    @override
    def should_poll(self) -> bool:
        """No polling — driven by the coordinator."""
        return False

    @property
    @override
    def available(self) -> bool:
        """True once the coordinator has completed at least one successful cycle."""
        return (
            self.coordinator.last_update_success and self.coordinator.data is not None
        )

    @property
    @override
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return entity state attributes."""
        data: CoordinatorData | None = self.coordinator.data

        if data is None or data.live is None:
            return {
                "status": "wait",
                "description": "Waiting for coordinator to complete first cycle.",
                "last_updated": None,
                "next_update": None,
                "unique_id": self._attr_unique_id,
            }

        cfg = data.cfg
        live = data.live

        # Guard against a partially-initialised coordinator snapshot where cfg
        # was not yet populated (should not happen after first cycle but
        # prevents AttributeError on None during startup race).
        if cfg is None:
            return {
                "status": "wait",
                "description": "Waiting for coordinator configuration to be loaded.",
                "last_updated": None,
                "next_update": None,
                "unique_id": self._attr_unique_id,
            }

        if live.missing_entities:
            return {
                "status": "error",
                "description": (
                    "Some of the required input sensors from the config flow is missing "
                    "or not reporting a state yet. Check your configuration and make sure "
                    "input sensors are configured correctly."
                ),
                "missing_input_entities_list": live.missing_entities_list,
                "last_updated": data.last_updated,
                "next_update": data.next_update,
                "unique_id": self._attr_unique_id,
            }

        extended = {}
        if cfg.extended_attributes:
            extended = {
                "import_electricity_price_sensor_entity": cfg.import_electricity_price_sensor,
                "export_electricity_price_sensor_entity": cfg.export_electricity_price_sensor,
                "ev_charger_power_entity": cfg.ev.power_entity,
                "ev_charger_status_entity": cfg.ev.status_entity,
                "ev_soc_entity": cfg.ev.soc_entity,
                "ev_connected_entity": cfg.ev.connected_entity,
                "ev_second_charger_power_entity": cfg.ev_second.power_entity,
                "ev_second_charger_status_entity": cfg.ev_second.status_entity,
                "ev_second_soc_entity": cfg.ev_second.soc_entity,
                "ev_second_connected_entity": cfg.ev_second.connected_entity,
                "force_working_mode_entity": live.force_working_mode,
                "house_consumption_power_entity": cfg.house_consumption_power,
                "hsem_huawei_solar_batteries_end_of_discharge_soc_entity": cfg.huawei_solar_batteries_end_of_discharge_soc,
                "huawei_solar_batteries_grid_charge_cutoff_soc_entity": cfg.huawei_solar_batteries_grid_charge_cutoff_soc,
                "huawei_solar_batteries_maximum_charging_power_entity": cfg.huawei_solar_batteries_maximum_charging_power,
                "huawei_solar_batteries_maximum_discharging_power_entity": cfg.huawei_solar_batteries_maximum_discharging_power,
                "huawei_solar_batteries_rated_capacity_max_entity": cfg.huawei_solar_batteries_rated_capacity,
                "huawei_solar_batteries_state_of_capacity_entity": cfg.huawei_solar_batteries_state_of_capacity,
                "huawei_solar_batteries_tou_charging_and_discharging_periods_entity": cfg.huawei_solar_batteries_tou_charging_and_discharging_periods,
                "huawei_solar_batteries_working_mode_entity": cfg.huawei_solar_batteries_working_mode,
                "huawei_solar_device_id_batteries_id": cfg.huawei_solar_device_id_batteries,
                "huawei_solar_device_id_inverter_1_id": cfg.huawei_solar_device_id_inverter_1,
                "huawei_solar_device_id_inverter_2_id": cfg.huawei_solar_device_id_inverter_2,
                "huawei_solar_inverter_active_power_control_state_entity": cfg.huawei_solar_inverter_active_power_control,
                "next_update": data.next_update,
                "read_only": cfg.read_only,
                "solar_production_power_entity": cfg.solar_production_power,
                "solcast_pv_forecast_forecast_today_entity": cfg.solcast_pv_forecast_forecast_today,
                "solcast_pv_forecast_forecast_tomorrow_entity": cfg.solcast_pv_forecast_forecast_tomorrow,
                "unique_id": self._attr_unique_id,
                "update_interval": cfg.update_interval,
                "recommendation_interval_minutes": cfg.recommendation_interval_minutes,
                "recommendation_interval_length": cfg.recommendation_interval_length,
            }

        attributes = {
            "batteries_current_capacity": live.battery_current_capacity_kwh,
            "batteries_usable_capacity": live.battery_usable_capacity_kwh,
            "batteries_recommended_min_price_threshold": calculate_recommended_threshold(
                purchase_price=cfg.batteries_purchase_price,
                expected_cycles=cfg.batteries_expected_cycles,
                usable_capacity=live.battery_usable_capacity_kwh,
                capacity_loss_pct=cfg.batteries_capacity_loss_pct,
            ),
            "batteries_capacity_loss_pct": cfg.batteries_capacity_loss_pct,
            "export_electricity_price_state": live.export_electricity_price,
            "import_electricity_price_state": live.import_electricity_price,
            "export_electricity_min_price": cfg.export_electricity_min_price,
            "electricity_price_update_interval": cfg.electricity_price_update_interval,
            "ev_charger_power_state": live.ev.power_w,
            "ev_charger_status_state": live.ev.is_charging,
            "ev_soc_state": live.ev.soc_pct,
            "ev_soc_target_state": live.ev.soc_target_pct,
            "ev_connected_state": live.ev.is_connected,
            "ev_allow_charge_past_target_soc": cfg.ev.allow_charge_past_target_soc,
            "ev_past_target_confidence_factor": cfg.ev.past_target_confidence_factor,
            "ev_charger_max_discharge_power_state": live.ev.max_discharge_power_w,
            "ev_charger_force_max_discharge_power": live.ev.force_max_discharge_power,
            "ev_second_enabled": cfg.ev_second_enabled,
            "ev_second_charger_power_state": live.ev_second.power_w,
            "ev_second_charger_status_state": live.ev_second.is_charging,
            "ev_second_soc_state": live.ev_second.soc_pct,
            "ev_second_soc_target_state": live.ev_second.soc_target_pct,
            "ev_second_connected_state": live.ev_second.is_connected,
            "ev_second_allow_charge_past_target_soc": cfg.ev_second.allow_charge_past_target_soc,
            "ev_second_past_target_confidence_factor": cfg.ev_second.past_target_confidence_factor,
            "ev_second_charger_max_discharge_power_state": live.ev_second.max_discharge_power_w,
            "ev_second_charger_force_max_discharge_power": live.ev_second.force_max_discharge_power,
            "force_working_mode_state": live.force_working_mode_state,
            "hourly_recommendation": data.hourly_recommendation,
            "hourly_recommendations": data.hourly_recommendations,
            "house_consumption_energy_weight_14d": cfg.house_consumption_energy_weight_14d,
            "house_consumption_energy_weight_1d": cfg.house_consumption_energy_weight_1d,
            "house_consumption_energy_weight_3d": cfg.house_consumption_energy_weight_3d,
            "house_consumption_energy_weight_7d": cfg.house_consumption_energy_weight_7d,
            "house_consumption_power_state": live.house_consumption_power_w,
            "house_power_includes_ev_charger_power": cfg.house_power_includes_ev_charger_power,
            "batteries_schedules_remaining_capacity_needed": data.batteries_schedules_remaining_capacity_needed,
            "batteries_schedules": data.batteries_schedules,
            "huawei_solar_batteries_charging_cutoff_capacity_state": live.huawei_batteries_charging_cutoff_capacity_pct,
            "huawei_solar_batteries_grid_charge_cutoff_soc_state": live.huawei_batteries_grid_charge_cutoff_soc_pct,
            "huawei_solar_batteries_maximum_charging_power_state": live.huawei_batteries_max_charge_power_w,
            "huawei_solar_batteries_maximum_discharging_power_state": live.huawei_batteries_max_discharge_power_w,
            "huawei_solar_batteries_rated_capacity_max_state": live.huawei_batteries_rated_capacity_wh,
            "huawei_solar_batteries_rated_capacity_min_state": live.battery_rated_capacity_min_kwh,
            "huawei_solar_batteries_state_of_capacity_state": live.huawei_batteries_soc_pct,
            "huawei_solar_batteries_tou_charging_and_discharging_periods_periods": live.tou_periods.periods,
            "huawei_solar_batteries_tou_charging_and_discharging_periods_state": live.tou_periods.raw_state,
            "huawei_solar_batteries_working_mode_state": live.huawei_batteries_working_mode,
            "huawei_solar_inverter_active_power_control_state_state": live.huawei_inverter_active_power_control,
            "huawei_solar_batteries_excess_pv_energy_use_in_tou_state": live.huawei_batteries_excess_pv_use_in_tou,
            "solcast_pv_forecast_forecast_likelihood": cfg.solcast_pv_forecast_forecast_likelihood,
            "last_updated": data.last_updated,
            "net_consumption_with_ev": live.net_consumption_with_ev_w,
            "net_consumption": live.net_consumption_w,
            "solar_production_power_state": live.solar_production_power_w,
            "months_winter": cfg.months_winter,
            "months_summer": cfg.months_summer,
            "seasonal_fill_mode": cfg.seasonal_fill_mode,
            "batteries_enable_excess_export": cfg.batteries_enable_excess_export,
            "batteries_excess_export_discharge_buffer": cfg.batteries_excess_export_discharge_buffer,
            "main_fuse_amps": cfg.main_fuse_amps,
        }

        apply_summary = data.apply_summary
        status = {
            "status": "read_only" if cfg.read_only else "ok",
            "degraded_mode": live.degraded_mode.value,
            "hardware_writes_blocked": not hardware_writes_allowed(live.degraded_mode),
            "apply_status": (
                apply_summary.overall_status.value if apply_summary else None
            ),
            "apply_failed_entities": (
                apply_summary.failed_entities if apply_summary else []
            ),
            "data_quality": data.data_quality.as_dict(),
        }

        return dict(sorted({**attributes, **extended, **status}.items()))

    # ------------------------------------------------------------------
    # HA lifecycle
    # ------------------------------------------------------------------

    @override
    async def async_added_to_hass(self) -> None:
        """Register the listener and queue an initial hardware-write pass."""
        await super().async_added_to_hass()
        if self.coordinator.data is not None:
            self._resolve_current_state(self.coordinator.data)
            self.async_write_ha_state()
            self._queue_hardware_update(self.coordinator.data)

    @override
    async def async_will_remove_from_hass(self) -> None:
        """Cancel any pending background update task before unloading.

        This prevents a stale task from issuing inverter/battery writes after
        the config entry has been unloaded.
        """
        self._unloading = True
        self._pending_update_data = None
        self._post_write_refresh_needed = False
        self._write_failure_backoff.clear()
        self._fully_fed_discharge_state.reset()
        self._cancel_update_task()
        await super().async_will_remove_from_hass()

    def _cancel_update_task(self) -> None:
        """Cancel ``_update_task`` if it exists and has not yet completed.

        Cancellation is silent — ``asyncio.CancelledError`` propagates only
        inside the task itself, which guards the hardware-write path, so no
        inverter command can be issued after this point.
        """
        if self._update_task is not None and not self._update_task.done():
            self._update_task.cancel()

    def _on_update_task_done(self, task: asyncio.Task) -> None:
        """Log any unhandled exception from the coordinator-update task.

        Registered as a ``done_callback`` on ``_update_task`` so that
        uncaught exceptions inside ``_async_on_coordinator_update()`` are
        recorded without breaking the task lifecycle.

        Cancelled tasks are ignored because cancellation is expected on unload
        or when a hard no-write safety gate supersedes an active write.
        """
        is_current_task = self._update_task is task
        if is_current_task:
            self._update_task = None
            self._active_hardware_intent = None

        if task.cancelled():
            exc = None
        else:
            exc = task.exception()
            if exc is not None:
                _LOGGER.error(
                    "Unhandled exception in working-mode update task: %s",
                    exc,
                )

        if (
            is_current_task
            and not self._unloading
            and self._pending_update_data is not None
        ):
            self._start_update_task()

    # ------------------------------------------------------------------
    # Coordinator callback
    # ------------------------------------------------------------------

    @override
    def _handle_coordinator_update(self) -> None:
        """Publish state immediately, then coalesce hardware work in one worker."""
        data = self.coordinator.data
        if data is not None:
            self._resolve_current_state(data)
        self.async_write_ha_state()
        if self._publishing_apply_summary:
            return
        if data is not None:
            self._coordinator_update_generation += 1
            self._queue_hardware_update(data)

    @staticmethod
    def _resolve_current_state(data: CoordinatorData) -> None:
        """Apply real-time overrides before publishing or queuing hardware."""
        live = data.live
        hourly_rec = data.hourly_recommendation
        if live is None or hourly_rec is None:
            return

        resolve_current_recommendation(
            hourly_rec,
            live,
            data.batteries_schedules_remaining_capacity_needed,
        )
        data.state = hourly_rec.recommendation
        _LOGGER.debug(
            "Current hourly recommendation: state=%s  "
            "ev_charger_calculated_power=%dW  "
            "ev_second_charger_calculated_power=%dW  "
            "ev_total_planned_load_kwh=%.3f  "
            "ev_planned_load_kwh=%.3f  ev_accounted_load_kwh=%.3f",
            hourly_rec.recommendation,
            hourly_rec.ev_charger_calculated_power,
            hourly_rec.ev_second_charger_calculated_power,
            hourly_rec.ev_total_planned_load_kwh,
            hourly_rec.ev_planned_load_kwh,
            hourly_rec.ev_accounted_load_kwh,
        )

    def _queue_hardware_update(self, data: CoordinatorData) -> None:
        """Keep the latest snapshot and safely redirect the write worker."""
        if self._unloading:
            return

        self._pending_update_data = data
        next_intent = self._hardware_intent(data)
        if self._update_task is not None and not self._update_task.done():
            if self._hard_no_write_gate(data) and not self._refresh_in_progress:
                # Explicit safety gates may interrupt a transaction. Safe
                # intent changes wait for a coherent post-write refresh.
                self._post_write_refresh_needed = False
                self._update_task.cancel()
            elif (
                not self._refresh_in_progress
                and self._active_hardware_intent is not None
                and next_intent != self._active_hardware_intent
            ):
                # The new snapshot was collected against potentially partial
                # hardware state. Complete the active transaction and replace
                # this snapshot with a fresh coordinator generation.
                self._post_write_refresh_needed = True
            return

        self._start_update_task()

    @staticmethod
    def _hard_no_write_gate(data: CoordinatorData) -> bool:
        """Return whether a snapshot deliberately forbids hardware writes."""
        cfg = data.cfg
        live = data.live
        return (
            cfg is None
            or live is None
            or cfg.read_only
            or not hardware_writes_allowed(live.degraded_mode)
        )

    def _start_update_task(self) -> None:
        """Create the sole hardware worker when pending data exists."""
        if self._unloading or self._pending_update_data is None:
            return
        self._update_task = self.hass.async_create_task(
            self._async_on_coordinator_update(),
            name="hsem_working_mode_update",
        )
        self._update_task.add_done_callback(self._on_update_task_done)

    @staticmethod
    def _hardware_intent(data: CoordinatorData) -> tuple[Any, ...]:
        """Return fields whose change makes an in-flight command obsolete."""
        cfg = data.cfg
        live = data.live
        rec = data.hourly_recommendation
        if cfg is None or live is None:
            return (None, data.state)
        phase_commands = (
            build_phase_aware_charge_commands(cfg, live, rec)
            if rec is not None
            else None
        )
        effective_rec = (
            phase_commands.recommendation if phase_commands is not None else rec
        )
        secondary_operations = (
            build_secondary_write_plan(cfg, live, effective_rec)
            if effective_rec is not None
            and cfg.secondary_storage.enabled
            and cfg.secondary_storage.control_enabled
            else []
        )
        secondary_intent = tuple(
            (
                operation.kind,
                operation.entity_id,
                (
                    _intent_scaled_int(operation.desired, 1000.0)
                    if isinstance(operation.desired, (int, float))
                    else operation.desired
                ),
            )
            for operation in secondary_operations
        )
        phase_intent = (
            (
                _intent_scaled_int(phase_commands.primary_grid_charge_power_w, 0.01),
                getattr(effective_rec, "secondary_storage_mode", None),
                _intent_scaled_int(
                    getattr(effective_rec, "secondary_storage_charge_current_a", None),
                    1000.0,
                ),
            )
            if phase_commands is not None
            else None
        )
        recommendation = getattr(rec, "recommendation", data.state)
        battery_capacity_intent = (
            None
            if recommendation == Recommendations.ForceBatteriesDischarge.value
            else _intent_energy_wh(live.battery_current_capacity_kwh)
        )
        return (
            getattr(cfg, "read_only", None),
            getattr(getattr(live, "degraded_mode", None), "value", None),
            (
                cfg.huawei_solar_device_id_inverter_1,
                cfg.huawei_solar_device_id_inverter_2,
                cfg.huawei_solar_inverter_active_power_control,
            ),
            desired_inverter_export_control(cfg, live, rec),
            getattr(cfg, "batteries_wait_mode_behavior", None),
            getattr(cfg, "phase_aware_charging_enabled", None),
            _intent_scaled_int(getattr(cfg, "main_fuse_amps", None), 1000.0),
            _intent_scaled_int(getattr(cfg, "main_fuse_phases", None), 1.0),
            (
                cfg.huawei_solar_device_id_batteries,
                cfg.huawei_solar_batteries_working_mode,
                cfg.huawei_solar_batteries_grid_charge_maximum_power,
                cfg.huawei_solar_batteries_maximum_discharging_power,
                cfg.huawei_solar_batteries_tou_charging_and_discharging_periods,
                cfg.huawei_solar_batteries_excess_pv_energy_use_in_tou,
                cfg.huawei_solar_batteries_forcible_charge,
            ),
            getattr(cfg.secondary_storage, "enabled", None),
            getattr(cfg.secondary_storage, "control_enabled", None),
            (
                cfg.secondary_storage.output_source_priority_entity,
                cfg.secondary_storage.charger_source_priority_entity,
                cfg.secondary_storage.max_charge_current_entity,
            ),
            getattr(rec, "start", None),
            getattr(rec, "end", None),
            recommendation,
            getattr(rec, "primary_battery_hold", False),
            _intent_energy_wh(getattr(rec, "batteries_charged_kwh", None)),
            _intent_energy_wh(getattr(rec, "batteries_discharged_kwh", None)),
            _intent_energy_wh(getattr(rec, "grid_import_kwh", None)),
            _intent_energy_wh(getattr(rec, "grid_export_kwh", None)),
            _intent_energy_wh(getattr(rec, "estimated_battery_capacity_kwh", None)),
            _intent_energy_wh(getattr(rec, "avg_house_consumption_kwh", None)),
            _intent_energy_wh(
                getattr(rec, "historical_avg_house_consumption_kwh", None)
            ),
            _intent_energy_wh(getattr(rec, "avg_house_consumption_1d_kwh", None)),
            _intent_energy_wh(getattr(rec, "avg_house_consumption_3d_kwh", None)),
            _intent_energy_wh(getattr(rec, "avg_house_consumption_7d_kwh", None)),
            _intent_energy_wh(getattr(rec, "avg_house_consumption_14d_kwh", None)),
            getattr(rec, "secondary_storage_mode", None),
            _intent_scaled_int(
                getattr(rec, "secondary_storage_charge_current_a", None), 1000.0
            ),
            _intent_energy_wh(data.current_required_battery),
            phase_intent,
            secondary_intent,
            (
                live.any_ev_charging,
                live.ev.force_max_discharge_power,
                live.ev_second.force_max_discharge_power,
                _intent_scaled_int(live.ev.max_discharge_power_w, 1.0),
                _intent_scaled_int(live.ev_second.max_discharge_power_w, 1.0),
                _intent_scaled_int(live.ev.power_w, 0.01),
                _intent_scaled_int(live.ev_second.power_w, 0.01),
                _intent_scaled_int(live.net_consumption_w, 0.01),
            ),
            battery_capacity_intent,
            _intent_scaled_int(live.huawei_batteries_rated_capacity_wh, 1.0),
            _intent_scaled_int(live.huawei_batteries_max_charge_power_w, 0.01),
            _intent_scaled_int(live.huawei_batteries_charge_discharge_power_w, 0.01),
            live.huawei_batteries_working_mode,
            live.huawei_batteries_forcible_charge_state,
        )

    async def _async_on_coordinator_update(self) -> None:
        """Drain the latest coordinator snapshots through one write worker."""
        try:
            while not self._unloading and self._pending_update_data is not None:
                data = self._pending_update_data
                self._pending_update_data = None
                intent = self._hardware_intent(data)
                self._active_hardware_intent = intent

                await self._async_apply_hardware_writes(data)

                if self._unloading:
                    return

                current = self.coordinator.data
                if (
                    current is not None
                    and data.apply_summary is not None
                    and self._hardware_intent(current) == intent
                ):
                    self._publishing_apply_summary = True
                    try:
                        self.coordinator.async_publish_apply_summary(data.apply_summary)
                    finally:
                        self._publishing_apply_summary = False

                if self._post_write_refresh_needed:
                    if not await self._async_refresh_after_superseded_write():
                        return
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._post_write_refresh_needed:
                self._pending_update_data = None
                self._post_write_refresh_needed = False
            _LOGGER.exception("Hardware-write task failed during coordinator update")

    async def _async_refresh_after_superseded_write(self) -> bool:
        """Replace a mid-write snapshot with one collected after completion.

        Return ``True`` only when the refresh publishes a newer successful
        listener generation. A failed or silent refresh drops the stale
        snapshot and stops this worker; a later external update can recover.
        """
        generation_before_refresh = self._coordinator_update_generation
        self._pending_update_data = None
        self._post_write_refresh_needed = False
        self._refresh_in_progress = True
        try:
            await self.coordinator.async_request_refresh()
        except asyncio.CancelledError:
            raise
        except Exception:
            self._pending_update_data = None
            _LOGGER.exception(
                "Post-write coordinator refresh failed; stale hardware intent dropped"
            )
            return False
        finally:
            self._refresh_in_progress = False

        if self._unloading:
            self._pending_update_data = None
            return False

        generation_advanced = (
            self._coordinator_update_generation > generation_before_refresh
        )
        fresh_data = self.coordinator.data
        if (
            not generation_advanced
            or not self.coordinator.last_update_success
            or fresh_data is None
        ):
            self._pending_update_data = None
            _LOGGER.warning(
                "Post-write coordinator refresh produced no successful new "
                "generation; stale hardware intent dropped"
            )
            return False

        # The refresh callback normally queued this object already. Assign it
        # explicitly so the worker always drains the coordinator's newest
        # coherent snapshot, never an intermediate listener snapshot.
        self._pending_update_data = fresh_data
        return True

    async def _async_apply_hardware_writes(self, data: CoordinatorData | None) -> None:
        """Perform inverter and battery hardware writes for the current slot.

        Writes are skipped when:
        - ``data`` is ``None``,
        - ``cfg.read_only`` is ``True``, or
        - the degraded mode is ``Error`` (critical entities missing).

        A real-time slot override is applied via :func:`resolve_current_recommendation`
        before issuing the hardware commands.

        Args:
            data: The latest :class:`CoordinatorData` snapshot from the coordinator,
                or ``None`` when the coordinator has no data yet.
        """
        if data is None:
            return

        cfg = data.cfg
        live = data.live

        if cfg is None or live is None:
            return

        hourly_rec = data.hourly_recommendation
        fully_fed_discharge_state = getattr(self, "_fully_fed_discharge_state", None)
        if not isinstance(fully_fed_discharge_state, FullyFedDischargeCapState):
            fully_fed_discharge_state = None
        if hourly_rec is None and fully_fed_discharge_state is not None:
            # Losing the selected recommendation invalidates the plan-derived
            # cap. If a plan later reappears, it must be accepted afresh.
            fully_fed_discharge_state.reset()

        # Gate hardware writes on read_only and degraded mode.
        writes_safe = hardware_writes_allowed(live.degraded_mode)
        combined_summary = CycleApplySummary()
        if cfg.read_only:
            if fully_fed_discharge_state is not None:
                fully_fed_discharge_state.reset()
            self._last_write_block_signature = None
            _LOGGER.debug("Hardware writes SKIPPED — read_only=True")
        elif not writes_safe:
            if fully_fed_discharge_state is not None:
                fully_fed_discharge_state.reset()
            block_signature = (
                live.degraded_mode.value,
                tuple(live.missing_entities_list),
            )
            if getattr(self, "_last_write_block_signature", None) != block_signature:
                _LOGGER.warning(
                    "Hardware writes BLOCKED — degraded mode: %s. Missing: %s",
                    live.degraded_mode.value,
                    live.missing_entities_list,
                )
                self._last_write_block_signature = block_signature
        else:
            self._last_write_block_signature = None
            inv_summary = await async_apply_inverter_power_control(
                self, cfg, live, hourly_rec
            )
            combined_summary.results.extend(inv_summary.results)

            phase_commands = (
                build_phase_aware_charge_commands(cfg, live, hourly_rec)
                if hourly_rec is not None
                else None
            )
            if phase_commands is not None and phase_commands.limits is not None:
                limits = phase_commands.limits
                live_phase_power_w = live.grid_phase_power_w
                measured_phase_power_w = (
                    live_phase_power_w
                    if phase_powers_valid(live_phase_power_w)
                    else (0.0, 0.0, 0.0)
                )
                _LOGGER.debug(
                    "Phase-aware charge limit: measured=%sW base=%sW "
                    "Huawei=%.0fW PowMr=%.0fA predicted=%sW target=%dA",
                    tuple(round(value) for value in measured_phase_power_w),
                    tuple(round(value) for value in limits.base_phase_power_w),
                    limits.primary_charge_power_w,
                    limits.secondary_charge_current_a,
                    tuple(round(value) for value in limits.predicted_phase_power_w),
                    cfg.main_fuse_amps,
                )

            # Block battery writes if the inverter write already failed.
            if (
                inv_summary.overall_status in {ApplyStatus.OK, ApplyStatus.SKIPPED}
                and hourly_rec is not None
            ):
                bat_summary = await async_apply_battery_settings(
                    self,
                    cfg,
                    live,
                    phase_commands.recommendation
                    if phase_commands is not None
                    else hourly_rec,
                    data.current_required_battery,
                    grid_charge_power_limit_w=(
                        phase_commands.primary_grid_charge_power_w
                        if phase_commands is not None
                        else None
                    ),
                    fully_fed_discharge_state=fully_fed_discharge_state,
                )
                combined_summary.results.extend(bat_summary.results)

            if (
                combined_summary.overall_status in {ApplyStatus.OK, ApplyStatus.SKIPPED}
                and hourly_rec is not None
            ):
                secondary_summary = await async_apply_secondary_storage(
                    self,
                    cfg,
                    live,
                    phase_commands.recommendation
                    if phase_commands is not None
                    else hourly_rec,
                )
                combined_summary.results.extend(secondary_summary.results)

        # Persist the apply summary onto the coordinator data so the status
        # sensor and extra_state_attributes can surface it to HA.
        data.apply_summary = combined_summary

    # ------------------------------------------------------------------
    # Legacy compatibility
    # ------------------------------------------------------------------

    @override
    async def async_update(self, event: Any | None = None) -> None:
        """Manually request a coordinator refresh.

        Kept for backwards compatibility with any callers that invoke
        ``async_update`` directly (e.g. HA service calls).
        """
        await self.coordinator.async_request_refresh()

    async def async_options_updated(self, config_entry: ConfigEntry) -> None:
        """Handle options update from configuration change.

        Delegates to the coordinator so all entities benefit simultaneously.
        """
        await self.coordinator.async_options_updated()
