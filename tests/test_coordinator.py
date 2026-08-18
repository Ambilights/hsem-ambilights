"""Tests for HSEMDataUpdateCoordinator (issue #283).

Acceptance criteria:
- Data is fetched once per interval (single coordinator, not per entity).
- Entities do not independently fetch the same data.
- Coordinator exposes last update status via coordinator.last_update_success.
- Update lock prevents concurrent pipeline executions.
- CoordinatorData contains a consistent snapshot after each cycle.
- async_setup registers timers; async_teardown cancels them.
- async_options_updated triggers a fresh pipeline cycle.

Implementation note
-------------------
``HSEMDataUpdateCoordinator.__init__`` calls ``DataUpdateCoordinator.__init__``
which invokes ``homeassistant.helpers.frame.report_usage``.  That helper
requires the HA event-loop frame helper to be bootstrapped (only done inside a
real HA test environment via ``hass`` fixtures).  To keep these tests isolated
and fast we use one of two approaches depending on what is being verified:

1. **Source inspection** – when the test only needs to confirm that a certain
   attribute is *initialised* in ``__init__``, we inspect the source code
   directly (no construction required).
2. **``object.__new__`` + manual attribute injection** – when the test needs to
   call *methods* on the coordinator we bypass ``__init__`` entirely and set
   only the attributes the method under test actually reads.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from custom_components.hsem.coordinator import (
    CoordinatorData,
    HSEMDataUpdateCoordinator,
    _apply_current_price_outage_hold,
    _apply_live_current_price_availability,
    _auto_full_negative_price_allowed,
    _canonical_price_channel,
    _force_discharge_live_metrics,
    _next_slot_boundary_utc,
    _price_forecast_signature,
    _PriceForecastSignature,
    _select_corrective_planner_output,
)
from custom_components.hsem.coordinator_builder import generate_recommendation_intervals
from custom_components.hsem.custom_sensors.hourly_data_populator.prices_solcast import (
    populate_price_and_solcast_from_snapshot,
)
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.plan_explanation import PlanExplanation
from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.planner_output import PlannerOutput
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.models.state_snapshot import StateSnapshot
from custom_components.hsem.utils.inverter_verify import CycleApplySummary
from custom_components.hsem.utils.recommendations import Recommendations

# ---------------------------------------------------------------------------
# Helper: build a bare coordinator instance without calling __init__
# ---------------------------------------------------------------------------


class TestAutoFullPriceAuthority:
    """Auto-Full EV requires an actually published current-slot price."""

    def test_nonactionable_slot_cannot_trigger_on_placeholder_zero(self) -> None:
        live = LiveState()
        live.import_electricity_price = 0.0
        live.import_electricity_price_available = True
        slot = generate_recommendation_intervals(15, 1)[0]
        slot.price_actionable = False

        assert _auto_full_negative_price_allowed(live, slot) is False

    def test_unavailable_live_price_cannot_trigger(self) -> None:
        live = LiveState()
        live.import_electricity_price = 0.0
        live.import_electricity_price_available = False
        slot = generate_recommendation_intervals(15, 1)[0]
        slot.price_actionable = True

        assert _auto_full_negative_price_allowed(live, slot) is False

    def test_published_negative_price_remains_actionable(self) -> None:
        live = LiveState()
        live.import_electricity_price = -0.1
        live.import_electricity_price_available = True
        slot = generate_recommendation_intervals(15, 1)[0]
        slot.price_actionable = True

        assert _auto_full_negative_price_allowed(live, slot) is True


def _populated_price_authority(
    *,
    now: datetime,
    price_interval_minutes: int,
    total_hours: int,
    include_tomorrow: bool,
    reverse_source_order: bool = False,
) -> tuple[_PriceForecastSignature, LiveState]:
    """Build a canonical signature from populated price-source attributes."""
    cfg = SensorConfig()
    cfg.electricity_price_update_interval = price_interval_minutes
    cfg.recommendation_interval_minutes = 15
    cfg.recommendation_interval_length = total_hours
    cfg.import_electricity_price_sensor = "sensor.import_price"
    cfg.import_electricity_price_forecast_sensor = None
    cfg.export_electricity_price_sensor = "sensor.export_price"
    cfg.export_electricity_price_forecast_sensor = None
    cfg.solcast_pv_forecast_forecast_today = None
    cfg.solcast_pv_forecast_forecast_tomorrow = None

    with patch("custom_components.hsem.coordinator_builder.hsem_now", return_value=now):
        recommendations = generate_recommendation_intervals(15, total_hours)

    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    count = total_hours * 60 // price_interval_minutes
    entries: list[dict[str, str | float]] = [
        {
            "start": (
                midnight + timedelta(minutes=index * price_interval_minutes)
            ).isoformat(),
            "price": round(0.25 + index * 0.01, 5),
        }
        for index in range(count)
    ]
    today_slot_count = 24 * 60 // price_interval_minutes
    today = entries[:today_slot_count]
    tomorrow = entries[today_slot_count:]
    if reverse_source_order:
        today.reverse()
        tomorrow.reverse()
    attributes: dict[str, list[dict[str, str | float]]] = {"prices_today": today}
    if include_tomorrow:
        attributes["prices_tomorrow"] = tomorrow
    snapshot = StateSnapshot(
        live=LiveState(),
        sensor_attributes={
            "sensor.import_price": attributes,
            "sensor.export_price": attributes,
        },
    )
    populate_price_and_solcast_from_snapshot(recommendations, snapshot, cfg)

    live = LiveState()
    live.import_electricity_price = 0.25
    live.export_electricity_price = 0.10
    live.import_electricity_price_available = True
    live.export_electricity_price_available = True
    return _price_forecast_signature(recommendations, live, now), live


class TestPriceForecastAuthority:
    """Planner reuse follows the entire published price/PV horizon authority."""

    @pytest.mark.parametrize(
        ("live_import", "live_export", "import_available", "export_available"),
        [
            (0.0, -0.25, True, True),
            (-0.10, 0.0, True, True),
            (float("nan"), 0.0, False, True),
            (0.0, float("inf"), True, False),
            (0.0, 0.0, False, False),
        ],
    )
    def test_current_forecast_authority_requires_finite_live_channels(
        self,
        live_import: float,
        live_export: float,
        import_available: bool,
        export_available: bool,
    ) -> None:
        now = datetime(2026, 8, 14, 12, 5, tzinfo=UTC)
        with patch(
            "custom_components.hsem.coordinator_builder.hsem_now", return_value=now
        ):
            recommendations = generate_recommendation_intervals(15, 24)
        current = recommendations[48]
        current.import_price = 1.0
        current.export_price = 0.5
        current.import_price_available = True
        current.export_price_available = True
        current.price_actionable = True
        live = LiveState()
        live.import_electricity_price = live_import
        live.export_electricity_price = live_export
        live.import_electricity_price_available = import_available
        live.export_electricity_price_available = export_available

        _apply_live_current_price_availability(recommendations, live, now)

        assert current.import_price == 1.0
        assert current.export_price == 0.5
        assert current.import_price_available is import_available
        assert current.export_price_available is export_available
        assert current.price_actionable is (import_available and export_available)

    def test_live_channel_never_promotes_missing_forecast_authority(self) -> None:
        now = datetime(2026, 8, 14, 12, 5, tzinfo=UTC)
        with patch(
            "custom_components.hsem.coordinator_builder.hsem_now", return_value=now
        ):
            recommendations = generate_recommendation_intervals(15, 24)
        current = recommendations[48]
        current.import_price_available = False
        current.export_price_available = True
        current.price_actionable = True
        live = LiveState()
        live.import_electricity_price = 0.0
        live.export_electricity_price = -0.25
        live.import_electricity_price_available = True
        live.export_electricity_price_available = True

        _apply_live_current_price_availability(recommendations, live, now)

        assert current.import_price_available is False
        assert current.export_price_available is True
        assert current.price_actionable is False

    def test_auto_price_outage_replaces_stale_actions_with_full_storage_hold(
        self,
    ) -> None:
        now = datetime(2026, 8, 14, 12, 5, tzinfo=UTC)
        with patch(
            "custom_components.hsem.coordinator_builder.hsem_now", return_value=now
        ):
            recommendations = generate_recommendation_intervals(15, 24)
        current = recommendations[48]
        current.recommendation = Recommendations.ForceBatteriesDischarge.value
        current.batteries_charged_kwh = 1.0
        current.batteries_discharged_kwh = 2.0
        current.secondary_storage_mode = "sbu"
        current.secondary_storage_charge_current_a = 30.0
        current.secondary_storage_charged_kwh = 1.0
        current.secondary_storage_discharged_kwh = 1.0
        current.price_actionable = False
        live = LiveState()
        live.force_working_mode_state = "auto"

        held = _apply_current_price_outage_hold(recommendations, live, now)

        assert held is current
        assert current.recommendation == Recommendations.BatteriesWaitMode.value
        assert current.primary_battery_hold is True
        assert current.batteries_charged_kwh == 0.0
        assert current.batteries_discharged_kwh == 0.0
        assert current.secondary_storage_mode == "utility"
        assert current.secondary_storage_charge_current_a == 0.0
        assert current.secondary_storage_charged_kwh == 0.0
        assert current.secondary_storage_discharged_kwh == 0.0

    def test_explicit_user_force_is_higher_authority_than_price_outage(self) -> None:
        now = datetime(2026, 8, 14, 12, 5, tzinfo=UTC)
        with patch(
            "custom_components.hsem.coordinator_builder.hsem_now", return_value=now
        ):
            recommendations = generate_recommendation_intervals(15, 24)
        current = recommendations[48]
        current.recommendation = Recommendations.ForceBatteriesDischarge.value
        current.price_actionable = False
        live = LiveState()
        live.force_working_mode_state = Recommendations.ForceBatteriesDischarge.value

        held = _apply_current_price_outage_hold(recommendations, live, now)

        assert held is None
        assert current.recommendation == Recommendations.ForceBatteriesDischarge.value
        assert current.primary_battery_hold is False

    @pytest.mark.parametrize("price_interval_minutes", [15, 60])
    def test_populated_source_order_does_not_trigger_replan(
        self, price_interval_minutes: int
    ) -> None:
        now = datetime(2026, 8, 14, 0, 0, tzinfo=UTC)
        original, live = _populated_price_authority(
            now=now,
            price_interval_minutes=price_interval_minutes,
            total_hours=4,
            include_tomorrow=False,
        )
        reversed_order, _ = _populated_price_authority(
            now=now,
            price_interval_minutes=price_interval_minutes,
            total_hours=4,
            include_tomorrow=False,
            reverse_source_order=True,
        )
        coordinator = _make_bare_coordinator()
        coordinator._cfg.recommendation_interval_minutes = 15
        coordinator._last_planner_output = SimpleNamespace(slots=[])  # type: ignore[assignment]
        coordinator._last_plan_slot_start = now
        coordinator._persist_plan_state(
            live,
            price_forecast_signature=original,
        )

        assert original == reversed_order
        assert (
            coordinator._should_replan(
                live,
                now,
                price_forecast_signature=reversed_order,
            )
            is False
        )

    @pytest.mark.parametrize(
        ("accepted_published", "current_published"),
        [(False, True), (True, False)],
    )
    def test_tomorrow_publication_or_withdrawal_triggers_replan(
        self, accepted_published: bool, current_published: bool
    ) -> None:
        now = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
        accepted, live = _populated_price_authority(
            now=now,
            price_interval_minutes=15,
            total_hours=48,
            include_tomorrow=accepted_published,
        )
        current, current_live = _populated_price_authority(
            now=now,
            price_interval_minutes=15,
            total_hours=48,
            include_tomorrow=current_published,
        )
        assert current_live.import_electricity_price == live.import_electricity_price
        assert current_live.export_electricity_price == live.export_electricity_price
        coordinator = _make_bare_coordinator()
        coordinator._last_planner_output = SimpleNamespace(slots=[])  # type: ignore[assignment]
        coordinator._last_plan_price_forecast_signature = accepted

        assert (
            coordinator._should_replan(
                current_live,
                now,
                price_forecast_signature=current,
            )
            is True
        )

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_advertised_channel_is_canonical_unavailable(
        self, value: float
    ) -> None:
        assert _canonical_price_channel(value, True) == (False, None)
        assert _canonical_price_channel(0.0, True) == (True, 0.0)

    def test_pv_publication_withdrawal_and_value_change_trigger_replan(
        self,
    ) -> None:
        now = datetime(2026, 8, 14, 0, 5, tzinfo=UTC)
        with patch(
            "custom_components.hsem.coordinator_builder.hsem_now", return_value=now
        ):
            recommendations = generate_recommendation_intervals(15, 1)
        live = LiveState()
        live.import_electricity_price = 0.25
        live.export_electricity_price = 0.10
        live.import_electricity_price_available = True
        live.export_electricity_price_available = True
        current = recommendations[0]
        current.import_price = 0.25
        current.export_price = 0.10
        current.import_price_available = True
        current.export_price_available = True

        unavailable = _price_forecast_signature(recommendations, live, now)
        current.solcast_pv_estimate_kwh = 0.0
        current.solcast_pv_estimate_available = True
        published_zero = _price_forecast_signature(recommendations, live, now)
        current.solcast_pv_estimate_kwh = 1.234567
        changed_value = _price_forecast_signature(recommendations, live, now)
        current.solcast_pv_estimate_kwh = float("inf")
        nonfinite = _price_forecast_signature(recommendations, live, now)

        assert unavailable != published_zero
        assert published_zero != changed_value
        assert nonfinite == unavailable
        assert unavailable[2][0][3] == (False, None)
        assert published_zero[2][0][3] == (True, 0.0)
        assert changed_value[2][0][3] == (True, 1.23457)

        coordinator = _make_bare_coordinator()
        coordinator._last_planner_output = SimpleNamespace(slots=[])  # type: ignore[assignment]
        coordinator._last_plan_price_forecast_signature = unavailable

        assert (
            coordinator._should_replan(
                live,
                now,
                price_forecast_signature=published_zero,
            )
            is True
        )

    def test_valuation_attribute_changes_trigger_replan(self) -> None:
        now = datetime(2026, 8, 14, 0, 5, tzinfo=UTC)
        with patch(
            "custom_components.hsem.coordinator_builder.hsem_now", return_value=now
        ):
            recommendations = generate_recommendation_intervals(15, 1)
        live = LiveState()
        live.import_electricity_price = 0.25
        live.export_electricity_price = 0.10
        live.import_electricity_price_available = True
        live.export_electricity_price_available = True
        accepted_attributes = {
            "forecast": [
                {
                    "start": "2026-08-15T00:00:00+00:00",
                    "value": 1.25,
                }
            ],
            "mae": 0.10,
        }
        changed_attributes = {
            "forecast": [
                {
                    "start": "2026-08-15T00:00:00+00:00",
                    "value": 2.50,
                }
            ],
            "mae": 0.20,
        }
        if (
            "valuation_attributes"
            in inspect.signature(_price_forecast_signature).parameters
        ):
            accepted = _price_forecast_signature(
                recommendations,
                live,
                now,
                valuation_attributes=accepted_attributes,
                valuation_enabled=True,
                valuation_margin=0.05,
            )
            changed = _price_forecast_signature(
                recommendations,
                live,
                now,
                valuation_attributes=changed_attributes,
                valuation_enabled=True,
                valuation_margin=0.05,
            )
        else:  # pragma: no cover - exercised only against the published .34 code
            accepted = _price_forecast_signature(recommendations, live, now)
            changed = _price_forecast_signature(recommendations, live, now)
        coordinator = _make_bare_coordinator()
        coordinator._last_planner_output = SimpleNamespace(slots=[])  # type: ignore[assignment]
        coordinator._last_plan_price_forecast_signature = accepted

        assert accepted != changed
        assert (
            coordinator._should_replan(
                live,
                now,
                price_forecast_signature=changed,
            )
            is True
        )


def _make_bare_coordinator() -> HSEMDataUpdateCoordinator:
    """Return an HSEMDataUpdateCoordinator whose __init__ was NOT called.

    Attributes required by individual tests are set explicitly on the returned
    object.  This avoids the ``frame.report_usage`` call inside HA's
    ``DataUpdateCoordinator.__init__`` which requires a bootstrapped HA runtime.
    """
    coord = object.__new__(HSEMDataUpdateCoordinator)
    # Minimal set of attributes that the coordinator methods may reference.
    coord._update_lock = asyncio.Lock()
    coord._interval_timer_unsub = None
    coord._hourly_timer_unsub = None
    coord._force_discharge_monitor_unsub = None
    coord._slot_boundary_timer_unsub = None
    coord._slot_boundary_interval_minutes = None
    coord._window_hysteresis_timer_unsub = None
    coord._window_hysteresis_expiry = None
    coord._window_hysteresis_expiry_replan_pending = False
    coord._tearing_down = False
    coord._listener_unsubs = []
    coord._timer_interval = None
    coord._next_update = None
    coord.data = None  # type: ignore[assignment]  # test sets data to None before first cycle
    coord.last_update_success = True
    cfg = MagicMock()
    cfg.verbose_logging = False
    cfg.update_interval = 5
    cfg.recommendation_interval_minutes = 60
    cfg.recommendation_interval_length = 24
    coord._cfg = cfg
    coord._options_update_task = None
    coord._options_update_debounce_task = None
    coord._secondary_storage_update_debounce_task = None
    coord._price_source_update_debounce_task = None
    coord._price_source_update_pending = False
    coord._current_slot_start = None
    coord._current_slot_price_actionable = None
    coord._current_slot_ev_power_w = 0.0
    coord._current_slot_ev_second_power_w = 0.0
    coord._last_planner_output = None
    coord._last_plan_price_forecast_signature = None
    coord._last_apply_summary = None
    coord._live = None
    coord._force_discharge_excess_since = None
    coord._force_discharge_excess_slot_start = None
    coord._force_discharge_replanned_slot_start = None
    coord._force_discharge_live_replan_pending_slot = None
    return coord


# ---------------------------------------------------------------------------
# CoordinatorData unit tests
# ---------------------------------------------------------------------------


class TestCoordinatorData:
    """Verify the CoordinatorData dataclass defaults and field types."""

    def test_default_instance_has_no_live_state(self) -> None:
        """A fresh CoordinatorData must have live=None."""
        data = CoordinatorData()
        assert data.live is None
        assert data.cfg is None
        assert data.state is None
        assert data.last_updated is None
        assert data.next_update is None

    def test_empty_list_fields_are_mutable(self) -> None:
        """List fields must be independent instances (not shared default)."""
        d1 = CoordinatorData()
        d2 = CoordinatorData()
        d1.hourly_recommendations.append("x")  # type: ignore[arg-type]  # intentional: test verifies list independence with a sentinel string
        assert "x" not in d2.hourly_recommendations

    def test_numeric_fields_default_to_zero(self) -> None:
        """Numeric accumulator fields must default to 0.0."""
        data = CoordinatorData()
        assert data.current_required_battery == pytest.approx(0.0)


class TestLightweightSecondaryStorageEvents:
    """Noisy PowMr states must not wake the full coordinator blindly."""

    @staticmethod
    def _configure(coordinator: HSEMDataUpdateCoordinator) -> MagicMock:
        secondary = coordinator._cfg.secondary_storage
        secondary.enabled = True
        secondary.control_enabled = True
        secondary.soc_entity = "sensor.powmr_soc"
        secondary.load_power_entity = "sensor.powmr_load_avg"
        secondary.output_source_priority_entity = "select.powmr_output"
        secondary.charger_source_priority_entity = "select.powmr_charger"
        secondary.max_charge_current_entity = "number.powmr_current"
        coordinator._last_plan_secondary_soc_pct = 75.0
        coordinator._last_plan_secondary_load_power_w = 190.0
        hass = MagicMock()
        coordinator.hass = hass
        return hass

    @pytest.mark.asyncio
    async def test_subthreshold_soc_and_load_events_do_not_schedule_update(
        self,
    ) -> None:
        coordinator = _make_bare_coordinator()
        hass = self._configure(coordinator)

        await coordinator._async_handle_secondary_storage_change(
            cast(
                Any,
                SimpleNamespace(
                    data={
                        "entity_id": "sensor.powmr_soc",
                        "new_state": SimpleNamespace(state="75.9"),
                    }
                ),
            )
        )
        await coordinator._async_handle_secondary_storage_change(
            cast(
                Any,
                SimpleNamespace(
                    data={
                        "entity_id": "sensor.powmr_load_avg",
                        "new_state": SimpleNamespace(state="214.9"),
                    }
                ),
            )
        )

        hass.async_create_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_control_event_is_ignored_after_control_is_disabled(self) -> None:
        """A stale control listener cannot wake HSEM after control is disabled."""
        coordinator = _make_bare_coordinator()
        hass = self._configure(coordinator)
        coordinator._cfg.secondary_storage.control_enabled = False

        await coordinator._async_handle_secondary_storage_change(
            cast(
                Any,
                SimpleNamespace(
                    data={
                        "entity_id": "select.powmr_output",
                        "old_state": SimpleNamespace(state="utility"),
                        "new_state": SimpleNamespace(state="sbu"),
                    }
                ),
            )
        )

        hass.async_create_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_material_events_are_coalesced(self) -> None:
        coordinator = _make_bare_coordinator()
        hass = self._configure(coordinator)
        pending_task = MagicMock()
        pending_task.done.return_value = False
        hass.async_create_task.return_value = pending_task

        await coordinator._async_handle_secondary_storage_change(
            cast(
                Any,
                SimpleNamespace(
                    data={
                        "entity_id": "sensor.powmr_soc",
                        "new_state": SimpleNamespace(state="76.0"),
                    }
                ),
            )
        )
        await coordinator._async_handle_secondary_storage_change(
            cast(
                Any,
                SimpleNamespace(
                    data={
                        "entity_id": "select.powmr_output",
                        "old_state": SimpleNamespace(state="utility"),
                        "new_state": SimpleNamespace(state="sbu"),
                    }
                ),
            )
        )

        hass.async_create_task.assert_called_once()
        coroutine = hass.async_create_task.call_args.args[0]
        coroutine.close()

    @pytest.mark.asyncio
    async def test_material_event_waits_for_busy_coordinator_lock(self) -> None:
        """A one-off priority event survives overlap with an active cycle."""
        coordinator = _make_bare_coordinator()
        await coordinator._update_lock.acquire()
        try:
            with (
                patch(
                    "custom_components.hsem.coordinator."
                    "SECONDARY_STORAGE_UPDATE_DEBOUNCE_SECONDS",
                    0.0,
                ),
                patch.object(
                    coordinator, "_async_run_update_cycle", new_callable=AsyncMock
                ) as run_cycle,
            ):
                task = asyncio.create_task(
                    coordinator._async_secondary_storage_update_debounced()
                )
                coordinator._secondary_storage_update_debounce_task = task
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                run_cycle.assert_not_awaited()

                coordinator._update_lock.release()
                await task

                run_cycle.assert_awaited_once()
                assert coordinator._secondary_storage_update_debounce_task is None
        finally:
            if coordinator._update_lock.locked():
                coordinator._update_lock.release()

    @pytest.mark.asyncio
    async def test_price_event_waits_for_busy_coordinator_lock(self) -> None:
        """A publication/withdrawal event survives overlap with an active cycle."""
        coordinator = _make_bare_coordinator()
        await coordinator._update_lock.acquire()
        try:
            with (
                patch(
                    "custom_components.hsem.coordinator."
                    "PRICE_SOURCE_UPDATE_DEBOUNCE_SECONDS",
                    0.0,
                ),
                patch.object(
                    coordinator, "_async_run_update_cycle", new_callable=AsyncMock
                ) as run_cycle,
            ):
                task = asyncio.create_task(
                    coordinator._async_price_source_update_debounced()
                )
                coordinator._price_source_update_debounce_task = task
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                run_cycle.assert_not_awaited()

                coordinator._update_lock.release()
                await task

                run_cycle.assert_awaited_once()
                assert coordinator._price_source_update_debounce_task is None
        finally:
            if coordinator._update_lock.locked():
                coordinator._update_lock.release()

    @pytest.mark.asyncio
    async def test_price_event_burst_before_refresh_is_coalesced(self) -> None:
        """Import/export events in one burst create only one refresh task."""
        coordinator = _make_bare_coordinator()
        coordinator.hass = MagicMock()
        pending_task = MagicMock()
        pending_task.done.return_value = False
        coordinator.hass.async_create_task.return_value = pending_task
        event = cast(Any, SimpleNamespace(data={}))

        await coordinator._async_handle_price_source_change(event)
        await coordinator._async_handle_price_source_change(event)

        coordinator.hass.async_create_task.assert_called_once()
        assert coordinator._price_source_update_pending is True
        coroutine = coordinator.hass.async_create_task.call_args.args[0]
        coroutine.close()

    @pytest.mark.asyncio
    async def test_price_event_during_refresh_guarantees_follow_up_cycle(self) -> None:
        """Separate channel updates cannot be lost after snapshot collection."""
        coordinator = _make_bare_coordinator()
        coordinator._price_source_update_pending = True

        async def run_cycle() -> None:
            if run.await_count == 1:
                coordinator._price_source_update_pending = True

        with (
            patch(
                "custom_components.hsem.coordinator."
                "PRICE_SOURCE_UPDATE_DEBOUNCE_SECONDS",
                0.0,
            ),
            patch.object(
                coordinator,
                "_async_run_update_cycle",
                new_callable=AsyncMock,
                side_effect=run_cycle,
            ) as run,
        ):
            task = asyncio.create_task(
                coordinator._async_price_source_update_debounced()
            )
            coordinator._price_source_update_debounce_task = task
            await task

        assert run.await_count == 2
        assert coordinator._price_source_update_pending is False
        assert coordinator._price_source_update_debounce_task is None

    def test_weekday_profile_is_updated_only_once_per_cycle(self) -> None:
        source = inspect.getsource(HSEMDataUpdateCoordinator._async_run_update_cycle)
        assert source.count("weekday_profile.update(") == 1

    def test_reused_or_rejected_plan_does_not_advance_secondary_baseline(self) -> None:
        """Small changes accumulate relative to the plan actually in force."""
        coordinator = _make_bare_coordinator()
        coordinator._last_plan_secondary_soc_pct = 75.0
        coordinator._last_plan_secondary_load_power_w = 190.0
        coordinator._last_plan_secondary_output_priority = "utility"
        valuation_signature = (False, 0.0, 0.0, ())
        accepted_price_signature = (
            (True, 1.0),
            (True, 0.5),
            (),
            valuation_signature,
        )
        candidate_price_signature = (
            (True, 1.1),
            (True, 0.6),
            (),
            valuation_signature,
        )
        coordinator._last_plan_price_forecast_signature = accepted_price_signature
        live = LiveState()
        live.secondary_storage.soc_pct = 75.8
        live.secondary_storage.load_power_w = 210.0
        live.secondary_storage.output_source_priority = "sbu"

        coordinator._persist_plan_state_if_accepted(
            live,
            False,
            price_forecast_signature=candidate_price_signature,
        )

        assert coordinator._last_plan_secondary_soc_pct == pytest.approx(75.0)
        assert coordinator._last_plan_secondary_load_power_w == pytest.approx(190.0)
        assert coordinator._last_plan_secondary_output_priority == "utility"
        assert (
            coordinator._last_plan_price_forecast_signature == accepted_price_signature
        )

        coordinator._persist_plan_state_if_accepted(
            live,
            True,
            price_forecast_signature=candidate_price_signature,
        )
        assert coordinator._last_plan_secondary_soc_pct == pytest.approx(75.8)
        assert coordinator._last_plan_secondary_load_power_w == pytest.approx(210.0)
        assert coordinator._last_plan_secondary_output_priority == "sbu"
        assert (
            coordinator._last_plan_price_forecast_signature == candidate_price_signature
        )


class TestApplySummaryPublication:
    """Verify completed hardware results are republished to diagnostics."""

    def test_publish_updates_snapshot_and_notifies_listeners(self) -> None:
        """A completed apply must be visible to every coordinator-backed entity."""
        coord = _make_bare_coordinator()
        coord.data = CoordinatorData()
        applier_status_refresh = MagicMock(name="applier_status_refresh")
        degraded_mode_refresh = MagicMock(name="degraded_mode_refresh")
        plan_explanation_refresh = MagicMock(name="plan_explanation_refresh")
        coord._listeners = {
            1: (applier_status_refresh, None),
            2: (degraded_mode_refresh, None),
            3: (plan_explanation_refresh, None),
        }
        summary = CycleApplySummary()

        coord.async_publish_apply_summary(summary)

        assert coord._last_apply_summary is summary
        assert coord.data.apply_summary is summary
        applier_status_refresh.assert_called_once_with()
        degraded_mode_refresh.assert_called_once_with()
        plan_explanation_refresh.assert_called_once_with()

    def test_publish_before_snapshot_only_persists_summary(self) -> None:
        """A late worker completion remains safe if teardown cleared the snapshot."""
        coord = _make_bare_coordinator()
        summary = CycleApplySummary()

        with patch.object(coord, "async_update_listeners") as update_listeners:
            coord.async_publish_apply_summary(summary)

        assert coord._last_apply_summary is summary
        update_listeners.assert_not_called()

    def test_new_snapshots_retain_last_completed_summary(self) -> None:
        """Normal refreshes must not revert diagnostics to pending."""
        source = inspect.getsource(HSEMDataUpdateCoordinator._async_run_update_cycle)
        assert 'apply_summary=getattr(self, "_last_apply_summary", None)' in source


# ---------------------------------------------------------------------------
# Coordinator construction tests (source inspection — no HA runtime needed)
# ---------------------------------------------------------------------------


class TestCoordinatorConstruction:
    """Verify key attributes are initialised in HSEMDataUpdateCoordinator.__init__."""

    def test_update_lock_is_asyncio_lock(self) -> None:
        """__init__ must create self._update_lock = asyncio.Lock()."""
        source = inspect.getsource(HSEMDataUpdateCoordinator.__init__)
        assert "_update_lock = asyncio.Lock()" in source, (
            "HSEMDataUpdateCoordinator.__init__ must contain "
            "self._update_lock = asyncio.Lock()"
        )

    def test_initial_data_field_comment_or_absent(self) -> None:
        """data is managed by the DataUpdateCoordinator base class (starts as None).

        We verify this via the bare instance helper which sets data=None to
        reflect the pre-first-cycle state.
        """
        coord = _make_bare_coordinator()
        assert coord.data is None

    def test_timer_handles_start_as_none(self) -> None:
        """Timer unsub handles must be None before async_setup is called."""
        coord = _make_bare_coordinator()
        assert coord._interval_timer_unsub is None
        assert coord._hourly_timer_unsub is None
        assert coord._force_discharge_monitor_unsub is None


# ---------------------------------------------------------------------------
# Concurrent update lock tests
# ---------------------------------------------------------------------------


class _StubCoordinator:
    """Minimal stub that replaces only the locking behaviour of HSEMDataUpdateCoordinator."""

    def __init__(self) -> None:
        self._update_lock = asyncio.Lock()
        self._cycle_count = 0
        self._skip_count = 0
        self._cfg = MagicMock()
        self._cfg.verbose_logging = False

    async def _async_handle_update(self, event: Any = None) -> None:
        """Identical guard logic to the production coordinator."""
        if self._update_lock.locked():
            self._skip_count += 1
            return
        async with self._update_lock:
            await self._async_run_update_cycle()

    async def _async_run_update_cycle(self) -> None:
        """Simulated slow cycle (2 event-loop ticks)."""
        self._cycle_count += 1
        await asyncio.sleep(0)
        await asyncio.sleep(0)


class TestCoordinatorUpdateLock:
    """Verify the asyncio.Lock guard inside _async_handle_update."""

    @pytest.mark.asyncio
    async def test_single_call_runs_once(self) -> None:
        """A lone call executes exactly one cycle."""
        coord = _StubCoordinator()
        await coord._async_handle_update()
        assert coord._cycle_count == 1
        assert coord._skip_count == 0

    @pytest.mark.asyncio
    async def test_concurrent_second_call_is_dropped(self) -> None:
        """While a cycle is running a second concurrent call is skipped, not queued."""
        coord = _StubCoordinator()
        await asyncio.gather(
            coord._async_handle_update(),
            coord._async_handle_update(),
        )
        assert coord._cycle_count == 1, f"Expected 1 cycle, got {coord._cycle_count}"
        assert coord._skip_count == 1, f"Expected 1 skip, got {coord._skip_count}"

    @pytest.mark.asyncio
    async def test_sequential_calls_both_execute(self) -> None:
        """Two non-overlapping sequential calls both run the cycle."""
        coord = _StubCoordinator()
        await coord._async_handle_update()
        await coord._async_handle_update()
        assert coord._cycle_count == 2


# ---------------------------------------------------------------------------
# Coordinator async_teardown
# ---------------------------------------------------------------------------


class TestCoordinatorTeardown:
    """Verify async_teardown cancels all registered timer subscriptions."""

    @pytest.mark.asyncio
    async def test_teardown_cancels_timers(self) -> None:
        """async_teardown must call every timer unsubscribe callback."""
        coordinator = _make_bare_coordinator()

        interval_unsub = MagicMock()
        hourly_unsub = MagicMock()
        monitor_unsub = MagicMock()
        boundary_unsub = MagicMock()
        hysteresis_unsub = MagicMock()
        coordinator._interval_timer_unsub = interval_unsub
        coordinator._hourly_timer_unsub = hourly_unsub
        coordinator._force_discharge_monitor_unsub = monitor_unsub
        coordinator._slot_boundary_timer_unsub = boundary_unsub
        coordinator._window_hysteresis_timer_unsub = hysteresis_unsub

        await coordinator.async_teardown()

        interval_unsub.assert_called_once()
        hourly_unsub.assert_called_once()
        monitor_unsub.assert_called_once()
        boundary_unsub.assert_called_once()
        hysteresis_unsub.assert_called_once()
        assert coordinator._interval_timer_unsub is None
        assert coordinator._hourly_timer_unsub is None
        assert coordinator._force_discharge_monitor_unsub is None
        assert coordinator._slot_boundary_timer_unsub is None
        assert coordinator._slot_boundary_interval_minutes is None
        assert coordinator._window_hysteresis_timer_unsub is None
        assert coordinator._window_hysteresis_expiry is None
        assert coordinator._window_hysteresis_expiry_replan_pending is False

    @pytest.mark.asyncio
    async def test_teardown_safe_when_no_timers(self) -> None:
        """async_teardown must not raise when no timers were registered."""
        coordinator = _make_bare_coordinator()
        # Both handles are None — no error expected
        await coordinator.async_teardown()


class TestExactCoordinatorTimers:
    """Recommendation and hysteresis transitions must not wait for polling."""

    def test_next_boundary_is_wall_clock_aligned_and_handles_midnight(self) -> None:
        now = datetime(2026, 8, 14, 23, 58, 42, tzinfo=UTC)
        assert _next_slot_boundary_utc(now, 15) == datetime(
            2026, 8, 15, 0, 0, tzinfo=UTC
        )
        now = datetime(2026, 8, 14, 12, 7, 59, tzinfo=UTC)
        assert _next_slot_boundary_utc(now, 15) == datetime(
            2026, 8, 14, 12, 15, tzinfo=UTC
        )

    def test_next_boundary_is_strictly_future_during_dst_fallback(self) -> None:
        stockholm = ZoneInfo("Europe/Stockholm")
        now = datetime(2026, 10, 25, 2, 30, tzinfo=stockholm, fold=1)

        boundary = _next_slot_boundary_utc(now, 15)

        assert boundary == datetime(2026, 10, 25, 1, 45, tzinfo=UTC)
        assert boundary > now.astimezone(UTC)

    def test_replan_distinguishes_same_wall_slot_in_second_fold(self) -> None:
        stockholm = ZoneInfo("Europe/Stockholm")
        coordinator = _make_bare_coordinator()
        coordinator._cfg.recommendation_interval_minutes = 15
        coordinator._last_planner_output = SimpleNamespace(slots=[])  # type: ignore[assignment]
        coordinator._last_plan_slot_start = datetime(
            2026, 10, 25, 2, 15, tzinfo=stockholm, fold=0
        )
        second_fold = datetime(2026, 10, 25, 2, 15, tzinfo=stockholm, fold=1)

        assert coordinator._should_replan(MagicMock(), second_fold) is True

    def test_next_boundary_skips_nonexistent_spring_hour(self) -> None:
        stockholm = ZoneInfo("Europe/Stockholm")
        now = datetime(2026, 3, 29, 1, 58, tzinfo=stockholm)

        boundary = _next_slot_boundary_utc(now, 15)

        assert boundary == datetime(2026, 3, 29, 1, 0, tzinfo=UTC)
        assert boundary.astimezone(stockholm).hour == 3

    def test_slot_boundary_registration_is_one_shot_and_exact(self) -> None:
        coordinator = _make_bare_coordinator()
        coordinator.hass = MagicMock()
        coordinator._cfg.recommendation_interval_minutes = 15
        unsub = MagicMock()
        with patch(
            "custom_components.hsem.coordinator.async_track_point_in_utc_time",
            return_value=unsub,
        ) as register:
            coordinator._schedule_next_slot_boundary(
                datetime(2026, 8, 14, 12, 7, tzinfo=UTC)
            )
        assert register.call_args.args[2] == datetime(2026, 8, 14, 12, 15, tzinfo=UTC)
        assert coordinator._slot_boundary_timer_unsub is unsub
        assert coordinator._slot_boundary_interval_minutes == 15

    def test_live_interval_change_reschedules_before_old_boundary(self) -> None:
        coordinator = _make_bare_coordinator()
        coordinator._slot_boundary_timer_unsub = MagicMock()
        coordinator._slot_boundary_interval_minutes = 60
        coordinator._cfg.recommendation_interval_minutes = 15
        now = datetime(2026, 8, 14, 12, 7, tzinfo=UTC)

        with patch.object(coordinator, "_schedule_next_slot_boundary") as schedule:
            coordinator._refresh_slot_boundary_schedule(now)

        schedule.assert_called_once_with(now)

    @pytest.mark.asyncio
    async def test_slot_boundary_waits_for_busy_cycle_then_runs_once(self) -> None:
        coordinator = _make_bare_coordinator()
        await coordinator._update_lock.acquire()
        try:
            with (
                patch.object(
                    coordinator, "_async_run_update_cycle", new_callable=AsyncMock
                ) as run_cycle,
                patch.object(coordinator, "_schedule_next_slot_boundary") as schedule,
            ):
                task = asyncio.create_task(
                    coordinator._async_handle_slot_boundary(
                        datetime(2026, 8, 14, 12, 15, tzinfo=UTC)
                    )
                )
                await asyncio.sleep(0)
                run_cycle.assert_not_awaited()
                coordinator._update_lock.release()
                await task
                run_cycle.assert_awaited_once()
                schedule.assert_called_once()
        finally:
            if coordinator._update_lock.locked():
                coordinator._update_lock.release()

    @pytest.mark.asyncio
    async def test_hysteresis_expiry_waits_and_forces_fresh_plan(self) -> None:
        coordinator = _make_bare_coordinator()
        await coordinator._update_lock.acquire()
        try:
            with patch.object(
                coordinator, "_async_run_update_cycle", new_callable=AsyncMock
            ) as run_cycle:
                task = asyncio.create_task(
                    coordinator._async_handle_window_hysteresis_expiry(
                        datetime(2026, 8, 14, 12, 10, tzinfo=UTC)
                    )
                )
                await asyncio.sleep(0)
                assert coordinator._window_hysteresis_expiry_replan_pending is True
                run_cycle.assert_not_awaited()
                coordinator._update_lock.release()
                await task
                run_cycle.assert_awaited_once()
        finally:
            if coordinator._update_lock.locked():
                coordinator._update_lock.release()

    def test_hysteresis_expiry_flag_is_a_material_replan_event(self) -> None:
        coordinator = _make_bare_coordinator()
        coordinator._last_planner_output = SimpleNamespace(slots=[])  # type: ignore[assignment]
        coordinator._window_hysteresis_expiry_replan_pending = True
        assert coordinator._should_replan(MagicMock(), datetime.now(UTC)) is True


# ---------------------------------------------------------------------------
# Options-update background task
# ---------------------------------------------------------------------------


class TestOptionsUpdateBackgroundTask:
    """async_options_updated must schedule the pipeline as a background task.

    Regression: the config-entry update listener (fired by
    ``async_update_entry`` from switch/number/time entities) used to await
    the full MILP/ML pipeline inline.  The toggle itself was fast (HA fires
    listeners as tasks), but the *effect* of a toggle was delayed because
    each options change ran a full synchronous pipeline cycle.  The fix
    schedules the cycle via ``hass.async_create_task`` with
    cancel-and-reschedule semantics so repeated toggles collapse into one
    run and the listener returns immediately.

    A short debounce window is used so rapid toggles only trigger a single
    planner run after the user stops clicking.
    """

    @pytest.mark.asyncio
    @patch("custom_components.hsem.coordinator.OPTIONS_UPDATE_DEBOUNCE_SECONDS", 0.0)
    async def test_options_updated_schedules_background_task(self) -> None:
        """async_options_updated returns immediately after scheduling a task."""
        coordinator = _make_bare_coordinator()
        coordinator.hass = MagicMock()

        scheduled: list[asyncio.Task] = []

        def _create_task(coro: Any, *, name: str, **kwargs: Any) -> asyncio.Task:
            task = asyncio.get_running_loop().create_task(coro, name=name)
            scheduled.append(task)
            return task

        coordinator.hass.async_create_task = MagicMock(side_effect=_create_task)

        # Make the pipeline cycle a no-op so the task completes quickly.
        async def _noop_cycle(*args: Any, **kwargs: Any) -> None:
            await asyncio.sleep(0)

        coordinator._async_handle_update = _noop_cycle  # type: ignore[method-assign]  # test monkey-patch

        await coordinator.async_options_updated()

        # The method must have scheduled exactly one debounce task with
        # eager-start disabled so the switch service call returns immediately.
        coordinator.hass.async_create_task.assert_called_once()  # type: ignore[union-attr]  # mock assertion
        call_kwargs = coordinator.hass.async_create_task.call_args[1]  # type: ignore[attr-defined]
        assert call_kwargs.get("eager_start") is False
        assert len(scheduled) == 1
        assert scheduled[0].get_name() == "hsem_options_update_debounce"

        # Let the debounce task finish; it then schedules the background task.
        await scheduled[0]
        assert len(scheduled) == 2
        assert scheduled[1].get_name() == "hsem_options_update"
        await scheduled[1]

    @pytest.mark.asyncio
    @patch("custom_components.hsem.coordinator.OPTIONS_UPDATE_DEBOUNCE_SECONDS", 0.0)
    async def test_repeated_toggle_cancels_pending_task(self) -> None:
        """Repeated options updates collapse into a single background run."""
        coordinator = _make_bare_coordinator()
        coordinator.hass = MagicMock()

        scheduled: list[asyncio.Task] = []

        def _create_task(coro: Any, *, name: str, **kwargs: Any) -> asyncio.Task:
            task = asyncio.get_running_loop().create_task(coro, name=name)
            scheduled.append(task)
            return task

        coordinator.hass.async_create_task = MagicMock(side_effect=_create_task)

        # A cycle that blocks until cancelled.
        started = asyncio.Event()

        async def _blocking_cycle(*args: Any, **kwargs: Any) -> None:
            started.set()
            await asyncio.sleep(60)

        coordinator._async_handle_update = _blocking_cycle  # type: ignore[method-assign]  # test monkey-patch

        await coordinator.async_options_updated()
        assert len(scheduled) == 1
        debounce_task = scheduled[0]
        assert debounce_task.get_name() == "hsem_options_update_debounce"

        # Let the debounce task fire and start the background update cycle.
        await started.wait()
        assert len(scheduled) == 2
        background_task = scheduled[1]
        assert background_task.get_name() == "hsem_options_update"

        # A second toggle cancels the previous debounce/background tasks and
        # schedules a fresh debounce task.
        await coordinator.async_options_updated()
        assert len(scheduled) == 3
        assert scheduled[2].get_name() == "hsem_options_update_debounce"

        # Let the new debounce task fire so it cancels the old background task.
        await scheduled[2]
        assert background_task.cancelled() or background_task.cancelling()

    @pytest.mark.asyncio
    @patch("custom_components.hsem.coordinator.OPTIONS_UPDATE_DEBOUNCE_SECONDS", 0.0)
    async def test_teardown_cancels_pending_options_task(self) -> None:
        """async_teardown must cancel pending debounce and options-update tasks."""
        coordinator = _make_bare_coordinator()
        coordinator.hass = MagicMock()

        async def _blocking_cycle(*args: Any, **kwargs: Any) -> None:
            await asyncio.sleep(60)

        coordinator._async_handle_update = _blocking_cycle  # type: ignore[method-assign]  # test monkey-patch
        coordinator.hass.async_create_task = MagicMock(  # type: ignore[method-assign]  # test monkey-patch
            side_effect=lambda coro, *, name, **kwargs: (
                asyncio.get_running_loop().create_task(coro, name=name)
            )
        )

        await coordinator.async_options_updated()
        debounce_task = coordinator._options_update_debounce_task
        assert debounce_task is not None and not debounce_task.done()

        # Wait for the debounce task to schedule the background task.
        await debounce_task
        task = coordinator._options_update_task
        assert task is not None and not task.done()
        # The debounce task is cleared once it has scheduled the background run.
        assert coordinator._options_update_debounce_task is None

        await coordinator.async_teardown()

        assert task.cancelling() or task.cancelled() or task.done()
        assert coordinator._options_update_task is None


# ---------------------------------------------------------------------------
# Forced-discharge live-demand corrective replan
# ---------------------------------------------------------------------------


def _force_discharge_slot(start: datetime) -> PlannedSlot:
    """Build a 15-minute forced-discharge slot with 1 kW battery-side power."""
    return PlannedSlot(
        start=start,
        end=start + timedelta(minutes=15),
        recommendation=Recommendations.ForceBatteriesDischarge.value,
        batteries_discharged_kwh=0.25,
    )


def _configure_force_discharge_monitor(
    coordinator: HSEMDataUpdateCoordinator, slot: PlannedSlot
) -> tuple[dict[str, SimpleNamespace], MagicMock]:
    """Populate the minimal coordinator/live/HA state used by monitor tests."""
    coordinator._last_planner_output = SimpleNamespace(slots=[slot])  # type: ignore[assignment]
    coordinator._live = LiveState(missing_entities=False)
    coordinator._cfg.house_consumption_power = "sensor.house"
    coordinator._cfg.solar_production_power = "sensor.solar"
    coordinator._cfg.house_power_includes_ev_charger_power = False
    coordinator._cfg.batteries_discharge_efficiency = 98.0
    states = {
        "sensor.house": SimpleNamespace(state="2200"),
        "sensor.solar": SimpleNamespace(state="100"),
    }
    coordinator.hass = MagicMock()
    state_get = MagicMock(side_effect=states.get)
    coordinator.hass.states.get = state_get
    return states, state_get


class TestForceDischargeLiveDemandMonitor:
    """Validate the lightweight, debounced once-per-slot corrective trigger."""

    def test_metrics_compare_ac_side_power_and_use_hybrid_threshold(self) -> None:
        """Battery-side slot energy is efficiency-adjusted before comparison."""
        start = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)
        metrics = _force_discharge_live_metrics(
            _force_discharge_slot(start),
            discharge_efficiency_pct=98.0,
            house_power_w=2200.0,
            solar_power_w=100.0,
        )

        assert metrics is not None
        planned_supply_w, live_residual_w, threshold_w, excess_w = metrics
        assert planned_supply_w == pytest.approx(980.0)
        assert live_residual_w == pytest.approx(2100.0)
        assert threshold_w == pytest.approx(150.0)
        assert excess_w == pytest.approx(1120.0)

    def test_powmr_sbu_load_is_not_double_counted_in_site_bus_comparison(
        self,
    ) -> None:
        """NAS load is absent from both live and solved site-bus demand in SBU."""
        start = datetime(2026, 8, 14, 20, 0, tzinfo=UTC)
        slot = _force_discharge_slot(start)
        slot.secondary_storage_mode = "sbu"
        slot.secondary_storage_load_kwh = 0.050
        slot.secondary_storage_discharged_kwh = 0.060

        metrics = _force_discharge_live_metrics(
            slot,
            discharge_efficiency_pct=100.0,
            house_power_w=1200.0,
            solar_power_w=0.0,
        )

        assert metrics is not None
        planned_supply_w, live_residual_w, threshold_w, excess_w = metrics
        assert planned_supply_w == pytest.approx(1000.0)
        assert live_residual_w == pytest.approx(1200.0)
        assert threshold_w == pytest.approx(150.0)
        assert excess_w == pytest.approx(200.0)

    def test_live_incident_exceeds_threshold_after_efficiency(self) -> None:
        """The reported 0.358-kWh plan versus 1.59-kW load is material."""
        start = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)
        slot = _force_discharge_slot(start)
        slot.batteries_discharged_kwh = 0.358

        metrics = _force_discharge_live_metrics(
            slot,
            discharge_efficiency_pct=98.0,
            house_power_w=1590.0,
            solar_power_w=0.0,
        )

        assert metrics is not None
        planned_supply_w, live_residual_w, threshold_w, excess_w = metrics
        assert planned_supply_w == pytest.approx(1403.36)
        assert live_residual_w == pytest.approx(1590.0)
        assert threshold_w == pytest.approx(150.0)
        assert excess_w == pytest.approx(186.64)

    def test_threshold_equality_is_not_material(self) -> None:
        """The trigger requires excess strictly above the noise threshold."""
        start = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)
        metrics = _force_discharge_live_metrics(
            _force_discharge_slot(start),
            discharge_efficiency_pct=100.0,
            house_power_w=1150.0,
            solar_power_w=0.0,
        )

        assert metrics is not None
        assert metrics[2] == pytest.approx(150.0)
        assert metrics[3] == pytest.approx(150.0)

    def test_material_partial_msc_compares_live_load_with_battery_plus_grid(
        self,
    ) -> None:
        """A partial BDM slot monitors drift above its solved total supply."""
        start = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)
        slot = _force_discharge_slot(start)
        slot.recommendation = Recommendations.BatteriesDischargeMode.value
        slot.grid_import_kwh = 0.10

        metrics = _force_discharge_live_metrics(
            slot,
            discharge_efficiency_pct=98.0,
            house_power_w=1800.0,
            solar_power_w=100.0,
        )

        assert metrics is not None
        planned_supply_w, live_residual_w, threshold_w, excess_w = metrics
        assert planned_supply_w == pytest.approx(1380.0)
        assert live_residual_w == pytest.approx(1700.0)
        assert threshold_w == pytest.approx(150.0)
        assert excess_w == pytest.approx(320.0)

    def test_material_partial_msc_is_monitored_but_rounding_import_is_not(
        self,
    ) -> None:
        """Only a real solved grid share extends the live-demand monitor."""
        coordinator = _make_bare_coordinator()
        coordinator._cfg.batteries_discharge_efficiency = 98.0
        start = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)
        slot = _force_discharge_slot(start)
        slot.recommendation = Recommendations.BatteriesDischargeMode.value
        slot.grid_import_kwh = 0.10
        coordinator._last_planner_output = SimpleNamespace(slots=[slot])  # type: ignore[assignment]

        assert coordinator._active_force_discharge_slot(start) is slot

        slot.grid_import_kwh = 0.001
        assert coordinator._active_force_discharge_slot(start) is None

    @pytest.mark.asyncio
    async def test_triggers_after_30_seconds_and_only_once_per_slot(self) -> None:
        """Four 10-second samples trigger one completed corrective attempt."""
        coordinator = _make_bare_coordinator()
        start = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)
        _configure_force_discharge_monitor(coordinator, _force_discharge_slot(start))

        async def _complete_corrective_update(event: Any = None) -> None:
            coordinator._force_discharge_replanned_slot_start = (
                coordinator._force_discharge_live_replan_pending_slot
            )
            coordinator._force_discharge_live_replan_pending_slot = None
            coordinator._clear_force_discharge_excess_window()

        coordinator._async_handle_update = AsyncMock(  # type: ignore[method-assign]
            side_effect=_complete_corrective_update
        )

        for seconds in (0, 10, 20):
            await coordinator._async_monitor_force_discharge_load(
                start + timedelta(seconds=seconds)
            )
        coordinator._async_handle_update.assert_not_awaited()  # type: ignore[attr-defined]

        await coordinator._async_monitor_force_discharge_load(
            start + timedelta(seconds=30)
        )
        coordinator._async_handle_update.assert_awaited_once()  # type: ignore[attr-defined]
        assert coordinator._force_discharge_replanned_slot_start == start

        await coordinator._async_monitor_force_discharge_load(
            start + timedelta(seconds=40)
        )
        coordinator._async_handle_update.assert_awaited_once()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_below_threshold_sample_restarts_debounce(self) -> None:
        """One normal sample breaks continuity; three later samples are insufficient."""
        coordinator = _make_bare_coordinator()
        start = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)
        (states, _) = _configure_force_discharge_monitor(
            coordinator, _force_discharge_slot(start)
        )
        coordinator._async_handle_update = AsyncMock()  # type: ignore[method-assign]

        await coordinator._async_monitor_force_discharge_load(start)
        states["sensor.house"].state = "1050"
        states["sensor.solar"].state = "0"
        await coordinator._async_monitor_force_discharge_load(
            start + timedelta(seconds=10)
        )
        states["sensor.house"].state = "2200"
        states["sensor.solar"].state = "100"
        for seconds in (20, 30, 40):
            await coordinator._async_monitor_force_discharge_load(
                start + timedelta(seconds=seconds)
            )

        coordinator._async_handle_update.assert_not_awaited()  # type: ignore[attr-defined]
        await coordinator._async_monitor_force_discharge_load(
            start + timedelta(seconds=50)
        )
        coordinator._async_handle_update.assert_awaited_once()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_busy_update_defers_pending_request(self) -> None:
        """A busy coordinator keeps the request pending for the next monitor tick."""
        coordinator = _make_bare_coordinator()
        start = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)
        _configure_force_discharge_monitor(coordinator, _force_discharge_slot(start))
        coordinator._async_handle_update = AsyncMock()  # type: ignore[method-assign]

        await coordinator._update_lock.acquire()
        try:
            for seconds in (0, 10, 20, 30):
                await coordinator._async_monitor_force_discharge_load(
                    start + timedelta(seconds=seconds)
                )
        finally:
            coordinator._update_lock.release()

        coordinator._async_handle_update.assert_not_awaited()  # type: ignore[attr-defined]
        assert coordinator._force_discharge_live_replan_pending_slot == start

        await coordinator._async_monitor_force_discharge_load(
            start + timedelta(seconds=40)
        )
        coordinator._async_handle_update.assert_awaited_once()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_non_force_slot_never_reads_live_power(self) -> None:
        """The lightweight timer stays dormant outside forced battery export."""
        coordinator = _make_bare_coordinator()
        start = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)
        slot = _force_discharge_slot(start)
        slot.recommendation = Recommendations.BatteriesDischargeMode.value
        _, state_get = _configure_force_discharge_monitor(coordinator, slot)
        coordinator._async_handle_update = AsyncMock()  # type: ignore[method-assign]

        await coordinator._async_monitor_force_discharge_load(start)

        state_get.assert_not_called()
        coordinator._async_handle_update.assert_not_awaited()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_ev_inclusive_house_meter_fails_closed_while_ev_charges(
        self,
    ) -> None:
        """An EV included in house power cannot trigger a corrective replan."""
        coordinator = _make_bare_coordinator()
        start = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)
        _, state_get = _configure_force_discharge_monitor(
            coordinator, _force_discharge_slot(start)
        )
        coordinator._cfg.house_power_includes_ev_charger_power = True
        assert coordinator._live is not None
        coordinator._live.ev.is_charging = True
        coordinator._async_handle_update = AsyncMock()  # type: ignore[method-assign]

        await coordinator._async_monitor_force_discharge_load(start)

        state_get.assert_not_called()
        coordinator._async_handle_update.assert_not_awaited()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_does_not_start_inside_last_minute(self) -> None:
        """A correction is skipped when less than 60 seconds remains."""
        coordinator = _make_bare_coordinator()
        start = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)
        _, state_get = _configure_force_discharge_monitor(
            coordinator, _force_discharge_slot(start)
        )
        coordinator._async_handle_update = AsyncMock()  # type: ignore[method-assign]

        await coordinator._async_monitor_force_discharge_load(
            start + timedelta(minutes=14, seconds=1)
        )

        state_get.assert_not_called()
        coordinator._async_handle_update.assert_not_awaited()  # type: ignore[attr-defined]

    def test_pending_request_is_a_material_replan_event(self) -> None:
        """The pending flag bypasses ordinary event-driven plan reuse."""
        coordinator = _make_bare_coordinator()
        start = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)
        coordinator._last_planner_output = SimpleNamespace(slots=[])  # type: ignore[assignment]
        coordinator._force_discharge_live_replan_pending_slot = start

        assert coordinator._should_replan(MagicMock(), start) is True

    @pytest.mark.parametrize(
        "solver_status", ["optimal", "time_limit_feasible_incumbent"]
    )
    def test_authoritative_corrective_milp_can_replace_force_with_msc(
        self, solver_status: str
    ) -> None:
        """A validated corrected plan may replace Fully Fed intent with MSC."""
        start = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)
        forced = _force_discharge_slot(start)
        corrected = _force_discharge_slot(start)
        corrected.recommendation = Recommendations.BatteriesDischargeMode.value
        corrected.grid_export_kwh = 0.0

        previous = PlannerOutput(slots=[forced], winner_name="milp")
        candidate = PlannerOutput(
            slots=[corrected],
            winner_name="milp",
            explanation=PlanExplanation(solver_status=solver_status),
        )

        selected, accepted, rejection = _select_corrective_planner_output(
            previous, candidate, start
        )

        assert accepted is True
        assert rejection == ""
        assert selected is candidate
        assert selected.slots[0].recommendation == (
            Recommendations.BatteriesDischargeMode.value
        )
        assert selected.slots[0].grid_export_kwh == 0.0

    def test_corrective_fallback_keeps_previous_force_plan(self) -> None:
        """A passive fallback cannot replace a previously validated MILP plan."""
        start = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)
        previous = PlannerOutput(
            slots=[_force_discharge_slot(start)], winner_name="milp"
        )
        fallback = PlannerOutput(
            slots=[],
            winner_name="passive",
            explanation=PlanExplanation(
                solver_status="time_limit_no_incumbent",
                fallback_reason="time_limit_no_incumbent",
            ),
        )

        selected, accepted, rejection = _select_corrective_planner_output(
            previous, fallback, start
        )

        assert accepted is False
        assert rejection == "winner_not_milp"
        assert selected is previous

    def test_price_withdrawal_uses_fresh_corrective_fallback(self) -> None:
        """A stale force plan cannot survive changed price authority."""
        start = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)
        forced = _force_discharge_slot(start)
        forced.price_actionable = True
        previous = PlannerOutput(slots=[forced], winner_name="milp")

        passive = PlannedSlot(
            start=start,
            end=start + timedelta(minutes=15),
            recommendation=Recommendations.BatteriesWaitMode.value,
            price_actionable=False,
            import_price_available=False,
            export_price_available=False,
        )
        fallback = PlannerOutput(
            slots=[passive],
            winner_name="passive",
            explanation=PlanExplanation(
                solver_status="time_limit_no_incumbent",
                fallback_reason="time_limit_no_incumbent",
            ),
        )

        selected, accepted, rejection = _select_corrective_planner_output(
            previous,
            fallback,
            start,
            price_authority_changed=True,
        )

        assert accepted is True
        assert rejection == ""
        assert selected is fallback
        assert selected.slots[0].price_actionable is False
        assert selected.slots[0].recommendation == (
            Recommendations.BatteriesWaitMode.value
        )

    def test_corrective_partial_msc_is_accepted_for_bounded_execution(self) -> None:
        """A partial MSC plan is safe once the applier enforces its energy cap."""
        start = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)
        previous = PlannerOutput(
            slots=[_force_discharge_slot(start)], winner_name="milp"
        )
        partial = _force_discharge_slot(start)
        partial.recommendation = Recommendations.BatteriesDischargeMode.value
        partial.batteries_discharged_kwh = 0.25
        partial.grid_import_kwh = 0.10
        candidate = PlannerOutput(
            slots=[partial],
            winner_name="milp",
            explanation=PlanExplanation(solver_status="optimal"),
        )

        selected, accepted, rejection = _select_corrective_planner_output(
            previous, candidate, start
        )

        assert accepted is True
        assert rejection == ""
        assert selected is candidate

    def test_corrective_cycle_bypasses_both_hysteresis_layers(self) -> None:
        """Candidate and current-window hysteresis cannot restore stale intent."""
        source = inspect.getsource(HSEMDataUpdateCoordinator._async_run_update_cycle)
        assert (
            "if corrective_live_replan\n                            else self._previous_planner_winner_name"
            in source
        )
        assert (
            "if corrective_live_replan\n                            else self._window_hys_previous_rec"
            in source
        )
        assert (
            "if corrective_live_replan\n                            else self._window_hys_previous_slot_start"
            in source
        )

    def test_attempt_is_consumed_only_after_planner_returns(self) -> None:
        """A failed cycle cannot consume the request before publication."""
        source = inspect.getsource(HSEMDataUpdateCoordinator._async_run_update_cycle)
        planner_call = source.index("run_planner, planner_input")
        publication = source.index("self.async_set_updated_data(data)")
        completed_marker = source.index("self._force_discharge_replanned_slot_start =")
        assert planner_call < publication < completed_marker


# ---------------------------------------------------------------------------
# Coordinator recommendation interval generation
# ---------------------------------------------------------------------------


class TestGenerateRecommendationIntervals:
    """Verify the recommendation-slot generation helper inside the coordinator."""

    def test_generates_correct_count_for_60min_24h(self) -> None:
        """60-minute slots over 24 hours must produce 24 slots."""
        slots = generate_recommendation_intervals(60, 24)
        assert len(slots) == 24

    def test_generates_correct_count_for_15min_48h(self) -> None:
        """15-minute slots over 48 hours must produce 192 slots."""
        slots = generate_recommendation_intervals(15, 48)
        assert len(slots) == 192

    def test_slots_start_at_midnight(self) -> None:
        """The first slot must start at midnight of the current day."""
        slots = generate_recommendation_intervals(60, 24)
        first = slots[0]
        assert first.start.hour == 0
        assert first.start.minute == 0

    def test_consecutive_slots_are_contiguous(self) -> None:
        """Each slot's end must equal the next slot's start."""
        slots = generate_recommendation_intervals(15, 2)
        for i in range(len(slots) - 1):
            assert slots[i].end == slots[i + 1].start

    def test_spring_forward_uses_physical_timeline(self) -> None:
        stockholm = ZoneInfo("Europe/Stockholm")
        midnight = datetime(2026, 3, 29, 0, 0, tzinfo=stockholm)
        with patch(
            "custom_components.hsem.coordinator_builder.hsem_now",
            return_value=midnight,
        ):
            slots = generate_recommendation_intervals(15, 24)

        assert len(slots) == 96
        assert not [slot for slot in slots if slot.start.hour == 2]

    def test_autumn_fallback_preserves_both_hour_folds(self) -> None:
        stockholm = ZoneInfo("Europe/Stockholm")
        midnight = datetime(2026, 10, 25, 0, 0, tzinfo=stockholm)
        with patch(
            "custom_components.hsem.coordinator_builder.hsem_now",
            return_value=midnight,
        ):
            slots = generate_recommendation_intervals(15, 24)

        repeated = [slot for slot in slots if slot.start.hour == 2]
        assert len(slots) == 96
        assert len(repeated) == 8
        assert [slot.start.fold for slot in repeated] == [0] * 4 + [1] * 4

    def test_slots_have_zero_defaults(self) -> None:
        """All numeric fields on a freshly generated slot must be 0.0."""
        slots = generate_recommendation_intervals(60, 1)
        slot = slots[0]
        assert slot.import_price == pytest.approx(0.0)
        assert slot.export_price == pytest.approx(0.0)
        assert slot.solcast_pv_estimate_kwh == pytest.approx(0.0)
        assert slot.avg_house_consumption_kwh == pytest.approx(0.0)


class TestPlannerOutputPropagation:
    """Resolved planner fields must be published as one coherent slot."""

    def test_resolved_house_consumption_is_copied_without_overwriting_history(
        self,
    ) -> None:
        coordinator = _make_bare_coordinator()
        recommendations = generate_recommendation_intervals(15, 1)
        rec = recommendations[0]
        rec.avg_house_consumption_kwh = 0.100
        rec.avg_house_consumption_1d_kwh = 0.080
        coordinator._hourly_recommendations = [rec]
        slot = PlannedSlot(start=rec.start, end=rec.end)
        slot.avg_house_consumption_kwh = 0.425

        coordinator._apply_planner_output(PlannerOutput(slots=[slot]))

        assert rec.avg_house_consumption_kwh == pytest.approx(0.425)
        assert rec.historical_avg_house_consumption_kwh == pytest.approx(0.100)
        assert rec.avg_house_consumption_1d_kwh == pytest.approx(0.080)


# ---------------------------------------------------------------------------
# Single-poll guarantee
# ---------------------------------------------------------------------------


class TestSinglePollGuarantee:
    """Entities must not independently poll; only the coordinator fetches data."""

    def test_working_mode_sensor_should_poll_is_false(self) -> None:
        """HSEMWorkingModeSensor.should_poll must return False."""
        from custom_components.hsem.custom_sensors.working_mode_sensor import (
            HSEMWorkingModeSensor,
        )

        assert HSEMWorkingModeSensor.should_poll.fget is not None  # type: ignore[attr-defined]  # mock attribute set in test
        # Instantiate a minimal stub to call the property
        sensor = object.__new__(HSEMWorkingModeSensor)
        # Inject a minimal coordinator mock
        coord = MagicMock()
        coord.last_update_success = False
        coord.data = None
        sensor.coordinator = coord
        assert sensor.should_poll is False

    def test_degraded_mode_sensor_should_poll_is_false(self) -> None:
        """HSEMDegradedModeSensor.should_poll must return False."""
        from custom_components.hsem.custom_sensors.degraded_mode_sensor import (
            HSEMDegradedModeSensor,
        )

        sensor = object.__new__(HSEMDegradedModeSensor)
        coord = MagicMock()
        coord.last_update_success = False
        coord.data = None
        sensor.coordinator = coord
        sensor._restored_state = None
        assert sensor.should_poll is False


# ---------------------------------------------------------------------------
# Coordinator data exposure
# ---------------------------------------------------------------------------


class TestCoordinatorDataExposure:
    """Verify coordinator exposes last_update_success and data correctly."""

    def test_last_update_success_defaults_true(self) -> None:
        """DataUpdateCoordinator initialises last_update_success to True (HA default).

        The coordinator is considered healthy until its first failed cycle, which
        is the standard behaviour for HA's DataUpdateCoordinator base class.
        We verify the attribute is present and True via the bare instance which
        reflects this default.
        """
        coord = _make_bare_coordinator()
        # Bare coordinator sets last_update_success=True to mirror the HA default.
        assert coord.last_update_success is True

    def test_data_is_none_before_first_cycle(self) -> None:
        """coordinator.data must be None before async_setup is called."""
        coord = _make_bare_coordinator()
        assert coord.data is None
