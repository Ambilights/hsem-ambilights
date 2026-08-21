"""DataUpdateCoordinator for the HSEM integration.

Single responsibility: run the shared HSEM polling pipeline once per interval
and expose the result as :class:`CoordinatorData` so that all subscribing
entities can read from one consistent snapshot.

Pipeline stages owned by the coordinator:

1. Reload config from the config entry.
2. Collect live HA entity states (:mod:`state_collector`).
3. Reset and generate recommendation time-slots.
4. Populate weighted house-consumption averages.
5. Populate electricity prices and Solcast PV estimates.
6. Run the pure-Python planner engine.
7. Resolve the current time-slot recommendation.

Hardware writes (inverter + battery commands) are **not** performed here; they
remain in :class:`~custom_components.hsem.custom_sensors.working_mode_sensor.HSEMWorkingModeSensor`
so that a "read_only" or "degraded mode" guard can still gate them at the entity
level.

Usage
-----
The coordinator is created in :func:`custom_components.hsem.__init__.async_setup_entry`
and stored on ``entry.runtime_data.coordinator``.  Each sensor platform retrieves
it from the config entry and passes it to the relevant entity constructors.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.event import (
    async_track_point_in_utc_time,
    async_track_time_change,
    async_track_time_interval,
)
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from custom_components.hsem.const import (
    EMA_ALPHA_NET_CONSUMPTION,
)
from custom_components.hsem.coordinator_builder import (
    build_planner_input,
    generate_recommendation_intervals,
)
from custom_components.hsem.custom_sensors.hourly_data_populator.consumption import (
    populate_avg_house_consumption_from_snapshot,
)
from custom_components.hsem.custom_sensors.hourly_data_populator.prices_solcast import (
    populate_price_and_solcast_from_snapshot,
)
from custom_components.hsem.custom_sensors.state_collector import (  # noqa: F401 — kept for backward compat
    async_collect_all_states,
    async_collect_live_state,
    build_sensor_config,
)
from custom_components.hsem.models.daily_metrics import DailyMetrics
from custom_components.hsem.models.daily_plan_vs_actual_tracker import (
    ActualPriceInterval,
    DailyPlanVsActualTracker,
)
from custom_components.hsem.models.data_quality import DataQuality
from custom_components.hsem.models.financial_tracker import FinancialTracker
from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.plan_explanation import PlanExplanation
from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.planner_input import PlannerInput
from custom_components.hsem.models.planner_output import PlannerOutput
from custom_components.hsem.models.price_source import PriceBackupStatus, PriceSource
from custom_components.hsem.models.savings_tracker import SavingsTracker
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.models.state_snapshot import StateSnapshot
from custom_components.hsem.planner import run_planner
from custom_components.hsem.planner.charge_scheduler import apply_window_hysteresis
from custom_components.hsem.planner.ev_planner import EVChargingPlan
from custom_components.hsem.planner.secondary_storage import (
    SECONDARY_MODE_CHARGE,
    SECONDARY_MODE_UTILITY,
)
from custom_components.hsem.utils.capacity_learner import CapacityLearner
from custom_components.hsem.utils.datetime_utils import (
    as_tz,
    now as hsem_now,
    slot_contains,
    slot_key,
    utc_key,
    utc_now_iso,
)
from custom_components.hsem.utils.degraded_mode import DegradedMode
from custom_components.hsem.utils.dynamic_floor import DynamicDischargeFloor
from custom_components.hsem.utils.forecast_tracker import ForecastTracker
from custom_components.hsem.utils.ha_helpers import ha_get_entity_state_and_convert
from custom_components.hsem.utils.inverter_verify import CycleApplySummary
from custom_components.hsem.utils.logger import (
    HSEM_LOGGER as _LOGGER,
    async_log,
    set_hsem_verbose,
)
from custom_components.hsem.utils.misc import (
    clamp_efficiency,
    ema_filter,
    get_config_value,
)
from custom_components.hsem.utils.prediction_tracker import (
    PredictionTracker,
    _action_label,
)
from custom_components.hsem.utils.price_forecast import build_price_forecast
from custom_components.hsem.utils.recommendations import CHARGE_RECS, Recommendations
from custom_components.hsem.utils.solar_corrector import SolarForecastCorrector
from custom_components.hsem.utils.units import (
    is_material_planned_energy_kwh,
    slot_duration_hours,
    usable_kwh_from_rated,
)
from custom_components.hsem.utils.weekday_profile import weekday_profile

if TYPE_CHECKING:
    from custom_components.hsem.ml.consumption_predictor import ConsumptionPredictor


# Seconds to wait after the last options change before scheduling a planner
# run.  Rapid switch/number/time toggles restart this timer, so the planner
# only rebuilds once after the user stops clicking.
OPTIONS_UPDATE_DEBOUNCE_SECONDS = 0.25

# Secondary-storage state entities can update every few seconds.  Their
# listener callback filters noisy telemetry against the last accepted plan,
# then coalesces genuinely material changes into one coordinator cycle.
SECONDARY_STORAGE_UPDATE_DEBOUNCE_SECONDS = 1.0
SECONDARY_STORAGE_SOC_REPLAN_DELTA_PCT = 1.0
SECONDARY_STORAGE_LOAD_REPLAN_DELTA_W = 25.0
PRICE_SOURCE_UPDATE_DEBOUNCE_SECONDS = 0.25
# Phase meters update roughly every ten seconds.  While a charge actuator is
# actually running, publish a fresh live snapshot promptly so the runtime cap
# tracks the real load, without re-solving the 192-slot horizon.  Both delays
# rate-limit that refresh so telemetry churn cannot drive continuous writes.
PHASE_SAFETY_UPDATE_DEBOUNCE_SECONDS = 2.0
PHASE_SAFETY_UPDATE_COOLDOWN_SECONDS = 8.0

# Lightweight live-demand monitor for active forced export and materially
# partial normal discharge. It is deliberately separate from the normal
# coordinator interval: only two HA states are read on each tick, and the
# expensive collection/planner pipeline runs only after a material excess has
# persisted for the debounce period.
FORCE_DISCHARGE_MONITOR_INTERVAL_SECONDS = 10
FORCE_DISCHARGE_EXCESS_DEBOUNCE_SECONDS = 30
FORCE_DISCHARGE_EXCESS_MIN_W = 150.0
FORCE_DISCHARGE_EXCESS_RELATIVE = 0.10
FORCE_DISCHARGE_REPLAN_MIN_REMAINING_SECONDS = 60
CORRECTIVE_MILP_SOLVER_STATUSES = frozenset(
    {"optimal", "time_limit_feasible_incumbent"}
)
LOAD_FORECAST_LIVE_DEMAND_THRESHOLD_W = 50.0
LOAD_FORECAST_ZERO_EPSILON_KWH = 1e-9

type _PriceChannelSignature = tuple[bool, float | None]
type _PriceSlotSignature = tuple[
    str,
    _PriceChannelSignature,
    _PriceChannelSignature,
    _PriceChannelSignature,
    PriceSource | None,
    PriceSource | None,
]
type _ValuationForecastSignature = tuple[
    bool,
    float,
    float,
    tuple[tuple[str, float], ...],
]
type _PriceForecastSignature = tuple[
    _PriceChannelSignature,
    _PriceChannelSignature,
    tuple[_PriceSlotSignature, ...],
    _ValuationForecastSignature,
]
type _LoadSlotSignature = tuple[str, float, float, float, float, float]
type _LoadForecastSignature = tuple[_LoadSlotSignature, ...]


@dataclass(frozen=True, slots=True)
class _ForecastAuthorityPlanState:
    """Accepted plan state restored when a late authority event aborts a cycle."""

    last_planner_input: PlannerInput | None
    last_planner_output: PlannerOutput | None
    last_plan_slot_start: datetime | None
    previous_planner_winner_name: str | None
    previous_planner_winner_score: float
    window_hys_previous_rec: str | None
    window_hys_previous_slot_start: datetime | None
    window_hysteresis_expiry: datetime | None
    current_slot_start: datetime | None
    current_slot_price_actionable: bool | None
    current_slot_ev_power_w: float
    current_slot_ev_second_power_w: float
    current_required_battery: float
    plan_explanation: PlanExplanation
    data_quality: DataQuality
    ev_charging_plan: EVChargingPlan | None
    ev_second_charging_plan: EVChargingPlan | None
    hourly_recommendation: HourlyRecommendation | None
    hourly_recommendations: list[HourlyRecommendation]


@dataclass(frozen=True)
class _LoadForecastReadiness:
    """Validated future consumption profile and its accepted-plan signature."""

    ready: bool
    reason: str | None
    signature: _LoadForecastSignature | None


def _assess_load_forecast(
    recommendations: Sequence[HourlyRecommendation],
    now: datetime,
    *,
    population_succeeded: bool,
    live_house_demand_w: float | None,
) -> _LoadForecastReadiness:
    """Validate future load provenance without rejecting genuine measured zero."""
    if not population_succeeded:
        return _LoadForecastReadiness(False, "source_unavailable", None)

    now_utc = utc_key(now)
    future_slots = sorted(
        (rec for rec in recommendations if utc_key(rec.end) > now_utc),
        key=lambda rec: utc_key(rec.start),
    )
    if not future_slots:
        return _LoadForecastReadiness(False, "missing_future_slots", None)

    signature_slots: list[_LoadSlotSignature] = []
    weighted_profile_is_zero = True
    for rec in future_slots:
        raw_values = (
            rec.avg_house_consumption_kwh,
            rec.avg_house_consumption_1d_kwh,
            rec.avg_house_consumption_3d_kwh,
            rec.avg_house_consumption_7d_kwh,
            rec.avg_house_consumption_14d_kwh,
        )
        canonical_values: list[float] = []
        for raw_value in raw_values:
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                return _LoadForecastReadiness(False, "invalid_future_values", None)
            number = float(raw_value)
            if not math.isfinite(number) or number < 0.0:
                return _LoadForecastReadiness(False, "invalid_future_values", None)
            canonical_values.append(round(number, 5))

        weighted, avg_1d, avg_3d, avg_7d, avg_14d = canonical_values
        if weighted > LOAD_FORECAST_ZERO_EPSILON_KWH:
            weighted_profile_is_zero = False
        signature_slots.append(
            (
                utc_key(rec.start).isoformat(),
                weighted,
                avg_1d,
                avg_3d,
                avg_7d,
                avg_14d,
            )
        )

    finite_live_demand_w = (
        float(live_house_demand_w)
        if live_house_demand_w is not None
        and not isinstance(live_house_demand_w, bool)
        and math.isfinite(live_house_demand_w)
        else None
    )
    if (
        weighted_profile_is_zero
        and finite_live_demand_w is not None
        and finite_live_demand_w > LOAD_FORECAST_LIVE_DEMAND_THRESHOLD_W
    ):
        return _LoadForecastReadiness(False, "zero_forecast_with_live_demand", None)

    return _LoadForecastReadiness(True, None, tuple(signature_slots))


def _load_forecast_signatures_match(
    current: _LoadForecastSignature | None,
    baseline: _LoadForecastSignature | None,
) -> bool:
    """Return whether two load signatures match within the canonical epsilon."""
    if current is None or baseline is None:
        return current is baseline
    if len(current) != len(baseline):
        return False
    for current_slot, baseline_slot in zip(current, baseline, strict=True):
        if current_slot[0] != baseline_slot[0]:
            return False
        if any(
            not math.isclose(
                current_value,
                baseline_value,
                rel_tol=0.0,
                abs_tol=LOAD_FORECAST_ZERO_EPSILON_KWH,
            )
            for current_value, baseline_value in zip(
                current_slot[1:], baseline_slot[1:], strict=True
            )
        ):
            return False
    return True


def _canonical_price_channel(
    value: float | None, available: bool
) -> _PriceChannelSignature:
    """Return a stable signature that distinguishes zero from unavailable."""
    if not available or value is None:
        return (False, None)
    number = float(value)
    if not math.isfinite(number):
        return (False, None)
    return (True, round(number, 5))


def _price_forecast_attributes(
    snapshot: StateSnapshot | None,
    cfg: SensorConfig,
) -> dict[str, Any] | None:
    """Return the price-forecast sensor's attributes from the cycle snapshot.

    Returns None when the feature is off, no sensor is configured, or the
    entity was unreadable — all of which the parser treats as "no forecast
    contribution" rather than as a reason to fail the cycle.
    """
    entity_id = cfg.price_forecast_valuation_sensor
    if not cfg.price_forecast_valuation_enabled or not entity_id:
        return None
    if snapshot is None:
        return None
    return snapshot.sensor_attributes.get(entity_id)


def _price_forecast_signature(
    recommendations: list[HourlyRecommendation],
    live: LiveState,
    now: datetime,
    *,
    valuation_attributes: dict[str, Any] | None = None,
    valuation_enabled: bool = False,
    valuation_margin: float = 0.0,
) -> _PriceForecastSignature:
    """Return the canonical live and future populated price/PV authority."""
    now_utc = utc_key(now)
    future_slots = sorted(
        (rec for rec in recommendations if utc_key(rec.end) > now_utc),
        key=lambda rec: utc_key(rec.start),
    )
    slots: tuple[_PriceSlotSignature, ...] = tuple(
        (
            utc_key(rec.start).isoformat(),
            _canonical_price_channel(rec.import_price, rec.import_price_available),
            _canonical_price_channel(rec.export_price, rec.export_price_available),
            _canonical_price_channel(
                rec.solcast_pv_estimate_kwh,
                rec.solcast_pv_estimate_available,
            ),
            rec.import_price_source if rec.import_price_available else None,
            rec.export_price_source if rec.export_price_available else None,
        )
        for rec in future_slots
    )
    valuation = build_price_forecast(
        valuation_attributes,
        enabled=valuation_enabled,
        margin=valuation_margin,
    )
    valuation_signature: _ValuationForecastSignature = (
        valuation.enabled,
        round(valuation.mae, 5),
        round(valuation.margin, 5),
        tuple(
            sorted(
                (
                    utc_key(point.start).isoformat(),
                    round(point.value, 5),
                )
                for point in valuation.points
            )
        ),
    )
    return (
        _canonical_price_channel(
            live.import_electricity_price,
            live.import_electricity_price_available,
        ),
        _canonical_price_channel(
            live.export_electricity_price,
            live.export_electricity_price_available,
        ),
        slots,
        valuation_signature,
    )


def _apply_live_current_price_availability(
    recommendations: list[HourlyRecommendation],
    live: LiveState,
    now: datetime,
) -> None:
    """Intersect current forecast authority with the live entity states.

    Forecast attributes can remain present after Home Assistant marks the
    underlying price entity unavailable.  They must not keep the current slot
    actionable: retain each populated forecast value, but withdraw its
    authority unless both the forecast channel and the live channel are
    currently available and finite.  A live channel never promotes a missing
    forecast channel.
    """
    current = next(
        (rec for rec in recommendations if slot_contains(rec.start, rec.end, now)),
        None,
    )
    if current is None:
        return

    current.import_price_available = bool(
        current.import_price_available
        and live.import_electricity_price_available
        and math.isfinite(live.import_electricity_price)
    )
    current.export_price_available = bool(
        current.export_price_available
        and live.export_electricity_price_available
        and math.isfinite(live.export_electricity_price)
    )
    current.price_actionable = bool(
        current.import_price_available and current.export_price_available
    )


def _set_strict_storage_hold(
    current: HourlyRecommendation,
) -> HourlyRecommendation:
    """Clear plan-derived storage motion and publish explicit hold intent."""
    current.recommendation = Recommendations.BatteriesWaitMode.value
    current.primary_battery_hold = True
    current.batteries_charged_kwh = 0.0
    current.batteries_discharged_kwh = 0.0
    current.secondary_storage_mode = SECONDARY_MODE_UTILITY
    current.secondary_storage_charge_current_a = 0.0
    current.secondary_storage_charged_kwh = 0.0
    current.secondary_storage_discharged_kwh = 0.0
    current.secondary_storage_grid_import_kwh = 0.0
    return current


def _apply_current_price_outage_hold(
    recommendations: list[HourlyRecommendation],
    live: LiveState,
    now: datetime,
) -> HourlyRecommendation | None:
    """Publish a strict current-slot storage hold during a price outage."""
    if str(live.force_working_mode_state).strip().lower() != "auto":
        return None
    current = next(
        (rec for rec in recommendations if slot_contains(rec.start, rec.end, now)),
        None,
    )
    if current is None or current.price_actionable:
        return None
    return _set_strict_storage_hold(current)


def _apply_load_forecast_hold(
    recommendations: list[HourlyRecommendation],
    live: LiveState,
    now: datetime,
    *,
    load_forecast_ready: bool,
) -> HourlyRecommendation | None:
    """Publish a strict current-slot hold while the load forecast is unsafe."""
    if load_forecast_ready:
        return None
    if str(live.force_working_mode_state).strip().lower() != "auto":
        return None
    current = next(
        (rec for rec in recommendations if slot_contains(rec.start, rec.end, now)),
        None,
    )
    if current is None:
        return None
    return _set_strict_storage_hold(current)


def _next_slot_boundary_utc(now: datetime, interval_minutes: int) -> datetime:
    """Return the next future wall-clock-aligned boundary in UTC.

    Iterating on the UTC timeline keeps the result strictly in the future at
    both sides of a DST fold while still testing alignment in local wall time.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    interval = max(int(interval_minutes), 1)
    candidate = now.astimezone(UTC).replace(second=0, microsecond=0) + timedelta(
        minutes=1
    )
    for _ in range(24 * 60 + 1):
        local_candidate = candidate.astimezone(now.tzinfo)
        elapsed_minutes = local_candidate.hour * 60 + local_candidate.minute
        if elapsed_minutes % interval == 0:
            return candidate
        candidate += timedelta(minutes=1)
    raise RuntimeError("unable to find the next recommendation boundary")


def _auto_full_negative_price_allowed(
    live: LiveState,
    slot: HourlyRecommendation | None,
) -> bool:
    """Return whether a published current price authorizes Auto-Full EV."""
    return bool(
        slot is not None
        and slot.price_actionable
        and live.import_electricity_price_available
        and live.import_electricity_price <= 0.0
    )


def _force_discharge_live_metrics(
    slot: PlannedSlot,
    *,
    discharge_efficiency_pct: float,
    house_power_w: float,
    solar_power_w: float,
) -> tuple[float, float, float, float] | None:
    """Return planned supply, live residual, threshold, and excess in watts.

    ``batteries_discharged_kwh`` is battery-side energy.  The live comparison
    is AC-side house demand after PV, so the planned delivery is multiplied by
    discharge efficiency before conversion to average slot power.
    """
    duration_h = slot_duration_hours(slot.start, slot.end)
    discharged_kwh = max(float(slot.batteries_discharged_kwh), 0.0)
    if duration_h <= 0.0 or discharged_kwh <= 1e-9:
        return None

    planned_ac_w = (
        discharged_kwh
        * clamp_efficiency(discharge_efficiency_pct)
        / duration_h
        * 1000.0
    )
    planned_grid_import_w = 0.0
    if slot.recommendation == Recommendations.BatteriesDischargeMode.value:
        grid_import_w = max(float(slot.grid_import_kwh), 0.0) / duration_h * 1000.0
        if is_material_planned_energy_kwh(slot.grid_import_kwh):
            planned_grid_import_w = grid_import_w

    planned_demand_w = planned_ac_w + planned_grid_import_w
    live_residual_w = max(house_power_w - solar_power_w, 0.0)
    threshold_w = max(
        FORCE_DISCHARGE_EXCESS_MIN_W,
        FORCE_DISCHARGE_EXCESS_RELATIVE * planned_demand_w,
    )
    excess_w = live_residual_w - planned_demand_w
    return planned_demand_w, live_residual_w, threshold_w, excess_w


def _corrective_output_rejection_reason(
    output: PlannerOutput,
    slot_start: datetime,
) -> str:
    """Return why a live-demand candidate must not replace the active plan."""
    if output.winner_name != "milp":
        return "winner_not_milp"
    if output.explanation.solver_status not in CORRECTIVE_MILP_SOLVER_STATUSES:
        return f"solver_{output.explanation.solver_status}"
    if output.explanation.fallback_reason:
        return f"fallback_{output.explanation.fallback_reason}"

    slot = next(
        (item for item in output.slots if utc_key(item.start) == utc_key(slot_start)),
        None,
    )
    if slot is None:
        return "active_slot_missing"

    if slot.recommendation == Recommendations.BatteriesDischargeMode.value:
        # Only switch Force -> MSC when the solved current slot allocates real
        # battery discharge. A partial allocation that intentionally retains
        # grid import is executable because the applier caps MSC to the solved
        # discharge energy; a zero-discharge label is not executable intent.
        if slot.batteries_discharged_kwh <= 1e-9:
            return "msc_without_planned_discharge"
    return ""


def _select_corrective_planner_output(
    previous: PlannerOutput,
    candidate: PlannerOutput,
    slot_start: datetime,
    *,
    price_authority_changed: bool = False,
) -> tuple[PlannerOutput, bool, str]:
    """Keep the prior plan unless the corrective candidate is authoritative."""
    if price_authority_changed:
        # Retaining a previously validated plan would also retain its stale
        # per-slot availability/actionability flags and could execute an old
        # force-export decision after prices were withdrawn.  The fresh
        # fallback is the only output derived from current price authority.
        return candidate, True, ""
    rejection_reason = _corrective_output_rejection_reason(candidate, slot_start)
    accepted = not rejection_reason
    return (candidate if accepted else previous), accepted, rejection_reason


# ---------------------------------------------------------------------------
# Lightweight slot for dynamic floor bridge computation
# ---------------------------------------------------------------------------


@dataclass
class _SimpleSlot:
    """Minimal slot for DynamicDischargeFloor.compute_floor().

    Carries only the fields needed by the bridge computation.
    """

    start: datetime
    end: datetime
    estimated_net_consumption_kwh: float
    batteries_charged_kwh: float
    recommendation: str | None


# ---------------------------------------------------------------------------
# Force-charge-now override helper
# ---------------------------------------------------------------------------


_EV_FORCE_CHARGE_RESCUED_STATES: frozenset[str] = frozenset(
    {
        "smart_charging_disabled",
        "waiting",
        "not_connected",
        "fully_charged",
    }
)
"""Plan states a forced charge overrides.

Every state except ``charging`` itself: the user has explicitly asked for
power now, so neither a deferred schedule ("waiting"), a disabled feature, a
connection the integration has not noticed, nor a target already believed met
should keep the charger off.
"""


def _apply_force_charge_now(
    *,
    config_entry: ConfigEntry,
    hourly_recommendations: list[HourlyRecommendation],
    ev_plan: EVChargingPlan | None,
    ev_second_plan: EVChargingPlan | None,
    now: datetime,
) -> None:
    """Apply the force-charge-now override to the current slot.

    When the user toggles ``hsem_ev_force_charge_now`` (or the second-EV
    equivalent), the current slot's recommendation is overridden to
    ``ev_smart_charging`` and the calculated charger power is set to the
    charger's maximum AC power.

    Crucially, force-charge works **even when smart charging is disabled**.
    The EV planner returns ``smart_charging_disabled`` with zero allocated
    power in that case, so this function also flips the plan state to
    ``charging`` so the plan sensor reflects the forced charge.

    Args:
        config_entry: The HSEM config entry (to read the force-charge switches).
        hourly_recommendations: The list of hourly recommendations to modify.
        ev_plan: The primary EV charging plan (may be ``None``).
        ev_second_plan: The second EV charging plan (may be ``None``).
        now: Current time (timezone-aware), used to locate the current slot.
    """
    force_primary = bool(get_config_value(config_entry, "hsem_ev_force_charge_now"))
    force_second = bool(
        get_config_value(config_entry, "hsem_ev_second_force_charge_now")
    )
    if not force_primary and not force_second:
        return

    now_slot = next(
        (r for r in hourly_recommendations if slot_contains(r.start, r.end, now)),
        None,
    )
    if now_slot is None:
        return

    if force_primary:
        now_slot.recommendation = Recommendations.EVSmartCharging.value
        pwr_kw = float(
            get_config_value(
                config_entry,
                "hsem_ev_planned_load_charger_power_kw",
            )
            or 0.0
        )
        now_slot.ev_charger_calculated_power = (
            round(pwr_kw * 1000) if pwr_kw > 0 else 0.0
        )
        # Flip the plan state so the sensor reports "charging" whenever the
        # user forces a charge.  Automations gate on this state, so any
        # non-charging state must be rescued — not just
        # "smart_charging_disabled".  "waiting" is the common case: the
        # planner has scheduled the charge for a later slot, and a forced
        # charge must override that, which is the whole point of the switch.
        if ev_plan is not None and ev_plan.state in _EV_FORCE_CHARGE_RESCUED_STATES:
            ev_plan.state = "charging"
        async_log(
            "debug",
            "[coordinator] Force-Charge-Now: primary EV "
            "→ overriding current slot to ev_smart_charging at %dW",
            now_slot.ev_charger_calculated_power,
        )

    if force_second:
        now_slot.recommendation = Recommendations.EVSmartCharging.value
        pwr_kw = float(
            get_config_value(
                config_entry,
                "hsem_ev_second_planned_load_charger_power_kw",
            )
            or 0.0
        )
        now_slot.ev_second_charger_calculated_power = (
            round(pwr_kw * 1000) if pwr_kw > 0 else 0.0
        )
        # Flip the plan state so the sensor reports "charging" (see the
        # primary-EV branch above for why every non-charging state is
        # rescued, not only "smart_charging_disabled").
        if (
            ev_second_plan is not None
            and ev_second_plan.state in _EV_FORCE_CHARGE_RESCUED_STATES
        ):
            ev_second_plan.state = "charging"
        async_log(
            "debug",
            "[coordinator] Force-Charge-Now: second EV "
            "→ overriding current slot to ev_smart_charging at %dW",
            now_slot.ev_second_charger_calculated_power,
        )


# ---------------------------------------------------------------------------
# Data payload exposed to subscriber entities
# ---------------------------------------------------------------------------


@dataclass
class CoordinatorData:
    """Snapshot of a single HSEM update cycle.

    All fields are read-only from the perspective of subscribing entities.
    The coordinator replaces this object atomically at the end of every cycle.

    Attributes:
        cfg: Configuration values read from the config entry.
        live: Live HA entity state snapshot collected at the start of the cycle.
        hourly_recommendations: Full list of planner recommendation slots.
        hourly_recommendation: The recommendation slot active *right now*, or
            ``None`` when no matching slot exists.
        current_required_battery: Required battery capacity from the planner (kWh).
        state: Working-mode recommendation string for the current slot, or one
            of the :class:`~utils.recommendations.Recommendations` sentinel values.
        last_updated: ISO-format timestamp of the cycle that produced this data.
        next_update: ISO-format timestamp of the *next* scheduled cycle.
    """

    cfg: SensorConfig | None = None
    live: LiveState | None = None
    hourly_recommendations: list[HourlyRecommendation] = field(default_factory=list)
    hourly_recommendation: HourlyRecommendation | None = None
    current_required_battery: float = 0.0
    state: str | None = None
    last_updated: str | None = None
    next_update: str | None = None
    #: Aggregated write-and-verify results from the most recent hardware apply cycle.
    #: ``None`` before the first hardware-write cycle completes.
    apply_summary: CycleApplySummary | None = None
    #: Human-readable explanation of why the selected plan was chosen.
    plan_explanation: PlanExplanation = field(default_factory=PlanExplanation)
    #: Structured data-quality report for price, PV, and load-forecast inputs.
    data_quality: DataQuality = field(default_factory=DataQuality)
    #: Selection/rejection status for the paired ENTSO-E published-price backup.
    entsoe_price_backup_status: PriceBackupStatus = field(
        default_factory=PriceBackupStatus
    )
    #: EV optimal charging plan for the primary EV (None when disabled).
    ev_charging_plan: EVChargingPlan | None = None
    #: EV optimal charging plan for the second EV (None when disabled).
    ev_second_charging_plan: EVChargingPlan | None = None
    #: ISO-format timestamp of the override expiry, or None when no timed
    #: override is active (issue #317).
    override_expiry: str | None = None
    #: Savings tracker with actual vs missed savings metrics.
    savings_tracker: SavingsTracker = field(default_factory=SavingsTracker)
    #: Prediction accuracy tracker reference (SoC/MAE/action-mix scorecard, issue #601).
    prediction_tracker: PredictionTracker | None = None
    #: Capacity learner for auto-detecting battery usable capacity from
    #: BMS kWh-remaining and SoC readings.
    capacity_learner: CapacityLearner = field(default_factory=CapacityLearner)
    #: Per-hour solar forecast accuracy factors (0-23 → factor).
    #: Used by the solar confidence diagnostic sensor (issue #602).
    solar_hour_factors: dict[int, float] = field(default_factory=dict)
    #: Effective dynamic discharge floor SoC percentage, or None when the
    #: feature is disabled.  Computed by DynamicDischargeFloor.compute_floor().
    effective_discharge_floor_pct: float | None = None
    #: Diagnostics dict from the dynamic floor computation, or None when
    #: the feature is disabled.
    effective_discharge_floor_diag: dict | None = None
    #: Financial tracker with cumulative import cost and export income.
    financial_tracker: FinancialTracker | None = None


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class HSEMDataUpdateCoordinator(DataUpdateCoordinator[CoordinatorData]):
    """DataUpdateCoordinator for HSEM.

    Manages the shared polling lifecycle:

    - Registers a periodic timer (``update_interval`` minutes from config) via
      :func:`~homeassistant.helpers.event.async_track_time_interval`.
    - Registers an hourly time-change listener at HH:00:10 to guarantee an
      update at the top of every hour even if the interval timer drifts.
    - Runs the full pipeline under an :class:`asyncio.Lock` so that concurrent
      triggers (e.g. a state-change event arriving during an in-progress cycle)
      are silently dropped rather than queued.

    Entities subscribe via
    :class:`~homeassistant.helpers.update_coordinator.CoordinatorEntity` and
    receive a push notification each time :attr:`data` is refreshed.
    """

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialise the coordinator.

        Args:
            hass: The Home Assistant instance.
            config_entry: The HSEM config entry whose options drive the pipeline.
        """
        super().__init__(
            hass,
            _LOGGER,
            name="HSEM",
            # DataUpdateCoordinator manages an internal timer; we build our own
            # interval timer below for dynamic interval support, so set None to
            # disable the built-in timer entirely (Bronze rule: appropriate-polling).
            update_interval=None,
        )
        self._config_entry = config_entry

        # Lock prevents concurrent executions of the update pipeline.
        self._update_lock = asyncio.Lock()

        # Timer handles — cancelled/re-registered when the interval changes.
        self._interval_timer_unsub: Callable[[], None] | None = None
        self._hourly_timer_unsub: Callable[[], None] | None = None
        self._force_discharge_monitor_unsub: Callable[[], None] | None = None
        self._slot_boundary_timer_unsub: Callable[[], None] | None = None
        self._slot_boundary_interval_minutes: int | None = None
        self._window_hysteresis_timer_unsub: Callable[[], None] | None = None
        self._window_hysteresis_expiry: datetime | None = None
        self._window_hysteresis_expiry_replan_pending: bool = False
        self._tearing_down: bool = False
        self._timer_interval: timedelta | None = None

        # Per-cycle mutable state (not exposed directly; packaged into CoordinatorData).
        self._cfg: SensorConfig = build_sensor_config(config_entry)
        self._live: LiveState | None = None
        self._snapshot: StateSnapshot | None = None
        self._hourly_recommendations: list[HourlyRecommendation] = []
        self._hourly_recommendation: HourlyRecommendation | None = None
        self._current_required_battery: float = 0.0
        self._next_update: str | None = None
        # Most recent completed hardware apply.  Hardware writes finish after
        # the coordinator snapshot is first published, so retain their result
        # independently and copy it into every later snapshot.
        self._last_apply_summary: CycleApplySummary | None = None

        # Entity resolution cache (persisted across cycles).
        self._force_working_mode_entity: str | None = None
        self._tracked_entities: set[str] = set()
        # Unsubscribe callbacks for state-change listeners registered via
        # state_collector._register_listeners.  Cancelled during async_teardown.
        self._listener_unsubs: list = []
        self._avg_house_consumption_entity_id_cache: dict[str, str] = {}
        # Most recent plan explanation produced by the planner engine.
        self._plan_explanation: PlanExplanation = PlanExplanation()
        # Most recent data quality report produced by the planner engine.
        self._data_quality: DataQuality = DataQuality()
        self._entsoe_price_backup_status: PriceBackupStatus = PriceBackupStatus()
        # Most recent EV charging plans from the planner engine.
        self._ev_charging_plan: EVChargingPlan | None = None
        self._ev_second_charging_plan: EVChargingPlan | None = None
        # Most recent planner input/output retained for diagnostics dumps.
        self._last_planner_input: PlannerInput | None = None
        self._last_planner_output: PlannerOutput | None = None
        self._price_source_update_debounce_task: asyncio.Task[None] | None = None
        self._price_source_update_pending: bool = False
        # Monotonic generation for registered price, PV, and valuation-source
        # events. A cycle captures this before collecting its snapshot so an
        # in-flight solve cannot publish commands after newer authority arrives.
        self._forecast_authority_generation: int = 0
        self._phase_safety_update_debounce_task: asyncio.Task[None] | None = None
        self._phase_safety_update_pending: bool = False

        # Previous planner winner name and score for hysteresis (issue #372).
        # Persisted across cycles so the planner can compare against the
        # previously active plan.
        self._previous_planner_winner_name: str | None = None
        self._previous_planner_winner_score: float = 0.0

        # Window-level hysteresis state (issue #315).
        # Persisted across cycles so the hold-time check can compare against
        # the previously active current-slot recommendation.

        # EMA-smoothed live net consumption (W).  Damped so transients
        # (støvsuger, kaffemaskine, cloud shadows) don't kill the EV
        # charging setpoint for the rest of a 15-minute slot.  Initialised
        # on the first cycle and updated every subsequent cycle.
        self._net_consumption_ema: float | None = None
        self._window_hys_previous_rec: str | None = None
        self._window_hys_previous_slot_start: datetime | None = None

        # Per-slot EV charger power freeze (issue #738).
        # The EV planner recomputes ev_charger_calculated_power whenever the
        # planner reruns, mixing live PV/consumption data into the current
        # slot. That makes the charger command oscillate inside a 15-minute
        # slot. We freeze the value at slot start and reuse it across replans
        # until the next slot begins. Explicit overrides (force-charge-now,
        # auto-full-EV) are applied on top of the frozen value each cycle.
        self._current_slot_start: datetime | None = None
        self._current_slot_price_actionable: bool | None = None
        self._current_slot_ev_power_w: float = 0.0
        self._current_slot_ev_second_power_w: float = 0.0

        # Event-driven re-planning — track state at last plan to avoid
        # re-solving the MILP when nothing material has changed.
        self._last_plan_ev_connected: bool | None = False
        self._last_plan_ev_charging: bool = False
        self._last_plan_ev_soc_below_target: bool = False
        self._last_plan_ev_second_connected: bool | None = False
        self._last_plan_ev_second_charging: bool = False
        self._last_plan_ev_second_soc_below_target: bool = False
        self._last_plan_force_mode: str = "auto"
        self._last_plan_slot_start: datetime | None = None
        self._last_plan_import_price: float | None = None
        self._last_plan_price_forecast_signature: _PriceForecastSignature | None = None
        self._last_plan_load_forecast_signature: _LoadForecastSignature | None = None
        # Invalid load data arms a durable recovery solve. It is cleared only
        # after a ready plan is accepted and published.
        self._load_forecast_recovery_replan_pending: bool = False
        self._last_load_forecast_readiness_reason: str | None = None
        # EV planned-load config that affects planner optimisation.
        self._last_plan_ev_target_soc: float | None = None
        self._last_plan_ev_smart_charging: bool | None = None
        self._last_plan_ev_deadline: datetime | None = None
        self._last_plan_ev2_target_soc: float | None = None
        self._last_plan_ev2_smart_charging: bool | None = None
        self._last_plan_ev2_deadline: datetime | None = None
        self._last_plan_secondary_soc_pct: float | None = None
        self._last_plan_secondary_load_power_w: float | None = None
        self._last_plan_secondary_output_priority: str | None = None

        # A 10-second lightweight monitor can request one corrective replan in
        # active forced-export or materially partial normal-discharge slots
        # when live residual demand stays above solved supply for 30 s.
        # The completed-slot marker is written only after the corrected
        # coordinator snapshot is published, so a busy coordinator or failed
        # update cannot consume the slot's one permitted attempt.
        self._force_discharge_excess_since: datetime | None = None
        self._force_discharge_excess_slot_start: datetime | None = None
        self._force_discharge_replanned_slot_start: datetime | None = None
        self._force_discharge_live_replan_pending_slot: datetime | None = None

        # Solar forecast accuracy auto-corrector (issue #602).
        self._solar_corrector: SolarForecastCorrector = SolarForecastCorrector()
        # Set of slot start times already fed to the solar corrector.
        self._solar_corrector_processed: set[datetime] = set()
        self._solar_corrector_processed_through: datetime | None = None

        # Forecast-vs-actual tracker (predicted-vs-actual tracking, issue #373).
        self._forecast_tracker: ForecastTracker = ForecastTracker(max_slots=2880)
        self._last_pv_power_w: float | None = None
        self._last_load_power_w: float | None = None
        # Prediction accuracy tracker — SoC/MAE/action-mix scorecard (issue #601).
        self._prediction_tracker: PredictionTracker = PredictionTracker(
            max_records=2880
        )
        # Expired unfinalised slots restored after a restart cannot be paired
        # with a later live SoC sample.
        self._prediction_restore_excluded: set[datetime] = set()
        # Daily plan-vs-actual tracker (diagnostic sensor with 90-day history).
        # The history file path is set in async_setup() once hass.config is available.
        self._daily_tracker: DailyPlanVsActualTracker = DailyPlanVsActualTracker()
        self._daily_tracker_initialized: bool = False
        # Savings tracker (actual vs missed savings with 90-day history).
        self._savings_tracker: SavingsTracker = SavingsTracker()
        self._savings_tracker_initialized: bool = False
        # Slot snapshots that priced the preceding physical charge interval.
        self._savings_price_slots: tuple[PlannedSlot, ...] = ()
        # Financial tracker — cumulative import cost and export income (never reset).
        # The history file path is set in async_setup() once hass.config is available.
        self._financial_tracker: FinancialTracker = FinancialTracker()
        self._financial_tracker_initialized: bool = False
        # Midnight timer unsubscribe handler for daily persistence.
        self._midnight_unsub: Callable[[], None] | None = None
        # Last slot end time accumulated from planner output (prevents double-counting).
        self._daily_plan_last_accumulated: datetime | None = None
        # Timestamp of the last actual-energy accumulation cycle.
        self._last_accumulation_ts: datetime | None = None
        # Override expiry timestamp for timed manual overrides (issue #317).
        # Set by set_temporary_override when duration_minutes is provided.
        # Checked on every update cycle; when expired, the override is cleared
        # automatically and the planner resumes control.
        self._override_expiry: datetime | None = None

        # Dynamic self-learning discharge floor (issue #600).
        self._dynamic_floor: DynamicDischargeFloor = DynamicDischargeFloor()
        self._effective_discharge_floor_pct: float | None = None
        self._effective_discharge_floor_diag: dict | None = None

        # Battery capacity learner (issue #605).
        self._capacity_learner: CapacityLearner = CapacityLearner()

        # ML consumption predictor — cached across cycles so the retrain
        # gate can skip re-fitting when no new history has arrived.
        self._ml_predictor: ConsumptionPredictor | None = None

        # Background task handle for option-change-triggered pipeline runs.
        # Tracked so repeated toggles cancel the pending run and so teardown
        # can cancel a still-running task (issue: switch toggles felt frozen
        # because the update listener awaited the full MILP/ML pipeline).
        self._options_update_task: asyncio.Task | None = None
        # Debounce task for option changes.  Rapid toggles restart this timer
        # so the planner only runs once after the user stops clicking.
        self._options_update_debounce_task: asyncio.Task | None = None
        self._secondary_storage_update_debounce_task: asyncio.Task | None = None

    @callback
    def async_publish_apply_summary(self, summary: CycleApplySummary) -> None:
        """Persist a completed hardware apply and refresh diagnostic entities.

        The hardware worker runs after the normal coordinator notification.
        Updating the current snapshot alone would therefore leave other
        coordinator-backed sensors stale until the next cycle.  Notify the
        listeners immediately; the working-mode entity suppresses hardware
        re-queueing while this diagnostics-only notification is in progress.
        """
        self._last_apply_summary = summary
        if self.data is None:
            return
        self.data.apply_summary = summary
        self.async_update_listeners()

    # ------------------------------------------------------------------
    # HA lifecycle
    # ------------------------------------------------------------------

    async def async_setup(self) -> None:
        """Register timers and run the first update cycle.

        Call this once after the coordinator is created (from
        :func:`~custom_components.hsem.__init__.async_setup_entry`).
        """
        # Initialise the financial tracker — lazy load from disk on first access.
        # This must happen before the first update cycle so the tracker
        # is available when accumulation runs.
        try:
            await self._init_financial_tracker()
        except Exception as e:
            async_log("error", "Failed to initialise financial tracker: %s", e)

        # Run an immediate first cycle so entities have data before first render.
        await self._async_handle_update(None)
        self._schedule_next_slot_boundary(hsem_now())

        # Hourly tick — guarantees a refresh at the top of every hour.
        self._hourly_timer_unsub = async_track_time_change(
            self.hass,
            self._async_handle_update,  # type: ignore[arg-type]  # HA stub expects Callable[[datetime], ...]; our callback also serves as coordinator update callback
            hour="*",
            minute=0,
            second=10,
        )
        self._force_discharge_monitor_unsub = async_track_time_interval(
            self.hass,
            self._async_monitor_force_discharge_load,
            timedelta(seconds=FORCE_DISCHARGE_MONITOR_INTERVAL_SECONDS),
        )

    async def async_teardown(self) -> None:
        """Cancel all registered timers and state-change listeners.

        Called from :func:`~custom_components.hsem.__init__.async_unload_entry`.
        """
        self._tearing_down = True
        # Cancel the base DataUpdateCoordinator's internal refresh timer
        # (set to 24 h as a fallback).  Without this the timer holds a
        # reference to the coordinator and prevents garbage collection.
        unsub_refresh = getattr(self, "_unsub_refresh", None)
        if unsub_refresh is not None:
            unsub_refresh()
        if self._hourly_timer_unsub is not None:
            self._hourly_timer_unsub()
            self._hourly_timer_unsub = None
        if self._interval_timer_unsub is not None:
            self._interval_timer_unsub()
            self._interval_timer_unsub = None
        if self._force_discharge_monitor_unsub is not None:
            self._force_discharge_monitor_unsub()
            self._force_discharge_monitor_unsub = None
        if self._slot_boundary_timer_unsub is not None:
            self._slot_boundary_timer_unsub()
            self._slot_boundary_timer_unsub = None
        self._slot_boundary_interval_minutes = None
        if self._window_hysteresis_timer_unsub is not None:
            self._window_hysteresis_timer_unsub()
            self._window_hysteresis_timer_unsub = None
        self._window_hysteresis_expiry = None
        self._window_hysteresis_expiry_replan_pending = False
        for unsub in self._listener_unsubs:
            unsub()
        self._listener_unsubs.clear()
        midnight = getattr(self, "_midnight_unsub", None)
        if midnight is not None:
            midnight()
            self._midnight_unsub = None

        # Cancel any pending options-update background task and debounce timer.
        task = getattr(self, "_options_update_task", None)
        if task is not None and not task.done():
            task.cancel()
            self._options_update_task = None
        debounce_task = getattr(self, "_options_update_debounce_task", None)
        if debounce_task is not None and not debounce_task.done():
            debounce_task.cancel()
            self._options_update_debounce_task = None
        secondary_task = getattr(self, "_secondary_storage_update_debounce_task", None)
        if secondary_task is not None and not secondary_task.done():
            secondary_task.cancel()
            self._secondary_storage_update_debounce_task = None
        price_task = getattr(self, "_price_source_update_debounce_task", None)
        if price_task is not None and not price_task.done():
            price_task.cancel()
            self._price_source_update_debounce_task = None
        phase_task = getattr(self, "_phase_safety_update_debounce_task", None)
        if phase_task is not None and not phase_task.done():
            phase_task.cancel()
            self._phase_safety_update_debounce_task = None
        self._price_source_update_pending = False
        self._phase_safety_update_pending = False

    async def async_options_updated(self) -> None:
        """Schedule a debounced pipeline re-run after an options change.

        Runs the update cycle as a **background task** so the caller (the
        config-entry update listener, triggered synchronously by
        ``async_update_entry`` from switch/number/time entities) returns
        immediately.  Without this, toggling a switch would block the HA
        service call until the entire read → plan (MILP/ML) → apply
        pipeline finished — making the UI feel frozen.

        A short debounce window is used so rapid switch/number/time toggles
        only trigger a single planner run after the user stops clicking.
        The background task is created with ``eager_start=False`` so the
        switch service call returns before any setup work begins.
        """
        if (
            self._options_update_debounce_task is not None
            and not self._options_update_debounce_task.done()
        ):
            self._options_update_debounce_task.cancel()
        self._options_update_debounce_task = self.hass.async_create_task(
            self._async_options_update_debounced(),
            name="hsem_options_update_debounce",
            eager_start=False,
        )

    async def _async_handle_secondary_storage_change(self, event: Event) -> None:
        """Coalesce material secondary-storage state changes into one update.

        Raw PowMr telemetry is deliberately handled outside the generic state
        listener path: SoC and the smoothed dedicated load only matter after
        crossing their existing replan thresholds, while hardware control
        entity changes remain promptly reactive. Battery net power is not a
        planner input and is therefore not registered by the state collector.
        """
        cfg = self._cfg.secondary_storage
        if not cfg.enabled:
            return

        entity_id = event.data.get("entity_id")
        new_state = event.data.get("new_state")
        if not entity_id or new_state is None:
            return

        material = False
        if entity_id == cfg.soc_entity:
            try:
                value = float(new_state.state)
            except TypeError, ValueError:
                return
            baseline = self._last_plan_secondary_soc_pct
            material = baseline is None or abs(value - baseline) >= (
                SECONDARY_STORAGE_SOC_REPLAN_DELTA_PCT
            )
        elif entity_id == cfg.load_power_entity:
            try:
                value = float(new_state.state)
            except TypeError, ValueError:
                return
            baseline = self._last_plan_secondary_load_power_w
            material = baseline is None or abs(value - baseline) >= (
                SECONDARY_STORAGE_LOAD_REPLAN_DELTA_W
            )
        elif entity_id in {
            cfg.output_source_priority_entity,
            cfg.charger_source_priority_entity,
            cfg.max_charge_current_entity,
        }:
            # A control listener registered before an options change may still
            # deliver events after hardware control is disabled. Ignore those
            # stale subscriptions; telemetry SoC/load remain planner inputs.
            if not cfg.control_enabled:
                return
            old_state = event.data.get("old_state")
            material = old_state is None or old_state.state != new_state.state

        if not material:
            return

        task = self._secondary_storage_update_debounce_task
        if task is not None and not task.done():
            return
        self._secondary_storage_update_debounce_task = self.hass.async_create_task(
            self._async_secondary_storage_update_debounced(),
            name="hsem_secondary_storage_update_debounce",
            eager_start=False,
        )

    async def _async_secondary_storage_update_debounced(self) -> None:
        """Run one full cycle after coalescing secondary-storage events."""
        try:
            await asyncio.sleep(SECONDARY_STORAGE_UPDATE_DEBOUNCE_SECONDS)
            # Unlike generic high-rate triggers, a material secondary control
            # event must not be dropped merely because another cycle currently
            # owns the coordinator lock. Wait for that cycle, then process the
            # event exactly once. Teardown cancellation interrupts both sleep
            # and lock acquisition safely.
            async with self._update_lock:
                await self._async_run_update_cycle()
        except asyncio.CancelledError:
            return
        finally:
            if self._secondary_storage_update_debounce_task is asyncio.current_task():
                self._secondary_storage_update_debounce_task = None

    async def _async_handle_phase_safety_change(self, _event: Event) -> None:
        """Refresh the live phase snapshot while a charge actuator is running.

        Phase-aware charging sizes the grid-charge cap from a meter snapshot
        taken just before the write.  Between coordinator cycles that snapshot
        goes stale, so a load that appears mid-slot is not seen until the next
        cycle.  This republishes the live state — and only the live state — so
        the working-mode entity recomputes the cap against current load.

        Does nothing unless a charge is actually in progress: an idle or
        discharging slot cannot overload the fuse by charging.
        """
        if not self._cfg.phase_aware_charging_enabled:
            return
        rec = self._hourly_recommendation
        if rec is None or not (
            rec.recommendation == Recommendations.BatteriesChargeGrid.value
            or rec.secondary_storage_mode == SECONDARY_MODE_CHARGE
        ):
            return

        # Record the request before checking for a running task, so a meter
        # change arriving during the debounce or cooldown is not dropped.  Some
        # sensors publish only on change, so losing the final event would leave
        # the cap stale until the next full cycle.
        self._phase_safety_update_pending = True
        task = self._phase_safety_update_debounce_task
        if task is not None and not task.done():
            return
        self._phase_safety_update_debounce_task = self.hass.async_create_task(
            self._async_phase_safety_update_debounced(),
            name="hsem_phase_safety_update_debounce",
            eager_start=False,
        )

    async def _async_phase_safety_update_debounced(self) -> None:
        """Publish fresh live snapshots without re-solving the horizon plan."""
        try:
            while self._phase_safety_update_pending and not self._tearing_down:
                self._phase_safety_update_pending = False
                await asyncio.sleep(PHASE_SAFETY_UPDATE_DEBOUNCE_SECONDS)
                async with self._update_lock:
                    await self._async_publish_live_phase_snapshot()
                await asyncio.sleep(PHASE_SAFETY_UPDATE_COOLDOWN_SECONDS)
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001 -- background safety task
            _LOGGER.warning("Phase-safety live refresh failed: %s", exc, exc_info=True)
        finally:
            if self._phase_safety_update_debounce_task is asyncio.current_task():
                self._phase_safety_update_debounce_task = None

    async def _async_publish_live_phase_snapshot(self) -> None:
        """Re-read live state and republish it against the accepted plan."""
        current_data = self.data
        cfg = current_data.cfg if current_data is not None else self._cfg
        if current_data is None or cfg is None:
            # No accepted plan to preserve yet; a normal cycle is cheaper than
            # inventing one.
            await self._async_run_update_cycle()
            return

        live, fwm_entity, new_unsubs = await async_collect_live_state(
            self,
            cfg,
            self._force_working_mode_entity,
            self._tracked_entities,
            entry_id=self._config_entry.entry_id,
        )
        self._force_working_mode_entity = fwm_entity
        self._listener_unsubs.extend(new_unsubs)
        self._live = live
        snapshot = self._snapshot
        if snapshot is not None:
            self._snapshot = replace(snapshot, live=live)

        # Reuse the accepted plan and swap in only the new hardware snapshot.
        self.async_set_updated_data(
            replace(current_data, cfg=cfg, live=live, last_updated=utc_now_iso())
        )

    async def _async_handle_price_source_change(self, _event: Event) -> None:
        """Coalesce price state/attribute changes into one durable refresh."""
        self._forecast_authority_generation = (
            getattr(self, "_forecast_authority_generation", 0) + 1
        )
        self._price_source_update_pending = True
        task = self._price_source_update_debounce_task
        if task is not None and not task.done():
            return
        self._price_source_update_debounce_task = self.hass.async_create_task(
            self._async_price_source_update_debounced(),
            name="hsem_price_source_update_debounce",
            eager_start=False,
        )

    def _forecast_authority_generation_changed(
        self,
        captured_generation: int,
        *,
        phase: str,
    ) -> bool:
        """Return whether newer forecast authority invalidated this cycle."""
        current_generation = getattr(self, "_forecast_authority_generation", 0)
        if current_generation == captured_generation:
            return False
        async_log(
            "debug",
            "[replan] Discarding stale coordinator cycle before %s: "
            "forecast authority generation advanced %d→%d; the queued "
            "source refresh will rebuild it from a fresh snapshot.",
            phase,
            captured_generation,
            current_generation,
        )
        return True

    def _capture_forecast_authority_plan_state(
        self,
    ) -> _ForecastAuthorityPlanState:
        """Snapshot accepted plan caches that a late stale cycle may mutate."""
        return _ForecastAuthorityPlanState(
            last_planner_input=self._last_planner_input,
            last_planner_output=self._last_planner_output,
            last_plan_slot_start=self._last_plan_slot_start,
            previous_planner_winner_name=self._previous_planner_winner_name,
            previous_planner_winner_score=self._previous_planner_winner_score,
            window_hys_previous_rec=self._window_hys_previous_rec,
            window_hys_previous_slot_start=self._window_hys_previous_slot_start,
            window_hysteresis_expiry=self._window_hysteresis_expiry,
            current_slot_start=self._current_slot_start,
            current_slot_price_actionable=self._current_slot_price_actionable,
            current_slot_ev_power_w=self._current_slot_ev_power_w,
            current_slot_ev_second_power_w=self._current_slot_ev_second_power_w,
            current_required_battery=self._current_required_battery,
            plan_explanation=self._plan_explanation,
            data_quality=self._data_quality,
            ev_charging_plan=self._ev_charging_plan,
            ev_second_charging_plan=self._ev_second_charging_plan,
            hourly_recommendation=self._hourly_recommendation,
            hourly_recommendations=self._hourly_recommendations,
        )

    def _restore_forecast_authority_plan_state(
        self,
        state: _ForecastAuthorityPlanState,
    ) -> None:
        """Roll back plan caches after a late authority generation abort."""
        self._last_planner_input = state.last_planner_input
        self._last_planner_output = state.last_planner_output
        self._last_plan_slot_start = state.last_plan_slot_start
        self._previous_planner_winner_name = state.previous_planner_winner_name
        self._previous_planner_winner_score = state.previous_planner_winner_score
        self._window_hys_previous_rec = state.window_hys_previous_rec
        self._window_hys_previous_slot_start = state.window_hys_previous_slot_start
        if self._window_hysteresis_expiry != state.window_hysteresis_expiry:
            self._cancel_window_hysteresis_expiry()
            if state.window_hysteresis_expiry is not None:
                self._schedule_window_hysteresis_expiry(state.window_hysteresis_expiry)
        self._current_slot_start = state.current_slot_start
        self._current_slot_price_actionable = state.current_slot_price_actionable
        self._current_slot_ev_power_w = state.current_slot_ev_power_w
        self._current_slot_ev_second_power_w = state.current_slot_ev_second_power_w
        self._current_required_battery = state.current_required_battery
        self._plan_explanation = state.plan_explanation
        self._data_quality = state.data_quality
        self._ev_charging_plan = state.ev_charging_plan
        self._ev_second_charging_plan = state.ev_second_charging_plan
        self._hourly_recommendation = state.hourly_recommendation
        self._hourly_recommendations = state.hourly_recommendations

    async def _async_price_source_update_debounced(self) -> None:
        """Refresh after publication/withdrawal without dropping busy events."""
        try:
            while True:
                await asyncio.sleep(PRICE_SOURCE_UPDATE_DEBOUNCE_SECONDS)
                # Events already coalesced into the snapshot about to be read
                # are consumed here. Any event arriving during collection,
                # solve, or publication sets this flag again and guarantees a
                # follow-up cycle with the newer source state.
                self._price_source_update_pending = False
                async with self._update_lock:
                    await self._async_run_update_cycle()
                if not self._price_source_update_pending:
                    break
        except asyncio.CancelledError:
            return
        finally:
            if self._price_source_update_debounce_task is asyncio.current_task():
                self._price_source_update_debounce_task = None

    async def _async_options_update_debounced(self) -> None:
        """Wait for the debounce window, then schedule the planner run."""
        try:
            await asyncio.sleep(OPTIONS_UPDATE_DEBOUNCE_SECONDS)
        except asyncio.CancelledError:
            # Superseded by a newer options update — not an error.
            async_log(
                "debug",
                "[coordinator] options-update debounce cancelled — "
                "superseded by a newer options change.",
            )
            return

        # Cancel any still-pending previous options-update task before
        # scheduling a fresh run with the latest option state.
        if (
            self._options_update_task is not None
            and not self._options_update_task.done()
        ):
            self._options_update_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._options_update_task

        self._options_update_task = self.hass.async_create_task(
            self._async_options_update_background(),
            name="hsem_options_update",
            eager_start=False,
        )
        self._options_update_debounce_task = None

    async def _async_options_update_background(self) -> None:
        """Background wrapper: run the update cycle, swallowing cancellation."""
        try:
            await self._async_handle_update(None)
        except asyncio.CancelledError:
            # Superseded by a newer options update — not an error.
            async_log(
                "debug",
                "[coordinator] options-update task cancelled — superseded by a "
                "newer options change.",
            )

    # ------------------------------------------------------------------
    # Internal update pipeline
    # ------------------------------------------------------------------

    async def _async_handle_update(self, event: Event | None = None) -> None:
        """Drop concurrent updates; run the update cycle while holding the lock."""
        if self._update_lock.locked():
            async_log(
                "debug",
                "------ Coordinator update skipped: a previous cycle is still running.",
            )
            return
        async with self._update_lock:
            await self._async_run_update_cycle()

    async def _async_run_update_cycle(self) -> None:
        """Execute the full collect → populate → plan cycle.

        On success, packages the results into a :class:`CoordinatorData` and
        calls :meth:`async_set_updated_data` to notify all subscriber entities.

        Raises:
            UpdateFailed: When an unrecoverable error occurs during the pipeline.
        """
        async_log("debug", "------ HSEM Coordinator: starting update cycle")
        forecast_authority_generation = getattr(
            self, "_forecast_authority_generation", 0
        )
        forecast_authority_plan_state = self._capture_forecast_authority_plan_state()
        now = hsem_now()
        corrective_live_replan = False
        corrective_request_slot: datetime | None = None
        corrective_output_accepted = False
        corrective_rejection_reason = ""
        corrective_candidate_winner = ""
        corrective_candidate_solver = "not_run"
        plan_state_should_persist = False
        normal_planner_output_to_commit: PlannerOutput | None = None
        price_forecast_signature: _PriceForecastSignature | None = None
        load_forecast_signature: _LoadForecastSignature | None = None
        price_authority_changed = False
        # An exact hysteresis-expiry callback must survive degraded or failed
        # cycles.  Consume it only after this cycle actually ran the planner
        # and published the resulting snapshot successfully.
        hysteresis_expiry_replan = getattr(
            self, "_window_hysteresis_expiry_replan_pending", False
        )
        hysteresis_expiry_replan_completed = False

        try:
            # 1. Reload config from the config entry.
            self._cfg = build_sensor_config(self._config_entry)
            cfg = self._cfg
            self._refresh_slot_boundary_schedule(now)

            # 2. Collect ALL HA entity states once into an immutable snapshot.
            #    This single call replaces the three-stage read pattern:
            #    async_collect_live_state → (populate consumption → populate price/solcast).
            (
                self._snapshot,
                self._force_working_mode_entity,
                new_unsubs,
            ) = await async_collect_all_states(
                self,
                cfg,
                self._force_working_mode_entity,
                self._tracked_entities,
                self._avg_house_consumption_entity_id_cache,
                entry_id=self._config_entry.entry_id,
            )
            self._listener_unsubs.extend(new_unsubs)
            self._live = self._snapshot.live
            live = self._live

            # Feed the capacity learner with BMS readings (issue #605).
            if (
                live.bms_kwh_remaining is not None
                and live.huawei_batteries_soc_pct is not None
            ):
                getattr(self, "_capacity_learner", CapacityLearner()).update(
                    live.bms_kwh_remaining, live.huawei_batteries_soc_pct
                )

            # Update the weekday/weekend consumption profile (issue #612).
            house_w = live.house_consumption_power_w
            if house_w is not None and house_w > 0:
                weekday_profile.update(
                    dow=now.weekday(),
                    slot=now.hour,
                    value_kwh=house_w / 1000.0,
                )

            # Apply EMA smoothing to live net consumption to damp transients
            # (støvsuger, kaffemaskine, cloud shadows) so they don't kill
            # the EV charging setpoint for the rest of a 15-minute slot.
            self._net_consumption_ema = ema_filter(
                live.net_consumption_w,
                self._net_consumption_ema,
                EMA_ALPHA_NET_CONSUMPTION,
            )
            # Swap the raw value with the EMA-smoothed value on the live
            # state object so all downstream code (PlannerInput builder,
            # forecast tracker, etc.) sees the damped signal.
            live.net_consumption_w = self._net_consumption_ema

            # -----------------------------------------------------------------------
            # Override expiry check (issue #317)
            # -----------------------------------------------------------------------
            # When a timed override was set via set_temporary_override with
            # duration_minutes, check if it has expired.  If so, auto-clear the
            # select entity back to "auto" so the planner resumes control.
            #
            # Also handle the case where the user manually cleared the override
            # before the expiry — clean up the stored expiry in that case too.
            if self._override_expiry is not None:
                if utc_key(now) >= utc_key(self._override_expiry):
                    async_log(
                        "debug",
                        "Timed override EXPIRED — clearing select entity to 'auto'.",
                    )
                    # Fire-and-forget: set the select entity back to "auto".
                    await self.hass.services.async_call(
                        "select",
                        "select_option",
                        {
                            "entity_id": live.force_working_mode,
                            "option": "auto",
                        },
                        blocking=True,
                    )
                    live.force_working_mode_state = "auto"
                    self._override_expiry = None
                elif live.force_working_mode_state == "auto":
                    # User manually cleared before expiry — remove the tracking.
                    async_log(
                        "debug",
                        "Override manually cleared before expiry — removing expiry tracking.",
                    )
                    self._override_expiry = None

            # 3. Reset and generate recommendation time-slots.
            self._hourly_recommendation = None
            self._hourly_recommendations = generate_recommendation_intervals(
                cfg.recommendation_interval_minutes,
                cfg.recommendation_interval_length,
            )

            # 4. Populate weighted house-consumption averages.
            #
            # Two paths are available:
            #   a) ML prediction — queries the HA recorder for historical
            #      energy data and uses a per-(DOW, hour) time-decay model.
            #   b) Legacy averaging sensors — reads HSEM's own 1d/3d/7d/14d
            #      RestoreEntity rolling-average sensors (the default).
            #
            # The ML path is enabled via `hsem_ml_consumption_enabled`.
            # When it fails (insufficient history, misconfigured, …), the
            # coordinator transparently falls back to the legacy pipeline.
            set_hsem_verbose(cfg.verbose_logging)

            if cfg.ml_consumption_enabled:
                # ML consumption prediction from recorder history.
                from custom_components.hsem.ml.populator import (
                    populate_ml_house_consumption,
                )

                (
                    consumption_ok,
                    self._ml_predictor,
                ) = await populate_ml_house_consumption(
                    self.hass,
                    self._hourly_recommendations,
                    cfg,
                    self._ml_predictor,
                )
                async_log(
                    "debug",
                    "[ml] populate_ml_house_consumption returned %s",
                    consumption_ok,
                )

                if not consumption_ok:
                    # Fallback: ML failed; try legacy avg sensors.
                    async_log(
                        "debug",
                        "[ml] ML consumption failed"
                        " — falling back to legacy avg sensors.",
                    )
                    consumption_ok = populate_avg_house_consumption_from_snapshot(
                        self._hourly_recommendations,
                        self._snapshot,
                        cfg,
                        self._avg_house_consumption_entity_id_cache,
                        entry_id=self._config_entry.entry_id,
                    )
            else:
                # Legacy averaging-sensor pipeline (default).
                consumption_ok = populate_avg_house_consumption_from_snapshot(
                    self._hourly_recommendations,
                    self._snapshot,
                    cfg,
                    self._avg_house_consumption_entity_id_cache,
                    entry_id=self._config_entry.entry_id,
                )
                async_log(
                    "debug",
                    "[avg] populate_avg_house_consumption_from_snapshot returned %s, "
                    "cache has %d entries, "
                    "snapshot has %d energy_avg values",
                    consumption_ok,
                    len(self._avg_house_consumption_entity_id_cache),
                    len(self._snapshot.energy_average_values),
                )

            load_forecast_readiness = _assess_load_forecast(
                self._hourly_recommendations,
                now,
                population_succeeded=consumption_ok,
                live_house_demand_w=live.house_consumption_power_w,
            )
            consumption_ok = load_forecast_readiness.ready
            load_forecast_signature = load_forecast_readiness.signature
            readiness_reason = load_forecast_readiness.reason
            previous_readiness_reason = getattr(
                self, "_last_load_forecast_readiness_reason", None
            )
            if consumption_ok:
                assert load_forecast_signature is not None
                if not self._data_quality.load_forecast_ready:
                    self._data_quality = replace(
                        self._data_quality,
                        load_forecast_ready=True,
                        load_forecast_reason=None,
                    )
                if previous_readiness_reason is not None:
                    async_log(
                        "info",
                        "[load] Forecast recovered (%s); a fresh plan is required.",
                        previous_readiness_reason,
                    )
            else:
                assert readiness_reason is not None
                self._load_forecast_recovery_replan_pending = True
                self._data_quality = DataQuality(
                    load_forecast_ready=False,
                    load_forecast_reason=readiness_reason,
                )
                if readiness_reason != previous_readiness_reason:
                    async_log(
                        "warning",
                        "[load] Forecast is not ready (%s); automatic control "
                        "will publish a strict storage hold.",
                        readiness_reason,
                    )
            self._last_load_forecast_readiness_reason = readiness_reason

            # Adjust timer based on missing-entities or pending-consumption status.
            if live.missing_entities or not consumption_ok:
                await self._set_update_interval(1)
            else:
                await self._set_update_interval()

            # 6. Determine working state: forced, missing, or full pipeline.
            state: str | None = None

            # Only a *critical* absence halts planning.  A non-critical one
            # (an EV sensor, a price feed) classifies as Degraded, which
            # utils.degraded_mode defines as "read-only calculations continue
            # on best-effort values" — so the battery keeps being optimised
            # while the unavailable feature sits out the cycle.
            if live.missing_entities and live.degraded_mode is not DegradedMode.Error:
                async_log(
                    "debug",
                    "Non-critical input entities missing (%s); "
                    "continuing on best-effort values.",
                    ", ".join(live.missing_entities_list) or "unknown",
                )

            if (
                live.degraded_mode is DegradedMode.Error
                and live.force_working_mode_state == "auto"
            ):
                state = Recommendations.MissingInputEntities.value
                async_log(
                    "debug",
                    "Critical input entities missing (%s), skipping calculations.",
                    ", ".join(live.missing_entities_list) or "unknown",
                )

            elif not consumption_ok and live.force_working_mode_state == "auto":
                # Energy average sensors not yet ready.  Still populate prices
                # and solcast below, but skip the planner (zeroed consumption
                # data would produce wrong results).
                pass  # handled below after price/solcast population

            elif live.force_working_mode_state != "auto":
                state = str(live.force_working_mode_state)
                async_log(
                    "debug",
                    "Force working mode is activated. Setting working mode to %s",
                    live.force_working_mode_state,
                )

            # 7. Populate electricity prices and Solcast PV estimates — always
            #    run, independent of consumption data.
            self._entsoe_price_backup_status = populate_price_and_solcast_from_snapshot(
                self._hourly_recommendations,
                self._snapshot,
                cfg,
            )
            _apply_live_current_price_availability(
                self._hourly_recommendations,
                live,
                now,
            )
            price_outage_hold = _apply_current_price_outage_hold(
                self._hourly_recommendations,
                live,
                now,
            )
            if price_outage_hold is not None:
                # Publish fail-safe intent even when another missing input or
                # pending consumption history prevents a planner run below.
                self._hourly_recommendation = price_outage_hold
                state = price_outage_hold.recommendation
            load_forecast_hold = _apply_load_forecast_hold(
                self._hourly_recommendations,
                live,
                now,
                load_forecast_ready=consumption_ok,
            )
            if load_forecast_hold is not None:
                assert readiness_reason is not None
                self._hourly_recommendation = load_forecast_hold
                state = load_forecast_hold.recommendation
                self._plan_explanation = PlanExplanation(
                    selected_strategy="safety_hold",
                    winner_name="safety_hold",
                    summary=(
                        "Battery held because the house-load forecast is not "
                        f"ready ({readiness_reason})."
                    ),
                    constraints=[f"load_forecast:{readiness_reason}"],
                    solver_status="not_run",
                    fallback_reason=readiness_reason,
                )
            price_forecast_attributes = _price_forecast_attributes(
                self._snapshot,
                cfg,
            )
            price_forecast_signature = _price_forecast_signature(
                self._hourly_recommendations,
                live,
                now,
                valuation_attributes=price_forecast_attributes,
                valuation_enabled=cfg.price_forecast_valuation_enabled,
                valuation_margin=cfg.price_forecast_valuation_margin,
            )
            price_authority_changed = price_forecast_signature != getattr(
                self, "_last_plan_price_forecast_signature", None
            )

            # -----------------------------------------------------------------------
            # Forecast-vs-actual accumulation (issue #373)
            # -----------------------------------------------------------------------
            # Every cycle, accumulate actual PV and load energy into the current
            # slot based on instantaneous power readings and elapsed time.
            self._accumulate_forecast_actuals(now, live)

            if (
                live.force_working_mode_state == "auto"
                and live.degraded_mode is not DegradedMode.Error
                and consumption_ok
            ):
                # Compute dynamic discharge floor BEFORE the planner runs
                # so it can influence the planner's discharge/export decisions.
                dynamic_floor_enabled = bool(
                    get_config_value(self._config_entry, "hsem_dynamic_discharge_floor")
                )
                if dynamic_floor_enabled:
                    # Compute usable kWh from the live inverter state.
                    rated_kwh = (
                        live.huawei_batteries_rated_capacity_wh or 0.0
                    ) / 1000.0
                    min_soc_pct = live.huawei_batteries_end_of_discharge_soc_pct or 0.0
                    max_soc_pct = (
                        live.huawei_batteries_charging_cutoff_capacity_pct or 100.0
                    )
                    _usable_kwh = usable_kwh_from_rated(
                        rated_kwh, min_soc_pct, max_soc_pct
                    )
                    _current_kwh = (
                        (live.huawei_batteries_soc_pct or 0.0) / 100.0 * _usable_kwh
                    )
                    # Build a lightweight slot list from hourly_recommendations
                    # for the bridge computation (they already have consumption
                    # and PV estimates populated by the populator).
                    _bridge_slots: list = []
                    for rec in self._hourly_recommendations:
                        _bridge_slots.append(
                            _SimpleSlot(
                                start=rec.start,
                                end=rec.end,
                                estimated_net_consumption_kwh=(
                                    rec.avg_house_consumption_kwh
                                    - rec.solcast_pv_estimate_kwh
                                ),
                                batteries_charged_kwh=rec.batteries_charged_kwh,
                                recommendation=rec.recommendation,
                            )
                        )
                    floor_pct, floor_diag = self._dynamic_floor.compute_floor(
                        now=now,
                        slots=_bridge_slots,
                        current_kwh=_current_kwh,
                        usable_kwh=_usable_kwh,
                        configured_min_soc_pct=min_soc_pct,
                    )
                    self._effective_discharge_floor_pct = floor_pct
                    self._effective_discharge_floor_diag = floor_diag

                    # Self-correct the safety margin.
                    if live.huawei_batteries_soc_pct is not None:
                        self._dynamic_floor.correct_margin(
                            live.huawei_batteries_soc_pct, floor_pct
                        )
                    _dynamic_floor_pct: float | None = floor_pct
                else:
                    self._effective_discharge_floor_pct = None
                    self._effective_discharge_floor_diag = None
                    _dynamic_floor_pct = None

                # 8. Run the pure-Python planner engine — only when all data
                #    is ready.  Skip when consumption averages are still
                #    pending (first cycle, sensor restore not done).
                #
                #    Event-driven re-planning: only re-solve the MILP when
                #    something material changed (EV state, slot boundary,
                #    price period, forced mode).  Between events, the
                #    previous plan is reused to prevent oscillation.

                # Collect session EV charge power for session-aware MILP
                # optimisation (issue #615).  When an EV is actively charging
                # in a forced-draw mode, its current charge power is treated
                # as certain demand for the next 2 hours in the MILP.
                ev_session_kw: dict[str, float] = {}
                if live.ev.is_charging and live.ev.power_w:
                    ev_session_kw["ev"] = (live.ev.power_w or 0.0) / 1000.0
                if (
                    cfg.ev_second_enabled
                    and live.ev_second.is_charging
                    and live.ev_second.power_w
                ):
                    ev_session_kw["ev_second"] = (
                        live.ev_second.power_w or 0.0
                    ) / 1000.0

                # Determine whether a full re-plan is needed.
                pending_force_slot = self._force_discharge_live_replan_pending_slot
                active_force_slot = self._active_force_discharge_slot(now)
                corrective_live_replan = (
                    pending_force_slot is not None
                    and active_force_slot is not None
                    and utc_key(active_force_slot.start) == utc_key(pending_force_slot)
                    and (utc_key(active_force_slot.end) - utc_key(now)).total_seconds()
                    >= FORCE_DISCHARGE_REPLAN_MIN_REMAINING_SECONDS
                    and live.degraded_mode is not DegradedMode.Error
                    and not (
                        cfg.house_power_includes_ev_charger_power
                        and live.any_ev_charging
                    )
                )
                if corrective_live_replan:
                    corrective_request_slot = pending_force_slot
                if pending_force_slot is not None and not corrective_live_replan:
                    self._force_discharge_live_replan_pending_slot = None
                    self._clear_force_discharge_excess_window()
                should_replan = self._should_replan(
                    live,
                    now,
                    price_forecast_signature=price_forecast_signature,
                    load_forecast_signature=load_forecast_signature,
                )

                if should_replan:
                    planner_input = build_planner_input(
                        cfg=cfg,
                        live=self._live,
                        hourly_recommendations=self._hourly_recommendations,
                        previous_winner_name=(
                            None
                            if corrective_live_replan
                            else self._previous_planner_winner_name
                        ),
                        previous_winner_score=(
                            0.0
                            if corrective_live_replan
                            else self._previous_planner_winner_score
                        ),
                        ev_session_kw=ev_session_kw if ev_session_kw else None,
                        dynamic_discharge_floor_pct=_dynamic_floor_pct,
                        capacity_learner=getattr(
                            self, "_capacity_learner", CapacityLearner()
                        ),
                        price_forecast_attributes=price_forecast_attributes,
                    )
                    # Wire the solar forecast corrector into the planner input so
                    # populate_solcast can apply per-hour accuracy corrections (issue #602).
                    planner_input.solar_corrector = self._solar_corrector
                    # Retain for diagnostics dumps (cleared on each cycle).
                    self._last_planner_input = planner_input

                    # Debug: log per-hour consumption total reaching the planner
                    # (after builder's *slots_per_hour scaling).
                    total_1d = sum(
                        c.avg_1d
                        for c in planner_input.consumption_averages
                        if c.avg_1d > 0
                    )
                    async_log(
                        "debug",
                        "[builder] consumption per-hour total reaching planner:"
                        " avg_1d=%.2f kWh"
                        " over %d hours",
                        total_1d,
                        len(planner_input.consumption_averages),
                    )

                    # Propagate the verbose-logging flag into the pure-Python
                    # planner so detailed slot-level decisions appear in the
                    # standard Home Assistant log when the user enables
                    # verbose logging.
                    set_hsem_verbose(cfg.verbose_logging)
                    # Run the planner in HA's executor pool.  The MILP/ML
                    # solver is CPU-bound; running it in the event-loop
                    # thread blocks the HA UI for the full solve duration.
                    candidate_planner_output = await self.hass.async_add_executor_job(
                        run_planner, planner_input
                    )
                    if self._forecast_authority_generation_changed(
                        forecast_authority_generation,
                        phase="planner solve",
                    ):
                        self._restore_forecast_authority_plan_state(
                            forecast_authority_plan_state
                        )
                        return
                    if hysteresis_expiry_replan:
                        hysteresis_expiry_replan_completed = True
                    if corrective_live_replan:
                        corrective_candidate_winner = (
                            candidate_planner_output.winner_name
                        )
                        corrective_candidate_solver = (
                            candidate_planner_output.explanation.solver_status
                        )
                        assert self._last_planner_output is not None
                        assert corrective_request_slot is not None
                        (
                            planner_output,
                            corrective_output_accepted,
                            corrective_rejection_reason,
                        ) = _select_corrective_planner_output(
                            self._last_planner_output,
                            candidate_planner_output,
                            corrective_request_slot,
                            price_authority_changed=price_authority_changed,
                        )
                        if not corrective_output_accepted:
                            # A passive/no-action fallback is not an economic
                            # correction. Keep executing the last validated
                            # plan and consume this slot's attempt only after
                            # the retained plan is successfully republished.
                            async_log(
                                "warning",
                                "[replan] Live-demand corrective solve did not "
                                "produce an authoritative MILP plan "
                                "(winner=%s solver=%s fallback=%s rejection=%s); "
                                "retaining "
                                "the previous plan for this slot.",
                                corrective_candidate_winner or "(none)",
                                corrective_candidate_solver,
                                candidate_planner_output.explanation.fallback_reason
                                or "(none)",
                                corrective_rejection_reason,
                            )
                            # The retained accepted output is mutable downstream.
                            # Isolate this corrective publication from its cache.
                            planner_output = deepcopy(planner_output)
                    else:
                        planner_output = candidate_planner_output
                        normal_planner_output_to_commit = planner_output
                        plan_state_should_persist = True

                    if corrective_live_replan and corrective_output_accepted:
                        plan_state_should_persist = True

                    for warning in planner_output.warnings:
                        async_log("debug", "[planner] %s", warning)

                    self._current_required_battery = (
                        planner_output.required_capacity_kwh
                    )
                    self._data_quality = planner_output.data_quality
                    self._ev_charging_plan = planner_output.ev_charging_plan
                    self._ev_second_charging_plan = (
                        planner_output.ev_second_charging_plan
                    )

                    # Warn when an EV is physically charging but no current or
                    # future slot carries ev_total_planned_load_kwh > 0.
                    if self._live.any_ev_charging:
                        has_planned = any(
                            s.ev_total_planned_load_kwh > 1e-9
                            for s in planner_output.slots
                            if utc_key(s.end) > utc_key(now)
                        )
                        if not has_planned:
                            async_log(
                                "debug",
                                "[planner] WARNING: EV is physically charging but no "
                                "current or future slot has ev_total_planned_load_kwh"
                                " > 0. The EV load is either outside the planning"
                                " window, smart charging is disabled, or"
                                " base_load_includes_ev is set but the plan produced"
                                " zero accounted load. Check EV plan state and slot"
                                " attributes.",
                            )
                else:
                    # No material changes — reuse the previous plan.
                    assert self._last_planner_output is not None, (
                        "_last_planner_output must be set when _should_replan"
                        " returns False"
                    )
                    planner_output = deepcopy(self._last_planner_output)
                    self._current_required_battery = (
                        planner_output.required_capacity_kwh
                    )
                    self._data_quality = planner_output.data_quality
                    self._ev_charging_plan = planner_output.ev_charging_plan
                    self._ev_second_charging_plan = (
                        planner_output.ev_second_charging_plan
                    )
                    async_log(
                        "debug",
                        "[replan] Skipping planner — no material changes detected."
                        " Reusing plan from %s.",
                        self._last_plan_slot_start.isoformat()
                        if self._last_plan_slot_start
                        else "(unknown)",
                    )

                # -----------------------------------------------------------------------
                # Window-level hysteresis — prevent rapid recommendation toggles
                # within a slot (issue #315).
                # -----------------------------------------------------------------------
                # Apply to planner output slots BEFORE _apply_planner_output so that
                # the held recommendation propagates to hourly_recommendations.
                window_hys_minutes = cfg.planner_window_hysteresis_minutes
                if window_hys_minutes > 0:
                    current_before_hysteresis = next(
                        (
                            s
                            for s in planner_output.slots
                            if slot_contains(s.start, s.end, now)
                        ),
                        None,
                    )
                    proposed_rec = (
                        current_before_hysteresis.recommendation
                        if current_before_hysteresis is not None
                        else None
                    )
                    held_rec, held_start = apply_window_hysteresis(
                        planner_output.slots,
                        now,
                        window_hysteresis_minutes=window_hys_minutes,
                        previous_current_recommendation=(
                            None
                            if corrective_live_replan
                            else self._window_hys_previous_rec
                        ),
                        previous_current_slot_start=(
                            None
                            if corrective_live_replan
                            else self._window_hys_previous_slot_start
                        ),
                    )
                    self._window_hys_previous_rec = held_rec
                    self._window_hys_previous_slot_start = held_start
                    if held_rec != proposed_rec and held_start is not None:
                        expiry = (
                            as_tz(held_start, now.tzinfo).astimezone(UTC)
                            + timedelta(minutes=window_hys_minutes)
                        ).astimezone(now.tzinfo)
                        self._schedule_window_hysteresis_expiry(expiry)
                    elif should_replan:
                        # A fresh authoritative result no longer needs a hold.
                        # Reused, already-mutated plans must leave an existing
                        # expiry callback intact.
                        self._cancel_window_hysteresis_expiry()
                else:
                    self._cancel_window_hysteresis_expiry()
                    # Feature disabled — still persist the current recommendation
                    # so that re-enabling picks up the right state.
                    for s in planner_output.slots:
                        if slot_contains(s.start, s.end, now):
                            self._window_hys_previous_rec = s.recommendation
                            self._window_hys_previous_slot_start = s.start
                            break

                # Freeze the current slot's per-EV charger power before
                # copying planner output to hourly recommendations. This keeps
                # the charger command stable across replans inside the same
                # 15-minute slot (issue #738).
                self._freeze_ev_charger_power_for_current_slot(planner_output, now)

                # Apply planner output (with hysteresis-applied slots) to
                # hourly_recommendations so the current slot resolution in
                # step 9 sees the held recommendation.
                self._apply_planner_output(planner_output)

                # 8b. Auto-Full EV on negative price override (issue #609).
                # When import price is ≤ 0 and the feature is enabled,
                # override the current slot to charge the EV at full power.
                auto_full_enabled = bool(
                    get_config_value(
                        self._config_entry, "hsem_ev_auto_full_negative_price"
                    )
                )
                if auto_full_enabled:
                    now_slot = next(
                        (
                            r
                            for r in self._hourly_recommendations
                            if slot_contains(r.start, r.end, now)
                        ),
                        None,
                    )
                    if _auto_full_negative_price_allowed(live, now_slot):
                        assert now_slot is not None
                        now_slot.recommendation = Recommendations.EVSmartCharging.value
                        pwr_kw = float(
                            get_config_value(
                                self._config_entry,
                                "hsem_ev_planned_load_charger_power_kw",
                            )
                            or 0.0
                        )
                        now_slot.ev_charger_calculated_power = (
                            round(pwr_kw * 1000) if pwr_kw > 0 else 0.0
                        )
                        async_log(
                            "debug",
                            "[coordinator] Auto-Full EV: negative price (%.4f) "
                            "→ overriding current slot to ev_smart_charging "
                            "at %dW",
                            live.import_electricity_price,
                            now_slot.ev_charger_calculated_power,
                        )

                # 8c. Force-charge-now override: when the user toggles the
                # "EV Force Charge Now" switch, override the current slot's
                # recommendation and calculated power to charge at max speed.
                # Force-charge works even when smart charging is disabled —
                # the plan state is flipped to "charging" so the plan sensor
                # reflects the forced charge instead of
                # "smart_charging_disabled".
                _apply_force_charge_now(
                    config_entry=self._config_entry,
                    hourly_recommendations=self._hourly_recommendations,
                    ev_plan=self._ev_charging_plan,
                    ev_second_plan=self._ev_second_charging_plan,
                    now=now,
                )

                # 9. Find the current time-slot recommendation.
                self._hourly_recommendations.sort(key=lambda x: utc_key(x.start))
                # now.tzinfo is guaranteed non-None because hsem_now() returns
                # a timezone-aware datetime; assert so pyright narrows the type.
                assert now.tzinfo is not None, (
                    "hsem_now() must return tz-aware datetime"
                )
                hourly_rec = next(
                    (
                        r
                        for r in self._hourly_recommendations
                        if slot_contains(r.start, r.end, now)
                    ),
                    None,
                )

                if hourly_rec is not None:
                    self._hourly_recommendation = hourly_rec
                    state = hourly_rec.recommendation

                # -----------------------------------------------------------------------
                # Register forecasts in the forecast tracker from the planner output.
                # -----------------------------------------------------------------------
                self._register_forecasts_from_planner(planner_output, now)

                # -----------------------------------------------------------------------
                # Daily plan-vs-actual accumulation from planner output.
                # -----------------------------------------------------------------------
                try:
                    await self._accumulate_daily_plan_actuals(now, live, planner_output)
                except Exception as e:
                    async_log(
                        "error",
                        "Daily plan-vs-actual accumulation failed — "
                        "continuing without updating daily metrics: %s",
                        e,
                    )

                # -----------------------------------------------------------------------
                # Financial tracker accumulation (issue #599).
                # -----------------------------------------------------------------------
                try:
                    await self._accumulate_financials(now, live)
                except Exception as e:
                    async_log(
                        "error",
                        "Financial tracker accumulation failed — "
                        "continuing without updating financial metrics: %s",
                        e,
                    )

                # -----------------------------------------------------------------------
                # Savings tracker accumulation (issue #604).
                # -----------------------------------------------------------------------
                try:
                    await self._accumulate_savings(now, live, planner_output)
                except Exception as e:
                    async_log(
                        "error",
                        "Savings tracker accumulation failed — "
                        "continuing without updating savings metrics: %s",
                        e,
                    )

        except asyncio.CancelledError:
            self._restore_forecast_authority_plan_state(forecast_authority_plan_state)
            raise
        except Exception as exc:
            self._restore_forecast_authority_plan_state(forecast_authority_plan_state)
            raise UpdateFailed(f"HSEM update cycle failed: {exc}") from exc

        if self._forecast_authority_generation_changed(
            forecast_authority_generation,
            phase="publication",
        ):
            self._restore_forecast_authority_plan_state(forecast_authority_plan_state)
            return

        # Final sort and timestamp.
        self._hourly_recommendations.sort(key=lambda x: utc_key(x.start))
        last_updated = utc_now_iso()

        data = CoordinatorData(
            cfg=self._cfg,
            live=self._live,
            hourly_recommendations=list(self._hourly_recommendations),
            hourly_recommendation=self._hourly_recommendation,
            current_required_battery=self._current_required_battery,
            state=state,
            last_updated=last_updated,
            next_update=self._next_update,
            apply_summary=getattr(self, "_last_apply_summary", None),
            plan_explanation=self._plan_explanation,
            data_quality=self._data_quality,
            entsoe_price_backup_status=self._entsoe_price_backup_status,
            ev_charging_plan=self._ev_charging_plan,
            ev_second_charging_plan=self._ev_second_charging_plan,
            override_expiry=(
                self._override_expiry.isoformat()
                if self._override_expiry is not None
                else None
            ),
            capacity_learner=getattr(self, "_capacity_learner", CapacityLearner()),
            solar_hour_factors=dict(
                getattr(self, "_solar_corrector", SolarForecastCorrector()).hour_factors
            ),
            effective_discharge_floor_pct=getattr(
                self, "_effective_discharge_floor_pct", None
            ),
            effective_discharge_floor_diag=(
                dict(getattr(self, "_effective_discharge_floor_diag", None) or {})
                if getattr(self, "_effective_discharge_floor_diag", None)
                else None
            ),
            financial_tracker=getattr(self, "_financial_tracker", None),
            prediction_tracker=getattr(self, "_prediction_tracker", None),
            savings_tracker=getattr(self, "_savings_tracker", SavingsTracker()),
        )

        # Notify all subscriber entities atomically.
        self.async_set_updated_data(data)
        if normal_planner_output_to_commit is not None:
            self._last_planner_output = normal_planner_output_to_commit
            self._last_plan_slot_start = now

        if hysteresis_expiry_replan_completed:
            self._window_hysteresis_expiry_replan_pending = False
        self._persist_plan_state_if_accepted(
            live,
            plan_state_should_persist,
            price_forecast_signature=price_forecast_signature,
            load_forecast_signature=load_forecast_signature,
        )
        if corrective_live_replan and corrective_request_slot is not None:
            # Commit the corrected plan and consume the once-per-slot request
            # only after the complete coordinator update was successfully
            # packaged and published. Any exception before this point leaves
            # the old cached plan and pending request intact for a later retry.
            if corrective_output_accepted:
                self._last_planner_output = planner_output
                self._last_plan_slot_start = now
            self._force_discharge_replanned_slot_start = corrective_request_slot
            self._force_discharge_live_replan_pending_slot = None
            self._clear_force_discharge_excess_window()

            effective_rec = self._hourly_recommendation
            async_log(
                "debug",
                "[replan] Live-demand correction published: slot=%s "
                "accepted=%s candidate_winner=%s candidate_solver=%s "
                "rejection=%s planned_recommendation=%s discharge=%.3f kWh "
                "import=%.3f kWh export=%.3f kWh.",
                corrective_request_slot.isoformat(),
                corrective_output_accepted,
                corrective_candidate_winner or "(none)",
                corrective_candidate_solver,
                corrective_rejection_reason or "(none)",
                effective_rec.recommendation if effective_rec is not None else "(none)",
                (
                    effective_rec.batteries_discharged_kwh
                    if effective_rec is not None
                    else 0.0
                ),
                effective_rec.grid_import_kwh if effective_rec is not None else 0.0,
                effective_rec.grid_export_kwh if effective_rec is not None else 0.0,
            )
        async_log("debug", "------ HSEM Coordinator: update cycle complete")

    # ------------------------------------------------------------------
    # DataUpdateCoordinator override
    # ------------------------------------------------------------------

    @override
    async def _async_update_data(self) -> CoordinatorData:
        """Called by DataUpdateCoordinator's internal timer (fallback only).

        The coordinator manages its own interval timer; this method acts as
        a safety-net in case the HA-managed polling fires.  It delegates to
        the same guarded handler to avoid double-execution.
        """
        await self._async_handle_update(None)
        # Return the last data if available, else an empty snapshot.
        return self.data if self.data is not None else CoordinatorData()

    # ------------------------------------------------------------------
    # Timer management
    # ------------------------------------------------------------------

    def _schedule_next_slot_boundary(self, now: datetime) -> None:
        """Schedule one exact callback at the next recommendation boundary."""
        if getattr(self, "_tearing_down", False):
            return
        if self._slot_boundary_timer_unsub is not None:
            self._slot_boundary_timer_unsub()
        interval_minutes = self._cfg.recommendation_interval_minutes
        boundary = _next_slot_boundary_utc(now, interval_minutes)
        self._slot_boundary_interval_minutes = interval_minutes
        self._slot_boundary_timer_unsub = async_track_point_in_utc_time(
            self.hass, self._async_handle_slot_boundary, boundary
        )

    def _refresh_slot_boundary_schedule(self, now: datetime) -> None:
        """Reschedule immediately when a live options update changes cadence."""
        if getattr(self, "_slot_boundary_timer_unsub", None) is None:
            return
        interval_minutes = self._cfg.recommendation_interval_minutes
        if getattr(self, "_slot_boundary_interval_minutes", None) != interval_minutes:
            self._schedule_next_slot_boundary(now)

    async def _async_handle_slot_boundary(self, now: datetime) -> None:
        """Run one non-droppable cycle at a recommendation-slot boundary."""
        self._slot_boundary_timer_unsub = None
        if self._tearing_down:
            return
        try:
            async with self._update_lock:
                if self._tearing_down:
                    return
                await self._async_run_update_cycle()
        finally:
            if not self._tearing_down:
                self._schedule_next_slot_boundary(hsem_now())

    def _cancel_window_hysteresis_expiry(self) -> None:
        """Cancel the pending exact hysteresis-expiry callback, if any."""
        if self._window_hysteresis_timer_unsub is not None:
            self._window_hysteresis_timer_unsub()
        self._window_hysteresis_timer_unsub = None
        self._window_hysteresis_expiry = None

    def _schedule_window_hysteresis_expiry(self, expiry: datetime) -> None:
        """Schedule one non-droppable planner run when a held action expires."""
        expiry_utc = expiry.astimezone(UTC)
        if (
            self._window_hysteresis_timer_unsub is not None
            and self._window_hysteresis_expiry == expiry_utc
        ):
            return
        self._cancel_window_hysteresis_expiry()
        if self._tearing_down:
            return
        self._window_hysteresis_expiry = expiry_utc
        self._window_hysteresis_timer_unsub = async_track_point_in_utc_time(
            self.hass, self._async_handle_window_hysteresis_expiry, expiry_utc
        )

    async def _async_handle_window_hysteresis_expiry(self, now: datetime) -> None:
        """Force a fresh plan exactly when a held recommendation may change."""
        self._window_hysteresis_timer_unsub = None
        self._window_hysteresis_expiry = None
        if self._tearing_down:
            return
        self._window_hysteresis_expiry_replan_pending = True
        async with self._update_lock:
            if self._tearing_down:
                return
            await self._async_run_update_cycle()

    async def _set_update_interval(self, override_minutes: int | None = None) -> None:
        """Register or re-register the periodic update timer.

        Args:
            override_minutes: Force a specific interval in minutes (e.g. 1 when
                entities are missing).  When ``None`` the value from config is used.
        """
        cfg = self._cfg
        minutes = (
            override_minutes if override_minutes is not None else cfg.update_interval
        )
        interval = timedelta(minutes=minutes)
        if self._timer_interval != interval:
            self._timer_interval = interval
            await self._register_interval_timer(interval)
        self._next_update = (hsem_now() + interval).isoformat()

    async def _register_interval_timer(self, interval: timedelta) -> None:
        """Cancel any existing interval timer and register a fresh one.

        Args:
            interval: The new polling cadence.
        """
        if self._interval_timer_unsub is not None:
            self._interval_timer_unsub()
            self._interval_timer_unsub = None
        self._interval_timer_unsub = async_track_time_interval(
            self.hass,
            self._async_handle_update,  # type: ignore[arg-type]  # HA stub expects Callable[[datetime], ...]; our callback also serves as coordinator update callback
            interval,
        )
        async_log(
            "debug",
            "HSEM Coordinator: update interval set to %s",
            interval,
        )

    def _active_force_discharge_slot(self, now: datetime) -> PlannedSlot | None:
        """Return an active slot needing live-demand correction, if any.

        Besides forced battery export, a materially partial normal-discharge
        allocation is monitored because its safe MSC cap intentionally follows
        the solved energy split. Rounding-level grid import is ignored.
        """
        output = self._last_planner_output
        if output is None or now.tzinfo is None:
            return None
        for slot in output.slots:
            if not (
                slot.batteries_discharged_kwh > 1e-9
                and slot_contains(slot.start, slot.end, now)
            ):
                continue
            if slot.recommendation == Recommendations.ForceBatteriesDischarge.value:
                return slot
            if (
                slot.recommendation == Recommendations.BatteriesDischargeMode.value
                and is_material_planned_energy_kwh(slot.grid_import_kwh)
            ):
                return slot
        return None

    def _clear_force_discharge_excess_window(self, reason: str | None = None) -> None:
        """Clear only the in-progress 30-second excess-demand debounce."""
        if reason is not None and self._force_discharge_excess_since is not None:
            async_log(
                "debug",
                "[replan] Live-demand debounce cleared: %s.",
                reason,
            )
        self._force_discharge_excess_since = None
        self._force_discharge_excess_slot_start = None

    async def _async_monitor_force_discharge_load(self, now: datetime) -> None:
        """Request one corrective replan for sustained excess live demand.

        This timer is intentionally lightweight. It reads only the configured
        house and PV power states, and invokes the full coordinator pipeline
        only after the excess survives three 10-second samples (30 seconds).
        """
        slot = self._active_force_discharge_slot(now)
        live = self._live
        if slot is None or live is None or live.degraded_mode is DegradedMode.Error:
            self._force_discharge_live_replan_pending_slot = None
            self._clear_force_discharge_excess_window()
            return

        # Never let an EV already included in the house-power sensor masquerade
        # as an unplanned base-load increase. EV state changes have their own
        # listeners and independently produce a fresh plan.
        if self._cfg.house_power_includes_ev_charger_power and live.any_ev_charging:
            self._force_discharge_live_replan_pending_slot = None
            self._clear_force_discharge_excess_window("EV load is ambiguous")
            return

        slot_start = slot.start
        if self._force_discharge_replanned_slot_start is not None and utc_key(
            self._force_discharge_replanned_slot_start
        ) == utc_key(slot_start):
            self._clear_force_discharge_excess_window()
            return

        remaining_seconds = (utc_key(slot.end) - utc_key(now)).total_seconds()
        if remaining_seconds < FORCE_DISCHARGE_REPLAN_MIN_REMAINING_SECONDS:
            self._force_discharge_live_replan_pending_slot = None
            self._clear_force_discharge_excess_window("less than 60 s remains")
            return

        # A pending request remains pending until run_planner returns. If the
        # update lock was busy or a previous update failed, retry on this tick
        # without consuming the slot's one successful attempt.
        if self._force_discharge_live_replan_pending_slot is not None and utc_key(
            self._force_discharge_live_replan_pending_slot
        ) == utc_key(slot_start):
            if not self._update_lock.locked():
                await self._async_handle_update(None)
            return

        try:
            house_power = ha_get_entity_state_and_convert(
                self, self._cfg.house_consumption_power, "float", 3
            )
            solar_power = ha_get_entity_state_and_convert(
                self, self._cfg.solar_production_power, "float", 3
            )
        except HomeAssistantError:
            self._clear_force_discharge_excess_window("live power is unavailable")
            return

        if not isinstance(house_power, int | float) or not isinstance(
            solar_power, int | float
        ):
            self._clear_force_discharge_excess_window("live power is unavailable")
            return

        metrics = _force_discharge_live_metrics(
            slot,
            discharge_efficiency_pct=self._cfg.batteries_discharge_efficiency,
            house_power_w=float(house_power),
            solar_power_w=float(solar_power),
        )
        if metrics is None:
            self._clear_force_discharge_excess_window()
            return
        planned_supply_w, live_residual_w, threshold_w, excess_w = metrics

        if excess_w <= threshold_w:
            self._clear_force_discharge_excess_window(
                "demand returned within threshold"
            )
            return

        if self._force_discharge_excess_slot_start is None or utc_key(
            self._force_discharge_excess_slot_start
        ) != utc_key(slot_start):
            self._force_discharge_excess_slot_start = slot_start
            self._force_discharge_excess_since = now
            async_log(
                "debug",
                "[replan] Live demand exceeds active discharge plan: "
                "slot=%s planned_supply=%.0fW live_residual=%.0fW excess=%.0fW "
                "threshold=%.0fW; starting 30 s debounce.",
                slot_start.isoformat(),
                planned_supply_w,
                live_residual_w,
                excess_w,
                threshold_w,
            )
            return

        excess_since = self._force_discharge_excess_since
        if (
            excess_since is None
            or (utc_key(now) - utc_key(excess_since)).total_seconds()
            < FORCE_DISCHARGE_EXCESS_DEBOUNCE_SECONDS
        ):
            return

        self._force_discharge_live_replan_pending_slot = slot_start
        async_log(
            "debug",
            "[replan] Sustained live demand exceeded active discharge plan for "
            "30 s: slot=%s planned_supply=%.0fW live_residual=%.0fW; requesting "
            "one corrective replan.",
            slot_start.isoformat(),
            planned_supply_w,
            live_residual_w,
        )
        if not self._update_lock.locked():
            await self._async_handle_update(None)

    # ------------------------------------------------------------------
    # Planner bridge helpers
    # ------------------------------------------------------------------

    def _should_replan(
        self,
        live: LiveState,
        now: datetime,
        *,
        price_forecast_signature: _PriceForecastSignature | None = None,
        load_forecast_signature: _LoadForecastSignature | None = None,
    ) -> bool:
        """Determine whether the planner should be re-run.

        Returns ``True`` when a material event occurred since the last plan:

        - EV connection state changed (plugged in or unplugged)
        - EV charging state changed (started or stopped)
        - EV SoC crossed the target threshold
        - Forced working mode changed
        - Crossed into a new recommendation slot
        - Import price changed significantly (new price period)

        Returns ``False`` when nothing material changed — the previous
        plan can be reused.
        """
        # First run — always plan.
        if self._last_planner_output is None:
            return True

        if getattr(self, "_load_forecast_recovery_replan_pending", False):
            async_log(
                "debug",
                "[replan] Load forecast recovered after a safety hold — re-planning.",
            )
            return True

        if load_forecast_signature is not None and not _load_forecast_signatures_match(
            load_forecast_signature,
            getattr(self, "_last_plan_load_forecast_signature", None),
        ):
            async_log(
                "debug",
                "[replan] Future load forecast changed — re-planning.",
            )
            return True

        if price_forecast_signature is not None and price_forecast_signature != getattr(
            self, "_last_plan_price_forecast_signature", None
        ):
            async_log(
                "debug",
                "[replan] Published price/PV forecast authority changed — re-planning.",
            )
            return True

        if getattr(self, "_window_hysteresis_expiry_replan_pending", False):
            async_log("debug", "[replan] Window hysteresis expired — re-planning.")
            return True

        if self._force_discharge_live_replan_pending_slot is not None:
            async_log(
                "debug",
                "[replan] Sustained live demand exceeded active discharge plan — "
                "re-planning.",
            )
            return True

        # Slot boundary crossed — new slot needs a fresh plan.
        if self._last_plan_slot_start is not None:
            slot_minutes = self._cfg.recommendation_interval_minutes
            if slot_key(now, slot_minutes) != slot_key(
                self._last_plan_slot_start, slot_minutes
            ):
                async_log(
                    "debug",
                    "[replan] Slot boundary crossed (last=%s, now=%s) — re-planning.",
                    self._last_plan_slot_start.isoformat(),
                    now.isoformat(),
                )
                return True

        # EV connection state changed.
        if live.ev.is_connected != self._last_plan_ev_connected:
            async_log(
                "debug",
                "[replan] EV connected state changed (%s → %s) — re-planning.",
                self._last_plan_ev_connected,
                live.ev.is_connected,
            )
            return True

        # EV charging state changed.
        if live.ev.is_charging != self._last_plan_ev_charging:
            async_log(
                "debug",
                "[replan] EV charging state changed (%s → %s) — re-planning.",
                self._last_plan_ev_charging,
                live.ev.is_charging,
            )
            return True

        # EV SoC crossed target threshold.
        ev_soc_below = (
            live.ev.soc_pct is not None
            and live.ev.soc_target_pct is not None
            and live.ev.soc_pct < live.ev.soc_target_pct
        )
        if ev_soc_below != self._last_plan_ev_soc_below_target:
            async_log(
                "debug",
                "[replan] EV SoC target threshold crossed — re-planning.",
            )
            return True

        # Second EV connection state changed.
        if live.ev_second.is_connected != self._last_plan_ev_second_connected:
            async_log(
                "debug",
                "[replan] EV2 connected state changed — re-planning.",
            )
            return True

        # Second EV charging state changed.
        if live.ev_second.is_charging != self._last_plan_ev_second_charging:
            async_log(
                "debug",
                "[replan] EV2 charging state changed — re-planning.",
            )
            return True

        # Second EV SoC crossed target threshold.
        ev2_soc_below = (
            live.ev_second.soc_pct is not None
            and live.ev_second.soc_target_pct is not None
            and live.ev_second.soc_pct < live.ev_second.soc_target_pct
        )
        if ev2_soc_below != self._last_plan_ev_second_soc_below_target:
            async_log(
                "debug",
                "[replan] EV2 SoC target threshold crossed — re-planning.",
            )
            return True

        # EV planned-load config changed (target SoC, smart charging, deadline).
        # These are live-state values that reflect the user's config choices.
        if self._last_plan_ev_target_soc is not None:
            cur_target = live.ev_planned_load_target_soc_pct or 80.0
            if abs(cur_target - self._last_plan_ev_target_soc) > 0.5:
                async_log(
                    "debug",
                    "[replan] EV target SoC changed (%.1f → %.1f) — re-planning.",
                    self._last_plan_ev_target_soc,
                    cur_target,
                )
                return True

        if self._last_plan_ev_smart_charging is not None:
            cur_smart = live.ev_planned_load_smart_charging_enabled
            if cur_smart != self._last_plan_ev_smart_charging:
                async_log(
                    "debug",
                    "[replan] EV smart charging toggled (%s → %s) — re-planning.",
                    self._last_plan_ev_smart_charging,
                    cur_smart,
                )
                return True

        if self._last_plan_ev_deadline is not None:
            cur_deadline = live.ev_planned_load_deadline
            if cur_deadline is None or utc_key(cur_deadline) != utc_key(
                self._last_plan_ev_deadline
            ):
                async_log(
                    "debug",
                    "[replan] EV deadline changed — re-planning.",
                )
                return True

        if self._last_plan_ev2_target_soc is not None:
            cur_target2 = live.ev_second_planned_load_target_soc_pct or 80.0
            if abs(cur_target2 - self._last_plan_ev2_target_soc) > 0.5:
                async_log(
                    "debug",
                    "[replan] EV2 target SoC changed (%.1f → %.1f) — re-planning.",
                    self._last_plan_ev2_target_soc,
                    cur_target2,
                )
                return True

        if self._last_plan_ev2_smart_charging is not None:
            cur_smart2 = live.ev_second_planned_load_smart_charging_enabled
            if cur_smart2 != self._last_plan_ev2_smart_charging:
                async_log(
                    "debug",
                    "[replan] EV2 smart charging toggled (%s → %s) — re-planning.",
                    self._last_plan_ev2_smart_charging,
                    cur_smart2,
                )
                return True

        if self._last_plan_ev2_deadline is not None:
            cur_deadline2 = live.ev_second_planned_load_deadline
            if cur_deadline2 is None or utc_key(cur_deadline2) != utc_key(
                self._last_plan_ev2_deadline
            ):
                async_log(
                    "debug",
                    "[replan] EV2 deadline changed — re-planning.",
                )
                return True

        if self._cfg.secondary_storage.enabled:
            secondary = live.secondary_storage
            if (
                self._last_plan_secondary_soc_pct is not None
                and secondary.soc_pct is not None
                and abs(secondary.soc_pct - self._last_plan_secondary_soc_pct)
                >= SECONDARY_STORAGE_SOC_REPLAN_DELTA_PCT
            ):
                async_log(
                    "debug",
                    "[replan] Secondary SoC changed (%.1f → %.1f) — re-planning.",
                    self._last_plan_secondary_soc_pct,
                    secondary.soc_pct,
                )
                return True
            if (
                self._last_plan_secondary_load_power_w is not None
                and secondary.load_power_w is not None
                and abs(secondary.load_power_w - self._last_plan_secondary_load_power_w)
                >= SECONDARY_STORAGE_LOAD_REPLAN_DELTA_W
            ):
                async_log(
                    "debug",
                    "[replan] Secondary dedicated load changed materially — re-planning.",
                )
                return True
            if (
                secondary.output_source_priority
                != self._last_plan_secondary_output_priority
            ):
                async_log(
                    "debug",
                    "[replan] Secondary output priority changed — re-planning.",
                )
                return True

        # Forced working mode changed.
        if live.force_working_mode_state != self._last_plan_force_mode:
            async_log(
                "debug",
                "[replan] Force working mode changed — re-planning.",
            )
            return True

        # Import price changed significantly (new price period).
        if self._last_plan_import_price is not None:
            price_delta = abs(
                (live.import_electricity_price or 0.0) - self._last_plan_import_price
            )
            if price_delta > 0.001:
                async_log(
                    "debug",
                    "[replan] Import price changed (%.4f → %.4f) — re-planning.",
                    self._last_plan_import_price,
                    live.import_electricity_price,
                )
                return True

        # Nothing material changed — stick to the plan.
        return False

    def _persist_plan_state(
        self,
        live: LiveState,
        *,
        price_forecast_signature: _PriceForecastSignature | None = None,
        load_forecast_signature: _LoadForecastSignature | None = None,
    ) -> None:
        """Record the current state after a successful plan run.

        Called after every planner run so ``_should_replan`` can compare
        against the state that existed when the plan was created.
        """
        self._last_plan_ev_connected = live.ev.is_connected
        self._last_plan_ev_charging = live.ev.is_charging
        self._last_plan_ev_soc_below_target = (
            live.ev.soc_pct is not None
            and live.ev.soc_target_pct is not None
            and live.ev.soc_pct < live.ev.soc_target_pct
        )
        self._last_plan_ev_second_connected = live.ev_second.is_connected
        self._last_plan_ev_second_charging = live.ev_second.is_charging
        self._last_plan_ev_second_soc_below_target = (
            live.ev_second.soc_pct is not None
            and live.ev_second.soc_target_pct is not None
            and live.ev_second.soc_pct < live.ev_second.soc_target_pct
        )
        self._last_plan_force_mode = live.force_working_mode_state
        self._last_plan_import_price = live.import_electricity_price
        # EV planned-load config values (target SoC, smart charging, deadline).
        self._last_plan_ev_target_soc = live.ev_planned_load_target_soc_pct or 80.0
        self._last_plan_ev_smart_charging = live.ev_planned_load_smart_charging_enabled
        self._last_plan_ev_deadline = live.ev_planned_load_deadline
        self._last_plan_ev2_target_soc = (
            live.ev_second_planned_load_target_soc_pct or 80.0
        )
        self._last_plan_ev2_smart_charging = (
            live.ev_second_planned_load_smart_charging_enabled
        )
        self._last_plan_ev2_deadline = live.ev_second_planned_load_deadline
        self._last_plan_secondary_soc_pct = live.secondary_storage.soc_pct
        self._last_plan_secondary_load_power_w = live.secondary_storage.load_power_w
        self._last_plan_secondary_output_priority = (
            live.secondary_storage.output_source_priority
        )
        if price_forecast_signature is not None:
            self._last_plan_price_forecast_signature = price_forecast_signature
        if load_forecast_signature is not None:
            self._last_plan_load_forecast_signature = load_forecast_signature
            self._load_forecast_recovery_replan_pending = False

    def _persist_plan_state_if_accepted(
        self,
        live: LiveState,
        plan_state_should_persist: bool,
        *,
        price_forecast_signature: _PriceForecastSignature | None = None,
        load_forecast_signature: _LoadForecastSignature | None = None,
    ) -> None:
        """Advance material-change baselines only for a published new plan."""
        if plan_state_should_persist:
            self._persist_plan_state(
                live,
                price_forecast_signature=price_forecast_signature,
                load_forecast_signature=load_forecast_signature,
            )

    def _freeze_ev_charger_power_for_current_slot(
        self, output: PlannerOutput, now: datetime
    ) -> None:
        """Freeze per-EV charger power for the current slot across replans.

        The EV planner recomputes ``ev_charger_calculated_power`` from live
        data whenever the planner reruns. Without freezing, the charger
        command oscillates inside a single 15-minute slot as clouds pass or
        the EV itself toggles on/off. We store the value computed at slot
        start and rewrite the current slot with that stored value on every
        subsequent replan until the next slot begins.

        Explicit overrides (force-charge-now, auto-full-EV) are applied
        after this freeze, so they can still change the current slot's
        command for as long as they remain active. When an override ends,
        the freeze restores the originally planned slot-start value.

        Args:
            output: Planner output whose current slot will be rewritten in
                place when we are still inside the same slot.
            now: Timezone-aware current datetime used to locate the current
                slot and compare against the stored slot-start time.
        """
        if now.tzinfo is None:
            return

        for slot in output.slots:
            s_start = as_tz(slot.start, now.tzinfo)
            if not slot_contains(slot.start, slot.end, now):
                continue

            price_actionable = bool(slot.price_actionable)
            new_slot = self._current_slot_start is None or utc_key(
                self._current_slot_start
            ) != utc_key(s_start)
            authority_changed = (
                getattr(self, "_current_slot_price_actionable", None)
                is not price_actionable
            )
            if new_slot or authority_changed:
                # A new slot or same-slot price-authority transition captures
                # fresh planner power. This prevents withdrawal from restoring
                # an old optional-EV command while preserving fixed-session
                # load that the nonactionable plan still publishes.
                self._current_slot_start = s_start
                self._current_slot_price_actionable = price_actionable
                self._current_slot_ev_power_w = slot.ev_charger_calculated_power
                self._current_slot_ev_second_power_w = (
                    slot.ev_second_charger_calculated_power
                )
                async_log(
                    "debug",
                    "[freeze] New EV power baseline for slot %s "
                    "(price_actionable=%s): primary=%dW second=%dW",
                    s_start.isoformat(),
                    price_actionable,
                    self._current_slot_ev_power_w,
                    self._current_slot_ev_second_power_w,
                )
            else:
                # Same slot — restore the frozen baseline so the charger
                # command does not chase live conditions.
                if (
                    abs(
                        slot.ev_charger_calculated_power - self._current_slot_ev_power_w
                    )
                    > 1e-9
                    or abs(
                        slot.ev_second_charger_calculated_power
                        - self._current_slot_ev_second_power_w
                    )
                    > 1e-9
                ):
                    async_log(
                        "debug",
                        "[freeze] Restoring frozen EV power for slot %s: "
                        "primary %dW→%dW, second %dW→%dW",
                        s_start.isoformat(),
                        slot.ev_charger_calculated_power,
                        self._current_slot_ev_power_w,
                        slot.ev_second_charger_calculated_power,
                        self._current_slot_ev_second_power_w,
                    )
                slot.ev_charger_calculated_power = self._current_slot_ev_power_w
                slot.ev_second_charger_calculated_power = (
                    self._current_slot_ev_second_power_w
                )
            break

    def _apply_planner_output(self, output: PlannerOutput) -> None:
        """Write :class:`PlannerOutput` decisions back into the recommendation list.

        The lookup normalises both sides to UTC with ``microsecond=0`` so that
        slots remain matched even when the recommendation list was created from
        ``hsem_now()`` while the planner slots were built from timedelta
        arithmetic (always zero microseconds).  Any recommendation slot that
        cannot be matched emits a warning so the mismatch is visible in logs.

        Args:
            output: The :class:`~planner.engine.PlannerOutput` returned by the
                planner engine.
        """
        slot_by_utc = {utc_key(s.start): s for s in output.slots}

        unmatched: list[str] = []
        for rec in self._hourly_recommendations:
            slot = slot_by_utc.get(utc_key(rec.start))
            if slot is None:
                unmatched.append(rec.start.isoformat())
                continue
            rec.recommendation = slot.recommendation
            # Publish the live-resolved current-slot load alongside the flows
            # derived from it. Keep the raw 1d/3d/7d/14d diagnostics intact.
            rec.historical_avg_house_consumption_kwh = rec.avg_house_consumption_kwh
            rec.avg_house_consumption_kwh = slot.avg_house_consumption_kwh
            rec.batteries_charged_kwh = slot.batteries_charged_kwh
            rec.batteries_discharged_kwh = slot.batteries_discharged_kwh
            rec.estimated_net_consumption_kwh = slot.estimated_net_consumption_kwh
            rec.ev_planned_load_kwh = slot.ev_planned_load_kwh
            rec.ev_accounted_load_kwh = slot.ev_accounted_load_kwh
            rec.ev_total_planned_load_kwh = slot.ev_total_planned_load_kwh
            rec.ev_charger_calculated_power = slot.ev_charger_calculated_power
            rec.ev_second_charger_calculated_power = (
                slot.ev_second_charger_calculated_power
            )
            rec.secondary_storage_load_kwh = slot.secondary_storage_load_kwh
            rec.secondary_storage_charged_kwh = slot.secondary_storage_charged_kwh
            rec.secondary_storage_discharged_kwh = slot.secondary_storage_discharged_kwh
            rec.secondary_storage_grid_import_kwh = (
                slot.secondary_storage_grid_import_kwh
            )
            rec.secondary_storage_estimated_capacity_kwh = (
                slot.secondary_storage_estimated_capacity_kwh
            )
            rec.secondary_storage_estimated_soc_pct = (
                slot.secondary_storage_estimated_soc_pct
            )
            rec.secondary_storage_charge_current_a = (
                slot.secondary_storage_charge_current_a
            )
            rec.secondary_storage_mode = slot.secondary_storage_mode
            rec.estimated_cost_currency = slot.estimated_cost_currency
            rec.estimated_battery_capacity_kwh = slot.estimated_battery_capacity_kwh
            rec.estimated_battery_soc_pct = slot.estimated_battery_soc_pct
            rec.grid_import_kwh = slot.grid_import_kwh
            rec.grid_export_kwh = slot.grid_export_kwh
            rec.primary_battery_hold = slot.primary_battery_hold
            rec.import_price_available = slot.import_price_available
            rec.export_price_available = slot.export_price_available
            rec.import_price_source = slot.import_price_source
            rec.export_price_source = slot.export_price_source
            rec.price_actionable = slot.price_actionable
            # Copy the planner's PV estimate so that solcast_pv_estimate,
            # estimated_net_consumption, and ev_planned_load_kwh are all
            # internally consistent in the final HourlyRecommendation output.
            # The planner may have applied confidence decay or other transforms
            # that differ from the raw value stored by the data populator.
            rec.solcast_pv_estimate_kwh = slot.solcast_pv_estimate_kwh

        if unmatched:
            async_log(
                "warning",
                "[HSEM] _apply_planner_output: %d recommendation slot(s) had no "
                "matching planner output slot — planner fields (ev_planned_load_kwh, "
                "ev_accounted_load_kwh, ev_total_planned_load_kwh, recommendation, …) "
                "will remain at default 0.0 for these slots. "
                "First unmatched rec.start: %s",
                len(unmatched),
                unmatched[0],
            )

        # Preserve the plan explanation and data quality for the next CoordinatorData snapshot.
        self._plan_explanation = output.explanation
        self._data_quality = output.data_quality

        # Persist the winning candidate name and score for hysteresis (issue #372).
        # The next planner run will compare against these values.
        if output.winner_name and output.candidates:
            winner_score = 0.0
            for c in output.candidates:
                if (
                    c.name == output.winner_name
                    and hasattr(c, "_cost")
                    and c._cost is not None
                ):
                    winner_score = c._cost.score
                    break
            self._previous_planner_winner_name = output.winner_name
            self._previous_planner_winner_score = winner_score

    # ------------------------------------------------------------------
    # Forecast-vs-actual tracking (issue #373)
    # ------------------------------------------------------------------

    def _accumulate_forecast_actuals(self, now: datetime, live: LiveState) -> None:
        """Accumulate measured energy and close eligible forecast baselines.

        The prior power sample represents ``[previous_timestamp, now)``.  The
        tracker splits that interval by physical UTC overlap, so a cycle that
        crosses a slot boundary cannot assign all energy to the new slot.

        Args:
            now: Current time (timezone-aware).
            live: The live HA entity state snapshot.
        """
        now_key = utc_key(now)

        if self._hourly_recommendations and (
            self._forecast_tracker.reconcile_unfinalised_layout(
                (
                    (recommendation.start, recommendation.end)
                    for recommendation in self._hourly_recommendations
                ),
                now=now,
            )
        ):
            # A 15↔60-minute options change invalidates every in-flight/future
            # baseline. Reset both endpoints so no prior-power interval bridges
            # the layout boundary; finalised accuracy history remains intact.
            self._last_accumulation_ts = None
            self._last_pv_power_w = None
            self._last_load_power_w = None
            getattr(self, "_prediction_restore_excluded", set()).clear()

        # This method runs before the planner.  Capture raw, future-only inputs
        # now so the post-planner registration cannot mistake corrected or
        # live-injected current-slot values for a pre-slot forecast.
        self._pre_plan_forecast_baselines = {
            utc_key(rec.start): (
                float(rec.solcast_pv_estimate_kwh),
                bool(rec.solcast_pv_estimate_available),
            )
            for rec in self._hourly_recommendations
            if utc_key(rec.start) > now_key
        }
        self._solar_corrector.set_reference_time(now)
        self._forecast_tracker.freeze_forecasts(now)

        previous_ts = self._last_accumulation_ts
        previous_pv_power_w = getattr(self, "_last_pv_power_w", None)
        previous_load_power_w = getattr(self, "_last_load_power_w", None)
        self._last_accumulation_ts = now
        pv_missing = any(
            "solar_production_power" in item for item in live.missing_entities_list
        )
        load_missing = any(
            "house_consumption_power" in item for item in live.missing_entities_list
        )
        pv_sample = None if pv_missing else live.solar_production_power_w
        load_sample = None if load_missing else live.house_consumption_power_w
        self._last_pv_power_w = (
            float(pv_sample)
            if pv_sample is not None and math.isfinite(float(pv_sample))
            else None
        )
        self._last_load_power_w = (
            float(load_sample)
            if load_sample is not None and math.isfinite(float(load_sample))
            else None
        )

        if (
            previous_ts is not None
            and previous_pv_power_w is not None
            and previous_load_power_w is not None
            and self._last_pv_power_w is not None
            and self._last_load_power_w is not None
        ):
            elapsed_seconds = (now_key - utc_key(previous_ts)).total_seconds()
            effective_interval = getattr(self, "_timer_interval", None)
            if isinstance(effective_interval, timedelta):
                cadence_seconds = effective_interval.total_seconds()
            else:
                cadence_seconds = float(self._cfg.update_interval) * 60.0
            # Never let an invalid/zero override make every real sample stale.
            # Two expected polls may be bridged; anything longer is rejected.
            cadence_seconds = max(cadence_seconds, 60.0)
            max_gap_seconds = 2.0 * cadence_seconds
            assigned_seconds = self._forecast_tracker.accumulate_power_interval(
                previous_ts,
                now,
                pv_power_w=previous_pv_power_w,
                load_power_w=previous_load_power_w,
                max_gap_seconds=max_gap_seconds,
            )
            if elapsed_seconds > max_gap_seconds:
                async_log(
                    "debug",
                    "[forecast_tracker] rejected stale %.1fs power sample gap "
                    "(limit %.1fs)",
                    elapsed_seconds,
                    max_gap_seconds,
                )
            elif assigned_seconds + 1.0 < max(elapsed_seconds, 0.0):
                async_log(
                    "debug",
                    "[forecast_tracker] only %.1fs of %.1fs sample interval "
                    "overlapped frozen records",
                    assigned_seconds,
                    elapsed_seconds,
                )

        # Finalise any slots whose end time has passed.
        newly_finalised_keys = {
            utc_key(frec.start)
            for frec in self._forecast_tracker.records
            if not frec.finalised and utc_key(frec.end) <= now_key
        }
        self._forecast_tracker.finalise_past_records(now)

        # -------------------------------------------------------------------
        # Solar forecast auto-correction (issue #602)
        # -------------------------------------------------------------------
        # Feed every newly-finalised forecast tracker record into the solar
        # corrector so it can learn per-hour accuracy factors and update the
        # intra-hour residual buffer.
        for frec in self._forecast_tracker.records:
            if not frec.finalised:
                continue
            processed_key = utc_key(frec.start)
            processed_through = getattr(
                self, "_solar_corrector_processed_through", None
            )
            corrector_watermark = self._solar_corrector.processed_through
            if isinstance(corrector_watermark, datetime) and (
                processed_through is None
                or utc_key(corrector_watermark) > utc_key(processed_through)
            ):
                processed_through = corrector_watermark
                self._solar_corrector_processed_through = corrector_watermark
            if processed_key in self._solar_corrector_processed or (
                processed_through is not None
                and processed_key <= utc_key(processed_through)
            ):
                continue

            # Mark incomplete / legacy-live records processed without teaching
            # the corrector, so they cannot repeatedly enter this loop.
            if not frec.accuracy_eligible:
                self._solar_corrector_processed.add(processed_key)
                self._solar_corrector.mark_processed(processed_key)
                if processed_through is None or processed_key > utc_key(
                    processed_through
                ):
                    self._solar_corrector_processed_through = processed_key
                continue

            raw_pv_kwh = frec.raw_forecast_pv_kwh
            assert raw_pv_kwh is not None

            self._solar_corrector.update_hour(
                frec.start.hour, raw_pv_kwh, frec.actual_pv_kwh
            )
            self._solar_corrector.update_residual(
                frec.forecast_pv_kwh, frec.actual_pv_kwh
            )
            self._solar_corrector_processed.add(processed_key)
            self._solar_corrector.mark_processed(processed_key)
            if processed_through is None or processed_key > utc_key(processed_through):
                self._solar_corrector_processed_through = processed_key

        # The persisted UTC watermark carries long-term replay protection.
        # Retain only keys still present in the tracker's bounded live buffer.
        tracked_keys = {
            utc_key(record.start) for record in self._forecast_tracker.records
        }
        self._solar_corrector_processed.intersection_update(tracked_keys)

        # -------------------------------------------------------------------
        # Prediction accuracy scorecard (issue #601)
        # -------------------------------------------------------------------
        # Feed completed slots into the prediction accuracy tracker so the
        # sensor can report SoC MAE, solar MAPE, and action mix.
        actual_soc_sample = live.huawei_batteries_soc_pct
        actual_soc = (
            float(actual_soc_sample)
            if actual_soc_sample is not None and math.isfinite(float(actual_soc_sample))
            else None
        )
        prediction_restore_excluded: set[datetime] = getattr(
            self, "_prediction_restore_excluded", set()
        )
        for frec in self._forecast_tracker.records:
            processed_key = utc_key(frec.start)
            if (
                not frec.finalised
                or not frec.prediction_eligible
                or processed_key not in newly_finalised_keys
                or processed_key in prediction_restore_excluded
                or actual_soc is None
            ):
                continue
            forecast_soc_pct = frec.forecast_soc_pct
            forecast_action = frec.forecast_action
            assert forecast_soc_pct is not None
            assert forecast_action is not None
            self._prediction_tracker.add_record(
                predicted_soc=forecast_soc_pct,
                actual_soc=actual_soc,
                predicted_pv=frec.forecast_pv_kwh,
                actual_pv=frec.actual_pv_kwh,
                predicted_load=frec.forecast_load_kwh,
                actual_load=frec.actual_load_kwh,
                action=forecast_action,
                slot_start=frec.start,
            )
        prediction_restore_excluded.difference_update(newly_finalised_keys)

    def _register_forecasts_from_planner(
        self, output: PlannerOutput, now: datetime
    ) -> None:
        """Register future planner forecasts against raw pre-plan baselines.

        Replans may refresh a future baseline, but the tracker freezes the last
        value observed before physical slot start.  Current/past planner values
        are excluded because current-slot PV/load may have been live-injected.

        Args:
            output: The :class:`~planner.engine.PlannerOutput` returned by the
                planner engine.
            now: Current time at which the planner output was observed.
        """
        now_key = utc_key(now)
        baselines = getattr(self, "_pre_plan_forecast_baselines", {})
        for slot in output.slots:
            slot_key_utc = utc_key(slot.start)
            if slot_key_utc <= now_key:
                continue
            baseline = baselines.get(slot_key_utc)
            if baseline is None:
                continue
            raw_pv_kwh, raw_pv_available = baseline
            if not raw_pv_available:
                continue

            pv_forecast = getattr(slot, "solcast_pv_estimate_kwh", 0.0)
            load_forecast = getattr(slot, "avg_house_consumption_kwh", 0.0)
            self._forecast_tracker.get_or_create_record(slot.start, slot.end)
            self._forecast_tracker.set_forecasts(
                start=slot.start,
                pv_kwh=pv_forecast,
                load_kwh=load_forecast,
                raw_pv_kwh=raw_pv_kwh,
                forecast_soc_pct=slot.estimated_battery_soc_pct,
                forecast_action=_action_label(slot.recommendation),
                observed_at=now,
            )

    # ------------------------------------------------------------------
    # Daily plan-vs-actual accumulation (issue #540)
    # ------------------------------------------------------------------

    async def _accumulate_daily_plan_actuals(
        self,
        now: datetime,
        live: LiveState,
        output: PlannerOutput,
    ) -> None:
        """Accumulate measured and planned values on HA's local timeline.

        Actual cumulative-meter deltas are sampled before rollover so an
        interval crossing midnight can be split between both dates. Plan slots
        are then recorded against the newly selected local day.
        """
        await self._init_daily_tracker()
        tracker = self._daily_tracker
        effective_interval = getattr(self, "_timer_interval", None)
        if isinstance(effective_interval, timedelta):
            cadence_seconds = effective_interval.total_seconds()
        else:
            cadence_seconds = float(self._cfg.update_interval) * 60.0
        cadence_seconds = max(cadence_seconds, 60.0)
        max_gap_seconds = 2.0 * cadence_seconds

        price_intervals = tuple(
            ActualPriceInterval(
                start=slot.start,
                end=slot.end,
                import_price=slot.price.import_price,
                export_price=slot.price.export_price,
                import_price_available=slot.import_price_available,
                export_price_available=slot.export_price_available,
            )
            for slot in output.slots
        )

        soc_pct = live.huawei_batteries_soc_pct
        rated_cap_kwh = (live.huawei_batteries_rated_capacity_wh or 0.0) / 1000.0
        tracker.accumulate_actual(
            grid_import_energy_kwh=live.grid_import_energy_kwh,
            grid_export_energy_kwh=live.grid_export_energy_kwh,
            pv_energy_kwh=live.pv_energy_kwh,
            soc_pct=soc_pct,
            rated_capacity_kwh=rated_cap_kwh,
            import_price=live.import_electricity_price,
            export_price=live.export_electricity_price,
            import_price_available=live.import_electricity_price_available,
            export_price_available=live.export_electricity_price_available,
            max_gap_seconds=max_gap_seconds,
            now=now,
            price_intervals=price_intervals,
        )

        rollover = await tracker.check_day_rollover(now)
        if rollover is not None:
            self._daily_plan_last_accumulated = None

        # Capture the plan once when its slot starts, after any local-date
        # rollover reset has selected the destination accumulator.
        self._daily_plan_last_accumulated = _accumulate_plan_for_slots(
            tracker,
            output.slots,
            now,
            self._daily_plan_last_accumulated,
        )

    # ------------------------------------------------------------------
    # Financial tracker accumulation (issue #599)
    # ------------------------------------------------------------------

    async def _init_financial_tracker(self) -> None:
        """Lazily initialise the financial tracker.

        Called once on the first access.  Loads the JSON history file.
        Failures are logged and leave the tracker with an empty history
        file path so the sensors show 'no data' rather than crashing the
        coordinator.
        """
        if getattr(self, "_financial_tracker_initialized", True):
            return

        try:
            config_dir = self.hass.config.config_dir
            self._financial_tracker.history_file = str(
                Path(config_dir) / ".storage" / "hsem_financial_history.json"
            )
            await self._load_financial_tracker()
            self._financial_tracker_initialized = True
        except Exception:
            async_log(
                "error",
                "Failed to initialise financial tracker "
                "(financial sensors will be unavailable)",
            )
            self._financial_tracker_initialized = True  # don't retry

    async def _load_financial_tracker(self) -> None:
        """Load financial tracker state from the JSON persistence file."""
        path = Path(self._financial_tracker.history_file)
        if not path.exists():
            return
        try:
            data = await asyncio.to_thread(FinancialTracker._read_history_file, path)
            if data is not None:
                loaded = FinancialTracker.from_dict(data)
                self._financial_tracker = loaded
        except Exception:
            async_log("error", "Failed to load financial tracker history")

    async def _persist_financial_tracker(self) -> bool:
        """Persist financial tracker state to disk atomically."""
        if not self._financial_tracker.history_file:
            return False
        data = self._financial_tracker.as_dict()
        path = Path(self._financial_tracker.history_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        return await asyncio.to_thread(FinancialTracker._write_history_file, data, path)

    async def _accumulate_financials(
        self,
        now: datetime,
        live: LiveState,
    ) -> None:
        """Accumulate import cost and export income into the financial tracker.

        Called each coordinator cycle after plan-vs-actual accumulation.
        Handles day rollover (snapshotting yesterday's totals) before
        accumulating the live cost deltas from the energy meters.

        Args:
            now: Current datetime (timezone-aware).
            live: Live HA entity state snapshot.
        """
        await self._init_financial_tracker()
        tracker = self._financial_tracker

        # Check and handle day rollover first.
        tracker.check_day_rollover(now)

        # Accumulate cost deltas from live meter readings.
        tracker.accumulate(
            grid_import_energy_kwh=live.grid_import_energy_kwh,
            grid_export_energy_kwh=live.grid_export_energy_kwh,
            import_price=live.import_electricity_price,
            export_price=live.export_electricity_price,
            import_price_available=live.import_electricity_price_available,
            export_price_available=live.export_electricity_price_available,
        )

    async def _accumulate_savings(
        self,
        now: datetime,
        live: LiveState,
        output: PlannerOutput,
    ) -> None:
        """Accumulate date-specific savings from measured interval deltas."""
        await self._init_savings_tracker()
        st = self._savings_tracker
        dt = self._daily_tracker
        savings_rollover = st.check_day_rollover(now.date().isoformat())
        effective_interval = getattr(self, "_timer_interval", None)
        if isinstance(effective_interval, timedelta):
            cadence_seconds = effective_interval.total_seconds()
        else:
            cadence_seconds = float(self._cfg.update_interval) * 60.0
        cadence_seconds = max(cadence_seconds, 60.0)
        max_gap_seconds = 2.0 * cadence_seconds

        prior_price_slots = getattr(self, "_savings_price_slots", ())
        self._savings_price_slots = tuple(replace(slot) for slot in output.slots)

        charge_sample = st.sample_charge_interval(
            now,
            live.huawei_batteries_charge_discharge_power_w,
            max_gap_seconds=max_gap_seconds,
        )
        charge_savings_by_date = self._compute_actual_charge_savings(
            output,
            charge_sample,
            prior_slots=prior_price_slots,
        )

        actual_by_date = dt.last_actual_delta_by_date
        switch_on = live.force_working_mode_state == "auto"
        all_dates = sorted(set(actual_by_date) | set(charge_savings_by_date))
        for day_str in all_dates:
            actual = actual_by_date.get(day_str, DailyMetrics())
            st.accumulate(
                export_revenue_delta=actual.grid_export_rev,
                charge_savings_delta=charge_savings_by_date.get(day_str, 0.0),
                baseline_cost_delta=actual.grid_import_cost,
                switch_on=switch_on,
                day=day_str,
            )

        # Consume this cycle's explicit deltas. If daily accumulation fails on
        # a later coordinator cycle, savings cannot accidentally reuse them.
        dt.last_actual_delta_by_date = {}
        if savings_rollover is not None and st.history_file:
            saved = await st.save_history()
            if not saved:
                async_log(
                    "warning",
                    "Failed to persist corrected savings rollover for %s",
                    savings_rollover.date,
                )

    @classmethod
    def _compute_actual_charge_savings(
        cls,
        output: PlannerOutput,
        charge_sample: tuple[datetime, datetime, float] | None,
        *,
        prior_slots: Sequence[PlannedSlot] = (),
    ) -> dict[str, float]:
        """Value measured charge over prior/current UTC-de-duplicated slots."""
        if charge_sample is None:
            return {}
        interval_start, interval_end, charged_kwh = charge_sample
        start_utc = utc_key(interval_start)
        end_utc = utc_key(interval_end)
        total_seconds = (end_utc - start_utc).total_seconds()
        if total_seconds <= 0 or charged_kwh <= 1e-9:
            return {}

        local_tz = interval_end.tzinfo
        if local_tz is None:
            return {}

        prior_authority = tuple(prior_slots)
        current_authority = tuple(output.slots)
        boundaries = {start_utc, end_utc}
        for slot in (*prior_authority, *current_authority):
            overlap_start = max(start_utc, utc_key(slot.start))
            overlap_end = min(end_utc, utc_key(slot.end))
            if overlap_end > overlap_start:
                boundaries.update((overlap_start, overlap_end))

        ordered_boundaries = sorted(boundaries)
        segments: list[
            tuple[datetime, datetime, PlannedSlot, tuple[PlannedSlot, ...], bool]
        ] = []
        for segment_start, segment_end in zip(
            ordered_boundaries, ordered_boundaries[1:], strict=False
        ):
            prior_slot = next(
                (
                    slot
                    for slot in prior_authority
                    if utc_key(slot.start) <= segment_start
                    and utc_key(slot.end) >= segment_end
                ),
                None,
            )
            if prior_slot is not None:
                segments.append(
                    (segment_start, segment_end, prior_slot, prior_authority, True)
                )
                continue
            current_slot = next(
                (
                    slot
                    for slot in current_authority
                    if utc_key(slot.start) <= segment_start
                    and utc_key(slot.end) >= segment_end
                ),
                None,
            )
            if current_slot is not None:
                segments.append(
                    (
                        segment_start,
                        segment_end,
                        current_slot,
                        current_authority,
                        False,
                    )
                )
        daily_averages: dict[tuple[bool, date], float] = {}
        savings_by_date: dict[str, float] = {}
        for (
            overlap_start,
            overlap_end,
            slot,
            source_slots,
            uses_prior,
        ) in segments:
            overlap_seconds = (overlap_end - overlap_start).total_seconds()
            import_price = slot.price.import_price
            if (
                overlap_seconds <= 0
                or slot.recommendation not in CHARGE_RECS
                or not slot.price_actionable
                or not slot.import_price_available
                or not math.isfinite(import_price)
            ):
                continue

            midpoint = overlap_start + (overlap_end - overlap_start) / 2
            local_date = midpoint.astimezone(local_tz).date()
            average_key = (uses_prior, local_date)
            avg_import_price = daily_averages.setdefault(
                average_key,
                cls._compute_daily_avg_import_price(
                    source_slots,
                    local_date,
                    interval_end,
                ),
            )
            if avg_import_price <= 0 or import_price >= avg_import_price:
                continue

            energy_share = charged_kwh * overlap_seconds / total_seconds
            day_str = local_date.isoformat()
            savings_by_date[day_str] = savings_by_date.get(day_str, 0.0) + (
                energy_share * (avg_import_price - import_price)
            )
        return savings_by_date

    @staticmethod
    def _compute_daily_avg_import_price(
        slots: Sequence[PlannedSlot],
        target_date: date,
        reference: datetime,
    ) -> float:
        """Compute a published-price mean for one injected HA-local date."""
        prices = [
            float(slot.price.import_price)
            for slot in slots
            if slot.start.astimezone(reference.tzinfo).date() == target_date
            and slot.import_price_available
            and math.isfinite(slot.price.import_price)
        ]
        return sum(prices) / len(prices) if prices else 0.0

    async def _init_savings_tracker(self) -> None:
        """Lazily initialise the savings tracker."""
        if getattr(self, "_savings_tracker_initialized", True):
            return

        try:
            config_dir = self.hass.config.config_dir
            self._savings_tracker.history_file = str(
                Path(config_dir) / ".storage" / "hsem_savings_history.json"
            )
            await self._savings_tracker.load_history()
            self._savings_tracker_initialized = True
        except Exception:
            async_log(
                "error",
                "Failed to initialise savings tracker "
                "(savings sensor will be unavailable)",
            )
            self._savings_tracker_initialized = True  # don't retry

    async def _init_daily_tracker(self) -> None:
        """Lazily initialise the daily plan-vs-actual tracker.

        Called once on the first access.  Registers the midnight timer
        and loads the history file.  Failures are logged and leave the
        tracker with an empty history file path so the sensor shows
        'no data' rather than crashing the coordinator.
        """
        if getattr(self, "_daily_tracker_initialized", True):
            return

        try:
            config_dir = self.hass.config.config_dir
            self._daily_tracker.history_file = str(
                Path(config_dir) / ".storage" / "hsem_daily_history.json"
            )
            await self._daily_tracker.load_history()

            self._midnight_unsub = async_track_time_change(
                self.hass,
                self._async_handle_midnight,
                hour=0,
                minute=0,
                second=0,
            )
            self._daily_tracker_initialized = True
        except Exception:
            async_log(
                "error",
                "Failed to initialise daily tracker (plan-vs-actual "
                "sensor will be unavailable)",
            )
            self._daily_tracker_initialized = True  # don't retry

    async def _async_handle_midnight(self, _now: datetime) -> None:
        """Persist a provisional old-day snapshot without resetting baselines.

        The first post-midnight cumulative-meter sample is needed to split the
        final physical interval across local midnight. The normal coordinator
        cycle performs the final rollover and idempotently replaces this
        crash-safe provisional record.
        """
        tracker = self._daily_tracker
        new_day = _now.date().isoformat()
        if tracker.history_file and tracker.today and tracker.today != new_day:
            prior_record = tracker._build_today_record()
            saved = await tracker._save_record_to_history(prior_record)
            if saved:
                async_log(
                    "info",
                    "Provisional daily plan-vs-actual record saved for %s",
                    tracker.today,
                )
            else:
                async_log(
                    "warning",
                    "Failed to save provisional daily record for %s",
                    tracker.today,
                )

        # Persist the financial tracker at midnight so daily log survives
        # HA restarts.
        financial = self._financial_tracker
        if financial.history_file:
            saved = await self._persist_financial_tracker()
            if saved:
                async_log(
                    "info",
                    "Financial tracker persisted for %s",
                    financial.today,
                )
            else:
                async_log(
                    "warning",
                    "Failed to persist financial tracker for %s",
                    financial.today,
                )

        # Persist savings tracker state at midnight.
        st = self._savings_tracker
        if st.history_file:
            saved = await st.save_history()
            if saved:
                async_log("info", "Savings tracker state saved for %s", st._today)
            else:
                async_log(
                    "warning",
                    "Failed to save savings tracker state for %s",
                    st._today,
                )


# ---------------------------------------------------------------------------
# Module-level helpers for daily plan-vs-actual accumulation
# ---------------------------------------------------------------------------


def _accumulate_plan_for_slots(
    tracker: DailyPlanVsActualTracker,
    slots: list,
    now: datetime,
    last_accumulated: datetime | None,
) -> datetime | None:
    """Accumulate each plan interval once on the physical UTC timeline.

    The marker is the end of plan coverage already booked. On an ordinary
    cadence, the current slot is captured in full at its first observation.
    If a live 15/60-minute cadence change creates an overlapping layout, only
    the still-uncovered fraction is added; an already-covered old interval is
    never counted twice.

    Completed past slots are a safety net for coordinator gaps. They are
    ignored on the first cycle so stale, zeroed startup slots cannot inflate
    the plan.

    Returns:
        The physical end of accumulated plan coverage.
    """
    coverage_end = last_accumulated
    now_key = utc_key(now)

    for slot in slots:
        slot_start = as_tz(slot.start, now.tzinfo) if hasattr(slot, "start") else None
        slot_end = as_tz(slot.end, now.tzinfo) if hasattr(slot, "end") else None
        if slot_start is None or slot_end is None:
            continue

        slot_start_key = utc_key(slot_start)
        slot_end_key = utc_key(slot_end)
        duration_seconds = (slot_end_key - slot_start_key).total_seconds()
        if duration_seconds <= 0:
            continue

        is_current = slot_contains(slot_start, slot_end, now)
        is_completed = slot_end_key <= now_key
        if not is_current and not is_completed:
            continue
        if not is_current and coverage_end is None:
            continue

        uncovered_start = slot_start_key
        if coverage_end is not None:
            uncovered_start = max(uncovered_start, utc_key(coverage_end))
        if uncovered_start >= slot_end_key:
            if is_current:
                return coverage_end
            continue

        uncovered_seconds = (slot_end_key - uncovered_start).total_seconds()
        _add_slot_to_tracker(
            tracker,
            slot,
            fraction=uncovered_seconds / duration_seconds,
        )
        coverage_end = slot_end
        if is_current:
            return coverage_end

    # Preserve the old startup safety marker when no current slot exists.
    return coverage_end or _last_completed_slot_end(slots, now)


def _add_slot_to_tracker(
    tracker: DailyPlanVsActualTracker,
    slot: object,
    fraction: float = 1.0,
) -> None:
    """Add a single slot's plan values to the tracker, scaled by *fraction*."""
    gi = (getattr(slot, "grid_import_kwh", 0.0) or 0.0) * fraction
    ge = (getattr(slot, "grid_export_kwh", 0.0) or 0.0) * fraction
    chg = (getattr(slot, "batteries_charged_kwh", 0.0) or 0.0) * fraction
    dis = (getattr(slot, "batteries_discharged_kwh", 0.0) or 0.0) * fraction
    pv = (getattr(slot, "solcast_pv_estimate_kwh", 0.0) or 0.0) * fraction
    slot_price = getattr(slot, "price", None)
    import_price = slot_price.import_price if slot_price is not None else 0.0
    export_price = slot_price.export_price if slot_price is not None else 0.0
    price_actionable = bool(getattr(slot, "price_actionable", True))
    import_price_available = price_actionable and bool(
        getattr(slot, "import_price_available", True)
    )
    export_price_available = price_actionable and bool(
        getattr(slot, "export_price_available", True)
    )
    cycle_kwh = abs(chg) + abs(dis)
    tracker.accumulate_plan(
        grid_import_kwh=gi,
        grid_export_kwh=ge,
        cycle_kwh=cycle_kwh,
        pv_kwh=pv,
        import_price=import_price,
        export_price=export_price,
        import_price_available=import_price_available,
        export_price_available=export_price_available,
    )


def _last_completed_slot_end(slots: list, now: datetime) -> datetime | None:
    """Return the end time of the most recent completed slot, or None."""
    last_end: datetime | None = None
    for slot in slots:
        slot_end = as_tz(slot.end, now.tzinfo) if hasattr(slot, "end") else None
        if slot_end is not None and utc_key(slot_end) <= utc_key(now):
            if last_end is None or utc_key(slot_end) > utc_key(last_end):
                last_end = slot_end
    return last_end
