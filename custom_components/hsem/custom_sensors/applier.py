"""Applier for HSEMWorkingModeSensor.

Single responsibility: translate the current :class:`HourlyRecommendation` and
:class:`LiveState` into hardware write calls on the Huawei Solar inverter and
battery pack.

This is the **only** module in the sensor pipeline that is allowed to call
``async_set_*`` hardware functions.  All decision logic lives in the planner
engine or the recommendation resolver; this module only executes the resulting
action plan.

Write-and-verify
----------------
Every hardware write is wrapped with :func:`~utils.inverter_verify.async_write_and_verify`:

1. Write the desired value via a Huawei Solar service call.
2. Wait :data:`~utils.inverter_verify.DEFAULT_SETTLE_SECONDS` for the inverter
   to persist the new value.
3. Read the entity state back from HA.
4. Accept if the read-back value matches within
   :data:`~utils.inverter_verify.DEFAULT_NUMERIC_TOLERANCE`.
5. Retry up to :data:`~utils.inverter_verify.DEFAULT_MAX_RETRIES` times on
   mismatch or transient read/write error.
6. After all retries, mark the result ``FAILED`` and **block further writes for
   this cycle** (the caller gates subsequent writes on the summary status).

Each top-level apply function returns a :class:`~utils.inverter_verify.CycleApplySummary`
that the :class:`~custom_sensors.applier_status_sensor.HSEMApplierStatusSensor` surfaces
to Home Assistant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.util import dt as dt_util

from custom_components.hsem.const import (
    DEFAULT_HSEM_BATTERIES_WAIT_MODE,
    DEFAULT_HSEM_EV_CHARGER_TOU_MODES,
    DEFAULT_HSEM_TOU_MODES_FORCE_CHARGE,
    GRID_EXPORT_LIMIT_WATT,
)
from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.utils.conversion import convert_to_int
from custom_components.hsem.utils.degraded_mode import hardware_writes_allowed
from custom_components.hsem.utils.ha_helpers import (
    async_set_number_value,
    async_set_select_option,
)
from custom_components.hsem.utils.huawei import (
    async_set_grid_export_power_pct,
    async_set_grid_export_power_watt,
    async_set_tou_periods,
    async_stop_forcible_discharge,
)
from custom_components.hsem.utils.inverter_verify import (
    ApplyResult,
    ApplyStatus,
    CycleApplySummary,
    async_write_and_verify,
    get_write_failure_backoff,
)
from custom_components.hsem.utils.logger import HSEM_LOGGER as _LOGGER
from custom_components.hsem.utils.misc import (
    generate_hash,
    get_max_discharge_power,
)
from custom_components.hsem.utils.recommendations import Recommendations
from custom_components.hsem.utils.units import (
    hours_ahead,
    is_material_planned_energy_kwh,
    slot_duration_hours,
)
from custom_components.hsem.utils.workingmodes import WorkingModes


@dataclass
class FullyFedDischargeCapState:
    """Per-entity state for reconciling a planned Fully Fed discharge cap.

    Huawei reports battery SoC only in coarse one-percent steps. Those samples
    are not precise enough to pace a 15-minute energy allocation, so this state
    latches the plan-derived command until the selected plan or hardware limit
    changes.
    """

    slot_start: datetime | None = None
    slot_end: datetime | None = None
    planned_discharge_kwh: float | None = None
    planned_end_capacity_kwh: float | None = None
    max_discharge_power_w: int | None = None
    commanded_cap_w: int | None = None

    def reset(self) -> None:
        """Forget the prior Fully Fed plan and command."""
        self.slot_start = None
        self.slot_end = None
        self.planned_discharge_kwh = None
        self.planned_end_capacity_kwh = None
        self.max_discharge_power_w = None
        self.commanded_cap_w = None


def _fmt_live_power_w(power_w: float | None) -> str:
    """Format a live power reading for log lines (``None`` → ``n/a``)."""
    if power_w is None:
        return "n/a"
    return f"{int(power_w)} W"


def _wait_mode_self_consumption_cap_w(
    battery_capacity_kwh: float,
    required_capacity_kwh: float,
    slot_hours: float,
    max_discharge_power_w: int,
) -> int:
    """Return the discharge power cap for wait-mode self-consumption.

    When the battery holds more energy than the planner has reserved for future
    expensive periods, the surplus may be used for normal household
    self-consumption.  The cap is the power required to consume exactly that
    surplus over the slot duration; the inverter will only draw what the house
    actually needs, so the battery will not discharge faster than the surplus
    allows.

    Args:
        battery_capacity_kwh: Current usable battery energy above the discharge
            floor (kWh).
        required_capacity_kwh: Energy the planner has reserved for future use
            (kWh).
        slot_hours: Duration of the current recommendation slot in hours.
        max_discharge_power_w: Maximum discharge power supported by the battery
            pack (W).

    Returns:
        Discharge power cap in watts.  ``0`` when there is no surplus above the
        reserve or the slot duration is invalid.
    """
    surplus_kwh = max(battery_capacity_kwh - required_capacity_kwh, 0.0)
    if surplus_kwh <= 1e-9 or slot_hours <= 1e-9:
        return 0
    cap_w = int(surplus_kwh / slot_hours * 1000.0)
    return min(cap_w, max_discharge_power_w)


def _fully_fed_discharge_cap_w(
    *,
    planned_discharge_kwh: float,
    slot_hours: float,
    remaining_slot_hours: float,
    max_discharge_power_w: int,
) -> int:
    """Return the stable plan-derived cap for a Fully Fed to Grid slot.

    Huawei's Fully Fed to Grid mode gives PV first use of the inverter's AC
    capacity and lets the battery fill only the remaining headroom. The
    maximum-discharge entity therefore acts as an upper bound on battery power.
    Dividing the planned battery energy by the full slot duration ensures that
    even a full-slot command cannot exceed the selected plan. The cap stays
    stable within the slot: Huawei's integer SoC samples represent about
    0.3 kWh on a 30 kWh battery and must not be mistaken for precise delivered
    energy when pacing a much smaller slot allocation.
    """
    if slot_hours <= 1e-9 or remaining_slot_hours <= 1e-9 or max_discharge_power_w <= 0:
        return 0

    planned_discharge_kwh = max(planned_discharge_kwh, 0.0)
    if planned_discharge_kwh <= 1e-9:
        return 0

    planned_power_w = planned_discharge_kwh / slot_hours * 1000.0
    safe_power_w = min(planned_power_w, max_discharge_power_w)
    return int(max(safe_power_w, 0.0) + 1e-6)


def _fully_fed_plan_is_unchanged(
    state: FullyFedDischargeCapState,
    rec: HourlyRecommendation,
) -> bool:
    """Return whether *state* describes the same selected export plan."""
    if (
        state.planned_discharge_kwh is None
        or state.planned_end_capacity_kwh is None
        or state.max_discharge_power_w is None
    ):
        return False
    return (
        state.slot_start == rec.start
        and state.slot_end == rec.end
        and abs(state.planned_discharge_kwh - rec.batteries_discharged_kwh) <= 1e-9
        and abs(state.planned_end_capacity_kwh - rec.estimated_battery_capacity_kwh)
        <= 1e-9
    )


def _record_fully_fed_cap_state(
    state: FullyFedDischargeCapState,
    rec: HourlyRecommendation,
    *,
    max_discharge_power_w: int,
    commanded_cap_w: int,
) -> None:
    """Record the selected plan and resulting command."""
    state.slot_start = rec.start
    state.slot_end = rec.end
    state.planned_discharge_kwh = rec.batteries_discharged_kwh
    state.planned_end_capacity_kwh = rec.estimated_battery_capacity_kwh
    state.max_discharge_power_w = max_discharge_power_w
    state.commanded_cap_w = commanded_cap_w


def _reconciled_fully_fed_discharge_cap_w(
    *,
    rec: HourlyRecommendation,
    remaining_slot_hours: float,
    battery_capacity_kwh: float,
    max_discharge_power_w: int,
    state: FullyFedDischargeCapState | None,
) -> tuple[int, str]:
    """Return a stable cap and the reconciliation trigger.

    A selected plan's original average battery power is held for the slot.
    Coarse Huawei SoC changes are deliberately ignored after the plan is
    latched. A callback at or after slot completion forces a zero-cap command.
    A newly selected plan that is already at or below its planned endpoint is
    rejected once as a stale-plan safety check.
    """
    slot_hours = slot_duration_hours(rec.start, rec.end)
    plan_unchanged = state is not None and _fully_fed_plan_is_unchanged(state, rec)
    hardware_limit_unchanged = (
        state is not None and state.max_discharge_power_w == max_discharge_power_w
    )

    if remaining_slot_hours <= 1e-9:
        if state is None:
            return 0, "slot_finished"
        _record_fully_fed_cap_state(
            state,
            rec,
            max_discharge_power_w=max_discharge_power_w,
            commanded_cap_w=0,
        )
        return 0, "slot_finished"

    if (
        plan_unchanged
        and hardware_limit_unchanged
        and state is not None
        and state.commanded_cap_w is not None
    ):
        return min(state.commanded_cap_w, max_discharge_power_w), "cached_plan"

    computed_cap_w = _fully_fed_discharge_cap_w(
        planned_discharge_kwh=rec.batteries_discharged_kwh,
        slot_hours=slot_hours,
        remaining_slot_hours=remaining_slot_hours,
        max_discharge_power_w=max_discharge_power_w,
    )
    target_reached_on_new_plan = (
        not plan_unchanged
        and battery_capacity_kwh <= max(rec.estimated_battery_capacity_kwh, 0.0) + 1e-9
    )
    if target_reached_on_new_plan:
        computed_cap_w = 0
        trigger = "new_plan_at_endpoint"
    else:
        trigger = (
            "stateless"
            if state is None
            else "hardware_limit_changed"
            if plan_unchanged
            else "plan_changed"
        )

    if state is None:
        return computed_cap_w, trigger

    _record_fully_fed_cap_state(
        state,
        rec,
        max_discharge_power_w=max_discharge_power_w,
        commanded_cap_w=computed_cap_w,
    )
    return computed_cap_w, trigger


def _is_forcible_discharge_active(state: str | None) -> bool:
    """Return whether Huawei reports a legacy forcible battery command."""
    if not state:
        return False
    return state.lower().strip() not in {
        "stopped",
        STATE_UNAVAILABLE,
        STATE_UNKNOWN,
    }


def compute_ev_discharge_cap_w(
    *,
    live_net_w: float | None,
    ev_power_available: bool,
    historical_w: int,
    sub_window_ws: list[int],
) -> int:
    """Compute the EV discharge cap in Watts (pure function, unit-testable).

    The cap limits battery discharge to the house-only load while an EV is
    charging, so 100 % of the EV load goes to the grid.

    Selection rules:

    - **Live reading available** (EV power sensor present): the historical
      baseline is the stable reference — the cap **is** the baseline.  The
      live reading (``house_w − ev_w``) must not move the cap in either
      direction: downward it ratchets toward zero when the CT clamp and the
      EV sensor disagree (beta8: 363→40 W staircase), and upward it swings
      with ordinary house noise (cooking, heat pump cycles), slowly
      draining the battery into what is supposed to be a grid-served EV
      session (v6.2.0-beta1: 652→1968→928 W swings emptied the battery
      before the 06:00 scheduled plan).  A short house spike covered from
      the grid costs a few øre; an empty battery at 06:00 costs the whole
      morning peak.
    - **No live reading** (boolean-only EV sensor): fall back to the
      smallest positive sub-window average — the 1d window recalibrates
      fastest after an upgrade or sensor configuration change.
    - **No history at all** (fresh install): trust the live reading.

    Args:
        live_net_w: ``net_consumption_w`` from the live state (EV power
            already subtracted), or ``None``.
        ev_power_available: Whether at least one EV power sensor reported
            a positive reading this cycle.
        historical_w: House baseline in Watts from the current slot's
            weighted average (0 when unavailable).
        sub_window_ws: Sub-window averages (1d/3d/7d/14d/weighted)
            converted to Watts.

    Returns:
        The discharge cap in Watts (≥ 0).
    """
    if live_net_w is not None and ev_power_available:
        if historical_w > 0:
            return historical_w
        return int(max(live_net_w, 0.0))
    best_w = 0
    for w in sub_window_ws:
        if w > 0 and (best_w == 0 or w < best_w):
            best_w = w
    return best_w


def _desired_battery_discharge_cap_w(
    *,
    cfg: SensorConfig,
    live: LiveState,
    rec: HourlyRecommendation,
    now: datetime,
    current_required_battery_kwh: float,
    max_discharge_power_w: int,
    fully_fed_discharge_state: FullyFedDischargeCapState | None,
) -> tuple[int, str]:
    """Return the desired Huawei discharge cap and a diagnostic reason."""
    recommendation = rec.recommendation

    if recommendation == Recommendations.ForceBatteriesDischarge.value:
        slot_hours = slot_duration_hours(rec.start, rec.end)
        remaining_slot_hours = hours_ahead(now, rec.end)
        cap_w, control = _reconciled_fully_fed_discharge_cap_w(
            rec=rec,
            remaining_slot_hours=remaining_slot_hours,
            battery_capacity_kwh=live.battery_current_capacity_kwh,
            max_discharge_power_w=max_discharge_power_w,
            state=fully_fed_discharge_state,
        )
        return (
            cap_w,
            "fully-fed battery export "
            f"(planned={rec.batteries_discharged_kwh:.3f} kWh, "
            f"slot={slot_hours:.3f} h, live={live.battery_current_capacity_kwh:.3f} "
            f"kWh, planned_end={rec.estimated_battery_capacity_kwh:.3f} kWh, "
            f"remaining={remaining_slot_hours:.3f} h, control={control})",
        )

    if recommendation == Recommendations.ForceExport.value:
        return 0, "PV-only fully-fed export"

    if recommendation == Recommendations.EVSmartCharging.value and (
        live.ev.force_max_discharge_power or live.ev_second.force_max_discharge_power
    ):
        return (
            int(
                max(
                    live.ev.max_discharge_power_w,
                    live.ev_second.max_discharge_power_w,
                )
            ),
            "EV V2H override",
        )

    if rec.primary_battery_hold:
        # The MILP explicitly allocated no primary-battery discharge. This
        # intent must also survive an EV display relabel: the live-EV branch
        # below otherwise derives a non-zero cap from historical house load.
        return 0, "planned battery hold"

    planned_discharge_kwh = float(getattr(rec, "batteries_discharged_kwh", 0.0) or 0.0)
    planned_grid_import_kwh = float(getattr(rec, "grid_import_kwh", 0.0) or 0.0)
    partial_self_consumption_cap_w: int | None = None
    if (
        recommendation == Recommendations.BatteriesDischargeMode.value
        and planned_discharge_kwh > 1e-9
        and planned_grid_import_kwh > 1e-9
    ):
        # MSC normally follows live house demand up to the Huawei hardware
        # maximum. When the selected plan deliberately retains more than the
        # 0.001-kWh publication residue as grid import, unrestricted MSC could
        # consume battery energy the MILP reserved for a later slot. Reuse the
        # stable full-slot cap for that intentional partial allocation. A
        # label-only runtime override, zero import, and rounding-only import
        # keep the normal hardware maximum below.
        slot_hours = slot_duration_hours(rec.start, rec.end)
        if is_material_planned_energy_kwh(planned_grid_import_kwh):
            partial_self_consumption_cap_w = _fully_fed_discharge_cap_w(
                planned_discharge_kwh=planned_discharge_kwh,
                slot_hours=slot_hours,
                remaining_slot_hours=hours_ahead(now, rec.end),
                max_discharge_power_w=max_discharge_power_w,
            )

    if live.any_ev_charging:
        slot_hours = slot_duration_hours(rec.start, rec.end)
        historical_w = (
            int(rec.avg_house_consumption_kwh / slot_hours * 1000.0)
            if slot_hours > 1e-9 and rec.avg_house_consumption_kwh > 1e-9
            else 0
        )
        live_net_w = live.net_consumption_w
        ev_power_available = (
            live.ev.power_w is not None and live.ev.power_w > 1e-9
        ) or (live.ev_second.power_w is not None and live.ev_second.power_w > 1e-9)
        sub_window_ws = [
            int(window_kwh / slot_hours * 1000.0)
            for window_kwh in (
                rec.avg_house_consumption_1d_kwh,
                rec.avg_house_consumption_3d_kwh,
                rec.avg_house_consumption_7d_kwh,
                rec.avg_house_consumption_14d_kwh,
                rec.avg_house_consumption_kwh,
            )
            if window_kwh > 1e-9 and slot_hours > 1e-9
        ]
        cap_w = compute_ev_discharge_cap_w(
            live_net_w=live_net_w,
            ev_power_available=ev_power_available,
            historical_w=historical_w,
            sub_window_ws=sub_window_ws,
        )
        if (
            cap_w > 0
            and current_required_battery_kwh > 1e-9
            and live.battery_current_capacity_kwh > 1e-9
            and live.battery_current_capacity_kwh <= current_required_battery_kwh
        ):
            cap_w = 0
        if partial_self_consumption_cap_w is not None:
            cap_w = min(cap_w, partial_self_consumption_cap_w)
        planned = (
            rec.ev_charger_calculated_power > 1e-9
            or rec.ev_second_charger_calculated_power > 1e-9
            or rec.ev_total_planned_load_kwh > 1e-9
        )
        reason = "EV active" if planned else "EV active (unplanned)"
        if partial_self_consumption_cap_w is not None:
            reason += "; planned partial self-consumption"
        return cap_w, reason

    if partial_self_consumption_cap_w is not None:
        return partial_self_consumption_cap_w, "planned partial self-consumption"

    if recommendation == Recommendations.BatteriesWaitMode.value:
        if cfg.batteries_wait_mode_behavior == "self_consumption_with_reserve":
            slot_hours = slot_duration_hours(rec.start, rec.end)
            cap_w = _wait_mode_self_consumption_cap_w(
                battery_capacity_kwh=live.battery_current_capacity_kwh,
                required_capacity_kwh=current_required_battery_kwh,
                slot_hours=slot_hours,
                max_discharge_power_w=max_discharge_power_w,
            )
            return cap_w, "wait-mode self-consumption reserve"
        return 0, "strict wait mode"

    return max_discharge_power_w, "normal hardware maximum"


def _should_force_export_for_ev(
    ev: Any,
    ev_cfg: Any,
    live: LiveState,
) -> bool:
    """Return True if the EV needs charging and export should be forced."""
    if not ev.is_connected:
        return False
    if (
        isinstance(ev.soc_pct, (int, float))
        and isinstance(ev.soc_target_pct, (int, float))
        and ev.soc_pct < ev.soc_target_pct
    ):
        return True
    if (
        isinstance(ev.soc_pct, (int, float))
        and ev_cfg.allow_charge_past_target_soc
        and ev.soc_pct < 100
        and live.huawei_batteries_soc_pct is not None
        and live.huawei_batteries_soc_pct >= 99.0
    ):
        return True
    return False


async def async_apply_inverter_power_control(
    sensor: Any,  # NOSONAR -- HA internal type; circular import risk
    cfg: SensorConfig,
    live: LiveState,
) -> CycleApplySummary:
    """Set the grid-export power percentage on all inverters.

    Decides whether to allow full export (100%) or block export (0%) based on
    the current export price, the minimum price threshold, and EV connection
    state.  Only issues a hardware write when the value differs from the current
    inverter state.

    Each write is wrapped with :func:`~utils.inverter_verify.async_write_and_verify`
    so that the inverter is polled after the write and the result is verified
    within tolerance.  If any write fails all retries, further writes within this
    cycle are blocked and the failure is recorded in the returned summary.

    This function includes its own safety gate as defense-in-depth.  Callers
    (``working_mode_sensor``) are expected to gate writes too, but this
    secondary check ensures no write ever reaches the inverter when
    ``cfg.read_only`` is ``True`` or the degraded mode is ``Error``.

    Args:
        sensor: ``HSEMWorkingModeSensor`` instance for HA access and logging.
        cfg: Current sensor configuration.
        live: Live state snapshot (prices, EV states, inverter control state).

    Returns:
        :class:`CycleApplySummary` with one :class:`ApplyResult` per inverter
        write attempted.  Returns an empty summary immediately when blocked.
    """
    summary = CycleApplySummary()

    # Defense-in-depth: block writes if read_only or degraded mode is Error.
    if cfg.read_only:
        _LOGGER.debug("async_apply_inverter_power_control: skipped — read_only=True")
        return summary
    if not hardware_writes_allowed(live.degraded_mode):
        _LOGGER.warning(
            f"async_apply_inverter_power_control: skipped — degraded mode: {live.degraded_mode.value}",
        )
        return summary

    export_price = live.export_electricity_price
    min_price = cfg.export_electricity_min_price

    if not isinstance(export_price, (int, float)):
        return summary
    if not isinstance(min_price, (int, float)):
        return summary

    export_pct = 100 if export_price >= min_price else 0

    # Allow export if EV is connected and needs charging
    if export_pct == 0 and _should_force_export_for_ev(live.ev, cfg.ev, live):
        export_pct = 100
    if export_pct == 0 and _should_force_export_for_ev(
        live.ev_second, cfg.ev_second, live
    ):
        export_pct = 100

    _LOGGER.debug(
        f"Determined export power percentage: {export_pct}% "
        f"(export={export_price}, min={min_price}, "
        f"ev1_connected={live.ev.is_connected}, ev2_connected={live.ev_second.is_connected})",
    )

    current_pct = _parse_power_control_pct(live.huawei_inverter_active_power_control)
    current_is_watt = _is_watt_limit(live.huawei_inverter_active_power_control)

    for inv_id in [
        cfg.huawei_solar_device_id_inverter_1,
        cfg.huawei_solar_device_id_inverter_2,
    ]:
        if inv_id is None:
            continue

        inv_entity = cfg.huawei_solar_inverter_active_power_control
        reader_fn = lambda inv=inv_entity: _parse_power_control_pct(
            sensor.hass.states.get(inv).state
            if inv and sensor.hass.states.get(inv) is not None
            else None
        )

        if export_pct == 0:
            # Block export → set a soft floor at GRID_EXPORT_LIMIT_WATT.
            desired = GRID_EXPORT_LIMIT_WATT
            if current_pct is not None and current_is_watt and current_pct == desired:
                continue  # already at the watt limit

            result = await async_write_and_verify(
                entity_id=inv_entity or f"inverter:{inv_id}",
                desired=desired,
                writer=lambda _id=inv_id, _w=desired: async_set_grid_export_power_watt(  # type: ignore[misc]  # mypy cannot infer lambda types with default parameters
                    sensor, _id, _w
                ),
                reader=reader_fn,
                backoff=get_write_failure_backoff(sensor),
            )
        else:
            # export_pct == 100 — Allow full export.
            if (
                current_pct is not None
                and not current_is_watt
                and current_pct == export_pct
            ):
                continue  # already at unlimited / 100 %

            result = await async_write_and_verify(
                entity_id=inv_entity or f"inverter:{inv_id}",
                desired=export_pct,
                writer=lambda _id=inv_id, _pct=export_pct: (  # type: ignore[misc]  # mypy cannot infer lambda types with default parameters
                    async_set_grid_export_power_pct(sensor, _id, _pct)
                ),
                reader=reader_fn,
                backoff=get_write_failure_backoff(sensor),
            )

        summary.results.append(result)

        if result.status not in {ApplyStatus.OK, ApplyStatus.SKIPPED}:
            return summary

    return summary


async def async_apply_battery_settings(
    sensor: Any,  # NOSONAR -- HA internal type; circular import risk
    cfg: SensorConfig,
    live: LiveState,
    rec: HourlyRecommendation,
    current_required_battery_kwh: float,
    grid_charge_power_limit_w: float | None = None,
    *,
    now: datetime | None = None,
    fully_fed_discharge_state: FullyFedDischargeCapState | None = None,
) -> CycleApplySummary:
    """Apply the working mode, TOU periods, and discharge power to the battery pack.

    Translates the ``rec.recommendation`` string into the correct Huawei Solar
    API calls.  Only issues writes when the hardware state actually needs to
    change (idempotent guard on each write).

    Each write is wrapped with :func:`~utils.inverter_verify.async_write_and_verify`
    so that the value is polled back from HA after the write and verified within
    tolerance.  If a write fails all retries it is recorded in the returned
    summary and further writes are blocked for this cycle.

    Args:
        sensor: ``HSEMWorkingModeSensor`` instance for HA access and logging.
        cfg: Current sensor configuration.
        live: Live state snapshot.
        rec: The current-interval recommendation.
        current_required_battery_kwh: Planner reserve in usable kWh. Used by
            wait-mode and EV reserve protection. Fully Fed export follows the
            selected slot's planned discharge energy; its end capacity is used
            only to reject a newly accepted stale plan.
        grid_charge_power_limit_w: Optional phase-safe Huawei battery charge
            power command in Watts. Applied before enabling forced grid
            charging. ``None`` retains the legacy hardware behavior.
        now: Optional timezone-aware application time for deterministic tests.
            Defaults to Home Assistant's current local time.
        fully_fed_discharge_state: Per-entity reconciliation state used to
            keep the selected plan-derived cap stable across coordinator wakes
            and coarse Huawei SoC changes.

    Returns:
        :class:`CycleApplySummary` with one :class:`ApplyResult` per write
        attempted.  Returns an empty summary immediately when blocked.
    """
    summary = CycleApplySummary()

    # Defense-in-depth: block writes if read_only or degraded mode is Error.
    if cfg.read_only:
        if fully_fed_discharge_state is not None:
            fully_fed_discharge_state.reset()
        _LOGGER.debug("async_apply_battery_settings: skipped — read_only=True")
        return summary
    if not hardware_writes_allowed(live.degraded_mode):
        if fully_fed_discharge_state is not None:
            fully_fed_discharge_state.reset()
        _LOGGER.warning(
            f"async_apply_battery_settings: skipped — degraded mode: {live.degraded_mode.value}",
        )
        return summary

    tou_modes = None
    working_mode = None
    recommendation = rec.recommendation
    ev_active = live.any_ev_charging
    if (
        recommendation != Recommendations.ForceBatteriesDischarge.value
        and fully_fed_discharge_state is not None
    ):
        fully_fed_discharge_state.reset()

    _rated_capacity = convert_to_int(live.huawei_batteries_rated_capacity_wh)
    max_discharge_power = get_max_discharge_power(
        _rated_capacity if _rated_capacity is not None else 0
    )

    if (
        recommendation == Recommendations.BatteriesChargeGrid.value
        and grid_charge_power_limit_w is not None
    ):
        charge_entity = cfg.huawei_solar_batteries_grid_charge_maximum_power
        if charge_entity is None:
            _LOGGER.error(
                "Phase-aware charge blocked: grid charge maximum power entity "
                "is not configured"
            )
            return summary
        desired_charge_w = max(grid_charge_power_limit_w, 0.0)
        if live.huawei_batteries_max_charge_power_w is not None:
            desired_charge_w = min(
                desired_charge_w,
                max(live.huawei_batteries_max_charge_power_w, 0.0),
            )
        desired_charge_w = float(int(desired_charge_w // 100.0) * 100)
        if live.huawei_batteries_grid_charge_max_power_w != desired_charge_w:
            _gce: str = charge_entity
            charge_result = await async_write_and_verify(
                entity_id=_gce,
                desired=desired_charge_w,
                writer=lambda: async_set_number_value(sensor, _gce, desired_charge_w),
                reader=lambda: _read_number_state(sensor, _gce),
                backoff=get_write_failure_backoff(sensor),
            )
            summary.results.append(charge_result)
            if charge_result.status not in {ApplyStatus.OK, ApplyStatus.SKIPPED}:
                return summary

    match recommendation:
        case Recommendations.ForceExport.value:
            working_mode = WorkingModes.FullyFedToGrid.value

        case Recommendations.BatteriesChargeGrid.value:
            tou_modes = DEFAULT_HSEM_TOU_MODES_FORCE_CHARGE
            working_mode = WorkingModes.TimeOfUse.value

        case Recommendations.EVSmartCharging.value:
            if (
                live.ev.force_max_discharge_power
                or live.ev_second.force_max_discharge_power
            ):
                working_mode = WorkingModes.MaximizeSelfConsumption.value
            else:
                tou_modes = DEFAULT_HSEM_EV_CHARGER_TOU_MODES
                working_mode = WorkingModes.TimeOfUse.value

        case Recommendations.BatteriesDischargeMode.value:
            working_mode = WorkingModes.MaximizeSelfConsumption.value

        case Recommendations.BatteriesChargeSolar.value:
            working_mode = WorkingModes.MaximizeSelfConsumption.value

        case Recommendations.ForceBatteriesDischarge.value:
            working_mode = WorkingModes.FullyFedToGrid.value

        case Recommendations.BatteriesWaitMode.value:
            # Strict wait keeps the battery idle in TOU mode.  Self-consumption
            # with reserve switches to MaximizeSelfConsumption so the house can
            # use surplus battery energy above the planner's required reserve.
            if (
                cfg.batteries_wait_mode_behavior == "self_consumption_with_reserve"
                and not rec.primary_battery_hold
            ):
                surplus = (
                    live.battery_current_capacity_kwh - current_required_battery_kwh
                )
                if surplus > 1e-9:
                    working_mode = WorkingModes.MaximizeSelfConsumption.value
                else:
                    tou_modes = DEFAULT_HSEM_BATTERIES_WAIT_MODE
                    working_mode = WorkingModes.TimeOfUse.value
            else:
                tou_modes = DEFAULT_HSEM_BATTERIES_WAIT_MODE
                working_mode = WorkingModes.TimeOfUse.value

        case _:
            # Unrecognised recommendation — nothing to apply.
            return summary

    wait_mode_self_consumption = (
        recommendation == Recommendations.BatteriesWaitMode.value
        and cfg.batteries_wait_mode_behavior == "self_consumption_with_reserve"
        and not rec.primary_battery_hold
        and working_mode == WorkingModes.MaximizeSelfConsumption.value
        and not ev_active
    )

    # The new Fully Fed path never starts forcible discharge. Clean up only a
    # command left active by an older release, a manual action, or a failed
    # transition before applying any new battery mode.
    stale_force_result = await _async_stop_stale_forcible_discharge(sensor, cfg, live)
    if stale_force_result is not None:
        summary.results.append(stale_force_result)
        if stale_force_result.status not in {ApplyStatus.OK, ApplyStatus.SKIPPED}:
            return summary

    desired_discharge_cap_w, discharge_cap_reason = _desired_battery_discharge_cap_w(
        cfg=cfg,
        live=live,
        rec=rec,
        now=now or dt_util.now(),
        current_required_battery_kwh=current_required_battery_kwh,
        max_discharge_power_w=max_discharge_power,
        fully_fed_discharge_state=fully_fed_discharge_state,
    )
    current_discharge_cap_w = live.huawei_batteries_max_discharge_power_w
    leaving_fully_fed = (
        live.huawei_batteries_working_mode == WorkingModes.FullyFedToGrid.value
        and working_mode != WorkingModes.FullyFedToGrid.value
    )

    # Entering/staying Fully Fed: apply the plan cap before the mode can draw
    # from the battery. Leaving Fully Fed: fail closed at 0 W; the normal cap is
    # restored only after the mode change has been verified.
    pre_mode_cap_w = 0 if leaving_fully_fed else desired_discharge_cap_w
    if (
        current_discharge_cap_w is None
        or abs(current_discharge_cap_w - pre_mode_cap_w) > 1.0
    ):
        cap_result = await _async_write_discharge_cap(sensor, cfg, pre_mode_cap_w)
        summary.results.append(cap_result)
        _LOGGER.debug(
            "Battery max discharge cap set to %d W before mode change (%s)",
            pre_mode_cap_w,
            "fail-safe Fully Fed exit" if leaving_fully_fed else discharge_cap_reason,
        )
        if cap_result.status not in {ApplyStatus.OK, ApplyStatus.SKIPPED}:
            return summary
        current_discharge_cap_w = float(pre_mode_cap_w)

    # Excess PV use in TOU — fed_to_grid for strict wait/fully-fed modes, charge
    # otherwise.  Wait-mode self-consumption keeps excess PV in the battery so
    # the surplus above the reserve can be used for household self-consumption.
    # Both export recommendations map to FullyFedToGrid. The power cap
    # distinguishes PV-only export from planned battery export.
    ev_smart_holds_primary = (
        recommendation == Recommendations.EVSmartCharging.value
        and rec.primary_battery_hold
    )
    desired_excess = (
        "charge"
        if wait_mode_self_consumption
        else (
            "fed_to_grid"
            if recommendation
            in (
                Recommendations.BatteriesWaitMode.value,
                Recommendations.ForceExport.value,
                Recommendations.ForceBatteriesDischarge.value,
            )
            or ev_smart_holds_primary
            else "charge"
        )
    )
    if live.huawei_batteries_excess_pv_use_in_tou != desired_excess:
        excess_entity = cfg.huawei_solar_batteries_excess_pv_energy_use_in_tou
        if excess_entity is None:
            _LOGGER.warning("Excess PV use entity not configured; skipping write.")
            return summary
        _ee: str = excess_entity  # narrowed for closure
        excess_result = await async_write_and_verify(
            entity_id=_ee,
            desired=desired_excess,
            writer=lambda: async_set_select_option(sensor, _ee, desired_excess),
            reader=lambda: _read_select_state(sensor, _ee),
            backoff=get_write_failure_backoff(sensor),
        )
        summary.results.append(excess_result)
        if excess_result.status not in {ApplyStatus.OK, ApplyStatus.SKIPPED}:
            return summary

    # TOU periods — no read-back verification (TOU period state is complex JSON;
    # hash comparison is sufficient; single attempt only).
    if (
        working_mode == WorkingModes.TimeOfUse.value
        and tou_modes
        and generate_hash(str(tou_modes))
        != generate_hash(str(live.tou_periods.periods))
    ):
        tou_entity = cfg.huawei_solar_batteries_tou_charging_and_discharging_periods
        battery_device_id = cfg.huawei_solar_device_id_batteries
        if tou_entity is None or battery_device_id is None:
            _LOGGER.warning(
                "TOU entity or battery device ID not configured; skipping write.",
            )
            return summary
        result = await async_write_and_verify(
            entity_id=tou_entity,
            desired=generate_hash(str(tou_modes)),
            writer=lambda: async_set_tou_periods(sensor, battery_device_id, tou_modes),
            reader=lambda: generate_hash(
                str(
                    sensor.hass.states.get(tou_entity).state
                    if tou_entity and sensor.hass.states.get(tou_entity) is not None
                    else ""
                )
            ),
            # TOU periods may take longer to propagate; skip equality check
            # since we always write when the hash differs.
            skip_if_equal=False,
            max_retries=2,
            backoff=get_write_failure_backoff(sensor),
        )
        summary.results.append(result)
        if result.status not in {ApplyStatus.OK, ApplyStatus.SKIPPED}:
            return summary

    # Working mode
    if working_mode and live.huawei_batteries_working_mode != working_mode:
        mode_entity = cfg.huawei_solar_batteries_working_mode
        if mode_entity is None:
            _LOGGER.warning("Working mode entity not configured; skipping write.")
            return summary
        _me: str = mode_entity  # narrowed for closure
        mode_result = await async_write_and_verify(
            entity_id=_me,
            desired=working_mode,
            writer=lambda: async_set_select_option(sensor, _me, working_mode),
            reader=lambda: _read_select_state(sensor, _me),
            backoff=get_write_failure_backoff(sensor),
        )
        summary.results.append(mode_result)
        if mode_result.status not in {ApplyStatus.OK, ApplyStatus.SKIPPED}:
            return summary

    if leaving_fully_fed and (
        current_discharge_cap_w is None
        or abs(current_discharge_cap_w - desired_discharge_cap_w) > 1.0
    ):
        restore_result = await _async_write_discharge_cap(
            sensor, cfg, desired_discharge_cap_w
        )
        summary.results.append(restore_result)
        _LOGGER.debug(
            "Battery max discharge cap restored to %d W after Fully Fed exit (%s)",
            desired_discharge_cap_w,
            discharge_cap_reason,
        )
        if restore_result.status not in {ApplyStatus.OK, ApplyStatus.SKIPPED}:
            return summary

    return summary


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _async_write_discharge_cap(
    sensor: Any,  # NOSONAR -- HA internal type; circular import risk
    cfg: SensorConfig,
    desired_cap_w: int,
) -> ApplyResult:
    """Write and verify the Huawei maximum battery-discharge power."""
    entity_id = cfg.huawei_solar_batteries_maximum_discharging_power
    if entity_id is None:
        message = "Maximum battery discharging power entity is not configured"
        _LOGGER.error(message)
        return ApplyResult(
            entity_id="number:maximum_discharging_power",
            desired=desired_cap_w,
            actual=None,
            status=ApplyStatus.FAILED,
            attempts=0,
            error_message=message,
        )

    return await async_write_and_verify(
        entity_id=entity_id,
        desired=desired_cap_w,
        writer=lambda: async_set_number_value(sensor, entity_id, desired_cap_w),
        reader=lambda: _read_number_state(sensor, entity_id),
        backoff=get_write_failure_backoff(sensor),
    )


async def _async_stop_stale_forcible_discharge(
    sensor: Any,  # NOSONAR -- HA internal type; circular import risk
    cfg: SensorConfig,
    live: LiveState,
) -> ApplyResult | None:
    """Stop and verify only a forcible command already active on Huawei."""
    if not _is_forcible_discharge_active(live.huawei_batteries_forcible_charge_state):
        return None

    device_id = cfg.huawei_solar_device_id_batteries
    entity_id = cfg.huawei_solar_batteries_forcible_charge
    if device_id is None:
        message = "Cannot stop stale forcible discharge: battery device ID is missing"
        _LOGGER.error(message)
        return ApplyResult(
            entity_id=entity_id or "forcible:batteries",
            desired=0.0,
            actual=1.0,
            status=ApplyStatus.FAILED,
            attempts=0,
            error_message=message,
        )

    def _read_stopped() -> float | None:
        if entity_id is None:
            return None
        state = sensor.hass.states.get(entity_id)
        if state is None or state.state in (
            STATE_UNKNOWN,
            STATE_UNAVAILABLE,
            "",
            None,
        ):
            return None
        return 1.0 if _is_forcible_discharge_active(str(state.state)) else 0.0

    result = await async_write_and_verify(
        entity_id=entity_id or f"forcible:{device_id}",
        desired=0.0,
        writer=lambda: async_stop_forcible_discharge(sensor, device_id),
        reader=_read_stopped,
        tolerance=0.0,
        max_retries=3,
        backoff=get_write_failure_backoff(sensor),
    )
    if result.attempts > 0:
        _LOGGER.warning(
            "Stopped legacy forcible battery command before applying %s: %s",
            live.huawei_batteries_working_mode or "next mode",
            result.status.value,
        )
    return result


# ---------------------------------------------------------------------------
# Read-back helpers (pure — no side effects)
# ---------------------------------------------------------------------------


def _read_number_state(
    sensor: Any, entity_id: str | None
) -> float | None:  # NOSONAR -- HA internal type; circular import risk
    """Read a number entity state from HA and return it as float, or None.

    Args:
        sensor: HSEM sensor instance with a ``hass`` attribute.
        entity_id: HA entity ID to read.

    Returns:
        Current numeric state, or ``None`` when the entity is unavailable.
    """
    if not entity_id:
        return None
    state = sensor.hass.states.get(entity_id)
    if state is None or state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE, None):
        return None
    try:
        return float(state.state)
    except ValueError, TypeError:
        return None


def _read_select_state(
    sensor: Any, entity_id: str | None
) -> str | None:  # NOSONAR -- HA internal type; circular import risk
    """Read a select entity state from HA and return it as a string, or None.

    Args:
        sensor: HSEM sensor instance with a ``hass`` attribute.
        entity_id: HA entity ID to read.

    Returns:
        Current option string, or ``None`` when unavailable.
    """
    if not entity_id:
        return None
    state = sensor.hass.states.get(entity_id)
    if state is None or state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE, None):
        return None
    return str(state.state)


def _parse_power_control_pct(state: str | None) -> int | None:
    """Parse the inverter active power control state string into a numeric value.

    Handles both percentage (``"Limited to 80%"`` → 80) and watt-based
    (``"Limited to 100W"`` → 100) formats.  ``"Unlimited"`` returns 100.

    Args:
        state: Raw string from the inverter entity (e.g. ``"Unlimited"``,
               ``"Limited to 80%"``, or ``"Limited to 100W"``).

    Returns:
        Integer value (percentage or watts), or ``None`` if the string
        cannot be parsed.
    """
    if not isinstance(state, str):
        return None
    normalized = state.strip().lower()
    # Accept any locale-independent representation of "unlimited" / no cap.
    if normalized in (
        "unlimited",
        "ikke begrænset",
        "onbeperkt",
        "unbegrenzt",
        "illimitato",
        "sin límite",
        "không giới hạn",
    ):
        return 100
    # Extract the numeric value regardless of surrounding translated text or
    # unit suffix (% or W).  This handles patterns like:
    #   "Limited to 80%"   →  80
    #   "Limited to 100W"  →  100
    #   "Begrenzt auf 80 %"  →  80
    #   "Beperkt tot 80%"  →  80
    match = re.search(r"(-?\d+(?:\.\d+)?)", normalized)
    if match:
        try:
            return int(round(float(match.group(1))))
        except ValueError, TypeError:
            pass
    return None


def _is_watt_limit(state: str | None) -> bool:
    """Check if the power control state represents a watt-based limit.

    Args:
        state: Raw string from the inverter entity (e.g. ``"Limited to 100W"``
               or ``"Limited to 80%"``).

    Returns:
        ``True`` if the state is a watt-based limit, ``False`` otherwise
        (percentage-based or unlimited).
    """
    if not isinstance(state, str):
        return False
    normalized = state.strip().lower()
    # Unlimited / percentage-based states never contain a watt indicator
    if normalized in (
        "unlimited",
        "ikke begrænset",
        "onbeperkt",
        "unbegrenzt",
        "illimitato",
        "sin límite",
        "không giới hạn",
    ):
        return False
    # Look for a number immediately followed (with optional whitespace) by "w"
    # Single quantifier avoids polynomial backtracking from stacked greedy quantifiers
    return bool(re.search(r"\d[\d\s]*w", normalized))
