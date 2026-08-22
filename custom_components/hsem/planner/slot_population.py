"""Slot population helpers for the HSEM planner.

Single responsibility: transform raw time-series inputs (prices, Solcast PV,
consumption averages) into fully populated :class:`PlannedSlot` objects.

All functions are pure — no I/O, no side effects beyond mutating the slot list
passed in.  No Home Assistant imports.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING, Any

from custom_components.hsem.models.hourly_consumption_average import (
    HourlyConsumptionAverage,
)
from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.planner_input import PlannerInput
from custom_components.hsem.models.price_point import PricePoint
from custom_components.hsem.models.solcast_slot import SolcastSlot
from custom_components.hsem.models.time_series import TimeSeriesIndex
from custom_components.hsem.planner import consumption_weighting as _weighting
from custom_components.hsem.planner.cost_helpers import grid_cash_flow_cost
from custom_components.hsem.utils.datetime_utils import as_tz, slot_contains, utc_key
from custom_components.hsem.utils.logger import log_planner
from custom_components.hsem.utils.prices import SlotPrice
from custom_components.hsem.utils.recommendations import Recommendations

if TYPE_CHECKING:
    from custom_components.hsem.utils.solar_corrector import SolarForecastCorrector

# Backward-compatible re-exports for existing integrations and tests.
BASELINE_7D_SHARE = _weighting.BASELINE_7D_SHARE
BASELINE_14D_SHARE = _weighting.BASELINE_14D_SHARE
CAP7_DOWN = _weighting.CAP7_DOWN
CAP7_UP = _weighting.CAP7_UP
CAP14_DOWN = _weighting.CAP14_DOWN
CAP14_UP = _weighting.CAP14_UP
CHANGE3_LIMIT_DOWN_FACTOR = _weighting.CHANGE3_LIMIT_DOWN_FACTOR
CHANGE3_LIMIT_UP_FACTOR = _weighting.CHANGE3_LIMIT_UP_FACTOR
CHANGE_LIMIT_DOWN_FACTOR = _weighting.CHANGE_LIMIT_DOWN_FACTOR
CHANGE_LIMIT_UP_FACTOR = _weighting.CHANGE_LIMIT_UP_FACTOR
IQR_OUTLIER_MULTIPLIER = _weighting.IQR_OUTLIER_MULTIPLIER
WINDOW_PEER_CLAMP_FACTOR = _weighting.WINDOW_PEER_CLAMP_FACTOR
WINDOW_PEER_CLAMP_FLOOR_KWH = _weighting.WINDOW_PEER_CLAMP_FLOOR_KWH
clamp_window_to_peer_median = _weighting.clamp_window_to_peer_median
detect_outliers_iqr = _weighting.detect_outliers_iqr
weighted_avg_consumption = _weighting.weighted_avg_consumption

# ---------------------------------------------------------------------------
# Slot generation
# ---------------------------------------------------------------------------


def build_time_series_index(inp: PlannerInput, now: datetime) -> TimeSeriesIndex:
    """Build the shared :class:`TimeSeriesIndex` for a planning run.

    The index is the single source of truth for slot boundaries.  All
    populate functions should derive their slot positions from the index
    rather than computing ``start.hour`` independently.

    Args:
        inp: Planner input containing interval and horizon settings.
        now: Timezone-aware current datetime.

    Returns:
        A fully constructed :class:`TimeSeriesIndex`.
    """
    log_planner(
        "debug",
        "[pop] build_time_series_index  now=%s  interval=%dmin  horizon=%dh",
        now.isoformat(),
        inp.interval_minutes,
        inp.interval_length_hours,
    )
    return TimeSeriesIndex.from_now(
        now,
        interval_minutes=inp.interval_minutes,
        horizon_hours=inp.interval_length_hours,
    )


def build_slots(inp: PlannerInput, now: datetime) -> list[PlannedSlot]:
    """Generate a chronologically ordered list of empty :class:`PlannedSlot` objects.

    Slot boundaries are derived from a :class:`TimeSeriesIndex` so that
    every slot is DST-safe and consistent with the shared time axis.

    Args:
        inp: Planner input containing interval settings.
        now: Timezone-aware current datetime.

    Returns:
        List of empty :class:`PlannedSlot` objects.
    """
    tsi = build_time_series_index(inp, now)
    slots = [PlannedSlot(start=meta.start, end=meta.end) for meta in tsi]
    log_planner(
        "debug",
        "[pop] build_slots  count=%d  horizon_start=%s  horizon_end=%s",
        len(slots),
        slots[0].start.isoformat() if slots else "N/A",
        slots[-1].end.isoformat() if slots else "N/A",
    )
    return slots


def index_by_hour(items: list, hour_attr: str = "hour") -> dict[int, Any]:
    """Build a dict keyed by the integer *hour* attribute of each item."""
    return {getattr(item, hour_attr): item for item in items}


# ---------------------------------------------------------------------------
# Time-series population
# ---------------------------------------------------------------------------


def _price_channel_available(value: float, advertised: bool) -> bool:
    """Return whether a source advertised a finite economic price."""
    return bool(advertised and math.isfinite(value))


def populate_prices(
    slots: list[PlannedSlot],
    price_points: list[PricePoint],
    tsi: TimeSeriesIndex | None = None,
) -> None:
    """Write import/export prices into each slot from ``price_points``.

    When a :class:`TimeSeriesIndex` is provided the prices are aligned via
    the shared slot index so that all series use the same time axis.  Missing
    Numeric values still default to 0 for downstream compatibility, but
    explicit availability flags distinguish an unpublished value from a
    genuine zero or negative price.

    Args:
        slots: Mutable list of planned slots to update.
        price_points: Per-hour price data.
        tsi: Optional shared time-series index.  When supplied, alignment is
            delegated to :meth:`TimeSeriesIndex.align_hourly_prices` so that
            missing slots are tracked centrally.
    """
    log_planner(
        "debug",
        "[pop] populate_prices  price_points=%d  tsi_provided=%s",
        len(price_points),
        tsi is not None,
    )
    if tsi is not None:
        # Sub-hourly path: when the points carry slot_in_day (issue #720),
        # key by (day_offset, slot_in_day) so quarter-hourly prices land on
        # their own slots instead of being fanned out from one hourly value.
        if any(pp.slot_in_day is not None for pp in price_points):
            imp_by_slot = {
                (pp.day_offset, pp.slot_in_day): pp
                for pp in price_points
                if pp.slot_in_day is not None
            }
            # Hourly fallback for slots the source does not cover (e.g. a
            # 60-min price source feeding 15-min slots).  Sub-hourly points
            # must never become an implicit fallback for adjacent quarters.
            imp_by_hour = {
                (pp.day_offset, pp.hour): pp
                for pp in price_points
                if pp.slot_in_day is None
            }
            for slot, meta in zip(slots, tsi.slots):
                key = (meta.key.day_offset, meta.key.slot_in_day)
                hour_key = (meta.key.day_offset, meta.hour)
                point = imp_by_slot.get(key, imp_by_hour.get(hour_key))
                imp_available = bool(
                    point
                    and _price_channel_available(
                        point.import_price, point.import_price_available
                    )
                )
                exp_available = bool(
                    point
                    and _price_channel_available(
                        point.export_price, point.export_price_available
                    )
                )
                slot.price = SlotPrice(
                    import_price=(
                        point.import_price
                        if point is not None and imp_available
                        else 0.0
                    ),
                    export_price=(
                        point.export_price
                        if point is not None and exp_available
                        else 0.0
                    ),
                )
                slot.import_price_available = imp_available
                slot.export_price_available = exp_available
                slot.import_price_source = (
                    point.import_price_source
                    if point is not None and imp_available
                    else None
                )
                slot.export_price_source = (
                    point.export_price_source
                    if point is not None and exp_available
                    else None
                )
                slot.price_actionable = imp_available and exp_available
                if not slot.price_actionable:
                    tsi.missing_slots.add(meta.key)
                    tsi.missing_price_slots.add(meta.key)
            return

        # Use (day_offset, hour) keys when any entry carries a non-zero
        # day_offset so that tomorrow's prices are not overwritten by today's.
        use_day_key = any(pp.day_offset != 0 for pp in price_points)
        imp_prices: dict[int, float] | dict[tuple[int, int], float]
        exp_prices: dict[int, float] | dict[tuple[int, int], float]
        if use_day_key:
            imp_prices = {
                (pp.day_offset, pp.hour): pp.import_price
                for pp in price_points
                if _price_channel_available(pp.import_price, pp.import_price_available)
            }
            exp_prices = {
                (pp.day_offset, pp.hour): pp.export_price
                for pp in price_points
                if _price_channel_available(pp.export_price, pp.export_price_available)
            }
            points_by_key = {(pp.day_offset, pp.hour): pp for pp in price_points}
            aligned_points = [
                points_by_key.get((meta.key.day_offset, meta.hour)) for meta in tsi
            ]
        else:
            imp_prices = {
                pp.hour: pp.import_price
                for pp in price_points
                if _price_channel_available(pp.import_price, pp.import_price_available)
            }
            exp_prices = {
                pp.hour: pp.export_price
                for pp in price_points
                if _price_channel_available(pp.export_price, pp.export_price_available)
            }
            points_by_hour = {pp.hour: pp for pp in price_points}
            aligned_points = [points_by_hour.get(meta.hour) for meta in tsi]
        aligned_imp, aligned_exp = tsi.align_hourly_prices(imp_prices, exp_prices)
        for slot, point, imp, exp in zip(
            slots, aligned_points, aligned_imp, aligned_exp
        ):
            slot.import_price_available = math.isfinite(imp)
            slot.export_price_available = math.isfinite(exp)
            slot.import_price_source = (
                point.import_price_source
                if point is not None and slot.import_price_available
                else None
            )
            slot.export_price_source = (
                point.export_price_source
                if point is not None and slot.export_price_available
                else None
            )
            slot.price_actionable = (
                slot.import_price_available and slot.export_price_available
            )
            slot.price = SlotPrice(
                import_price=imp if slot.import_price_available else 0.0,
                export_price=exp if slot.export_price_available else 0.0,
            )
        return

    price_by_hour = index_by_hour(price_points)
    for slot in slots:
        pt = price_by_hour.get(slot.start.hour)
        imp_available = bool(
            pt and _price_channel_available(pt.import_price, pt.import_price_available)
        )
        exp_available = bool(
            pt and _price_channel_available(pt.export_price, pt.export_price_available)
        )
        slot.price = SlotPrice(
            import_price=pt.import_price if pt is not None and imp_available else 0.0,
            export_price=pt.export_price if pt is not None and exp_available else 0.0,
        )
        slot.import_price_available = imp_available
        slot.export_price_available = exp_available
        slot.import_price_source = (
            pt.import_price_source if pt is not None and imp_available else None
        )
        slot.export_price_source = (
            pt.export_price_source if pt is not None and exp_available else None
        )
        slot.price_actionable = imp_available and exp_available


def populate_solcast(
    slots: list[PlannedSlot],
    solcast_slots: list[SolcastSlot],
    interval_minutes: int,
    tsi: TimeSeriesIndex | None = None,
    corrector: SolarForecastCorrector | None = None,
) -> None:
    """Write PV estimates into each slot, scaled to the slot duration.

    Solcast data is provided per *hour*; if the slot duration is shorter
    (e.g. 15 min) the estimate is divided proportionally.

    When a :class:`TimeSeriesIndex` is provided the PV series is aligned via
    the shared slot index and missing slots are tracked centrally.

    When *corrector* is provided the raw PV estimate is corrected using the
    learned per-hour accuracy factor and intra-hour residual before being
    written to the slot.  This keeps the raw Solcast data unchanged (the
    correction is only applied at consumption time).

    Args:
        slots: Mutable list of planned slots to update.
        solcast_slots: Per-hour Solcast PV estimate data.
        interval_minutes: Slot width in minutes.
        tsi: Optional shared time-series index.
        corrector: Optional :class:`~custom_components.hsem.utils.solar_corrector.SolarForecastCorrector`
            instance.  When provided the per-slot PV estimate is corrected
            before being written.  When ``None`` the raw estimate is used
            unchanged (backward compatible).
    """
    log_planner(
        "debug",
        "[pop] populate_solcast  solcast_slots=%d  interval=%dmin  tsi_provided=%s",
        len(solcast_slots),
        interval_minutes,
        tsi is not None,
    )
    if tsi is not None:
        # Slot-addressed forecasts preserve physical identity across an autumn
        # DST fold. Keep support for explicitly hour-granular callers as a
        # fallback, but never fan one slot-addressed point into its neighbours.
        if any(sc.slot_in_day is not None for sc in solcast_slots):
            pv_by_slot = {
                (sc.day_offset, sc.slot_in_day): sc
                for sc in solcast_slots
                if sc.slot_in_day is not None
            }
            point_by_hour = {
                (sc.day_offset, sc.hour): sc
                for sc in solcast_slots
                if sc.slot_in_day is None
            }
            for i, (slot, meta) in enumerate(zip(slots, tsi.slots)):
                point = pv_by_slot.get(
                    (meta.key.day_offset, meta.key.slot_in_day),
                    point_by_hour.get((meta.key.day_offset, meta.hour)),
                )
                if (
                    point is None
                    or not point.pv_estimate_available
                    or not math.isfinite(point.pv_estimate)
                ):
                    tsi.missing_slots.add(meta.key)
                    tsi.missing_pv_slots.add(meta.key)
                    raw_estimate = 0.0
                else:
                    raw_estimate = point.pv_estimate * meta.slot_fraction
                if corrector is not None and raw_estimate > 0:
                    slot.solcast_pv_estimate_kwh = round(
                        corrector.get_corrected_pv(
                            slot.start.hour,
                            raw_estimate,
                            slots_ahead=corrector.slots_ahead_for(
                                slot.start,
                                interval_minutes,
                                fallback=i,
                            ),
                        ),
                        3,
                    )
                else:
                    slot.solcast_pv_estimate_kwh = round(raw_estimate, 3)
            return

        # Use (day_offset, hour) keys when any entry carries a non-zero
        # day_offset so that tomorrow's PV forecast is not shadowed by today's.
        pv_by_hour: dict[int, float] | dict[tuple[int, int], float]
        if any(sc.day_offset != 0 for sc in solcast_slots):
            pv_by_hour = {
                (sc.day_offset, sc.hour): sc.pv_estimate
                for sc in solcast_slots
                if sc.pv_estimate_available and math.isfinite(sc.pv_estimate)
            }
        else:
            pv_by_hour = {
                sc.hour: sc.pv_estimate
                for sc in solcast_slots
                if sc.pv_estimate_available and math.isfinite(sc.pv_estimate)
            }
        aligned = tsi.align_hourly_pv(pv_by_hour)
        for i, (slot, val) in enumerate(zip(slots, aligned)):
            raw_estimate = val if math.isfinite(val) else 0.0
            if corrector is not None and raw_estimate > 0:
                slot.solcast_pv_estimate_kwh = round(
                    corrector.get_corrected_pv(
                        slot.start.hour,
                        raw_estimate,
                        slots_ahead=corrector.slots_ahead_for(
                            slot.start,
                            interval_minutes,
                            fallback=i,
                        ),
                    ),
                    3,
                )
            else:
                slot.solcast_pv_estimate_kwh = round(raw_estimate, 3)
        return

    solcast_by_hour = index_by_hour(solcast_slots)
    scale = 60.0 / interval_minutes  # e.g. 4 for 15-min slots

    for i, slot in enumerate(slots):
        sc = solcast_by_hour.get(slot.start.hour)
        raw_estimate = (
            round(sc.pv_estimate / scale, 3)
            if sc is not None
            and sc.pv_estimate_available
            and math.isfinite(sc.pv_estimate)
            else 0.0
        )
        if corrector is not None and raw_estimate > 0:
            slot.solcast_pv_estimate_kwh = round(
                corrector.get_corrected_pv(
                    slot.start.hour,
                    raw_estimate,
                    slots_ahead=corrector.slots_ahead_for(
                        slot.start,
                        interval_minutes,
                        fallback=i,
                    ),
                ),
                3,
            )
        else:
            slot.solcast_pv_estimate_kwh = raw_estimate


def populate_consumption(
    slots: list[PlannedSlot],
    averages: list[HourlyConsumptionAverage],
    w1: int,
    w3: int,
    w7: int,
    w14: int,
    interval_minutes: int,
    tsi: TimeSeriesIndex | None = None,
) -> None:
    """Compute and write spike-aware weighted consumption into each slot.

    When a :class:`TimeSeriesIndex` is provided each sub-series (1d, 3d, 7d,
    14d) is individually aligned via the shared slot axis so that missing
    hours are tracked centrally rather than silently defaulted to zero.

    Args:
        slots: Mutable list of planned slots to update.
        averages: Per-hour historical consumption averages.
        w1..w14: Configured integer weights (percent).
        interval_minutes: Slot width in minutes.
        tsi: Optional shared time-series index.
    """
    log_planner(
        "debug",
        "[pop] populate_consumption  averages=%d  weights=%d/%d/%d/%d  "
        "interval=%dmin  tsi_provided=%s",
        len(averages),
        w1,
        w3,
        w7,
        w14,
        interval_minutes,
        tsi is not None,
    )
    if tsi is not None:
        # Use (day_offset, hour) keys when any entry carries a non-zero
        # day_offset so that tomorrow's consumption forecast is not overwritten
        # by today's cyclical averages.
        avg_1d: dict[int, float] | dict[tuple[int, int], float]
        avg_3d: dict[int, float] | dict[tuple[int, int], float]
        avg_7d: dict[int, float] | dict[tuple[int, int], float]
        avg_14d: dict[int, float] | dict[tuple[int, int], float]
        if any(ca.day_offset != 0 for ca in averages):
            avg_1d = {(ca.day_offset, ca.hour): ca.avg_1d for ca in averages}
            avg_3d = {(ca.day_offset, ca.hour): ca.avg_3d for ca in averages}
            avg_7d = {(ca.day_offset, ca.hour): ca.avg_7d for ca in averages}
            avg_14d = {(ca.day_offset, ca.hour): ca.avg_14d for ca in averages}
        else:
            avg_1d = {ca.hour: ca.avg_1d for ca in averages}
            avg_3d = {ca.hour: ca.avg_3d for ca in averages}
            avg_7d = {ca.hour: ca.avg_7d for ca in averages}
            avg_14d = {ca.hour: ca.avg_14d for ca in averages}
        aligned_1d = tsi.align_hourly_load(avg_1d)
        aligned_3d = tsi.align_hourly_load(avg_3d)
        aligned_7d = tsi.align_hourly_load(avg_7d)
        aligned_14d = tsi.align_hourly_load(avg_14d)
        for i, (slot, v1, v3, v7, v14) in enumerate(
            zip(slots, aligned_1d, aligned_3d, aligned_7d, aligned_14d)
        ):
            if any(math.isnan(v) for v in (v1, v3, v7, v14)):
                continue  # missing data — leave defaults
            # Reverse the slot_fraction scaling: TSI already applied it;
            # weighted_avg_consumption expects hourly values, so undo scaling.
            sf = tsi.slots[i].slot_fraction
            if abs(sf) < 1e-9:
                continue
            h1 = v1 / sf
            h3 = v3 / sf
            h7 = v7 / sf
            h14 = v14 / sf
            hourly_avg, _ = weighted_avg_consumption(h1, h3, h7, h14, w1, w3, w7, w14)
            slot.avg_house_consumption_kwh = round(hourly_avg * sf, 3)
            slot.avg_house_consumption_1d_kwh = round(v1, 3)
            slot.avg_house_consumption_3d_kwh = round(v3, 3)
            slot.avg_house_consumption_7d_kwh = round(v7, 3)
            slot.avg_house_consumption_14d_kwh = round(v14, 3)
        return

    avg_by_hour = index_by_hour(averages)
    scale = 60.0 / interval_minutes

    for slot in slots:
        h = slot.start.hour
        ca: HourlyConsumptionAverage | None = avg_by_hour.get(h)
        if ca is None:
            continue

        hourly_avg, _ = weighted_avg_consumption(
            ca.avg_1d, ca.avg_3d, ca.avg_7d, ca.avg_14d, w1, w3, w7, w14
        )
        slot_avg = round(hourly_avg / scale, 3)
        slot.avg_house_consumption_kwh = slot_avg
        slot.avg_house_consumption_1d_kwh = round(ca.avg_1d / scale, 3)
        slot.avg_house_consumption_3d_kwh = round(ca.avg_3d / scale, 3)
        slot.avg_house_consumption_7d_kwh = round(ca.avg_7d / scale, 3)
        slot.avg_house_consumption_14d_kwh = round(ca.avg_14d / scale, 3)


def populate_net_consumption(slots: list[PlannedSlot]) -> None:
    """Compute effective net consumption per slot.

    Formula::

        effective_net_load_kwh
            = avg_house_consumption + ev_planned_load_kwh - solcast_pv_estimate

    The ``ev_planned_load_kwh`` field is already populated when EV planned
    load integration is active (and ``base_load_includes_ev`` is False).
    When EV integration is disabled ``ev_planned_load_kwh`` defaults to 0.0
    so the formula degrades to the original ``avg_consumption - pv_estimate``.

    Args:
        slots: Mutable list of planned slots to update.
    """
    log_planner(
        "debug",
        "[pop] populate_net_consumption  slots=%d",
        len(slots),
    )
    for slot in slots:
        slot.estimated_net_consumption_kwh = round(
            slot.avg_house_consumption_kwh
            + slot.ev_planned_load_kwh
            - slot.solcast_pv_estimate_kwh,
            3,
        )


def populate_estimated_cost(
    slots: list[PlannedSlot],
    *,
    export_min_price: float = 0.0,
) -> None:
    """Populate the pre-flow baseline ``estimated_cost_currency``.

    Positive net consumption is provisional grid import; negative net
    consumption is provisional grid export. Finite negative prices remain
    authoritative, so import can be a credit and export can be a cost.

    When ``export_min_price > 0``, export prices below the threshold are
    treated as 0 to match the physical inverter behaviour (the applier
    blocks all export below ``export_min_price`` via
    ``GRID_EXPORT_LIMIT_WATT``).

    Args:
        slots: Mutable list of planned slots to update.
        export_min_price: Minimum export price for grid power control.
            Export prices below this are clamped to 0.  Defaults to 0.0.
    """
    log_planner(
        "debug",
        "[pop] populate_estimated_cost  slots=%d  export_min_price=%.4f",
        len(slots),
        export_min_price,
    )
    for slot in slots:
        net = slot.estimated_net_consumption_kwh
        slot.estimated_cost_currency = round(
            grid_cash_flow_cost(
                max(net, 0.0),
                max(-net, 0.0),
                slot.price.import_price,
                slot.price.export_price,
                price_actionable=slot.price_actionable,
                export_min_price=export_min_price,
            ),
            4,
        )


def mark_time_passed(slots: list[PlannedSlot], now: datetime) -> None:
    """Mark past slots as ``TimePassed``.

    Args:
        slots: Mutable list of planned slots to update.
        now: Timezone-aware current datetime.
    """
    past_count = 0
    for slot in slots:
        if utc_key(slot.end) < utc_key(now):
            slot.recommendation = Recommendations.TimePassed.value
            past_count += 1
    log_planner(
        "debug",
        "[pop] mark_time_passed  total=%d  past=%d  now=%s",
        len(slots),
        past_count,
        now.isoformat(),
    )


def populate_battery_capacity(
    slots: list[PlannedSlot],
    now: datetime,
    current_capacity: float,
    usable_capacity: float,
) -> None:
    """Forward-simulate battery SoC through all slots.

    Args:
        slots: Mutable list of planned slots to update.
        now: Timezone-aware current datetime.
        current_capacity: Currently available battery energy in kWh.
        usable_capacity: Maximum usable battery energy in kWh.
    """
    log_planner(
        "debug",
        "[pop] populate_battery_capacity  current=%.3f kWh  usable=%.3f kWh  slots=%d",
        current_capacity,
        usable_capacity,
        len(slots),
    )
    previous_capacity = 0.0

    for slot in slots:
        slot_start = as_tz(slot.start, now.tzinfo)

        if slot_contains(slot.start, slot.end, now):
            cap = max(
                current_capacity
                - slot.estimated_net_consumption_kwh
                + slot.batteries_charged_kwh,
                0.0,
            )
        elif utc_key(slot_start) >= utc_key(now):
            cap = max(
                previous_capacity
                - slot.estimated_net_consumption_kwh
                + slot.batteries_charged_kwh,
                0.0,
            )
        else:
            cap = 0.0

        cap = min(cap, usable_capacity)
        slot.estimated_battery_capacity_kwh = round(cap, 3)
        previous_capacity = cap

    for slot in slots:
        if slot.estimated_battery_capacity_kwh > 0 and usable_capacity > 0:
            slot.estimated_battery_soc_pct = round(
                slot.estimated_battery_capacity_kwh / usable_capacity * 100, 2
            )


def usable_capacity(
    rated_kwh: float,
    soc_pct: float,
    end_of_discharge_soc_pct: float,
    max_soc_pct: float = 100.0,
) -> tuple[float, float]:
    """Return ``(usable_kwh, current_kwh)`` given rated capacity and SoC limits.

    ``usable_kwh`` is the energy available in the range
    ``[end_of_discharge_soc, max_soc]``.
    ``current_kwh`` is the energy currently stored above the discharge floor,
    clamped to ``usable_kwh``.

    Args:
        rated_kwh: Nameplate capacity in kWh.
        soc_pct: Current state of charge as a percentage (0-100).
        end_of_discharge_soc_pct: Minimum allowed SoC as a percentage (0-100).
        max_soc_pct: Maximum allowed SoC as a percentage (0-100).  Defaults to
            100 % (no restriction beyond nameplate capacity).

    Returns:
        ``(usable_kwh, current_kwh)`` tuple, both non-negative.
    """
    effective_max_soc = min(max(max_soc_pct, end_of_discharge_soc_pct), 100.0)
    usable = rated_kwh * (effective_max_soc - end_of_discharge_soc_pct) / 100
    current = rated_kwh * (soc_pct / 100) - rated_kwh * end_of_discharge_soc_pct / 100
    result = max(usable, 0.0), min(max(current, 0.0), max(usable, 0.0))
    log_planner(
        "debug",
        "[pop] usable_capacity  rated=%.2f  soc=%.1f%%  usable=%.3f  current=%.3f",
        rated_kwh,
        soc_pct,
        result[0],
        result[1],
    )
    return result
