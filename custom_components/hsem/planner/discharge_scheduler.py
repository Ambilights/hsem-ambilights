"""Discharge scheduling for the HSEM planner.

Single responsibility: decide *when* to discharge the battery
based on discharge-window schedules, price signals, and seasonal strategy.

All functions are pure — no I/O, no Home Assistant imports.  They mutate the
:class:`PlannedSlot` list passed in and return nothing (or a scalar result).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta

from custom_components.hsem.const import (
    SEASONAL_FILL_MODE_FORECAST,
    SEASONAL_FILL_MODE_MONTHS,
    SEASONAL_FILL_MODES,
)
from custom_components.hsem.models.battery_schedule_input import BatteryScheduleInput
from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.utils.datetime_utils import as_tz, utc_key
from custom_components.hsem.utils.logger import log_planner
from custom_components.hsem.utils.misc import clamp_efficiency
from custom_components.hsem.utils.recommendations import (
    DISCHARGE_RECS as _DISCHARGE_RECS,
    Recommendations,
)
from custom_components.hsem.utils.time_windows import next_window_start_dt

# ---------------------------------------------------------------------------
# Discharge schedule detection
# ---------------------------------------------------------------------------


def apply_discharge_schedules(
    slots: list[PlannedSlot],
    battery_schedules: list[BatteryScheduleInput],
    now: datetime,
) -> None:
    """Mark slots inside each enabled discharge window as ``BatteriesDischargeMode``.

    Also populates ``_needed_capacity`` and ``_avg_import_price`` as dynamic
    attributes on each :class:`BatteryScheduleInput` so the charge planner can
    read them without an extra pass.

    Args:
        slots: Mutable list of planned slots.
        battery_schedules: Schedule configurations to evaluate.
        now: Timezone-aware current datetime.
    """
    log_planner(
        "debug",
        "[disch] apply_discharge_schedules  schedules=%d  now=%s",
        len(battery_schedules),
        now.isoformat(),
    )
    for sched in battery_schedules:
        if not sched.enabled:
            continue

        # Determine the last slot end in the planning horizon so we know how
        # many days to cover.  We apply the discharge window once per calendar
        # day that falls within [now, horizon_end].
        future_slots = [s for s in slots if utc_key(s.end) > utc_key(now)]
        if not future_slots:
            continue
        horizon_end = as_tz(future_slots[-1].end, now.tzinfo)

        # Collect all occurrences of this schedule window within the horizon.
        # Start from the first upcoming occurrence and advance one day at a time.
        # Each occurrence is stored so apply_charge_schedules can schedule
        # pre-charge independently per window occurrence.
        window_start_abs = next_window_start_dt(now, sched.start)
        occurrences: list[tuple[datetime, datetime, float, float]] = []
        sched_total_net = 0.0

        while utc_key(window_start_abs) < utc_key(horizon_end):
            if sched.end > sched.start:
                window_end_abs = datetime.combine(
                    window_start_abs.date(), sched.end
                ).replace(tzinfo=now.tzinfo)
            else:
                # Cross-midnight discharge window
                window_end_abs = datetime.combine(
                    (window_start_abs + timedelta(days=1)).date(), sched.end
                ).replace(tzinfo=now.tzinfo)

            for slot in slots:
                slot_start = as_tz(slot.start, now.tzinfo)
                slot_end = as_tz(slot.end, now.tzinfo)
                if utc_key(slot_end) <= utc_key(now):
                    continue
                if utc_key(slot_start) >= utc_key(window_start_abs) and utc_key(
                    slot_end
                ) <= utc_key(window_end_abs):
                    slot.recommendation = Recommendations.BatteriesDischargeMode.value

            # Capture per-occurrence capacity and avg price.
            #
            # Battery-relevant net consumption excludes EV planned load:
            #   battery_net = avg_house_consumption - pv
            #
            # When base_load_includes_ev=False, estimated_net_consumption includes
            # ev_planned_load_kwh.  The EV draws directly from grid/PV, not from
            # the home battery, so including it in occ_needed would over-inflate
            # the pre-charge target and cause the price-spread guard in
            # _apply_grid_charge to reject otherwise profitable charge slots.
            #
            # ev_accounted_load_kwh is already captured in avg_house_consumption
            # (base_load_includes_ev=True), so no correction is needed for that
            # case — the battery must cover it.
            occ_net = 0.0
            occ_prices: list[float] = []
            for s in slots:
                s_start = as_tz(s.start, now.tzinfo)
                s_end = as_tz(s.end, now.tzinfo)
                if (
                    s.recommendation == Recommendations.BatteriesDischargeMode.value
                    and s.price_actionable
                    and utc_key(s_start) >= utc_key(window_start_abs)
                    and utc_key(s_end) <= utc_key(window_end_abs)
                ):
                    # Subtract extra EV load (injected, base_load_includes_ev=False)
                    # so the battery only targets house coverage.
                    battery_net = (
                        s.estimated_net_consumption_kwh - s.ev_planned_load_kwh
                    )
                    occ_net += battery_net
                    occ_prices.append(s.price.import_price)

            occ_needed = max(occ_net, 0.0)
            occ_avg_price = (
                round(sum(occ_prices) / len(occ_prices), 3) if occ_prices else 0.0
            )
            # Store: (window_start, window_end, needed_kwh, avg_discharge_price)
            occurrences.append(
                (window_start_abs, window_end_abs, occ_needed, occ_avg_price)
            )
            sched_total_net += occ_net

            # Advance to the same window start on the following calendar day
            window_start_abs += timedelta(days=1)

        # _occurrences: per-day data consumed by apply_charge_schedules
        sched._occurrences = occurrences
        # _needed_capacity: aggregate across all occurrences (used by coordinator)
        sched._needed_capacity = max(sched_total_net, 0.0)
        # _avg_import_price: average across all occurrences
        all_occ_prices = [avg for _, _, _, avg in occurrences if avg > 0]
        sched._avg_import_price = (
            round(sum(all_occ_prices) / len(all_occ_prices), 3)
            if all_occ_prices
            else 0.0
        )


# ---------------------------------------------------------------------------
# Excess export
# ---------------------------------------------------------------------------


def apply_force_export_policy(
    slots: list[PlannedSlot],
    export_min_price: float,
) -> None:
    """Preclassify PV-only export slots before reserve and fill decisions."""
    for slot in slots:
        if (
            slot.price_actionable
            and slot.price.export_price > slot.price.import_price
            and slot.price.export_price >= export_min_price
            and slot.recommendation is None
        ):
            slot.recommendation = Recommendations.ForceExport.value


def calculate_required_battery_until_solar(
    slots: list[PlannedSlot],
    now: datetime,
    usable_capacity: float,
    discharge_buffer_pct: float,
    discharge_efficiency_pct: float = 100.0,
    max_discharge_per_slot: float | None = None,
) -> float:
    """Estimate battery capacity needed until the first solar surplus slot.

    Slots are sorted by start time before scanning, so calling code does not
    need to guarantee chronological order.

    Scans forward from *now* and accumulates positive net-consumption until
    a slot with negative net-consumption (solar surplus) is found.

    Args:
        slots: List of planned slots.
        now: Timezone-aware current datetime.
        usable_capacity: Maximum usable battery energy in kWh.
        discharge_buffer_pct: Safety buffer as a percentage of usable capacity.
        discharge_efficiency_pct: Battery-to-AC efficiency percentage.
        max_discharge_per_slot: Battery-side energy limit per slot. ``None``
            means the battery can serve the full non-EV load.

    Returns:
        Required battery capacity in kWh (including safety buffer).
    """
    discharge_efficiency = clamp_efficiency(discharge_efficiency_pct)
    required = 0.0
    for slot in sorted(slots, key=lambda s: utc_key(s.start)):
        if utc_key(slot.start) < utc_key(now):
            continue
        if not slot.price_actionable:
            # Unknown-price slots are an authoritative storage Hold. Their PV
            # cannot promise a refill that changes earlier priced decisions.
            break
        if slot.estimated_net_consumption_kwh < 0:
            if slot.primary_battery_hold or slot.recommendation in (
                Recommendations.ForceBatteriesDischarge.value,
                Recommendations.ForceExport.value,
            ):
                continue
            break
        if slot.estimated_net_consumption_kwh > 0:
            if slot.ev_total_planned_load_kwh > 1e-9:
                continue
            battery_load = slot.estimated_net_consumption_kwh / discharge_efficiency
            if max_discharge_per_slot is not None:
                battery_load = min(
                    battery_load,
                    max(max_discharge_per_slot, 0.0),
                )
            required += battery_load

    # Cap at usable_capacity — the battery can't hold more than this
    # anyway.  Without this cap, a multi-hour gap until the next solar
    # surplus (e.g. overnight) would make required exceed usable_capacity
    # and block excess export entirely.
    buffer_kwh = usable_capacity * (discharge_buffer_pct / 100)
    if required >= usable_capacity - 1e-9:
        result = usable_capacity
    else:
        result = round(min(required + buffer_kwh, usable_capacity), 3)
    log_planner(
        "debug",
        "[disch] calculate_required_battery_until_solar  required=%.3f  buffer=%.3f  result=%.3f",
        required,
        buffer_kwh,
        result,
    )
    return result


def best_alternative_import_price(
    slots: list[PlannedSlot],
    after: datetime,
    refill_kwh: float,
) -> float:
    """Return what a stored kWh is worth if kept instead of exported now.

    Exporting battery energy and buying it back later is a loss whenever the
    later import price exceeds the export price.  Both sides of that choice
    move the same kWh out of the battery through the same inverter, so the
    efficiency and cycle-wear terms cancel and the comparison reduces to
    ``export_price`` against the highest import price the battery could still
    displace.

    The scan stops once forecast solar surplus would put ``refill_kwh`` back
    into the battery: beyond that point the energy held would merely displace
    free PV rather than paid import, so keeping it buys nothing.  This is why
    the scan cannot simply stop at the first surplus slot — a single sunny
    quarter-hour does not refill a battery, and stopping there undervalues
    every evening hour that follows.

    Args:
        slots: All planned slots; scanned in chronological order.
        after: Only slots starting strictly after this instant are considered.
        refill_kwh: Energy that solar must return before held energy stops
            being the marginal source.  Non-positive disables the cutoff.

    Returns:
        The highest displaceable import price, or ``0.0`` when no later slot
        needs the battery before solar refills it.
    """
    best = 0.0
    surplus_kwh = 0.0
    for slot in sorted(slots, key=lambda s: utc_key(s.start)):
        if utc_key(slot.start) <= utc_key(after):
            continue
        if not slot.price_actionable:
            # An unpublished price is not evidence that the energy is
            # worthless; it simply cannot be compared.  Skip it.
            continue
        net = slot.estimated_net_consumption_kwh
        if net is None:
            continue
        if net < 0:
            surplus_kwh += -net
            if refill_kwh > 1e-9 and surplus_kwh >= refill_kwh:
                break
            continue
        price = slot.price.import_price
        if price is not None and float(price) > best:
            best = float(price)
    return best


def apply_excess_export(
    slots: list[PlannedSlot],
    now: datetime,
    current_capacity: float,
    required_capacity: float,
    export_price_threshold: float,
    warnings: list[str],
    *,
    export_min_price: float = 0.0,
    recommended_threshold: float = 0.0,
    battery_export_min_price: float = 0.0,
) -> None:
    """Mark high-export-price future slots for forced battery discharge.

    Only triggered when the battery holds more energy than needed until
    the next solar surplus.  Grid-charged batteries require a minimum price
    difference; solar-charged batteries export opportunistically but still
    require ``export_price >= max(export_min_price, recommended_threshold,
    battery_export_min_price)`` — the highest of the user-configured minimum
    (inverter physical floor), the cycle-wear cost, and the per-slot hard
    floor for intentional battery-to-grid export (issue #752).

    Args:
        slots: Mutable list of planned slots.
        now: Timezone-aware current datetime.
        current_capacity: Current available battery energy in kWh.
        required_capacity: Energy needed until next solar surplus (kWh).
        export_price_threshold: Minimum export-minus-import price delta for
            grid-charged batteries.
        warnings: Mutable list to append diagnostic messages to.
        export_min_price: Minimum export price (local currency/kWh) below
            which forced discharge is never triggered.  Sourced from
            ``hsem_export_electricity_min_price``.
        recommended_threshold: Battery cycle-wear cost per kWh (depreciation)
            from :func:`~custom_components.hsem.utils.misc.calculate_recommended_threshold`.
            Used as a floor — exporting below this price costs more in
            battery wear than it earns in revenue.
        battery_export_min_price: Per-slot hard floor for intentional
            battery-to-grid export (issue #752).  ``0.0`` disables the
            guard — fully backward compatible.
    """
    # battery_discharge_budget_kwh is the kWh the battery can export beyond what is
    # already needed to cover future house load.  Solar surplus in a slot does NOT
    # add to this budget: solar is a separate energy flow and is already accounted for
    # in estimated_net_consumption.  Only positive net consumption (house load > solar)
    # draws down the battery, so we drain the budget by max(net, 0) per slot.
    #
    battery_discharge_budget_kwh = float("inf")  # let concentrate + SoC handle limits
    effective_min_export_price = max(
        export_min_price, recommended_threshold, battery_export_min_price
    )
    log_planner(
        "debug",
        "[disch] apply_excess_export  budget=%.3f  current=%.3f  required=%.3f  "
        "price_threshold=%.4f  recommended_threshold=%.4f  "
        "battery_export_min_price=%.4f  effective_min_export_price=%.4f",
        battery_discharge_budget_kwh,
        current_capacity,
        required_capacity,
        export_price_threshold,
        recommended_threshold,
        battery_export_min_price,
        effective_min_export_price,
    )
    if battery_discharge_budget_kwh < 0:
        log_planner(
            "debug",
            "[disch] apply_excess_export  skipped — budget < 0",
        )
        return

    # Force discharge profitability per slot:
    #   profit = export × battery - house × import - charge - cycle
    # where battery ≈ min(max_discharge, net_demand) and house ≈ net_demand.
    # Conservative: require export ≥ import + cycle_wear when house > 0,
    # or export ≥ cycle_wear when PV covers house (net < 0).
    # Energy the battery holds beyond what it needs before solar returns.  Used
    # as the refill yardstick: once forecast PV would put this much back, held
    # energy stops being the marginal source and selling it costs nothing.
    exportable_kwh = max(current_capacity - required_capacity, 0.0)

    candidates = sorted(
        (
            s
            for s in slots
            if utc_key(s.start) >= utc_key(now)
            and s.price_actionable
            and s.recommendation
            in (
                None,
                Recommendations.BatteriesDischargeMode.value,
            )
            and (
                # PV already covers the house this slot, so nothing discharged
                # here offsets an import *now* — it is sold.  That only pays if
                # the price beats what the same stored kWh would save later:
                # exporting at 0.66 to buy back at 0.96 four hours on is a loss
                # no static floor can detect.  ``refill_kwh`` is the energy
                # under consideration, so the comparison ends once solar would
                # have replaced it anyway (issue #752 follow-up).
                (
                    s.estimated_net_consumption_kwh is not None
                    and s.estimated_net_consumption_kwh < 0
                    and s.price.export_price
                    >= best_alternative_import_price(slots, s.start, exportable_kwh)
                )
                or (
                    # No PV surplus: must cover house import cost
                    s.estimated_net_consumption_kwh is not None
                    and s.price.export_price
                    >= s.price.import_price + recommended_threshold
                )
            )
            and s.price.export_price >= effective_min_export_price
        ),
        key=lambda x: x.price.export_price,
        reverse=True,
    )

    for s in candidates:
        if battery_discharge_budget_kwh < 0:
            break
        s.recommendation = Recommendations.ForceBatteriesDischarge.value
        warnings.append(
            f"ForceBatteriesDischarge at {s.start.isoformat()}: export={s.price.export_price}"
        )
        battery_discharge_budget_kwh -= max(s.estimated_net_consumption_kwh, 0.0)


# ---------------------------------------------------------------------------
# Discharge concentration — avoid wasting battery on cheap slots
# ---------------------------------------------------------------------------


def concentrate_discharge_on_expensive_slots(
    slots: list[PlannedSlot],
    now: datetime,
    current_kwh: float,
    usable_kwh: float,
    max_discharge_per_slot: float | None,
    discharge_efficiency_pct: float = 100.0,
) -> None:
    """Clear cheap discharge slots the battery cannot fully serve, per calendar day.

    ``apply_discharge_schedules`` and ``apply_optimization_strategy`` mark
    *every* slot in a discharge window as ``BatteriesDischargeMode``, but
    the battery can only cover a fraction of them.  Without concentration
    the SoC simulation greedily discharges in the *first* (cheapest) slots
    and runs out before the most expensive ones.

    This function ranks all ``BatteriesDischargeMode`` slots by import price
    (descending) and clears the recommendation on the cheapest slots that
    exceed the battery's discharge capacity, turning them into grid-import
    slots (marked ``BatteriesWaitMode``).  The most expensive slots keep
    their discharge recommendation.

    **Per-day budget pools:** slots are grouped by calendar day and each day
    receives its own independent ``usable_kwh`` budget.  This avoids overly
    conservative behaviour where slots on day N+1 compete with slots on day N
    for the same capacity pool — the battery is recharged by solar between
    discharge windows on different days.

    The estimate within each day is conservative: it assumes the battery
    starts at full capacity and there is no incoming charge between discharge
    slots on the same day.

    Args:
        slots: Mutable list of planned slots.
        now: Timezone-aware current datetime.
        current_kwh: Energy currently stored above the discharge floor (kWh).
        usable_kwh: Maximum usable energy above the discharge floor (kWh).
            Applied as a **per-day** budget for each calendar day that has
            discharge slots.
        max_discharge_per_slot: Maximum energy dischargeable per slot (kWh).
            ``None`` means unlimited (inverter default).
        discharge_efficiency_pct: Discharge-side efficiency (0-100 %).
    """
    log_planner(
        "debug",
        "[disch] concentrate_discharge_on_expensive_slots  usable=%.3f  current=%.3f  "
        "max_discharge=%s",
        usable_kwh,
        current_kwh,
        f"{max_discharge_per_slot:.3f}" if max_discharge_per_slot is not None else "∞",
    )
    discharge_eff = clamp_efficiency(discharge_efficiency_pct)

    # Collect all future discharge slots (both BatteriesDischargeMode and
    # ForceBatteriesDischarge — issue #425 Bug I fix).
    discharge_slots = [
        s
        for s in slots
        if s.recommendation in _DISCHARGE_RECS
        and s.price_actionable
        and utc_key(s.end) > utc_key(now)
    ]
    if not discharge_slots:
        return

    # Group slots by calendar day — each day gets its own independent budget
    # because the battery is recharged by solar between discharge windows on
    # different days.
    by_day: dict[date, list[PlannedSlot]] = defaultdict(list)
    for s in discharge_slots:
        by_day[as_tz(s.start, now.tzinfo).date()].append(s)

    # Log the per-day grouping so the operator can verify that each day
    # gets its own independent battery budget.
    day_summaries: list[str] = []
    for day in sorted(by_day):
        day_summaries.append(f"{day}({len(by_day[day])} slots)")
    log_planner(
        "debug",
        "[disch] concentrate: %d discharge slots grouped into %d day(s): %s  "
        "usable_per_day=%.3f",
        len(discharge_slots),
        len(by_day),
        ", ".join(day_summaries),
        usable_kwh,
    )

    # Sort ALL discharge slots by import price across all days (descending).
    # Each day gets its own independent usable_kwh budget because the
    # battery is recharged by solar between days — day N's discharge
    # does not reduce day N+1's capacity.
    discharge_slots.sort(key=lambda s: s.price.import_price, reverse=True)

    total_kept = 0
    total_cleared = 0
    keep_set: set[int] = set()
    per_day_used: dict[date, float] = defaultdict(float)

    for s in discharge_slots:
        slot_day = as_tz(s.start, now.tzinfo).date()
        slot_demand = max(s.estimated_net_consumption_kwh, 0.0)
        battery_needed = slot_demand / discharge_eff if discharge_eff > 1e-9 else 0.0
        if max_discharge_per_slot is not None:
            battery_needed = min(battery_needed, max_discharge_per_slot)

        day_remaining = usable_kwh - per_day_used[slot_day]
        if battery_needed <= day_remaining:
            per_day_used[slot_day] += battery_needed
            keep_set.add(id(s))
        else:
            continue

    total_kept = 0
    total_cleared = 0

    for s in discharge_slots:
        if id(s) in keep_set:
            total_kept += 1
        else:
            total_cleared += 1
            log_planner(
                "debug",
                "concentrate: clearing discharge at %s→%s  price=%.4f  day=%s",
                s.start.strftime("%d %H:%M"),
                s.end.strftime("%H:%M"),
                s.price.import_price,
                as_tz(s.start, now.tzinfo).strftime("%Y-%m-%d"),
            )
            s.recommendation = Recommendations.BatteriesWaitMode.value
            s.batteries_charged_kwh = 0.0

    log_planner(
        "debug",
        "[disch] concentrate_discharge_on_expensive_slots DONE  "
        "days=%d  kept=%d  cleared=%d  usable_per_day=%.3f  total_budget=%.3f",
        len(by_day),
        total_kept,
        total_cleared,
        usable_kwh,
        usable_kwh * len(by_day),
    )


# ---------------------------------------------------------------------------
# Seasonal optimization
# ---------------------------------------------------------------------------


def _storable_solar_refill_kwh(
    slot: PlannedSlot,
    charge_efficiency: float,
    max_charge_per_slot: float | None,
) -> float:
    """Return battery-side solar refill allowed by the slot power limit."""
    stored = max(-slot.estimated_net_consumption_kwh, 0.0) * charge_efficiency
    if max_charge_per_slot is not None:
        stored = min(stored, max(max_charge_per_slot, 0.0))
    return stored


def _battery_load_kwh(
    slot: PlannedSlot,
    discharge_efficiency: float,
    max_discharge_per_slot: float | None,
) -> float:
    """Return battery-side energy needed to serve this non-EV load slot."""
    if slot.ev_total_planned_load_kwh > 1e-9:
        return 0.0
    needed = max(slot.estimated_net_consumption_kwh, 0.0) / discharge_efficiency
    if max_discharge_per_slot is not None:
        needed = min(needed, max(max_discharge_per_slot, 0.0))
    return needed


def _forced_discharge_target_kwh(
    slot: PlannedSlot,
    *,
    available_capacity_kwh: float,
    required_capacity_kwh: float,
    max_discharge_per_slot: float | None,
) -> float:
    """Return the battery-side forced draw executable in this slot."""
    if slot.primary_battery_hold or slot.ev_total_planned_load_kwh > 1e-9:
        return 0.0

    planned = max(slot.batteries_discharged_kwh, 0.0)
    if planned <= 1e-9:
        planned = max(available_capacity_kwh - required_capacity_kwh, 0.0)
    if max_discharge_per_slot is not None:
        planned = min(planned, max(max_discharge_per_slot, 0.0))
    return min(planned, max(available_capacity_kwh, 0.0))


def _future_forced_discharge_commitments(
    slots: list[PlannedSlot],
    *,
    usable_capacity: float,
    charge_efficiency: float,
    max_charge_per_slot: float | None,
    max_discharge_per_slot: float | None,
) -> list[float]:
    """Return future forced draws net of executable intervening refill."""
    commitments_after = [0.0] * len(slots)
    running_commitment = 0.0
    for index in range(len(slots) - 1, -1, -1):
        commitments_after[index] = running_commitment
        slot = slots[index]
        if not slot.price_actionable:
            running_commitment = 0.0
            continue
        if (
            slot.recommendation == Recommendations.ForceBatteriesDischarge.value
            and not slot.primary_battery_hold
            and slot.ev_total_planned_load_kwh <= 1e-9
        ):
            target = max(slot.batteries_discharged_kwh, 0.0)
            if target <= 1e-9:
                target = usable_capacity
            if max_discharge_per_slot is not None:
                target = min(target, max(max_discharge_per_slot, 0.0))
            running_commitment += target
            continue
        if (
            slot.primary_battery_hold
            or slot.recommendation == Recommendations.ForceExport.value
        ):
            continue
        refill_kwh = _storable_solar_refill_kwh(
            slot,
            charge_efficiency,
            max_charge_per_slot,
        )
        if slot.recommendation == Recommendations.BatteriesChargeGrid.value:
            scheduled_charge_kwh = max(slot.batteries_charged_kwh, 0.0)
            if max_charge_per_slot is not None:
                scheduled_charge_kwh = min(
                    scheduled_charge_kwh,
                    max(max_charge_per_slot, 0.0),
                )
            refill_kwh = max(refill_kwh, scheduled_charge_kwh)
        running_commitment = max(running_commitment - refill_kwh, 0.0)
    return commitments_after


def _forward_refill_forecast(
    slots: list[PlannedSlot],
    *,
    charge_efficiency_pct: float = 100.0,
    max_charge_per_slot: float | None = None,
) -> tuple[list[float], list[bool]]:
    """Return forward PV-surplus energy and Solcast usability per slot.

    Each value describes slots strictly *after* the corresponding slot and
    stops at any primary-hold, forced-export, or price-authority boundary.
    A reverse pass resets both running values at each boundary, keeping the
    calculation O(n) for the full horizon.

    Args:
        slots: Chronologically ordered planner slots.

    Returns:
        Two lists aligned with ``slots``: forecast refill energy in kWh and a
        flag indicating whether any positive Solcast value exists in that
        forward window.
    """
    charge_efficiency = clamp_efficiency(charge_efficiency_pct)
    refill_after_kwh = [0.0] * len(slots)
    forecast_usable_after = [False] * len(slots)
    running_refill_kwh = 0.0
    running_forecast_usable = False
    refill_boundaries = {
        Recommendations.ForceBatteriesDischarge.value,
        Recommendations.ForceExport.value,
    }

    for index in range(len(slots) - 1, -1, -1):
        refill_after_kwh[index] = running_refill_kwh
        forecast_usable_after[index] = running_forecast_usable

        slot = slots[index]
        if not slot.price_actionable:
            # The unknown-price tail is a primary-battery Hold.  Do not expose
            # its PV or net surplus as refill headroom for an earlier priced
            # slot; that promised charge is forbidden later in finalization.
            running_refill_kwh = 0.0
            running_forecast_usable = False
            continue
        if slot.primary_battery_hold or slot.recommendation in refill_boundaries:
            running_refill_kwh = 0.0
            running_forecast_usable = False
            continue

        running_refill_kwh += _storable_solar_refill_kwh(
            slot,
            charge_efficiency,
            max_charge_per_slot,
        )
        running_forecast_usable = (
            running_forecast_usable or slot.solcast_pv_estimate_kwh > 0.0
        )

    return refill_after_kwh, forecast_usable_after


def _month_based_fill_recommendation(
    rec: PlannedSlot,
    current_month: int,
    months_winter: list[int],
) -> str:
    """Return the legacy calendar-based recommendation for one idle slot."""
    if current_month in months_winter:
        return Recommendations.BatteriesWaitMode.value
    if rec.estimated_net_consumption_kwh < 0.0:
        return Recommendations.BatteriesChargeSolar.value
    return Recommendations.BatteriesDischargeMode.value


def apply_optimization_strategy(
    slots: list[PlannedSlot],
    now: datetime,
    current_capacity: float,
    usable_capacity: float,
    required_capacity: float,
    months_winter: list[int],
    export_min_price: float = 0.0,
    seasonal_fill_mode: str = SEASONAL_FILL_MODE_FORECAST,
    charge_efficiency_pct: float = 100.0,
    discharge_efficiency_pct: float = 100.0,
    max_charge_per_slot: float | None = None,
    max_discharge_per_slot: float | None = None,
) -> None:
    """Apply final optimization logic to remaining unassigned slots.

    Decision priority per unassigned slot:

    1. Export price > import price **and** export price ≥ ``export_min_price``
       → ``ForceExport``
    2. Solar surplus → ``BatteriesChargeSolar`` (until battery full)
    3. Future forced export pending and battery above required → ``BatteriesWaitMode``
    4. In forecast mode, discharge when forward PV refill headroom is positive;
       otherwise wait.  A current genuine PV surplus still charges from solar.
    5. In months mode, or when forward Solcast is unusable, apply the legacy
       winter-wait / summer-self-consumption rule.

    Args:
        slots: Mutable list of planned slots.
        now: Timezone-aware current datetime.
        current_capacity: Current available battery energy in kWh.
        usable_capacity: Maximum usable battery energy in kWh.
        required_capacity: Energy required until next solar surplus (kWh).
        months_winter: List of month integers (1-12) treated as winter.
        export_min_price: Minimum export price required to trigger
            ``ForceExport``.  Slots where export price is below this
            threshold are not marked for export even if export > import.
            Defaults to ``0.0`` (any positive export price qualifies).
        seasonal_fill_mode: ``forecast`` (default) uses forward PV refill
            headroom.  ``months`` preserves the legacy calendar rule.
    """
    effective_fill_mode = seasonal_fill_mode
    if effective_fill_mode not in SEASONAL_FILL_MODES:
        log_planner(
            "warning",
            "[disch] invalid seasonal_fill_mode=%s; falling back to %s",
            effective_fill_mode,
            SEASONAL_FILL_MODE_MONTHS,
        )
        effective_fill_mode = SEASONAL_FILL_MODE_MONTHS

    log_planner(
        "debug",
        "[disch] apply_optimization_strategy  current=%.3f  usable=%.3f  "
        "required=%.3f  export_min_price=%.4f  seasonal_fill_mode=%s",
        current_capacity,
        usable_capacity,
        required_capacity,
        export_min_price,
        effective_fill_mode,
    )
    current_month = now.month
    charge_efficiency = clamp_efficiency(charge_efficiency_pct)
    discharge_efficiency = clamp_efficiency(discharge_efficiency_pct)

    apply_force_export_policy(slots, export_min_price)

    # Solar charging per calendar day — each day gets its own
    # usable_capacity budget so tomorrow's solar charging isn't
    # blocked by today's full battery.
    # Group unassigned future slots by calendar day.
    by_day: dict[date, list[PlannedSlot]] = defaultdict(list)
    for s in slots:
        if s.recommendation is None and utc_key(s.start) >= utc_key(now):
            by_day[as_tz(s.start, now.tzinfo).date()].append(s)

    for day_slots in by_day.values():
        day_budget = usable_capacity
        day_charged = 0.0
        for rec in sorted(
            day_slots,
            key=lambda x: (
                not x.price_actionable,
                x.price.export_price if x.price_actionable else 0.0,
                utc_key(x.start),
            ),
        ):
            if day_charged >= day_budget:
                break
            # Only charge from solar when there is an actual PV surplus
            # (negative net consumption).  A small positive house load
            # must not be treated as a solar-charging opportunity —
            # otherwise the planner labels grid-charging slots as
            # BatteriesChargeSolar (issue #720).
            if rec.estimated_net_consumption_kwh < 0.0:
                slot_energy = min(
                    _storable_solar_refill_kwh(
                        rec, charge_efficiency, max_charge_per_slot
                    ),
                    day_budget - day_charged,
                )
                day_charged += slot_energy
                rec.recommendation = Recommendations.BatteriesChargeSolar.value
                rec.batteries_charged_kwh = round(slot_energy, 3)

    # Precompute the forward refill signal once.  The running sum resets at
    # each forced battery-discharge slot, so later PV cannot be promised
    # across a planned export event.
    refill_after_kwh, forecast_usable_after = _forward_refill_forecast(
        slots,
        charge_efficiency_pct=charge_efficiency_pct,
        max_charge_per_slot=max_charge_per_slot,
    )
    projected_capacity_kwh = min(max(current_capacity, 0.0), usable_capacity)
    future_forced_commitment_kwh = _future_forced_discharge_commitments(
        slots,
        usable_capacity=usable_capacity,
        charge_efficiency=charge_efficiency,
        max_charge_per_slot=max_charge_per_slot,
        max_discharge_per_slot=max_discharge_per_slot,
    )

    # Fill remaining unassigned slots.
    for index, rec in enumerate(slots):
        if rec.recommendation is not None:
            if not rec.primary_battery_hold and rec.recommendation in (
                Recommendations.BatteriesChargeSolar.value,
                Recommendations.BatteriesChargeGrid.value,
                Recommendations.EVSmartCharging.value,
            ):
                available_headroom_kwh = max(
                    usable_capacity - projected_capacity_kwh,
                    0.0,
                )
                solar_absorption_kwh = _storable_solar_refill_kwh(
                    rec,
                    charge_efficiency,
                    max_charge_per_slot,
                )
                scheduled_charge_kwh = 0.0
                if rec.recommendation == Recommendations.BatteriesChargeGrid.value:
                    scheduled_charge_kwh = max(rec.batteries_charged_kwh, 0.0)
                    if max_charge_per_slot is not None:
                        scheduled_charge_kwh = min(
                            scheduled_charge_kwh,
                            max(max_charge_per_slot, 0.0),
                        )
                projected_capacity_kwh += min(
                    max(solar_absorption_kwh, scheduled_charge_kwh),
                    available_headroom_kwh,
                )
            elif rec.recommendation == Recommendations.BatteriesDischargeMode.value:
                projected_capacity_kwh -= min(
                    projected_capacity_kwh,
                    _battery_load_kwh(
                        rec,
                        discharge_efficiency,
                        max_discharge_per_slot,
                    ),
                )
            elif rec.recommendation == Recommendations.ForceBatteriesDischarge.value:
                projected_capacity_kwh -= _forced_discharge_target_kwh(
                    rec,
                    available_capacity_kwh=projected_capacity_kwh,
                    required_capacity_kwh=required_capacity,
                    max_discharge_per_slot=max_discharge_per_slot,
                )
            projected_capacity_kwh = min(
                max(projected_capacity_kwh, 0.0),
                usable_capacity,
            )
            continue

        slot_load_kwh = _battery_load_kwh(
            rec,
            discharge_efficiency,
            max_discharge_per_slot,
        )
        future_commitment_kwh = future_forced_commitment_kwh[index]
        minimum_after_load_kwh = min(
            required_capacity + future_commitment_kwh,
            usable_capacity,
        )
        if (
            future_commitment_kwh > 1e-9
            and slot_load_kwh > 1e-9
            and projected_capacity_kwh - slot_load_kwh + 1e-9 < minimum_after_load_kwh
        ):
            rec.recommendation = Recommendations.BatteriesWaitMode.value
            continue

        if effective_fill_mode == SEASONAL_FILL_MODE_FORECAST and (
            rec.estimated_net_consumption_kwh < 0.0 or forecast_usable_after[index]
        ):
            refill_forecast_kwh = refill_after_kwh[index]
            projected_excess_kwh = projected_capacity_kwh - required_capacity
            headroom_kwh = projected_excess_kwh + refill_forecast_kwh
            if rec.estimated_net_consumption_kwh < 0.0:
                stored_refill_kwh = min(
                    _storable_solar_refill_kwh(
                        rec, charge_efficiency, max_charge_per_slot
                    ),
                    max(usable_capacity - projected_capacity_kwh, 0.0),
                )
                rec.batteries_charged_kwh = round(stored_refill_kwh, 3)
                projected_capacity_kwh += stored_refill_kwh
                recommendation = Recommendations.BatteriesChargeSolar.value
            elif (
                slot_load_kwh > 1e-9
                and headroom_kwh + 1e-9 >= slot_load_kwh
                and projected_capacity_kwh + 1e-9 >= slot_load_kwh
            ):
                recommendation = Recommendations.BatteriesDischargeMode.value
                projected_capacity_kwh -= slot_load_kwh
            else:
                recommendation = Recommendations.BatteriesWaitMode.value

            rec.recommendation = recommendation
            log_planner(
                "debug",
                "[disch] seasonal_fill slot=%s  net=%.3f  "
                "refill_forecast=%.3f  required=%.3f  current=%.3f  "
                "headroom=%.3f  -> %s",
                rec.start.isoformat(),
                rec.estimated_net_consumption_kwh,
                refill_forecast_kwh,
                required_capacity,
                current_capacity,
                headroom_kwh,
                recommendation,
            )
            continue

        fallback_reason = (
            "configured_months"
            if effective_fill_mode == SEASONAL_FILL_MODE_MONTHS
            else "forecast_unusable"
        )
        recommendation = _month_based_fill_recommendation(
            rec,
            current_month,
            months_winter,
        )
        rec.recommendation = recommendation
        if recommendation == Recommendations.BatteriesDischargeMode.value:
            projected_capacity_kwh -= min(
                projected_capacity_kwh,
                slot_load_kwh,
            )
        elif recommendation == Recommendations.BatteriesChargeSolar.value:
            stored_refill_kwh = min(
                _storable_solar_refill_kwh(
                    rec,
                    charge_efficiency,
                    max_charge_per_slot,
                ),
                max(usable_capacity - projected_capacity_kwh, 0.0),
            )
            rec.batteries_charged_kwh = round(stored_refill_kwh, 3)
            projected_capacity_kwh += stored_refill_kwh
        log_planner(
            "debug",
            "[disch] seasonal_fill slot=%s  mode=months  reason=%s  net=%.3f  -> %s",
            rec.start.isoformat(),
            fallback_reason,
            rec.estimated_net_consumption_kwh,
            recommendation,
        )
