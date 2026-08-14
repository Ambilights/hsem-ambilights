"""Rebuild user-facing EV plans from authoritative MILP slot decisions."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from custom_components.hsem.utils.datetime_utils import slot_contains
from custom_components.hsem.utils.misc import clamp_efficiency
from custom_components.hsem.utils.units import slot_duration_hours


def rebuild_ev_plan_from_slots(
    original_plan: Any,
    slots: list,
    now: datetime,
    charger_efficiency_pct: float = 100.0,
    *,
    is_second: bool = False,
) -> Any:
    """Rebuild an EV charging plan from MILP-decided per-EV slot fields.

    The import of the EV plan dataclasses is deliberately local so this helper
    can remain separate while ``ev_planner`` re-exports the public function.
    """
    from custom_components.hsem.planner.ev_planner import (  # noqa: PLC0415
        EVChargingPlan,
        EVChargingSlot,
    )

    eff = clamp_efficiency(charger_efficiency_pct)
    charging_slots: list[EVChargingSlot] = []
    planned_load_by_slot: dict[str, float] = {}
    current_slot_planned_load_kwh = 0.0
    total_charged_kwh = 0.0
    power_field = (
        "ev_second_charger_calculated_power"
        if is_second
        else "ev_charger_calculated_power"
    )

    for slot in slots:
        power_w = getattr(slot, power_field, 0.0)
        if power_w < 1e-9:
            continue

        slot_hours = slot_duration_hours(slot.start, slot.end)
        ac_load = power_w * slot_hours / 1000.0
        dc_kwh = ac_load * eff
        total_charged_kwh += dc_kwh

        pv_kwh = max(getattr(slot, "solcast_pv_estimate_kwh", 0.0), 0.0)
        house_kwh = max(getattr(slot, "avg_house_consumption_kwh", 0.0), 0.0)
        surplus_kwh = max(pv_kwh - house_kwh, 0.0)
        solar_used_ac = min(ac_load, surplus_kwh)
        solar_used_dc = solar_used_ac * eff
        import_needed = max(dc_kwh - solar_used_dc, 0.0)
        raw_import_price = getattr(getattr(slot, "price", None), "import_price", 0.0)
        import_price = (
            float(raw_import_price)
            if getattr(slot, "price_actionable", True)
            and math.isfinite(float(raw_import_price))
            else 0.0
        )

        charging_slots.append(
            EVChargingSlot(
                start=slot.start,
                end=slot.end,
                estimated_charged_kwh=round(dc_kwh, 3),
                ac_load_kwh=round(ac_load, 3),
                solar_surplus_kwh=round(solar_used_dc, 3),
                import_needed_kwh=round(import_needed, 3),
                import_price=import_price,
                estimated_cost=round(ac_load * import_price, 4),
            )
        )
        planned_load_by_slot[slot.start.isoformat()] = dc_kwh

        if slot_contains(slot.start, slot.end, now):
            current_slot_planned_load_kwh = dc_kwh

    if charging_slots:
        state = "charging" if current_slot_planned_load_kwh > 1e-9 else "waiting"
    elif original_plan.state == "fully_charged":
        state = "fully_charged"
    else:
        state = original_plan.state

    data_quality = dict(original_plan.data_quality)
    unmet_target_kwh = max(original_plan.total_kwh_needed - total_charged_kwh, 0.0)
    if unmet_target_kwh > 1e-9:
        data_quality["unmet_target_kwh"] = round(unmet_target_kwh, 3)
    else:
        data_quality.pop("unmet_target_kwh", None)
        if data_quality.get("warning") in {
            "No candidate slots before deadline",
            (
                "EV target cannot be fully scheduled before the deadline using "
                "published-price slots"
            ),
        }:
            data_quality.pop("warning", None)

    return EVChargingPlan(
        state=state,
        ev_connected=original_plan.ev_connected,
        base_load_includes_ev=original_plan.base_load_includes_ev,
        current_soc_pct=original_plan.current_soc_pct,
        target_soc_pct=original_plan.target_soc_pct,
        battery_capacity_kwh=original_plan.battery_capacity_kwh,
        charger_power_kw=original_plan.charger_power_kw,
        charger_min_power_w=original_plan.charger_min_power_w,
        total_kwh_needed=original_plan.total_kwh_needed,
        deadline=original_plan.deadline,
        charging_slots=charging_slots,
        planned_load_by_slot=planned_load_by_slot,
        current_slot_planned_load_kwh=round(current_slot_planned_load_kwh, 3),
        data_quality=data_quality,
    )
