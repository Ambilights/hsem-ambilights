"""Charge/discharge scheduling passes used by the planner engine."""

from __future__ import annotations

from datetime import datetime

from custom_components.hsem.models.planner_input import PlannerInput
from custom_components.hsem.planner.charging.opportunistic_charge import (
    apply_opportunistic_charge,
)
from custom_components.hsem.planner.discharge_scheduler import (
    apply_force_export_policy,
    apply_optimization_strategy,
    calculate_required_battery_until_solar,
)
from custom_components.hsem.planner.slot_population import (
    mark_time_passed,
    populate_battery_capacity,
)
from custom_components.hsem.utils.logger import log_planner
from custom_components.hsem.utils.misc import clamp_efficiency
from custom_components.hsem.utils.units import (
    max_energy_per_slot_kwh,
)


def _schedule_slots(
    slots: list,
    inp: PlannerInput,
    now: datetime,
    current_kwh: float,
    usable_kwh: float,
    rt: float,
    effective_cycle_cost: float,
    warnings: list[str],
) -> tuple[float, float | None, float, float, list[str]]:
    """Run all charge/discharge scheduling passes."""
    mark_time_passed(slots, now)
    charge_eff = clamp_efficiency(inp.battery_charge_efficiency_pct)
    max_charge_per_slot = max_energy_per_slot_kwh(
        inp.battery_max_charge_power_w,
        inp.interval_minutes,
        efficiency_fraction=charge_eff,
    )
    apply_opportunistic_charge(
        slots,
        now,
        current_kwh,
        usable_kwh,
        max_charge_per_slot,
        rt,
        cycle_cost_per_kwh=effective_cycle_cost,
    )
    max_discharge_per_slot: float | None = None
    if inp.battery_max_discharge_power_w is not None:
        max_discharge_per_slot = max_energy_per_slot_kwh(
            inp.battery_max_discharge_power_w,
            inp.interval_minutes,
        )
    max_soc_kwh = usable_kwh
    populate_battery_capacity(slots, now, current_kwh, usable_kwh)
    apply_force_export_policy(slots, inp.export_min_price)
    required_capacity = calculate_required_battery_until_solar(
        slots,
        now,
        usable_kwh,
        inp.excess_export_discharge_buffer_pct,
        discharge_efficiency_pct=inp.battery_discharge_efficiency_pct,
        max_discharge_per_slot=max_discharge_per_slot,
    )
    log_planner(
        "debug",
        "[core] _schedule_slots  pass=after_scheduling  mcps=%.3f  mdps=%s  "
        "max_soc=%.3f  rc=%.3f",
        max_charge_per_slot,
        (
            f"{max_discharge_per_slot:.3f}"
            if max_discharge_per_slot is not None
            else "∞"
        ),
        max_soc_kwh,
        required_capacity,
    )
    log_planner(
        "debug",
        "[core] _schedule_slots  pass=excess_export  heuristic=retired  "
        "enabled=%s  authority=MILP",
        inp.excess_export_enabled,
    )
    apply_optimization_strategy(
        slots,
        now,
        current_kwh,
        usable_kwh,
        required_capacity,
        inp.months_winter,
        export_min_price=inp.export_min_price,
        seasonal_fill_mode=inp.seasonal_fill_mode,
        charge_efficiency_pct=inp.battery_charge_efficiency_pct,
        discharge_efficiency_pct=inp.battery_discharge_efficiency_pct,
        max_charge_per_slot=max_charge_per_slot,
        max_discharge_per_slot=max_discharge_per_slot,
    )
    log_planner(
        "debug",
        "[core] _schedule_slots DONE  mcps=%.3f  mdps=%s  max_soc=%.3f  rc=%.3f  "
        "warnings=%d",
        max_charge_per_slot,
        (
            f"{max_discharge_per_slot:.3f}"
            if max_discharge_per_slot is not None
            else "∞"
        ),
        max_soc_kwh,
        required_capacity,
        len(warnings),
    )
    return (
        max_charge_per_slot,
        max_discharge_per_slot,
        max_soc_kwh,
        required_capacity,
        warnings,
    )
