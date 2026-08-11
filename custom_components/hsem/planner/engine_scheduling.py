"""Charge/discharge scheduling passes used by the planner engine."""

from __future__ import annotations

from datetime import datetime

from custom_components.hsem.models.planner_input import PlannerInput
from custom_components.hsem.planner.charging.arbitrage_charge import (
    apply_arbitrage_grid_charge,
)
from custom_components.hsem.planner.charging.opportunistic_charge import (
    apply_opportunistic_charge,
)
from custom_components.hsem.planner.charging.pre_charge import apply_charge_schedules
from custom_components.hsem.planner.discharge_scheduler import (
    apply_discharge_schedules,
    apply_excess_export,
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
    roundtrip_loss_pct,
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
    apply_discharge_schedules(slots, inp.battery_schedules, now)
    log_planner(
        "debug",
        "[core] _schedule_slots  pass=discharge_schedules  slots=%d",
        len(slots),
    )
    charge_eff = clamp_efficiency(inp.battery_charge_efficiency_pct)
    roundtrip_loss = roundtrip_loss_pct(
        inp.battery_charge_efficiency_pct,
        inp.battery_discharge_efficiency_pct,
    )
    max_charge_per_slot = max_energy_per_slot_kwh(
        inp.battery_max_charge_power_w,
        inp.interval_minutes,
        efficiency_fraction=charge_eff,
    )
    apply_charge_schedules(
        slots,
        inp.battery_schedules,
        now,
        max_charge_per_slot,
        current_kwh=current_kwh,
        usable_kwh=usable_kwh,
        cycle_cost_per_kwh=effective_cycle_cost,
        recommended_threshold=rt,
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
    apply_arbitrage_grid_charge(
        slots,
        inp.battery_schedules,
        now,
        current_kwh,
        usable_kwh,
        max_charge_per_slot,
        conversion_loss_pct=roundtrip_loss,
        cycle_cost_per_kwh=effective_cycle_cost,
        recommended_threshold=rt,
    )
    max_discharge_per_slot: float | None = None
    if inp.battery_max_discharge_power_w is not None:
        max_discharge_per_slot = max_energy_per_slot_kwh(
            inp.battery_max_discharge_power_w,
            inp.interval_minutes,
        )
    max_soc_kwh = usable_kwh
    populate_battery_capacity(slots, now, current_kwh, usable_kwh)
    required_capacity = calculate_required_battery_until_solar(
        slots, now, usable_kwh, inp.excess_export_discharge_buffer_pct
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
    if inp.excess_export_enabled:
        apply_excess_export(
            slots,
            now,
            current_kwh,
            required_capacity,
            inp.excess_export_price_threshold,
            warnings,
            export_min_price=inp.export_min_price,
            recommended_threshold=rt,
        )
        log_planner(
            "debug",
            "[core] _schedule_slots  pass=excess_export  enabled=True",
        )
    else:
        log_planner(
            "debug",
            "[core] _schedule_slots  pass=excess_export  enabled=False  "
            "→ MILP no_export constraint active (battery will not export to grid)",
        )
    apply_optimization_strategy(
        slots,
        now,
        current_kwh,
        usable_kwh,
        required_capacity,
        inp.months_winter,
        export_min_price=inp.export_min_price,
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
