"""Pure helpers for three-phase grid-import protection.

Positive phase power means import and negative phase power means export.  The
Huawei battery/inverter is treated as a balanced three-phase device while the
optional secondary charger is assigned to one configured phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeGuard

from custom_components.hsem.utils.misc import clamp_efficiency

NOMINAL_PHASE_VOLTAGE_V = 230.0
PHASE_COUNT = 3

POWMR_OUTPUT_UTILITY = "Utility first"
POWMR_OUTPUT_SBU = "SBU priority"
POWMR_CHARGER_UTILITY = "Solar and Utility"
POWMR_CHARGER_SOLAR_ONLY = "Only Solar"

PhasePowers = tuple[float, float, float]


@dataclass(frozen=True)
class PhaseChargeLimits:
    """Safe battery charge commands derived from live per-phase power."""

    primary_charge_power_w: float
    secondary_charge_current_a: float
    base_phase_power_w: PhasePowers
    predicted_phase_power_w: PhasePowers


def phase_powers_valid(
    values: tuple[float | None, float | None, float | None],
) -> TypeGuard[PhasePowers]:
    """Return whether all three signed phase readings are finite numbers."""
    return all(value is not None and math.isfinite(value) for value in values)


def secondary_site_power_delta_w(
    *,
    battery_net_power_w: float | None,
    load_power_w: float | None,
    charge_efficiency_pct: float,
    base_load_includes_dedicated_load: bool,
    output_source_priority: str | None,
    charger_source_priority: str | None,
) -> float:
    """Return the secondary branch's signed contribution to site-grid power.

    The value follows the same topology as the MILP: charging adds AC import;
    SBU removes the dedicated load from the site bus; utility mode leaves an
    already-accounted load unchanged.  When house history excludes the load,
    utility mode adds it explicitly instead.
    """
    load_w = max(load_power_w or 0.0, 0.0)
    charge_ac_w = 0.0
    if (
        charger_source_priority == POWMR_CHARGER_UTILITY
        and battery_net_power_w is not None
        and battery_net_power_w > 1e-9
    ):
        charge_ac_w = battery_net_power_w / clamp_efficiency(charge_efficiency_pct)

    sbu = output_source_priority == POWMR_OUTPUT_SBU
    if base_load_includes_dedicated_load:
        return charge_ac_w - (load_w if sbu else 0.0)
    return charge_ac_w + (0.0 if sbu else load_w)


def phase_imbalance_w(
    phase_power_w: PhasePowers,
    *,
    secondary_site_delta_w: float = 0.0,
    secondary_grid_phase: int = 3,
) -> PhasePowers:
    """Return zero-sum non-balanced offsets after removing the secondary branch.

    Balanced Huawei charge/discharge and PV production disappear when the mean
    is removed.  The explicit secondary adjustment prevents a currently
    charging PowMr from being counted both in the observed imbalance and again
    in future MILP decisions.
    """
    phase_index = min(max(secondary_grid_phase, 1), PHASE_COUNT) - 1
    adjusted = list(phase_power_w)
    adjusted[phase_index] -= secondary_site_delta_w
    mean_w = sum(adjusted) / PHASE_COUNT
    return tuple(value - mean_w for value in adjusted)  # type: ignore[return-value]


def phase_flows_from_total_w(
    total_grid_power_w: float,
    imbalance_w: PhasePowers,
    *,
    secondary_site_delta_w: float = 0.0,
    secondary_grid_phase: int = 3,
) -> PhasePowers:
    """Split total signed grid power into topology-aware phase flows."""
    phase_index = min(max(secondary_grid_phase, 1), PHASE_COUNT) - 1
    balanced_remainder_w = (total_grid_power_w - secondary_site_delta_w) / PHASE_COUNT
    return tuple(
        balanced_remainder_w
        + imbalance_w[index]
        + (secondary_site_delta_w if index == phase_index else 0.0)
        for index in range(PHASE_COUNT)
    )  # type: ignore[return-value]


def _primary_site_power_w(
    battery_power_w: float,
    charge_efficiency_pct: float,
    discharge_efficiency_pct: float,
) -> float:
    """Convert signed Huawei battery power to signed AC-site power."""
    if battery_power_w > 1e-9:
        return battery_power_w / clamp_efficiency(charge_efficiency_pct)
    if battery_power_w < -1e-9:
        return battery_power_w * clamp_efficiency(discharge_efficiency_pct)
    return 0.0


def _floor_step(value: float, step: float) -> float:
    """Round a non-negative command down to a supported hardware step."""
    if value <= 1e-9 or step <= 1e-9:
        return 0.0
    return math.floor((value + 1e-9) / step) * step


def compute_phase_charge_limits(
    *,
    measured_phase_power_w: PhasePowers,
    fuse_amps: float,
    desired_primary_charge_power_w: float,
    primary_is_controlled: bool,
    primary_actual_battery_power_w: float,
    primary_charge_efficiency_pct: float,
    primary_discharge_efficiency_pct: float,
    desired_secondary_charge_current_a: float,
    secondary_actual_site_delta_w: float,
    secondary_desired_noncharge_site_delta_w: float,
    secondary_grid_phase: int,
    secondary_nominal_voltage_v: float,
    secondary_charge_efficiency_pct: float,
    secondary_min_charge_current_a: float,
    secondary_max_charge_current_a: float,
    secondary_charge_current_step_a: float,
) -> PhaseChargeLimits:
    """Allocate live phase headroom to Huawei first and PowMr second.

    Existing controllable charge/discharge contributions are removed from the
    meter snapshot before new commands are calculated.  This avoids a feedback
    loop where a running charger consumes its own apparent headroom.  The
    resulting commands target the rated fuse current; no intentional overload
    allowance is used.
    """
    phase_index = min(max(secondary_grid_phase, 1), PHASE_COUNT) - 1
    limit_w = max(fuse_amps, 0.0) * NOMINAL_PHASE_VOLTAGE_V
    base = list(measured_phase_power_w)

    if primary_is_controlled:
        primary_actual_site_w = _primary_site_power_w(
            primary_actual_battery_power_w,
            primary_charge_efficiency_pct,
            primary_discharge_efficiency_pct,
        )
        for index in range(PHASE_COUNT):
            base[index] -= primary_actual_site_w / PHASE_COUNT
    base[phase_index] -= secondary_actual_site_delta_w
    base[phase_index] += secondary_desired_noncharge_site_delta_w

    primary_eff = clamp_efficiency(primary_charge_efficiency_pct)
    desired_primary_dc_w = max(desired_primary_charge_power_w, 0.0)
    primary_ac_headroom_w = PHASE_COUNT * max(
        min(limit_w - phase_w for phase_w in base),
        0.0,
    )
    primary_dc_limit_w = primary_ac_headroom_w * primary_eff
    primary_dc_target_w = _floor_step(
        min(desired_primary_dc_w, primary_dc_limit_w),
        100.0,
    )
    primary_ac_target_w = primary_dc_target_w / primary_eff

    secondary_eff = clamp_efficiency(secondary_charge_efficiency_pct)
    secondary_headroom_ac_w = max(
        limit_w - base[phase_index] - primary_ac_target_w / PHASE_COUNT,
        0.0,
    )
    secondary_dc_limit_w = secondary_headroom_ac_w * secondary_eff
    secondary_current_limit_a = (
        secondary_dc_limit_w / secondary_nominal_voltage_v
        if secondary_nominal_voltage_v > 1e-9
        else 0.0
    )
    secondary_target_a = _floor_step(
        min(
            max(desired_secondary_charge_current_a, 0.0),
            max(secondary_current_limit_a, 0.0),
            max(secondary_max_charge_current_a, 0.0),
        ),
        secondary_charge_current_step_a,
    )
    if secondary_target_a < secondary_min_charge_current_a - 1e-9:
        secondary_target_a = 0.0

    secondary_ac_target_w = (
        secondary_target_a * secondary_nominal_voltage_v / secondary_eff
        if secondary_target_a > 1e-9
        else 0.0
    )
    predicted = tuple(
        base[index]
        + primary_ac_target_w / PHASE_COUNT
        + (secondary_ac_target_w if index == phase_index else 0.0)
        for index in range(PHASE_COUNT)
    )
    return PhaseChargeLimits(
        primary_charge_power_w=primary_dc_target_w,
        secondary_charge_current_a=secondary_target_a,
        base_phase_power_w=tuple(base),  # type: ignore[arg-type]
        predicted_phase_power_w=predicted,  # type: ignore[arg-type]
    )
