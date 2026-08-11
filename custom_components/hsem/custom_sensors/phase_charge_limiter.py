"""Translate a planned battery charge into phase-safe live commands."""

from __future__ import annotations

from dataclasses import dataclass, replace

from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.planner.secondary_storage import (
    SECONDARY_MODE_CHARGE,
    SECONDARY_MODE_SBU,
    SECONDARY_MODE_UTILITY,
)
from custom_components.hsem.utils.phase_power import (
    PhaseChargeLimits,
    compute_phase_charge_limits,
    phase_powers_valid,
    secondary_site_power_delta_w,
)
from custom_components.hsem.utils.recommendations import Recommendations
from custom_components.hsem.utils.units import slot_duration_hours


@dataclass(frozen=True)
class PhaseAwareChargeCommands:
    """Hardware commands after applying live per-phase headroom."""

    recommendation: HourlyRecommendation
    primary_grid_charge_power_w: float | None = None
    limits: PhaseChargeLimits | None = None


def build_phase_aware_charge_commands(
    cfg: SensorConfig,
    live: LiveState,
    rec: HourlyRecommendation,
) -> PhaseAwareChargeCommands:
    """Return Huawei-first commands that target, but never plan above, the fuse.

    This is intentionally a runtime correction on top of the horizon MILP.
    The MILP uses a forecast and the latest phase imbalance; this function uses
    the newest meter snapshot immediately before writes and therefore protects
    against appliance changes since the plan was solved.
    """
    live_phase_power_w = live.grid_phase_power_w
    if (
        not cfg.phase_aware_charging_enabled
        or cfg.main_fuse_phases != 3
        or cfg.main_fuse_amps <= 0
        or not phase_powers_valid(live_phase_power_w)
    ):
        return PhaseAwareChargeCommands(recommendation=rec)

    measured_phase_power_w = live_phase_power_w
    slot_hours = slot_duration_hours(rec.start, rec.end)
    primary_grid_charge = (
        rec.recommendation == Recommendations.BatteriesChargeGrid.value
    )
    desired_primary_w = 0.0
    if primary_grid_charge and slot_hours > 1e-9:
        desired_primary_w = max(rec.batteries_charged_kwh, 0.0) * 1000.0 / slot_hours
        if live.huawei_batteries_max_charge_power_w is not None:
            desired_primary_w = min(
                desired_primary_w,
                max(live.huawei_batteries_max_charge_power_w, 0.0),
            )

    secondary_charge = rec.secondary_storage_mode == SECONDARY_MODE_CHARGE
    desired_secondary_a = (
        max(rec.secondary_storage_charge_current_a, 0.0) if secondary_charge else 0.0
    )
    secondary_delta_w = 0.0
    secondary_load_w = max(live.secondary_storage.load_power_w or 0.0, 0.0)
    if cfg.secondary_storage.enabled:
        secondary_delta_w = secondary_site_power_delta_w(
            battery_net_power_w=live.secondary_storage.battery_net_power_w,
            load_power_w=live.secondary_storage.load_power_w,
            charge_efficiency_pct=cfg.secondary_storage.charge_efficiency_pct,
            base_load_includes_dedicated_load=(
                cfg.secondary_storage.base_load_includes_dedicated_load
            ),
            output_source_priority=live.secondary_storage.output_source_priority,
            charger_source_priority=live.secondary_storage.charger_source_priority,
        )
    if rec.secondary_storage_mode == SECONDARY_MODE_SBU:
        desired_secondary_noncharge_delta_w = (
            -secondary_load_w
            if cfg.secondary_storage.base_load_includes_dedicated_load
            else 0.0
        )
    else:
        desired_secondary_noncharge_delta_w = (
            0.0
            if cfg.secondary_storage.base_load_includes_dedicated_load
            else secondary_load_w
        )

    limits = compute_phase_charge_limits(
        measured_phase_power_w=measured_phase_power_w,
        fuse_amps=float(cfg.main_fuse_amps),
        desired_primary_charge_power_w=desired_primary_w,
        primary_is_controlled=primary_grid_charge,
        primary_actual_battery_power_w=(
            live.huawei_batteries_charge_discharge_power_w or 0.0
        ),
        primary_charge_efficiency_pct=cfg.batteries_charge_efficiency,
        primary_discharge_efficiency_pct=cfg.batteries_discharge_efficiency,
        desired_secondary_charge_current_a=desired_secondary_a,
        secondary_actual_site_delta_w=secondary_delta_w,
        secondary_desired_noncharge_site_delta_w=(desired_secondary_noncharge_delta_w),
        secondary_grid_phase=cfg.secondary_storage.grid_phase,
        secondary_nominal_voltage_v=cfg.secondary_storage.nominal_voltage_v,
        secondary_charge_efficiency_pct=(cfg.secondary_storage.charge_efficiency_pct),
        secondary_min_charge_current_a=(cfg.secondary_storage.min_charge_current_a),
        secondary_max_charge_current_a=(cfg.secondary_storage.max_charge_current_a),
        secondary_charge_current_step_a=10.0,
    )

    adjusted = rec
    if secondary_charge:
        if limits.secondary_charge_current_a > 1e-9:
            adjusted = replace(
                rec,
                secondary_storage_charge_current_a=(limits.secondary_charge_current_a),
            )
        else:
            adjusted = replace(
                rec,
                secondary_storage_mode=SECONDARY_MODE_UTILITY,
                secondary_storage_charge_current_a=0.0,
                secondary_storage_charged_kwh=0.0,
            )

    primary_limit_w = limits.primary_charge_power_w if primary_grid_charge else None
    return PhaseAwareChargeCommands(
        recommendation=adjusted,
        primary_grid_charge_power_w=primary_limit_w,
        limits=limits,
    )
