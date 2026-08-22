"""Phase-aware planner/runtime helpers for Huawei plus PowMr charging."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from custom_components.hsem.coordinator_builder import build_planner_input
from custom_components.hsem.custom_sensors.phase_charge_limiter import (
    build_phase_aware_charge_commands,
)
from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.planner.secondary_storage import (
    SECONDARY_MODE_CHARGE,
    SECONDARY_MODE_SBU,
    SECONDARY_MODE_UTILITY,
)
from custom_components.hsem.utils.phase_power import (
    POWMR_CHARGER_UTILITY,
    POWMR_OUTPUT_SBU,
    POWMR_OUTPUT_UTILITY,
    compute_phase_charge_limits,
    phase_flows_from_total_w,
    phase_imbalance_w,
    secondary_site_power_delta_w,
)
from custom_components.hsem.utils.recommendations import Recommendations

_NOW = datetime(2026, 8, 11, 20, 0, tzinfo=ZoneInfo("Europe/Stockholm"))


def _rec(
    *,
    primary_charge_kwh: float = 0.0,
    secondary_mode: str = SECONDARY_MODE_UTILITY,
    secondary_current_a: float = 0.0,
) -> HourlyRecommendation:
    """Build a complete 15-minute recommendation for runtime tests."""
    recommendation = (
        Recommendations.BatteriesChargeGrid.value
        if primary_charge_kwh > 1e-9
        else Recommendations.BatteriesWaitMode.value
    )
    return HourlyRecommendation(
        start=_NOW,
        end=_NOW + timedelta(minutes=15),
        avg_house_consumption_kwh=0.0,
        avg_house_consumption_1d_kwh=0.0,
        avg_house_consumption_3d_kwh=0.0,
        avg_house_consumption_7d_kwh=0.0,
        avg_house_consumption_14d_kwh=0.0,
        batteries_charged_kwh=primary_charge_kwh,
        batteries_discharged_kwh=0.0,
        estimated_battery_capacity_kwh=0.0,
        estimated_battery_soc_pct=0.0,
        estimated_cost_currency=0.0,
        estimated_net_consumption_kwh=0.0,
        export_price=0.0,
        grid_export_kwh=0.0,
        grid_import_kwh=0.0,
        import_price=0.0,
        recommendation=recommendation,
        solcast_pv_estimate_kwh=0.0,
        secondary_storage_charge_current_a=secondary_current_a,
        secondary_storage_mode=secondary_mode,
    )


def _config() -> SensorConfig:
    """Return the user's verified three-phase/PowMr topology."""
    cfg = SensorConfig()
    cfg.main_fuse_amps = 16
    cfg.main_fuse_phases = 3
    cfg.phase_aware_charging_enabled = True
    cfg.batteries_charge_efficiency = 100.0
    cfg.batteries_discharge_efficiency = 100.0
    cfg.secondary_storage.enabled = True
    cfg.secondary_storage.grid_phase = 3
    cfg.secondary_storage.nominal_voltage_v = 25.6
    cfg.secondary_storage.min_charge_current_a = 10.0
    cfg.secondary_storage.max_charge_current_a = 60.0
    cfg.secondary_storage.charge_efficiency_pct = 93.0
    cfg.secondary_storage.base_load_includes_dedicated_load = True
    return cfg


def test_phase_imbalance_removes_current_powmr_charge_once() -> None:
    """Active L3 PowMr charge must not be counted again in future slots."""
    battery_charge_w = 25.6 * 60.0
    charge_ac_w = battery_charge_w / 0.93
    measured = (1000.0, 1000.0, 1200.0 + charge_ac_w)
    delta_w = secondary_site_power_delta_w(
        battery_net_power_w=battery_charge_w,
        load_power_w=200.0,
        charge_efficiency_pct=93.0,
        base_load_includes_dedicated_load=True,
        output_source_priority=POWMR_OUTPUT_UTILITY,
        charger_source_priority=POWMR_CHARGER_UTILITY,
    )

    imbalance = phase_imbalance_w(
        measured,
        secondary_site_delta_w=delta_w,
        secondary_grid_phase=3,
    )

    assert imbalance == pytest.approx((-200.0 / 3.0, -200.0 / 3.0, 400.0 / 3.0))
    rebuilt = phase_flows_from_total_w(
        total_grid_power_w=sum(measured),
        imbalance_w=imbalance,
        secondary_site_delta_w=delta_w,
        secondary_grid_phase=3,
    )
    assert rebuilt == pytest.approx(measured)


def test_huawei_gets_balanced_headroom_before_powmr() -> None:
    """The 5/10/3 A example must cap Huawei first, then use remaining L3."""
    limits = compute_phase_charge_limits(
        measured_phase_power_w=(5.0 * 230.0, 10.0 * 230.0, 3.0 * 230.0),
        fuse_amps=16.0,
        desired_primary_charge_power_w=10000.0,
        primary_is_controlled=True,
        primary_actual_battery_power_w=0.0,
        primary_charge_efficiency_pct=100.0,
        primary_discharge_efficiency_pct=100.0,
        desired_secondary_charge_current_a=60.0,
        secondary_actual_site_delta_w=0.0,
        secondary_desired_noncharge_site_delta_w=0.0,
        secondary_grid_phase=3,
        secondary_nominal_voltage_v=25.6,
        secondary_charge_efficiency_pct=93.0,
        secondary_min_charge_current_a=10.0,
        secondary_max_charge_current_a=60.0,
        secondary_charge_current_step_a=10.0,
    )

    assert limits.primary_charge_power_w == pytest.approx(4100.0)
    assert limits.secondary_charge_current_a == pytest.approx(50.0)
    assert max(limits.predicted_phase_power_w) <= 16.0 * 230.0 + 1e-6


def test_existing_charge_does_not_ratchet_commands_down() -> None:
    """Removing current controlled draws preserves a stable command target."""
    cfg = _config()
    cfg.secondary_storage.charge_efficiency_pct = 100.0
    rec = _rec(
        primary_charge_kwh=1.5,
        secondary_mode=SECONDARY_MODE_CHARGE,
        secondary_current_a=40.0,
    )
    live = LiveState()
    live.grid_phase_power_w = (2500.0, 2500.0, 3524.0)
    live.huawei_batteries_charge_discharge_power_w = 6000.0
    live.huawei_batteries_max_charge_power_w = 10000.0
    live.secondary_storage.battery_net_power_w = 1024.0
    live.secondary_storage.load_power_w = 200.0
    live.secondary_storage.output_source_priority = POWMR_OUTPUT_UTILITY
    live.secondary_storage.charger_source_priority = POWMR_CHARGER_UTILITY

    commands = build_phase_aware_charge_commands(cfg, live, rec)

    assert commands.primary_grid_charge_power_w == pytest.approx(6000.0)
    assert commands.recommendation.secondary_storage_charge_current_a == pytest.approx(
        40.0
    )
    assert commands.limits is not None
    assert commands.limits.predicted_phase_power_w == pytest.approx(
        live.grid_phase_power_w
    )


def test_powmr_charge_is_disabled_when_no_full_step_fits() -> None:
    """Sub-10 A L3 headroom must select utility instead of rounding up."""
    cfg = _config()
    rec = _rec(
        secondary_mode=SECONDARY_MODE_CHARGE,
        secondary_current_a=60.0,
    )
    live = LiveState()
    live.grid_phase_power_w = (1000.0, 1000.0, 3600.0)
    live.secondary_storage.load_power_w = 200.0
    live.secondary_storage.output_source_priority = POWMR_OUTPUT_UTILITY

    commands = build_phase_aware_charge_commands(cfg, live, rec)

    assert commands.recommendation.secondary_storage_mode == SECONDARY_MODE_UTILITY
    assert commands.recommendation.secondary_storage_charge_current_a == pytest.approx(
        0.0
    )


def test_sbu_transition_frees_l3_headroom_for_huawei() -> None:
    """A planned SBU transition removes only the dedicated load from L3."""
    cfg = _config()
    rec = _rec(primary_charge_kwh=2.5, secondary_mode=SECONDARY_MODE_SBU)
    live = LiveState()
    live.grid_phase_power_w = (2000.0, 2000.0, 2200.0)
    live.huawei_batteries_max_charge_power_w = 10000.0
    live.huawei_batteries_charge_discharge_power_w = 0.0
    live.secondary_storage.load_power_w = 200.0
    live.secondary_storage.output_source_priority = POWMR_OUTPUT_UTILITY

    commands = build_phase_aware_charge_commands(cfg, live, rec)

    assert commands.limits is not None
    assert commands.limits.base_phase_power_w == pytest.approx((2000.0, 2000.0, 2000.0))
    assert max(commands.limits.predicted_phase_power_w) <= 16.0 * 230.0 + 1e-6


def test_disabled_phase_feature_still_carries_plan_derived_huawei_cap() -> None:
    """Aggregate-fuse users execute the selected Huawei energy allocation."""
    cfg = _config()
    cfg.phase_aware_charging_enabled = False
    rec = _rec(
        primary_charge_kwh=0.4,
        secondary_mode=SECONDARY_MODE_CHARGE,
        secondary_current_a=60.0,
    )

    commands = build_phase_aware_charge_commands(cfg, LiveState(), rec)

    assert commands.recommendation is rec
    assert commands.primary_grid_charge_power_w == pytest.approx(1600.0)
    assert commands.limits is None


def test_plan_derived_huawei_cap_stays_in_full_slot_dc_frame() -> None:
    """A late replan cannot compress full-slot stored energy above the fuse."""
    cfg = _config()
    cfg.phase_aware_charging_enabled = False
    cfg.main_fuse_amps = 1
    cfg.batteries_charge_efficiency = 90.0
    rec = _rec(primary_charge_kwh=0.1725)
    live = LiveState()
    live.huawei_batteries_max_charge_power_w = 10000.0

    commands = build_phase_aware_charge_commands(cfg, live, rec)

    # 0.1725 kWh DC over the 15-minute projection is 690 W DC. Neither a
    # ten-minute-late call nor charge efficiency may inflate this actuator cap.
    assert commands.primary_grid_charge_power_w == pytest.approx(690.0)
    assert commands.limits is None


def test_live_phase_imbalance_reaches_planner_input() -> None:
    """The coordinator bridge must remove the balanced mean exactly once."""
    cfg = _config()
    cfg.secondary_storage.enabled = False
    live = LiveState()
    live.grid_phase_power_w = (1150.0, 2300.0, 690.0)

    with patch(
        "custom_components.hsem.coordinator_builder.hsem_now",
        return_value=_NOW,
    ):
        planner_input = build_planner_input(
            cfg=cfg,
            live=live,
            hourly_recommendations=[_rec()],
            previous_winner_name=None,
            previous_winner_score=0.0,
        )

    assert planner_input.phase_aware_charging_enabled is True
    assert planner_input.grid_phase_power_imbalance_w == pytest.approx(
        (-230.0, 920.0, -690.0)
    )


def test_secondary_site_delta_for_sbu_matches_history_topology() -> None:
    """SBU removes the NAS from site flow only when history includes it."""
    included = secondary_site_power_delta_w(
        battery_net_power_w=-250.0,
        load_power_w=200.0,
        charge_efficiency_pct=93.0,
        base_load_includes_dedicated_load=True,
        output_source_priority=POWMR_OUTPUT_SBU,
        charger_source_priority=None,
    )
    excluded = secondary_site_power_delta_w(
        battery_net_power_w=-250.0,
        load_power_w=200.0,
        charge_efficiency_pct=93.0,
        base_load_includes_dedicated_load=False,
        output_source_priority=POWMR_OUTPUT_SBU,
        charger_source_priority=None,
    )

    assert included == pytest.approx(-200.0)
    assert excluded == pytest.approx(0.0)
