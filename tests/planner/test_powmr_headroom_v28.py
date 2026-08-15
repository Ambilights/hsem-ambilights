"""PowMr may not charge on headroom the Huawei is about to stop providing.

``compute_phase_charge_limits`` removes the Huawei's own contribution from the
meter snapshot so ``base`` means "house load without the battery".  That removal
was gated on the Huawei itself being grid-charged, which left this sequence
unprotected:

    Huawei discharging, masking the house load
      -> meter shows almost no import
      -> plan says Huawei Wait + PowMr Charge
      -> limiter reads the low import as spare capacity
      -> async_apply_battery_settings stops the discharge FIRST
      -> async_apply_secondary_storage then starts the PowMr

Reproduced on the real configuration: 3 kW/phase of house load behind a 9 kW
discharge shows ~60 W/phase on the meter, authorising the PowMr's full 60 A and
landing L3 at 20.2 A on a 16 A fuse.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from custom_components.hsem.custom_sensors.phase_charge_limiter import (
    PhaseAwareChargeCommands,
    build_phase_aware_charge_commands,
)
from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.planner.secondary_storage import (
    SECONDARY_MODE_CHARGE,
    SECONDARY_MODE_UTILITY,
)
from custom_components.hsem.utils.recommendations import Recommendations

FUSE_A = 16
PHASE_LIMIT_W = FUSE_A * 230.0
VOLT = 25.6
EFF = 0.93
# The Huawei's 9 kW discharge covers 2940 W/phase of a 3000 W/phase load.
MASKED_METER_W = 60.0
TRUE_HOUSE_W = 3000.0


def _commands(
    *,
    battery_power_w: float | None,
    recommendation: str = Recommendations.BatteriesWaitMode.value,
    secondary_mode: str = SECONDARY_MODE_CHARGE,
) -> PhaseAwareChargeCommands:
    cfg = SensorConfig()
    cfg.phase_aware_charging_enabled = True
    cfg.main_fuse_amps = FUSE_A
    cfg.main_fuse_phases = 3
    cfg.secondary_storage.enabled = True
    cfg.secondary_storage.grid_phase = 3
    cfg.secondary_storage.nominal_voltage_v = VOLT
    cfg.secondary_storage.charge_efficiency_pct = EFF * 100
    cfg.secondary_storage.min_charge_current_a = 10.0
    cfg.secondary_storage.max_charge_current_a = 60.0
    live = LiveState()
    live.grid_phase_power_w = (
        MASKED_METER_W,
        MASKED_METER_W,
        MASKED_METER_W,
    )
    live.huawei_batteries_charge_discharge_power_w = battery_power_w
    live.huawei_batteries_max_charge_power_w = 10000.0
    start = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)
    rec = HourlyRecommendation(
        start=start,
        end=start + timedelta(minutes=15),
        avg_house_consumption_kwh=0.75,
        avg_house_consumption_1d_kwh=0.75,
        avg_house_consumption_3d_kwh=0.75,
        avg_house_consumption_7d_kwh=0.75,
        avg_house_consumption_14d_kwh=0.75,
        batteries_charged_kwh=0.0,
        batteries_discharged_kwh=0.0,
        estimated_battery_capacity_kwh=6.0,
        estimated_battery_soc_pct=25.0,
        estimated_cost_currency=0.0,
        estimated_net_consumption_kwh=0.75,
        export_price=0.1,
        grid_export_kwh=0.0,
        grid_import_kwh=0.75,
        import_price=1.0,
        recommendation=recommendation,
        solcast_pv_estimate_kwh=0.0,
    )
    rec.secondary_storage_mode = secondary_mode
    rec.secondary_storage_charge_current_a = 60.0
    return build_phase_aware_charge_commands(cfg, live, rec)


class TestHuaweiContributionIsRemoved:
    """``base`` must mean house load without the battery, for either actuator."""

    def test_the_masked_load_is_reconstructed(self) -> None:
        cmds = _commands(battery_power_w=-9000.0)

        assert cmds.limits is not None
        assert cmds.limits.base_phase_power_w[2] == pytest.approx(
            TRUE_HOUSE_W, abs=50.0
        )

    def test_powmr_is_throttled_to_fit_the_real_load(self) -> None:
        cmds = _commands(battery_power_w=-9000.0)
        amps = cmds.recommendation.secondary_storage_charge_current_a
        powmr_ac_w = amps * VOLT / EFF

        assert amps < 60.0
        assert TRUE_HOUSE_W + powmr_ac_w <= PHASE_LIMIT_W

    def test_the_unfixed_reference_would_have_overloaded(self) -> None:
        """Pins the defect: full current lands L3 above the fuse."""
        powmr_ac_w = 60.0 * VOLT / EFF

        assert (TRUE_HOUSE_W + powmr_ac_w) / 230.0 > FUSE_A

    def test_a_huawei_grid_charge_still_works(self) -> None:
        cmds = _commands(
            battery_power_w=-9000.0,
            recommendation=Recommendations.BatteriesChargeGrid.value,
        )

        assert cmds.limits is not None
        assert cmds.limits.base_phase_power_w[2] == pytest.approx(
            TRUE_HOUSE_W, abs=50.0
        )


class TestBatteryTelemetryRequiredForEitherActuator:
    """The guard must cover a PowMr-only charge, not just a Huawei one."""

    @pytest.mark.parametrize("reading", [None, math.nan])
    def test_powmr_only_charge_is_blocked(self, reading: float | None) -> None:
        cmds = _commands(battery_power_w=reading)

        assert cmds.recommendation.secondary_storage_mode == SECONDARY_MODE_UTILITY
        assert cmds.recommendation.secondary_storage_charge_current_a == 0.0

    def test_a_non_charging_slot_is_unaffected(self) -> None:
        cmds = _commands(battery_power_w=None, secondary_mode=SECONDARY_MODE_UTILITY)

        assert cmds.recommendation.secondary_storage_mode == SECONDARY_MODE_UTILITY
        assert cmds.primary_grid_charge_power_w is None
