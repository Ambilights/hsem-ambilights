"""The grid meter's sign convention, and what happens when it is unusable.

``utils.phase_power`` documents "positive means import, negative means
export", and every consumer of ``LiveState.grid_phase_power_w`` relies on it.
The Huawei power meter reports the reverse.  Verified on live hardware over a
three-hour overnight window with zero PV and an idle battery — conditions in
which the house can only import — where all 932 meter samples were negative
and none positive.

Fed in unnegated, the fuse-headroom calculation *adds* the house import to the
fuse allowance instead of subtracting it, so the headroom it reports grows as
the house draws more.  The protection relaxes exactly when it must clamp.

Covers two fixes:

1. ``state_collector`` negates the meter at the read boundary, so the pure
   helpers keep their documented contract.
2. ``applier`` no longer re-arms grid charging at the hardware maximum when
   phase-aware charging is enabled but the phase telemetry is unusable.  A
   sensor dropout must not silently remove the fuse protection the user opted
   into; the plan's own charge power is used instead.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.hsem.custom_sensors.applier import (
    async_apply_battery_settings,
)
from custom_components.hsem.custom_sensors.state_collector import _negate_optional
from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.utils.degraded_mode import DegradedMode
from custom_components.hsem.utils.inverter_verify import ApplyResult, ApplyStatus
from custom_components.hsem.utils.phase_power import (
    PhaseChargeLimits,
    compute_phase_charge_limits,
)
from custom_components.hsem.utils.recommendations import Recommendations
from custom_components.hsem.utils.workingmodes import WorkingModes

CHARGE_ENTITY = "number.batteries_grid_charge_maximum_power"
FUSE_AMPS = 16.0
PHASE_LIMIT_W = FUSE_AMPS * 230.0  # 3680 W


class TestNegateOptional:
    """``None`` must survive so ``phase_powers_valid`` can still reject it."""

    def test_none_stays_none(self) -> None:
        assert _negate_optional(None) is None

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(-541.0, 541.0), (1084.0, -1084.0), (0.0, 0.0)],
    )
    def test_values_are_negated(self, value: float, expected: float) -> None:
        assert _negate_optional(value) == pytest.approx(expected)


class TestHeadroomRespondsToLoad:
    """Headroom must shrink as the house draws more, not grow."""

    @staticmethod
    def _limits(
        phase_w: tuple[float, float, float], desired_w: float
    ) -> PhaseChargeLimits:
        return compute_phase_charge_limits(
            measured_phase_power_w=phase_w,
            fuse_amps=FUSE_AMPS,
            desired_primary_charge_power_w=desired_w,
            primary_is_controlled=True,
            primary_actual_battery_power_w=0.0,
            primary_charge_efficiency_pct=100.0,
            primary_discharge_efficiency_pct=100.0,
            desired_secondary_charge_current_a=0.0,
            secondary_actual_site_delta_w=0.0,
            secondary_desired_noncharge_site_delta_w=0.0,
            secondary_grid_phase=3,
            secondary_nominal_voltage_v=48.0,
            secondary_charge_efficiency_pct=100.0,
            secondary_min_charge_current_a=10.0,
            secondary_max_charge_current_a=100.0,
            secondary_charge_current_step_a=10.0,
        )

    def test_heavy_import_clamps_a_large_request(self) -> None:
        """3 kW/phase of load leaves 680 W/phase, so 10 kW must be cut to ~2 kW."""
        meter_reading_w = -3000.0  # what the Huawei meter reports when importing
        normalised = _negate_optional(meter_reading_w)
        assert normalised is not None

        limits = self._limits((normalised, normalised, normalised), 10000.0)

        expected_w = 3 * (PHASE_LIMIT_W - 3000.0)  # 2040 W
        assert limits.primary_charge_power_w == pytest.approx(2000.0)  # 100 W step
        assert limits.primary_charge_power_w < expected_w

    def test_unnegated_reading_would_have_authorised_the_full_request(self) -> None:
        """Documents the defect: the raw meter value removes the clamp entirely."""
        limits = self._limits((-3000.0, -3000.0, -3000.0), 10000.0)

        assert limits.primary_charge_power_w == pytest.approx(10000.0)

    def test_export_grants_more_headroom_than_import(self) -> None:
        """The two directions must not be interchangeable."""
        importing = self._limits((2000.0, 2000.0, 2000.0), 10000.0)
        exporting = self._limits((-2000.0, -2000.0, -2000.0), 10000.0)

        assert exporting.primary_charge_power_w > importing.primary_charge_power_w

    def test_load_at_the_fuse_leaves_no_headroom(self) -> None:
        at_limit = self._limits((PHASE_LIMIT_W, PHASE_LIMIT_W, PHASE_LIMIT_W), 10000.0)

        assert at_limit.primary_charge_power_w == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Fallback when phase telemetry is unusable
# ---------------------------------------------------------------------------


def _config(*, phase_aware: bool) -> SensorConfig:
    cfg = SensorConfig()
    cfg.read_only = False
    cfg.phase_aware_charging_enabled = phase_aware
    cfg.main_fuse_amps = int(FUSE_AMPS)
    cfg.main_fuse_phases = 3
    cfg.huawei_solar_batteries_maximum_discharging_power = (
        "number.batteries_maximum_discharging_power"
    )
    cfg.huawei_solar_batteries_working_mode = "select.batteries_working_mode"
    cfg.huawei_solar_batteries_excess_pv_energy_use_in_tou = (
        "select.batteries_excess_pv"
    )
    cfg.huawei_solar_batteries_tou_charging_and_discharging_periods = "sensor.tou"
    cfg.huawei_solar_batteries_grid_charge_maximum_power = CHARGE_ENTITY
    cfg.huawei_solar_device_id_batteries = "battery-device-id"
    return cfg


def _live() -> LiveState:
    """A live state whose grid charge limit is disarmed and telemetry unusable."""
    live = LiveState()
    live._degraded_mode = DegradedMode.OK
    live.huawei_batteries_rated_capacity_wh = 30000.0
    live.huawei_batteries_max_discharge_power_w = 10000.0
    live.huawei_batteries_max_charge_power_w = 10000.0
    live.huawei_batteries_grid_charge_max_power_w = 0.0  # disarmed, needs re-arm
    live.huawei_batteries_working_mode = WorkingModes.TimeOfUse.value
    live.huawei_batteries_excess_pv_use_in_tou = "charge"
    live.huawei_batteries_forcible_charge_state = "Stopped"
    live.battery_current_capacity_kwh = 6.0
    live.grid_phase_power_w = (None, None, None)  # meter unreadable
    return live


def _recommendation(charged_kwh: float) -> HourlyRecommendation:
    start = datetime(2026, 8, 15, 5, 0, tzinfo=UTC)
    return HourlyRecommendation(
        start=start,
        end=start + timedelta(minutes=15),
        avg_house_consumption_kwh=0.25,
        avg_house_consumption_1d_kwh=0.25,
        avg_house_consumption_3d_kwh=0.25,
        avg_house_consumption_7d_kwh=0.25,
        avg_house_consumption_14d_kwh=0.25,
        batteries_charged_kwh=charged_kwh,
        batteries_discharged_kwh=0.0,
        estimated_battery_capacity_kwh=6.0,
        estimated_battery_soc_pct=25.0,
        estimated_cost_currency=0.0,
        estimated_net_consumption_kwh=0.25,
        export_price=0.16,
        grid_export_kwh=0.0,
        grid_import_kwh=0.25,
        import_price=1.024,
        recommendation=Recommendations.BatteriesChargeGrid.value,
        solcast_pv_estimate_kwh=0.0,
    )


def _ok(**kwargs: Any) -> ApplyResult:
    return ApplyResult(
        entity_id=kwargs["entity_id"],
        desired=kwargs["desired"],
        actual=kwargs["desired"],
        status=ApplyStatus.OK,
        attempts=1,
    )


async def _apply(
    cfg: SensorConfig, live: LiveState, rec: HourlyRecommendation
) -> list[tuple[str, object]]:
    sensor = MagicMock()
    sensor.hass = MagicMock()
    with patch(
        "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
        new_callable=AsyncMock,
        side_effect=_ok,
    ) as verifier:
        await async_apply_battery_settings(
            sensor, cfg, live, rec, current_required_battery_kwh=5.0, now=rec.start
        )
    return [
        (call.kwargs["entity_id"], call.kwargs["desired"])
        for call in verifier.await_args_list
    ]


class TestUnusableTelemetryFallback:
    """A meter dropout must fail closed, not re-arm the charge."""

    @staticmethod
    def _armed_live() -> LiveState:
        """A live state whose grid-charge limit is currently armed."""
        live = _live()
        live.huawei_batteries_grid_charge_max_power_w = 9600.0
        return live

    @pytest.mark.asyncio
    async def test_an_armed_limit_is_disarmed(self) -> None:
        """Clamping to planned power was not protection.

        The plan is per-phase constrained only when the phase imbalance is
        valid — the same condition the runtime limiter needs — and a
        full-power slot plans 2.5 kWh, i.e. the 10 kW hardware maximum.  So
        the clamp collapsed to no protection exactly when it mattered.
        """
        writes = await _apply(
            _config(phase_aware=True), self._armed_live(), _recommendation(2.5)
        )

        assert [d for e, d in writes if e == CHARGE_ENTITY] == [0.0]

    @pytest.mark.asyncio
    async def test_a_modest_planned_charge_is_blocked_too(self) -> None:
        """Not a cap on the plan — the charge is refused outright."""
        writes = await _apply(
            _config(phase_aware=True), self._armed_live(), _recommendation(0.4)
        )

        assert [d for e, d in writes if e == CHARGE_ENTITY] == [0.0]

    @pytest.mark.asyncio
    async def test_an_already_disarmed_limit_needs_no_write(self) -> None:
        """Idempotent: 0 W is already the safe value."""
        writes = await _apply(_config(phase_aware=True), _live(), _recommendation(2.5))

        assert [d for e, d in writes if e == CHARGE_ENTITY] == []

    @pytest.mark.asyncio
    async def test_phase_aware_disabled_still_restores_hardware_max(self) -> None:
        """Users without the feature keep the original re-arm behaviour."""
        writes = await _apply(_config(phase_aware=False), _live(), _recommendation(0.4))
        charge_writes = [d for e, d in writes if e == CHARGE_ENTITY]

        assert charge_writes == [10000.0]

    @pytest.mark.asyncio
    async def test_a_live_limit_still_wins_over_the_fallback(self) -> None:
        """When the limiter did produce a value, that value is used verbatim."""
        sensor = MagicMock()
        sensor.hass = MagicMock()
        with patch(
            "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
            new_callable=AsyncMock,
            side_effect=_ok,
        ) as verifier:
            rec = _recommendation(0.4)
            await async_apply_battery_settings(
                sensor,
                _config(phase_aware=True),
                _live(),
                rec,
                current_required_battery_kwh=5.0,
                grid_charge_power_limit_w=700.0,
                now=rec.start,
            )
        charge_writes = [
            call.kwargs["desired"]
            for call in verifier.await_args_list
            if call.kwargs["entity_id"] == CHARGE_ENTITY
        ]

        assert charge_writes == [700.0]
