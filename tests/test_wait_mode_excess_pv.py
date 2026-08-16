"""A wait slot must not export surplus PV nobody decided to sell.

Bug
---
``excess_pv_energy_use_in_tou`` was written from the recommendation *label*:
``BatteriesWaitMode`` mapped to ``fed_to_grid`` unless the wait-mode
self-consumption path happened to be active, which required the battery to
already hold energy above the planner's reserve.

The setting latches in hardware until the next coordinator cycle, so it is
applied from one snapshot and then governs however much surplus appears over
the following minutes.  That asymmetry is the defect:

    "charge" with no surplus      costs nothing
    "fed_to_grid" with surplus    sells it at the export price

Worse, the gate was inverted for this question.  ``surplus = capacity -
required`` decides whether the battery may *discharge* to serve the house;
using it to also decide whether the battery may *absorb* PV means the lower
the battery is relative to what it needs, the more eagerly HSEM exports.

Observed on the target installation: the 12:45 slot latched ``fed_to_grid`` at
a 179 W PV deficit, PV then climbed to 7.6 kW over seven minutes, and the
surplus left at an export price of 0.066 against an import price of 0.907 with
the battery at 44 %.  Across that day 0.419 kWh went out under ``fed_to_grid``
while the battery never rose above 46 %.

Fix
---
Exporting is the intent in exactly two modes, and ``apply_excess_export``
already prices the whole horizon in step 3 of ``run_planner`` and marks those
slots itself.  A plain wait slot is not an export decision, so it now keeps
``charge``.

``primary_battery_hold`` is deliberately still ``fed_to_grid``: that is a
validated MILP idle allocation, an explicit zero-energy decision for the
primary battery, and charging into it would execute energy absent from the
solved flow fields.

``test_wait_below_reserve_absorbs_surplus`` and
``test_strict_wait_absorbs_surplus`` fail against v6.2.2-powmr.31, which writes
``fed_to_grid`` in both cases.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.hsem.const import DEFAULT_HSEM_BATTERIES_WAIT_MODE
from custom_components.hsem.custom_sensors.applier import async_apply_battery_settings
from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.utils.degraded_mode import DegradedMode
from custom_components.hsem.utils.inverter_verify import ApplyResult, ApplyStatus
from custom_components.hsem.utils.recommendations import Recommendations
from custom_components.hsem.utils.workingmodes import WorkingModes

EXCESS_PV_ENTITY = "select.batteries_excess_pv"


def _cfg(wait_behavior: str = "self_consumption_with_reserve") -> SensorConfig:
    cfg = SensorConfig()
    cfg.read_only = False
    cfg.batteries_wait_mode_behavior = wait_behavior
    cfg.huawei_solar_batteries_maximum_discharging_power = (
        "number.batteries_max_discharge"
    )
    cfg.huawei_solar_batteries_working_mode = "select.batteries_working_mode"
    cfg.huawei_solar_batteries_excess_pv_energy_use_in_tou = EXCESS_PV_ENTITY
    return cfg


def _live(capacity_kwh: float = 6.0) -> LiveState:
    live = LiveState()
    live._degraded_mode = DegradedMode.OK
    live.huawei_batteries_rated_capacity_wh = 30000.0
    live.huawei_batteries_max_discharge_power_w = 10000.0
    live.huawei_batteries_working_mode = WorkingModes.MaximizeSelfConsumption.value
    live.huawei_batteries_excess_pv_use_in_tou = "fed_to_grid"
    live.huawei_batteries_forcible_charge_state = "Stopped"
    live.battery_current_capacity_kwh = capacity_kwh
    live.tou_periods.periods = list(DEFAULT_HSEM_BATTERIES_WAIT_MODE)
    return live


def _rec(recommendation: str, *, primary_battery_hold: bool) -> HourlyRecommendation:
    start = datetime(2026, 8, 16, 12, 45, tzinfo=UTC)
    return HourlyRecommendation(
        start=start,
        end=start + timedelta(minutes=15),
        avg_house_consumption_kwh=0.25,
        avg_house_consumption_1d_kwh=0.25,
        avg_house_consumption_3d_kwh=0.25,
        avg_house_consumption_7d_kwh=0.25,
        avg_house_consumption_14d_kwh=0.25,
        batteries_charged_kwh=0.0,
        batteries_discharged_kwh=0.0,
        estimated_battery_capacity_kwh=6.0,
        estimated_battery_soc_pct=44.0,
        estimated_cost_currency=0.0,
        estimated_net_consumption_kwh=0.25,
        export_price=0.066,
        grid_export_kwh=0.0,
        grid_import_kwh=0.0,
        import_price=0.907,
        recommendation=recommendation,
        solcast_pv_estimate_kwh=2.0,
        primary_battery_hold=primary_battery_hold,
    )


def _ok_write(**kwargs: Any) -> ApplyResult:
    return ApplyResult(
        entity_id=kwargs["entity_id"],
        desired=kwargs["desired"],
        actual=kwargs["desired"],
        status=ApplyStatus.OK,
        attempts=1,
    )


async def _excess_pv_write(
    cfg: SensorConfig,
    live: LiveState,
    rec: HourlyRecommendation,
    required_kwh: float,
) -> object | None:
    """Run the applier and return the effective excess-PV setting.

    The applier skips a write when the live value already matches what it
    wants, so "no write" means the live value stands rather than "no
    decision" — the fallback below reports what the hardware would be left on.
    """
    sensor = MagicMock()
    sensor.hass = MagicMock()
    with patch(
        "custom_components.hsem.custom_sensors.applier.async_write_and_verify",
        new_callable=AsyncMock,
        side_effect=_ok_write,
    ) as verifier:
        await async_apply_battery_settings(
            sensor,
            cfg,
            live,
            rec,
            current_required_battery_kwh=required_kwh,
            now=rec.start,
        )
    for call in verifier.await_args_list:
        if call.kwargs["entity_id"] == EXCESS_PV_ENTITY:
            written: object = call.kwargs["desired"]
            return written
    unchanged: object = live.huawei_batteries_excess_pv_use_in_tou
    return unchanged


class TestWaitSlotAbsorbsSurplus:
    """A wait slot keeps surplus PV rather than selling it."""

    @pytest.mark.asyncio
    async def test_wait_below_reserve_absorbs_surplus(self) -> None:
        """Below the reserve is exactly when the surplus should be kept.

        Fails against v31: the inverted gate sends it to the grid precisely
        when the battery is short of what the planner says it needs.
        """
        written = await _excess_pv_write(
            _cfg(),
            _live(capacity_kwh=6.0),
            _rec(Recommendations.BatteriesWaitMode.value, primary_battery_hold=False),
            required_kwh=9.0,
        )
        assert written == "charge"

    @pytest.mark.asyncio
    async def test_strict_wait_absorbs_surplus(self) -> None:
        """Strict wait keeps the battery idle, but idle is not the same as export.

        Fails against v31, which mapped every strict wait slot to fed_to_grid.
        """
        written = await _excess_pv_write(
            _cfg(wait_behavior="strict"),
            _live(capacity_kwh=6.0),
            _rec(Recommendations.BatteriesWaitMode.value, primary_battery_hold=False),
            required_kwh=9.0,
        )
        assert written == "charge"

    @pytest.mark.asyncio
    async def test_wait_above_reserve_still_absorbs(self) -> None:
        """The already-working case is unchanged."""
        written = await _excess_pv_write(
            _cfg(),
            _live(capacity_kwh=6.0),
            _rec(Recommendations.BatteriesWaitMode.value, primary_battery_hold=False),
            required_kwh=5.0,
        )
        assert written == "charge"


class TestExportIntentPreserved:
    """Only a real export decision still routes surplus to the grid."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "recommendation",
        [
            Recommendations.ForceExport.value,
            Recommendations.ForceBatteriesDischarge.value,
        ],
    )
    async def test_force_modes_still_feed_grid(self, recommendation: str) -> None:
        written = await _excess_pv_write(
            _cfg(),
            _live(),
            _rec(recommendation, primary_battery_hold=False),
            required_kwh=5.0,
        )
        assert written == "fed_to_grid"

    @pytest.mark.asyncio
    async def test_wait_with_primary_hold_still_feeds_grid(self) -> None:
        """A validated MILP idle allocation is an explicit zero-energy decision.

        Charging into it would execute energy absent from the solved flow
        fields — the plan-versus-command divergence class of v27/v28.
        """
        written = await _excess_pv_write(
            _cfg(),
            _live(),
            _rec(Recommendations.BatteriesWaitMode.value, primary_battery_hold=True),
            required_kwh=5.0,
        )
        assert written == "fed_to_grid"

    @pytest.mark.asyncio
    async def test_ev_relabelled_hold_still_feeds_grid(self) -> None:
        """engine_core relabels EV-carrying slots; the hold must still win."""
        live = _live()
        live.ev.is_charging = True
        live.ev.power_w = 7000.0
        live.net_consumption_w = 800.0
        written = await _excess_pv_write(
            _cfg(),
            live,
            _rec(Recommendations.EVSmartCharging.value, primary_battery_hold=True),
            required_kwh=5.0,
        )
        assert written == "fed_to_grid"


class TestOtherModesUnchanged:
    """Charging and discharging modes keep absorbing surplus as before."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "recommendation",
        [
            Recommendations.BatteriesChargeSolar.value,
            Recommendations.BatteriesDischargeMode.value,
        ],
    )
    async def test_normal_modes_charge(self, recommendation: str) -> None:
        written = await _excess_pv_write(
            _cfg(),
            _live(),
            _rec(recommendation, primary_battery_hold=False),
            required_kwh=5.0,
        )
        assert written == "charge"
