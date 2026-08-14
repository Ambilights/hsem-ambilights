"""Tests for topology-aware dedicated-load secondary storage."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.secondary_storage_config import (
    SecondaryStorageConfig,
)
from custom_components.hsem.planner import milp_optimizer
from custom_components.hsem.planner.cost_function import CostWeights, score_plan
from custom_components.hsem.planner.milp import _secondary_diagnostics
from custom_components.hsem.planner.milp._secondary_diagnostics import (
    SecondaryResultSummary,
)
from custom_components.hsem.planner.milp._secondary_storage import (
    _allocate_secondary_variables,
    _write_secondary_results,
)
from custom_components.hsem.planner.milp_optimizer import (
    is_scipy_available,
    solve_milp,
)
from custom_components.hsem.planner.secondary_storage import (
    SECONDARY_MODE_CHARGE,
    SECONDARY_MODE_SBU,
    SECONDARY_MODE_UTILITY,
)
from custom_components.hsem.utils.prices import SlotPrice
from custom_components.hsem.utils.recommendations import Recommendations

_NOW = datetime(2026, 8, 11, 0, 0, tzinfo=ZoneInfo("Europe/Stockholm"))

pytestmark = pytest.mark.skipif(
    not is_scipy_available(),
    reason="scipy not available in this environment",
)


def _slots(
    prices: list[float],
    *,
    house_load_kwh: float = 0.0,
) -> list[PlannedSlot]:
    """Return hourly slots with no PV and the requested import prices."""
    return [
        PlannedSlot(
            start=_NOW + timedelta(hours=index),
            end=_NOW + timedelta(hours=index + 1),
            price=SlotPrice(import_price=price, export_price=0.0),
            avg_house_consumption_kwh=house_load_kwh,
        )
        for index, price in enumerate(prices)
    ]


def _quarter_hour_slots(prices: list[float]) -> list[PlannedSlot]:
    """Return production-width 15-minute slots for horizon-size tests."""
    return [
        PlannedSlot(
            start=_NOW + timedelta(minutes=15 * index),
            end=_NOW + timedelta(minutes=15 * (index + 1)),
            price=SlotPrice(import_price=price, export_price=0.0),
            avg_house_consumption_kwh=0.025,
        )
        for index, price in enumerate(prices)
    ]


def _powmr(**overrides: float | bool | None) -> SecondaryStorageConfig:
    """Return the known PowMr/NAS topology used by the fork."""
    values: dict = {
        "enabled": True,
        "capacity_kwh": 15.0,
        "current_soc_pct": 20.0,
        "min_soc_pct": 20.0,
        "max_soc_pct": 100.0,
        "nominal_voltage_v": 24.0,
        "load_power_w": 100.0,
        "max_charge_current_a": 60.0,
        "min_charge_current_a": 10.0,
        "charge_current_step_a": 10.0,
        "charge_efficiency_pct": 93.0,
        "discharge_efficiency_pct": 93.0,
        "inverter_standby_power_w": 55.0,
        "cycle_cost_per_kwh": 0.0,
        "replacement_price_per_kwh": 0.40,
        "base_load_includes_dedicated_load": False,
        "allow_primary_battery_transfer": False,
    }
    values.update(overrides)
    return SecondaryStorageConfig(**values)


def _solve(
    slots: list[PlannedSlot],
    secondary: SecondaryStorageConfig,
    *,
    primary_current_kwh: float = 0.0,
    no_export: bool = True,
    min_export_price: float = 0.0,
    battery_export_min_price: float = 0.0,
    primary_max_discharge_per_slot: float = 2.0,
) -> tuple[list[PlannedSlot], dict]:
    """Solve a small deterministic secondary-storage scenario."""
    result = solve_milp(
        slots,
        _NOW,
        current_kwh=primary_current_kwh,
        usable_kwh=9.0,
        max_charge_per_slot=2.0,
        max_discharge_per_slot=primary_max_discharge_per_slot,
        charge_efficiency_pct=97.0,
        discharge_efficiency_pct=97.0,
        time_discount_rate=1.0,
        no_export=no_export,
        min_export_price=min_export_price,
        battery_export_min_price=battery_export_min_price,
        secondary_storage=secondary,
    )
    assert result is not None
    return result


def test_secondary_charges_cheap_and_serves_nas_when_expensive() -> None:
    """The battery should charge cheaply and use SBU during expensive slots."""
    result, diagnostics = _solve(
        _slots([0.05, 0.05, 1.00, 1.00, 1.00, 1.00]),
        _powmr(),
    )

    charge_hours = {
        slot.start.hour
        for slot in result
        if slot.secondary_storage_mode == SECONDARY_MODE_CHARGE
    }
    sbu_hours = {
        slot.start.hour
        for slot in result
        if slot.secondary_storage_mode == SECONDARY_MODE_SBU
    }

    assert charge_hours & {0, 1}
    assert sbu_hours & {2, 3, 4, 5}
    assert diagnostics["secondary_charge_slots"] > 0
    assert diagnostics["secondary_sbu_slots"] > 0
    assert all(
        slot.secondary_storage_charge_current_a % 10.0 == pytest.approx(0.0)
        for slot in result
        if slot.secondary_storage_mode == SECONDARY_MODE_CHARGE
    )


def test_secondary_never_exports_or_over_supplies_dedicated_load() -> None:
    """SBU discharge must equal NAS demand plus losses, never grid export."""
    config = _powmr(current_soc_pct=80.0)
    result, _diagnostics = _solve(_slots([1.0, 1.0, 1.0]), config)

    sbu_slots = [
        slot for slot in result if slot.secondary_storage_mode == SECONDARY_MODE_SBU
    ]
    assert sbu_slots
    expected_draw = 0.100 / 0.93 + 0.055
    for slot in sbu_slots:
        assert slot.secondary_storage_discharged_kwh == pytest.approx(
            expected_draw,
            abs=0.001,
        )
        assert slot.secondary_storage_grid_import_kwh == pytest.approx(0.0)
        assert slot.grid_export_kwh == pytest.approx(0.0)


def test_secondary_soc_never_crosses_twenty_percent_reserve() -> None:
    """The hard state equation must preserve the configured 3 kWh reserve."""
    result, _diagnostics = _solve(
        _slots([1.0] * 12),
        _powmr(current_soc_pct=20.0, replacement_price_per_kwh=0.0),
    )

    for slot in result:
        assert slot.secondary_storage_estimated_soc_pct >= 20.0


def test_secondary_rejects_impossible_initial_soc() -> None:
    """Pure planner inputs outside the physical 0-100% range are invalid."""
    assert _powmr(current_soc_pct=150.0).valid is False


def test_secondary_utility_load_is_counted_once_in_site_import() -> None:
    """The extended MILP bus already contains PowMr load; output must not re-add it."""
    result, _diagnostics = _solve(
        _slots([1.0]),
        _powmr(
            current_soc_pct=20.0,
            replacement_price_per_kwh=0.0,
        ),
    )

    assert result[0].secondary_storage_mode == "utility"
    assert result[0].secondary_storage_grid_import_kwh == pytest.approx(0.100)
    assert result[0].grid_import_kwh == pytest.approx(0.100)


def test_secondary_sbu_removes_included_load_from_site_import() -> None:
    """With the default history topology, SBU removes exactly the NAS load."""
    result, _diagnostics = _solve(
        _slots([1.0], house_load_kwh=0.100),
        _powmr(
            current_soc_pct=80.0,
            base_load_includes_dedicated_load=True,
        ),
    )

    assert result[0].secondary_storage_mode == SECONDARY_MODE_SBU
    assert result[0].secondary_storage_grid_import_kwh == pytest.approx(0.0)
    assert result[0].grid_import_kwh == pytest.approx(0.0)
    assert result[0].grid_export_kwh == pytest.approx(0.0)


def _apply_sbu_to_primary_slot(
    slot: PlannedSlot,
    *,
    battery_export_min_price: float,
    primary_site_discharge_limited: bool = False,
) -> dict[str, float | int] | None:
    """Apply one deterministic SBU write-out to an existing primary slot."""
    config = _powmr(
        current_soc_pct=80.0,
        base_load_includes_dedicated_load=True,
    )
    slot.secondary_storage_load_kwh = 0.100
    layout, n_vars = _allocate_secondary_variables(0, 1)
    result_x = np.zeros(n_vars)
    result_x[layout["discharge"]] = 0.100 / 0.93 + 0.055
    result_x[layout["sbu_mode"]] = 1.0

    return _write_secondary_results(
        [slot],
        result_x=result_x,
        layout=layout,
        config=config,
        future_idx=[0],
        minimum_action_kwh=1e-4,
        battery_export_min_price=battery_export_min_price,
        primary_site_discharge_limited=np.asarray(
            [primary_site_discharge_limited], dtype=bool
        ),
    )


def test_sbu_relabels_zero_import_primary_charge_as_solar() -> None:
    """SBU load removal must not leave a zero-import forced-grid label."""
    slot = _slots([1.0], house_load_kwh=0.100)[0]
    slot.recommendation = Recommendations.BatteriesChargeGrid.value
    slot.batteries_charged_kwh = 0.400
    slot.grid_import_kwh = 0.100
    original_charge = slot.batteries_charged_kwh

    _apply_sbu_to_primary_slot(slot, battery_export_min_price=0.50)

    assert slot.grid_import_kwh == pytest.approx(0.0)
    assert slot.grid_export_kwh == pytest.approx(0.0)
    assert slot.recommendation == Recommendations.BatteriesChargeSolar.value
    assert slot.batteries_charged_kwh == original_charge


def test_sbu_one_wh_import_residue_relabels_primary_charge_as_solar() -> None:
    """Post-SBU 0.001 kWh import residue must not force grid/TOU mode."""
    slot = _slots([1.0], house_load_kwh=0.100)[0]
    slot.recommendation = Recommendations.BatteriesChargeGrid.value
    slot.batteries_charged_kwh = 0.400
    slot.grid_import_kwh = 0.101

    _apply_sbu_to_primary_slot(slot, battery_export_min_price=0.50)

    assert slot.grid_import_kwh == pytest.approx(0.001)
    assert slot.recommendation == Recommendations.BatteriesChargeSolar.value


def test_sbu_two_wh_import_keeps_primary_grid_charge() -> None:
    """Post-SBU import above publication residue must keep grid/TOU mode."""
    slot = _slots([1.0], house_load_kwh=0.100)[0]
    slot.recommendation = Recommendations.BatteriesChargeGrid.value
    slot.batteries_charged_kwh = 0.400
    slot.grid_import_kwh = 0.102

    _apply_sbu_to_primary_slot(slot, battery_export_min_price=0.50)

    assert slot.grid_import_kwh == pytest.approx(0.002)
    assert slot.recommendation == Recommendations.BatteriesChargeGrid.value


def test_sbu_relabels_primary_discharge_export_as_force() -> None:
    """Allowed SBU-created export must use an executable Huawei mode."""
    slot = _slots([1.0], house_load_kwh=0.100)[0]
    slot.price = SlotPrice(import_price=1.0, export_price=1.00)
    slot.recommendation = Recommendations.BatteriesDischargeMode.value
    slot.batteries_discharged_kwh = 0.400
    original_discharge = slot.batteries_discharged_kwh

    result = _apply_sbu_to_primary_slot(slot, battery_export_min_price=0.50)

    assert result is not None
    assert slot.grid_import_kwh == pytest.approx(0.0)
    assert slot.grid_export_kwh == pytest.approx(0.100)
    assert slot.recommendation == Recommendations.ForceBatteriesDischarge.value
    assert slot.batteries_discharged_kwh == original_discharge


def test_sbu_one_wh_export_residue_keeps_self_consumption() -> None:
    """Post-SBU 0.001 kWh export residue must not enable Fully Fed mode."""
    slot = _slots([1.0], house_load_kwh=0.100)[0]
    slot.price = SlotPrice(import_price=1.0, export_price=1.00)
    slot.recommendation = Recommendations.BatteriesDischargeMode.value
    slot.batteries_discharged_kwh = 0.400
    slot.grid_import_kwh = 0.099

    result = _apply_sbu_to_primary_slot(slot, battery_export_min_price=0.50)

    assert result is not None
    assert slot.grid_export_kwh == pytest.approx(0.001)
    assert slot.recommendation == Recommendations.BatteriesDischargeMode.value


def test_sbu_two_wh_export_uses_forced_discharge() -> None:
    """Post-SBU export above publication residue must enable Fully Fed mode."""
    slot = _slots([1.0], house_load_kwh=0.100)[0]
    slot.price = SlotPrice(import_price=1.0, export_price=1.00)
    slot.recommendation = Recommendations.BatteriesDischargeMode.value
    slot.batteries_discharged_kwh = 0.400
    slot.grid_import_kwh = 0.098

    result = _apply_sbu_to_primary_slot(slot, battery_export_min_price=0.50)

    assert result is not None
    assert slot.grid_export_kwh == pytest.approx(0.002)
    assert slot.recommendation == Recommendations.ForceBatteriesDischarge.value


def test_sbu_rejects_material_export_in_site_limited_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blocked export is an invalid candidate, not a cosmetic BDM label."""
    messages: list[tuple] = []
    monkeypatch.setattr(
        "custom_components.hsem.planner.milp._secondary_storage.log_planner",
        lambda *args: messages.append(args),
    )
    slot = _slots([1.0], house_load_kwh=0.100)[0]
    slot.price = SlotPrice(import_price=1.0, export_price=0.49)
    slot.recommendation = Recommendations.BatteriesDischargeMode.value
    slot.batteries_discharged_kwh = 0.400

    result = _apply_sbu_to_primary_slot(
        slot,
        battery_export_min_price=0.50,
        primary_site_discharge_limited=True,
    )

    assert result is None
    assert messages
    assert messages[0][0] == "warning"
    assert "invariant failed" in messages[0][1]


@pytest.mark.parametrize(
    ("no_export", "min_export_price", "battery_export_min_price"),
    [
        (True, 0.0, 0.0),
        (False, 0.50, 0.0),
        (False, 0.0, 0.50),
    ],
)
def test_sbu_and_primary_discharge_cannot_create_blocked_export(
    no_export: bool,
    min_export_price: float,
    battery_export_min_price: float,
) -> None:
    """The solver must cap Huawei against the load SBU removes."""
    slot = _slots([10.0], house_load_kwh=0.200)[0]
    slot.price = SlotPrice(import_price=10.0, export_price=0.40)

    planned, _diagnostics = _solve(
        [slot],
        _powmr(
            current_soc_pct=80.0,
            base_load_includes_dedicated_load=True,
            replacement_price_per_kwh=0.0,
            inverter_standby_power_w=0.0,
            discharge_efficiency_pct=100.0,
        ),
        primary_current_kwh=1.0,
        no_export=no_export,
        min_export_price=min_export_price,
        battery_export_min_price=battery_export_min_price,
        primary_max_discharge_per_slot=0.103,
    )

    solved = planned[0]
    assert solved.secondary_storage_mode == SECONDARY_MODE_SBU
    assert solved.batteries_discharged_kwh > 0.0
    # House load is 0.200 kWh and SBU removes the included 0.100 kWh NAS
    # load, leaving exactly 0.100 kWh AC that Huawei may serve without export.
    assert solved.batteries_discharged_kwh * 0.97 == pytest.approx(
        0.100,
        abs=0.001,
    )
    assert solved.grid_import_kwh == pytest.approx(0.0)
    assert solved.grid_export_kwh == pytest.approx(0.0)
    assert solved.recommendation == Recommendations.BatteriesDischargeMode.value


def test_site_limited_pv_export_with_utility_mode_is_not_rejected() -> None:
    """PV-only export remains valid when Huawei contributes no discharge."""
    slot = _slots([1.0], house_load_kwh=0.100)[0]
    slot.solcast_pv_estimate_kwh = 0.300
    slot.price = SlotPrice(import_price=1.0, export_price=0.40)

    planned, _diagnostics = _solve(
        [slot],
        _powmr(
            current_soc_pct=100.0,
            base_load_includes_dedicated_load=True,
            replacement_price_per_kwh=10.0,
        ),
        primary_current_kwh=9.0,
        no_export=True,
    )

    solved = planned[0]
    assert solved.secondary_storage_mode == SECONDARY_MODE_UTILITY
    assert solved.batteries_discharged_kwh == pytest.approx(0.0)
    assert solved.secondary_storage_discharged_kwh == pytest.approx(0.0)
    assert solved.grid_export_kwh == pytest.approx(0.200)


def test_profitable_sbu_can_reveal_pv_export_without_primary_discharge() -> None:
    """SBU may remove only the demand present and reveal surplus PV."""
    slot = _slots([10.0], house_load_kwh=0.100)[0]
    slot.solcast_pv_estimate_kwh = 0.050
    slot.price = SlotPrice(import_price=10.0, export_price=0.40)

    planned, _diagnostics = _solve(
        [slot],
        _powmr(
            current_soc_pct=80.0,
            base_load_includes_dedicated_load=True,
            replacement_price_per_kwh=0.0,
            inverter_standby_power_w=0.0,
            discharge_efficiency_pct=100.0,
        ),
        primary_current_kwh=0.0,
        no_export=True,
    )

    solved = planned[0]
    assert solved.secondary_storage_mode == SECONDARY_MODE_SBU
    assert solved.batteries_discharged_kwh == pytest.approx(0.0)
    assert solved.grid_import_kwh == pytest.approx(0.0)
    assert solved.grid_export_kwh == pytest.approx(0.050)


def test_mixed_history_cannot_model_powmr_backfeed() -> None:
    """SBU may subtract only the dedicated load present in the house forecast."""
    result, _diagnostics = _solve(
        _slots([1.0], house_load_kwh=0.050),
        _powmr(
            current_soc_pct=80.0,
            base_load_includes_dedicated_load=True,
        ),
    )

    assert result[0].secondary_storage_mode == SECONDARY_MODE_SBU
    assert result[0].grid_import_kwh == pytest.approx(0.0)
    assert result[0].grid_export_kwh == pytest.approx(0.0)


def test_secondary_charge_blocks_huawei_discharge_transfer() -> None:
    """Huawei discharge must be zero whenever PowMr utility charging is active."""
    result, _diagnostics = _solve(
        _slots([0.05, 0.05, 1.00, 1.00], house_load_kwh=1.0),
        _powmr(),
        primary_current_kwh=5.0,
    )

    charge_slots = [
        slot for slot in result if slot.secondary_storage_mode == SECONDARY_MODE_CHARGE
    ]
    assert charge_slots
    for slot in charge_slots:
        assert slot.batteries_discharged_kwh == pytest.approx(0.0)


def test_secondary_solves_full_192_slot_horizon() -> None:
    """The integer mode/current extension must solve a production-size horizon."""
    prices = [0.05] * 48 + [1.0] * 48 + [0.10] * 48 + [0.80] * 48

    result, diagnostics = _solve(
        _quarter_hour_slots(prices),
        _powmr(base_load_includes_dedicated_load=True),
    )

    assert len(result) == 192
    assert diagnostics["secondary_charge_slots"] > 0
    assert diagnostics["secondary_sbu_slots"] > 0


def test_secondary_and_primary_export_reserve_share_one_milp() -> None:
    """PowMr integer variables and the primary export binary must coexist."""
    slots = _slots([5.0, 10.0, 10.0, 10.0])
    slots[0].price = SlotPrice(import_price=5.0, export_price=5.0)

    solved = solve_milp(
        slots,
        _NOW,
        current_kwh=9.0,
        usable_kwh=9.0,
        max_charge_per_slot=2.0,
        max_discharge_per_slot=2.0,
        charge_efficiency_pct=97.0,
        discharge_efficiency_pct=97.0,
        time_discount_rate=1.0,
        excess_export_discharge_buffer_pct=15.0,
        secondary_storage=_powmr(),
    )

    assert solved is not None
    result, diagnostics = solved
    assert len(result) == len(slots)
    assert diagnostics["battery_export_reserve_active"] is True
    assert diagnostics["battery_export_reserve_slots"] >= 1
    reserve_shortfall = max(
        diagnostics["battery_export_reserve_kwh"]
        - diagnostics["battery_export_reserve_min_checkpoint_soc_kwh"],
        0.0,
    )
    assert reserve_shortfall == pytest.approx(0.0, abs=1e-6)
    assert "secondary_result" in diagnostics


def test_disabled_secondary_preserves_existing_solution() -> None:
    """A disabled config must be numerically identical to no secondary config."""
    slots = _slots([0.05, 0.50, 1.00], house_load_kwh=0.4)
    without_secondary = solve_milp(
        slots,
        _NOW,
        current_kwh=2.0,
        usable_kwh=9.0,
        max_charge_per_slot=2.0,
        max_discharge_per_slot=2.0,
        time_discount_rate=1.0,
        no_export=True,
    )
    disabled_secondary = solve_milp(
        slots,
        _NOW,
        current_kwh=2.0,
        usable_kwh=9.0,
        max_charge_per_slot=2.0,
        max_discharge_per_slot=2.0,
        time_discount_rate=1.0,
        no_export=True,
        secondary_storage=SecondaryStorageConfig(enabled=False),
    )

    assert without_secondary is not None
    assert disabled_secondary is not None
    plain_slots, _plain_diag = without_secondary
    disabled_slots, _disabled_diag = disabled_secondary
    assert [
        (
            slot.recommendation,
            slot.batteries_charged_kwh,
            slot.batteries_discharged_kwh,
            slot.grid_import_kwh,
            slot.grid_export_kwh,
        )
        for slot in plain_slots
    ] == [
        (
            slot.recommendation,
            slot.batteries_charged_kwh,
            slot.batteries_discharged_kwh,
            slot.grid_import_kwh,
            slot.grid_export_kwh,
        )
        for slot in disabled_slots
    ]


def test_secondary_result_log_has_one_line_per_successful_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One enabled successful solve must emit one compact aggregate line."""
    solve_lines: list[str] = []
    result_lines: list[str] = []

    def capture_solve(_level: str, message: str, *args: object) -> None:
        rendered = message % args
        if "[milp] solve_milp" in rendered:
            solve_lines.append(rendered)

    def capture_result(_level: str, message: str, *args: object) -> None:
        rendered = message % args
        if "[milp] secondary_result" in rendered:
            result_lines.append(rendered)

    monkeypatch.setattr(milp_optimizer, "log_planner", capture_solve)
    monkeypatch.setattr(_secondary_diagnostics, "log_planner", capture_result)

    _solve(
        _slots([0.05, 0.05, 1.00, 1.00]),
        _powmr(),
    )

    assert len(solve_lines) == 1
    assert len(result_lines) == 1
    line = result_lines[0]
    assert "sbu_slots=" in line
    assert "charge_slots=" in line
    assert "utility_slots=" in line
    assert "terminal_credit=" in line
    assert "reason=scheduled" in line
    assert "\n" not in line


def test_secondary_result_log_absent_when_secondary_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inactive secondary model must not emit a zero-valued result line."""
    result_lines: list[str] = []

    def capture_result(_level: str, message: str, *args: object) -> None:
        rendered = message % args
        if "[milp] secondary_result" in rendered:
            result_lines.append(rendered)

    monkeypatch.setattr(_secondary_diagnostics, "log_planner", capture_result)
    result = solve_milp(
        _slots([0.05, 1.00]),
        _NOW,
        current_kwh=0.0,
        usable_kwh=9.0,
        max_charge_per_slot=2.0,
        max_discharge_per_slot=2.0,
        no_export=True,
        secondary_storage=SecondaryStorageConfig(enabled=False),
    )

    assert result is not None
    assert result_lines == []


def test_secondary_result_costs_match_authoritative_scorer() -> None:
    """Logged loss, wear, and terminal terms must equal ``score_plan``."""
    config = _powmr(cycle_cost_per_kwh=0.05)
    planned, diagnostics = _solve(
        _slots([0.05, 0.05, 1.00, 1.00]),
        config,
    )
    breakdown = score_plan(
        planned,
        CostWeights(
            secondary_storage_enabled=True,
            secondary_storage_charge_efficiency_pct=(config.charge_efficiency_pct),
            secondary_storage_discharge_efficiency_pct=(
                config.discharge_efficiency_pct
            ),
            secondary_storage_cycle_cost_per_kwh=config.cycle_cost_per_kwh,
            secondary_storage_replacement_price_per_kwh=(
                config.replacement_price_per_kwh
            ),
        ),
        now=_NOW,
    )
    summary = diagnostics["secondary_result"]

    assert summary["conversion_loss"] == pytest.approx(
        breakdown.secondary_conversion_loss_cost,
        abs=1e-6,
    )
    assert summary["cycle_cost"] == pytest.approx(
        breakdown.secondary_cycle_cost,
        abs=1e-6,
    )
    assert summary["terminal_credit"] == pytest.approx(
        -breakdown.secondary_terminal_soc_value,
        abs=1e-6,
    )
    assert summary["net"] == pytest.approx(
        summary["sbu_saving"]
        - summary["charge_cost"]
        - summary["cycle_cost"]
        - summary["conversion_loss"]
        + summary["terminal_credit"],
    )
    assert summary["sbu_slots"] + summary["charge_slots"] + summary[
        "utility_slots"
    ] == len(planned)


def test_secondary_result_summary_does_not_change_solved_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacing diagnostics with a no-op summary must leave slots identical."""
    slots = _slots([0.05, 0.05, 1.00, 1.00])
    config = _powmr(cycle_cost_per_kwh=0.05)
    with_summary, _diagnostics = _solve(slots, config)

    dummy = SecondaryResultSummary(
        sbu_slots=0,
        charge_slots=0,
        utility_slots=0,
        sbu_energy_kwh=0.0,
        charge_energy_kwh=0.0,
        sbu_saving=0.0,
        charge_cost=0.0,
        cycle_cost=0.0,
        conversion_loss=0.0,
        terminal_credit=0.0,
        net=0.0,
        soc_start_pct=0.0,
        soc_end_pct=0.0,
        reason="unknown",
    )
    monkeypatch.setattr(
        _secondary_diagnostics,
        "build_secondary_result_summary",
        lambda *_args, **_kwargs: dummy,
    )
    monkeypatch.setattr(_secondary_diagnostics, "log_secondary_result", lambda _s: None)

    without_summary, _diagnostics = _solve(slots, config)

    assert [asdict(slot) for slot in with_summary] == [
        asdict(slot) for slot in without_summary
    ]


def test_parked_secondary_identifies_terminal_value_as_dominant() -> None:
    """A terminal penalty that blocks otherwise useful SBU is explained."""
    planned, diagnostics = _solve(
        _slots([1.00, 1.00, 1.00], house_load_kwh=0.10),
        _powmr(
            current_soc_pct=80.0,
            cycle_cost_per_kwh=0.05,
            replacement_price_per_kwh=2.0,
            base_load_includes_dedicated_load=True,
        ),
    )

    assert all(slot.secondary_storage_mode == "utility" for slot in planned)
    assert diagnostics["secondary_result"]["reason"] == "terminal_credit_wins"
