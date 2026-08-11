"""Tests for topology-aware dedicated-load secondary storage."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.secondary_storage_config import (
    SecondaryStorageConfig,
)
from custom_components.hsem.planner.milp_optimizer import (
    is_scipy_available,
    solve_milp,
)
from custom_components.hsem.planner.secondary_storage import (
    SECONDARY_MODE_CHARGE,
    SECONDARY_MODE_SBU,
)
from custom_components.hsem.utils.prices import SlotPrice

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
) -> tuple[list[PlannedSlot], dict]:
    """Solve a small deterministic secondary-storage scenario."""
    result = solve_milp(
        slots,
        _NOW,
        current_kwh=primary_current_kwh,
        usable_kwh=9.0,
        max_charge_per_slot=2.0,
        max_discharge_per_slot=2.0,
        charge_efficiency_pct=97.0,
        discharge_efficiency_pct=97.0,
        time_discount_rate=1.0,
        no_export=True,
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
