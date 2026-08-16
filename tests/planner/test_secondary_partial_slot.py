"""Hand-calculated PowMr regressions for a partially elapsed current slot."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from inspect import signature
from typing import cast

import pytest

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.secondary_storage_config import (
    SecondaryStorageConfig,
)
from custom_components.hsem.planner.milp._secondary_diagnostics import (
    _minimum_secondary_discharge_kwh,
)
from custom_components.hsem.planner.milp_optimizer import (
    is_scipy_available,
    solve_milp,
)
from custom_components.hsem.planner.secondary_storage import (
    SECONDARY_MODE_CHARGE,
    SECONDARY_MODE_SBU,
    apply_secondary_utility_bypass,
)
from custom_components.hsem.utils.prices import SlotPrice

_SLOT_START = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
_NOW = _SLOT_START + timedelta(minutes=10)
_CURRENT_HOURS = 5.0 / 60.0
_FULL_HOURS = 15.0 / 60.0

pytestmark = pytest.mark.skipif(
    not is_scipy_available(),
    reason="scipy not available in this environment",
)


def _slots(import_price: float) -> list[PlannedSlot]:
    """Return one five-minute remainder followed by one full 15-minute slot."""
    return [
        PlannedSlot(
            start=_SLOT_START + timedelta(minutes=15 * index),
            end=_SLOT_START + timedelta(minutes=15 * (index + 1)),
            price=SlotPrice(import_price=import_price, export_price=0.0),
        )
        for index in range(2)
    ]


def _config(**overrides: float | bool | None) -> SecondaryStorageConfig:
    """Return a valid 15 kWh PowMr configuration for exact arithmetic."""
    values: dict[str, float | bool | None] = {
        "enabled": True,
        "capacity_kwh": 15.0,
        "current_soc_pct": 20.0,
        "min_soc_pct": 20.0,
        "max_soc_pct": 100.0,
        "nominal_voltage_v": 24.0,
        "load_power_w": 0.0,
        "max_charge_current_a": 20.0,
        "min_charge_current_a": 10.0,
        "charge_current_step_a": 10.0,
        "charge_efficiency_pct": 100.0,
        "discharge_efficiency_pct": 100.0,
        "inverter_standby_power_w": 0.0,
        "cycle_cost_per_kwh": 0.0,
        "replacement_price_per_kwh": 0.0,
        "base_load_includes_dedicated_load": False,
        "allow_primary_battery_transfer": False,
        "grid_phase": 3,
    }
    values.update(overrides)
    return SecondaryStorageConfig(**values)  # type: ignore[arg-type]


def _solve(
    slots: list[PlannedSlot],
    config: SecondaryStorageConfig,
) -> tuple[list[PlannedSlot], dict]:
    """Solve with an empty primary battery so only PowMr serves its branch."""
    result = solve_milp(
        slots,
        _NOW,
        current_kwh=0.0,
        usable_kwh=9.0,
        max_charge_per_slot=1.0,
        max_discharge_per_slot=1.0,
        charge_efficiency_pct=100.0,
        discharge_efficiency_pct=100.0,
        time_discount_rate=1.0,
        no_export=True,
        secondary_storage=config,
    )
    assert result is not None
    return result


def test_utility_bypass_uses_only_current_slot_remainder() -> None:
    """600 W uses 0.050 kWh in five minutes, then 0.150 kWh in 15."""
    slots = _slots(import_price=2.0)
    for slot in slots:
        slot.grid_import_kwh = 0.200

    apply_secondary_utility_bypass(
        slots,
        _config(load_power_w=600.0, current_soc_pct=50.0),
        _NOW,
    )

    assert [slot.secondary_storage_load_kwh for slot in slots] == pytest.approx(
        [0.050, 0.150]
    )
    assert [slot.secondary_storage_grid_import_kwh for slot in slots] == pytest.approx(
        [0.050, 0.150]
    )
    assert [slot.grid_import_kwh for slot in slots] == pytest.approx([0.250, 0.350])
    assert [slot.estimated_cost_currency for slot in slots] == pytest.approx(
        [0.500, 0.700]
    )
    assert [slot.secondary_storage_estimated_capacity_kwh for slot in slots] == (
        pytest.approx([4.500, 4.500])
    )
    assert [slot.secondary_storage_estimated_soc_pct for slot in slots] == (
        pytest.approx([50.0, 50.0])
    )


def test_milp_sbu_scales_load_overhead_grid_flow_and_soc() -> None:
    """SBU uses five minutes now and the full duration only in the next slot.

    At 600 W load, 80 % discharge efficiency, and 120 W inverter overhead:
    current draw = 0.050 / 0.80 + 0.120 * 5/60 = 0.0725 kWh;
    next draw = 0.150 / 0.80 + 0.120 * 15/60 = 0.2175 kWh.
    """
    planned, diagnostics = _solve(
        _slots(import_price=10.0),
        _config(
            current_soc_pct=60.0,
            load_power_w=600.0,
            discharge_efficiency_pct=80.0,
            inverter_standby_power_w=120.0,
        ),
    )

    assert [slot.secondary_storage_mode for slot in planned] == [
        SECONDARY_MODE_SBU,
        SECONDARY_MODE_SBU,
    ]
    assert [slot.secondary_storage_load_kwh for slot in planned] == pytest.approx(
        [0.050, 0.150]
    )
    assert [slot.secondary_storage_discharged_kwh for slot in planned] == (
        pytest.approx([0.072, 0.217])
    )
    assert [slot.secondary_storage_grid_import_kwh for slot in planned] == (
        pytest.approx([0.0, 0.0])
    )
    assert [slot.grid_import_kwh for slot in planned] == pytest.approx([0.0, 0.0])
    assert [slot.grid_export_kwh for slot in planned] == pytest.approx([0.0, 0.0])
    assert [slot.secondary_storage_estimated_capacity_kwh for slot in planned] == (
        pytest.approx([5.9275, 5.7100], abs=0.0005)
    )
    assert [slot.secondary_storage_estimated_soc_pct for slot in planned] == (
        pytest.approx([59.5167, 58.0667], abs=0.005)
    )
    assert diagnostics["secondary_total_discharged_kwh"] == pytest.approx(0.290)


def test_secondary_diagnostics_use_current_slot_remainder() -> None:
    """The parked explanation measures standby draw over five minutes, not 15."""
    slots = _slots(import_price=10.0)
    config = _config(
        load_power_w=600.0,
        discharge_efficiency_pct=80.0,
        inverter_standby_power_w=120.0,
    )
    apply_secondary_utility_bypass(slots, config, _NOW)

    if "now" in signature(_minimum_secondary_discharge_kwh).parameters:
        current_draw = _minimum_secondary_discharge_kwh(slots[0], config, _NOW)
        future_draw = _minimum_secondary_discharge_kwh(slots[1], config, _NOW)
    else:  # pragma: no cover - exercised only against the published .34 code
        legacy_draw = cast(Callable[..., float], _minimum_secondary_discharge_kwh)
        current_draw = legacy_draw(slots[0], config)
        future_draw = legacy_draw(slots[1], config)

    assert current_draw == pytest.approx(0.0725)
    assert future_draw == pytest.approx(0.2175)


def test_milp_charge_step_energy_and_current_use_effective_duration() -> None:
    """A 24 V, 20 A command stores 0.040 kWh now and 0.120 kWh next."""
    planned, diagnostics = _solve(
        _slots(import_price=0.01),
        _config(
            min_charge_current_a=20.0,
            max_charge_current_a=20.0,
            replacement_price_per_kwh=10.0,
        ),
    )

    assert [slot.secondary_storage_mode for slot in planned] == [
        SECONDARY_MODE_CHARGE,
        SECONDARY_MODE_CHARGE,
    ]
    expected_energy = [
        24.0 * 20.0 * _CURRENT_HOURS / 1000.0,
        24.0 * 20.0 * _FULL_HOURS / 1000.0,
    ]
    assert [slot.secondary_storage_charged_kwh for slot in planned] == pytest.approx(
        expected_energy
    )
    assert [slot.secondary_storage_charge_current_a for slot in planned] == (
        pytest.approx([20.0, 20.0])
    )
    assert [slot.secondary_storage_grid_import_kwh for slot in planned] == (
        pytest.approx(expected_energy)
    )
    assert [slot.grid_import_kwh for slot in planned] == pytest.approx(expected_energy)
    assert [slot.grid_export_kwh for slot in planned] == pytest.approx([0.0, 0.0])
    assert [slot.secondary_storage_estimated_capacity_kwh for slot in planned] == (
        pytest.approx([0.040, 0.160])
    )
    assert [slot.secondary_storage_estimated_soc_pct for slot in planned] == (
        pytest.approx([20.2667, 21.0667], abs=0.005)
    )
    assert diagnostics["secondary_total_charged_kwh"] == pytest.approx(0.160)


def test_included_load_retains_elapsed_current_slot_energy_balance() -> None:
    """SBU removes only the remaining included load from the full live projection.

    At 12:10 the 0.150 kWh live projection contains 0.100 kWh already elapsed
    and 0.050 kWh still controllable. Switching the PowMr to SBU removes only
    that remaining 0.050 kWh, so the current slot keeps 0.100 kWh grid import.
    """
    slots = _slots(import_price=10.0)
    for slot in slots:
        slot.avg_house_consumption_kwh = 0.150
        slot.estimated_net_consumption_kwh = 0.150

    planned, _diagnostics = _solve(
        slots,
        _config(
            current_soc_pct=60.0,
            load_power_w=600.0,
            base_load_includes_dedicated_load=True,
        ),
    )

    current = planned[0]
    assert current.secondary_storage_mode == SECONDARY_MODE_SBU
    assert current.secondary_storage_load_kwh == pytest.approx(0.050)
    assert current.grid_import_kwh == pytest.approx(0.100)
    assert current.grid_export_kwh == pytest.approx(0.0)
    assert current.grid_import_kwh - current.grid_export_kwh == pytest.approx(
        current.avg_house_consumption_kwh - current.secondary_storage_load_kwh
    )


def test_phase_fuse_rates_partial_powmr_charge_at_full_slot_power() -> None:
    """A 3 kW L3 base leaves two, not six, 10 A PowMr current steps."""
    slots = _slots(import_price=0.01)
    for slot in slots:
        slot.avg_house_consumption_kwh = 0.750
        slot.estimated_net_consumption_kwh = 0.750

    config = _config(
        nominal_voltage_v=25.6,
        min_charge_current_a=10.0,
        max_charge_current_a=60.0,
        replacement_price_per_kwh=10.0,
    )
    result = solve_milp(
        slots,
        _NOW,
        current_kwh=9.0,
        usable_kwh=9.0,
        max_charge_per_slot=1.0,
        max_discharge_per_slot=1.0,
        charge_efficiency_pct=100.0,
        discharge_efficiency_pct=100.0,
        time_discount_rate=1.0,
        no_export=True,
        main_fuse_amps=16.0,
        main_fuse_phases=3,
        phase_power_imbalance_w=(-1000.0, -1000.0, 2000.0),
        secondary_storage=config,
    )

    assert result is not None
    planned, diagnostics = result
    current_a = planned[0].secondary_storage_charge_current_a
    assert planned[0].secondary_storage_mode == SECONDARY_MODE_CHARGE
    assert current_a == pytest.approx(20.0)
    assert planned[0].secondary_storage_charged_kwh == pytest.approx(
        25.6 * 20.0 * _CURRENT_HOURS / 1000.0,
        abs=0.0005,
    )
    assert 3000.0 + config.nominal_voltage_v * current_a <= 16.0 * 230.0 + 1e-6
    assert diagnostics["max_phase_import_kwh"] == pytest.approx(0.878)
