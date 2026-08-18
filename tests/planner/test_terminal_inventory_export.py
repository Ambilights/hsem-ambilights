"""Regression coverage for destination-aware primary-battery export valuation.

Unless a test says otherwise, energy fields are kWh per slot, battery charge and
discharge are DC-side kWh, grid/source exports are AC-side kWh, and prices are
local currency/kWh.  The focused solves use explicit 60- or 15-minute slots so
their power limits and hand calculations stay visible in each test.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from custom_components.hsem.models.hourly_consumption_average import (
    HourlyConsumptionAverage,
)
from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.planner_input import PlannerInput
from custom_components.hsem.models.planner_output import PlannerOutput
from custom_components.hsem.models.price_forecast import (
    ForecastPricePoint,
    PriceForecast,
)
from custom_components.hsem.models.price_point import PricePoint
from custom_components.hsem.models.secondary_storage_config import (
    SecondaryStorageConfig,
)
from custom_components.hsem.models.solcast_slot import SolcastSlot
from custom_components.hsem.planner import engine_core, run_planner
from custom_components.hsem.planner.cost_function import CostWeights, score_plan
from custom_components.hsem.planner.future_value import (
    replacement_price_from_next_discharge,
)
from custom_components.hsem.planner.milp_optimizer import (
    is_scipy_available,
    solve_milp,
)
from custom_components.hsem.planner.secondary_storage import SECONDARY_MODE_SBU
from custom_components.hsem.utils.diagnostics import build_diagnostics_dump
from custom_components.hsem.utils.prices import SlotPrice

_TZ = ZoneInfo("Europe/Stockholm")
_NOW = datetime(2026, 8, 17, 0, 0, tzinfo=_TZ)

pytestmark = pytest.mark.skipif(
    not is_scipy_available(),
    reason="scipy not available in this environment",
)


def _slot(
    index: int,
    *,
    duration_minutes: int = 60,
    import_price: float = 1.0,
    export_price: float = 0.0,
    house_kwh: float = 0.0,
    pv_kwh: float = 0.0,
    price_actionable: bool = True,
) -> PlannedSlot:
    """Return one explicitly-sized slot with internally consistent net load."""
    start = _NOW + timedelta(minutes=index * duration_minutes)
    return PlannedSlot(
        start=start,
        end=start + timedelta(minutes=duration_minutes),
        price=SlotPrice(
            import_price=import_price,
            export_price=export_price,
        ),
        price_actionable=price_actionable,
        import_price_available=price_actionable,
        export_price_available=price_actionable,
        avg_house_consumption_kwh=house_kwh,
        solcast_pv_estimate_kwh=pv_kwh,
        estimated_net_consumption_kwh=house_kwh - pv_kwh,
    )


def _solve(
    slots: list[PlannedSlot],
    *,
    current_kwh: float,
    usable_kwh: float,
    max_charge_kwh: float,
    max_discharge_kwh: float,
    replacement_price: float | None,
    efficiency_pct: float = 100.0,
    cycle_cost: float = 0.0,
    no_export: bool = False,
    secondary: SecondaryStorageConfig | None = None,
) -> tuple[list[PlannedSlot], dict]:
    """Solve one deterministic primary/optional-secondary scenario."""
    solved = solve_milp(
        slots,
        _NOW,
        current_kwh=current_kwh,
        usable_kwh=usable_kwh,
        max_charge_per_slot=max_charge_kwh,
        max_discharge_per_slot=max_discharge_kwh,
        cycle_cost_per_kwh=cycle_cost,
        charge_efficiency_pct=efficiency_pct,
        discharge_efficiency_pct=efficiency_pct,
        replacement_price_per_kwh=replacement_price,
        time_discount_rate=1.0,
        no_export=no_export,
        secondary_storage=secondary,
    )
    assert solved is not None
    return solved


def _assert_export_source_balance(slots: list[PlannedSlot]) -> None:
    """Assert the public AC export-source identity at published precision."""
    for slot in slots:
        assert slot.primary_battery_export_kwh >= 0.0
        assert slot.pv_export_kwh >= 0.0
        assert round(
            slot.primary_battery_export_kwh + slot.pv_export_kwh,
            3,
        ) == round(slot.grid_export_kwh, 3), slot.start.isoformat()


def _established_terminal_export_input() -> PlannerInput:
    """Return the exact 24-hour, 10 kWh release-comparison reproduction."""
    prices = {
        17: (1.685, 0.688),
        19: (1.646, 0.658),
        20: (1.628, 0.643),
        22: (0.963, 0.143),
        23: (0.922, 0.078),
    }
    houses = {17: 0.5, 19: 0.5, 20: 0.6, 22: 0.6, 23: 0.42}
    pvs = {17: 0.72, 19: 0.4}
    return PlannerInput(
        now_iso="2024-06-15T16:00:00+02:00",
        interval_minutes=60,
        interval_length_hours=24,
        battery_soc_pct=100,
        battery_rated_capacity_kwh=10,
        battery_end_of_discharge_soc_pct=0,
        battery_max_soc_pct=100,
        battery_max_charge_power_w=5000,
        battery_max_discharge_power_w=None,
        battery_charge_efficiency_pct=100,
        battery_discharge_efficiency_pct=100,
        battery_purchase_price=0,
        battery_expected_cycles=6000,
        battery_cycle_cost_per_kwh=0,
        consumption_averages=[
            HourlyConsumptionAverage(
                hour=hour,
                avg_1d=houses.get(hour, 0.05),
                avg_3d=houses.get(hour, 0.05),
                avg_7d=houses.get(hour, 0.05),
                avg_14d=houses.get(hour, 0.05),
            )
            for hour in range(24)
        ],
        price_points=[
            PricePoint(
                hour=hour,
                import_price=prices.get(hour, (1, 0))[0],
                export_price=prices.get(hour, (1, 0))[1],
            )
            for hour in range(24)
        ],
        solcast_slots=[
            SolcastSlot(hour=hour, pv_estimate=pvs.get(hour, 0)) for hour in range(24)
        ],
        excess_export_enabled=True,
        months_winter=[1, 2, 3, 4, 10, 11, 12],
        is_read_only=True,
    )


def test_established_ten_kwh_export_at_0688_is_blocked() -> None:
    """Do not dump a 10 kWh battery at 0.688 before dearer known imports.

    This is the exact 60-minute release reproduction.  The old plan discharged
    all 10.000 DC kWh at 17:00 and exported it at 0.688 currency/kWh.  With
    0.720 kWh PV and 0.500 kWh load, 0.050 kWh first refills the battery energy
    used at 16:00.  The corrected 17:00 flow charges 0.050 DC kWh, exports only
    0.170 AC kWh PV, discharges 0, and leaves at least 7 kWh in the final slot
    after the horizon's roughly 2.7 kWh of local demand.
    """
    output = run_planner(_established_terminal_export_input())
    hour_17 = next(
        slot
        for slot in output.slots
        if slot.start.date().isoformat() == "2024-06-15" and slot.start.hour == 17
    )

    assert output.winner_name == "milp"
    assert hour_17.batteries_discharged_kwh == pytest.approx(0.0, abs=0.001)
    assert hour_17.batteries_charged_kwh == pytest.approx(0.050, abs=0.001)
    assert hour_17.grid_export_kwh == pytest.approx(0.170, abs=0.001)
    assert output.slots[-1].estimated_battery_capacity_kwh > 7.0


def test_saved_live_economics_discharges_only_into_local_load() -> None:
    """Reproduce the 15-minute 05:30 live economics without a battery dump.

    The observed slot had import 2.945, export 1.697, replacement value
    3.361545 currency/kWh, 0.475 kWh house demand, and 0.008 kWh PV.  At 98%
    discharge efficiency, local service requires 0.467/0.98 = 0.477 DC kWh.
    The corrected plan exports 0 AC kWh from that discharge, then may refill
    during the later 1.595 import-price slot.
    """
    slots = [
        _slot(
            0,
            duration_minutes=15,
            import_price=2.945,
            export_price=1.697,
            house_kwh=0.475,
            pv_kwh=0.008,
        ),
        _slot(
            1,
            duration_minutes=15,
            import_price=2.273,
            export_price=1.000,
            house_kwh=0.500,
        ),
        _slot(
            2,
            duration_minutes=15,
            import_price=1.595,
            export_price=0.500,
            house_kwh=0.500,
        ),
    ]

    planned, _diagnostics = _solve(
        slots,
        current_kwh=10.0,
        usable_kwh=10.0,
        max_charge_kwh=2.5,
        max_discharge_kwh=2.5,
        replacement_price=3.361545,
        efficiency_pct=98.0,
        cycle_cost=0.092593,
    )

    assert planned[0].batteries_discharged_kwh == pytest.approx(0.477, abs=0.001)
    assert planned[0].grid_export_kwh == pytest.approx(0.0, abs=0.001)
    assert sum(slot.batteries_charged_kwh for slot in planned) > 0.0


def test_free_pv_refill_allows_export_below_terminal_value() -> None:
    """A free refill makes a below-R export profitable over two 60-min slots.

    Start and finish with 1.000 DC kWh.  Slot 0 sells 1.000 AC kWh at 2.000;
    slot 1 stores 1.000 kWh of otherwise-worthless PV.  The 3.000 terminal
    debit and credit cancel, leaving exactly 2.000 currency of export revenue.
    """
    planned, _diagnostics = _solve(
        [
            _slot(0, import_price=3.2, export_price=2.0),
            _slot(1, import_price=0.2, export_price=0.0, pv_kwh=1.0),
        ],
        current_kwh=1.0,
        usable_kwh=1.0,
        max_charge_kwh=1.0,
        max_discharge_kwh=1.0,
        replacement_price=3.0,
    )

    assert planned[0].grid_export_kwh == pytest.approx(1.0, abs=0.001)
    assert planned[1].batteries_charged_kwh == pytest.approx(1.0, abs=0.001)
    assert sum(
        s.batteries_charged_kwh - s.batteries_discharged_kwh for s in planned
    ) == pytest.approx(0.0, abs=0.001)
    _assert_export_source_balance(planned)


@pytest.mark.parametrize(
    ("export_price", "should_cycle"),
    [(2.0, True), (0.8, False)],
    ids=("spread-pays-losses-and-wear", "spread-does-not-pay-losses-and-wear"),
)
def test_grid_refill_only_when_spread_pays_losses_and_wear(
    export_price: float,
    should_cycle: bool,
) -> None:
    """Require a profitable 60-minute export/refill round trip.

    One DC kWh at 90% efficiency exports 0.9 AC kWh.  Refilling costs
    1/0.9 * 0.5 = 0.5556, charge loss costs 0.05, export-side discharge
    loss costs 0.1 times the export price, and two throughput legs cost
    2 * 0.05 = 0.10.  Export 2.0 earns 1.8 against 0.9056 of non-revenue
    cost and cycles; export 0.8 earns 0.72 against 0.7856 and must leave
    SoC and all flows unchanged.
    """
    planned, _diagnostics = _solve(
        [
            _slot(0, import_price=2.5, export_price=export_price),
            _slot(1, import_price=0.5, export_price=0.0),
        ],
        current_kwh=1.0,
        usable_kwh=1.0,
        max_charge_kwh=1.0,
        max_discharge_kwh=1.0,
        replacement_price=3.0,
        efficiency_pct=90.0,
        cycle_cost=0.05,
    )

    if should_cycle:
        assert planned[0].grid_export_kwh == pytest.approx(0.900, abs=0.001)
        assert planned[0].batteries_discharged_kwh == pytest.approx(1.0, abs=0.001)
        assert planned[1].batteries_charged_kwh == pytest.approx(1.0, abs=0.001)
        assert planned[1].grid_import_kwh == pytest.approx(1.111, abs=0.001)
    else:
        assert sum(s.grid_export_kwh for s in planned) == pytest.approx(0.0, abs=0.001)
        assert sum(s.batteries_discharged_kwh for s in planned) == pytest.approx(
            0.0,
            abs=0.001,
        )
        assert sum(s.batteries_charged_kwh for s in planned) == pytest.approx(
            0.0,
            abs=0.001,
        )


@pytest.mark.parametrize(
    ("pv_refill_kwh", "max_charge_kwh", "refill_import_price"),
    [(0.4, 1.0, 4.0), (1.0, 0.4, 0.2)],
    ids=("insufficient-refill-energy", "power-limited-refill"),
)
def test_export_is_limited_to_energy_that_can_refill_before_horizon_end(
    pv_refill_kwh: float,
    max_charge_kwh: float,
    refill_import_price: float,
) -> None:
    """Block the unrefillable share of a below-R export over two 60-min slots.

    The battery starts at 1.000 kWh and export 2.000 is below R=3.000.  Either
    only 0.400 kWh PV exists with grid refill uneconomic at 4.000, or the
    charger can store only 0.400 kWh.  Exactly that 0.400 kWh may be sold and
    restored; the remaining 0.600 kWh must stay.
    """
    planned, _diagnostics = _solve(
        [
            _slot(0, import_price=3.0, export_price=2.0),
            _slot(
                1,
                import_price=refill_import_price,
                export_price=0.0,
                pv_kwh=pv_refill_kwh,
            ),
        ],
        current_kwh=1.0,
        usable_kwh=1.0,
        max_charge_kwh=max_charge_kwh,
        max_discharge_kwh=1.0,
        replacement_price=3.0,
    )

    assert planned[0].grid_export_kwh == pytest.approx(0.400, abs=0.001)
    assert planned[0].batteries_discharged_kwh == pytest.approx(0.400, abs=0.001)
    assert planned[1].batteries_charged_kwh == pytest.approx(0.400, abs=0.001)
    assert sum(
        s.batteries_charged_kwh - s.batteries_discharged_kwh for s in planned
    ) == pytest.approx(0.0, abs=0.001)


def test_refill_after_dear_load_is_too_late_to_justify_export() -> None:
    """Keep 1 kWh for a 4.000 import before a later free-PV refill.

    Across three 60-minute slots, selling first at 2.000 would force 1.000 kWh
    of grid import at 4.000 before the PV arrives.  The correct flow is 0 export,
    1.000 DC kWh discharged into that load, then 1.000 kWh PV charged; start and
    final SoC are both 1.000 kWh and the avoided cost exceeds sale revenue by 2.
    """
    planned, _diagnostics = _solve(
        [
            _slot(0, import_price=3.0, export_price=2.0),
            _slot(1, import_price=4.0, export_price=0.0, house_kwh=1.0),
            _slot(2, import_price=0.2, export_price=0.0, pv_kwh=1.0),
        ],
        current_kwh=1.0,
        usable_kwh=1.0,
        max_charge_kwh=1.0,
        max_discharge_kwh=1.0,
        replacement_price=3.0,
    )

    assert planned[0].grid_export_kwh == pytest.approx(0.0, abs=0.001)
    assert planned[1].grid_import_kwh == pytest.approx(0.0, abs=0.001)
    assert planned[1].batteries_discharged_kwh == pytest.approx(1.0, abs=0.001)
    assert planned[2].batteries_charged_kwh == pytest.approx(1.0, abs=0.001)

    # Same prices and energy, but the free refill now arrives before demand.
    # The battery can sell, refill, and still serve the dear load, so timing
    # alone changes slot-0 export from 0 to 1 kWh.
    refill_first, _diagnostics = _solve(
        [
            _slot(0, import_price=3.0, export_price=2.0),
            _slot(1, import_price=0.2, export_price=0.0, pv_kwh=1.0),
            _slot(2, import_price=4.0, export_price=0.0, house_kwh=1.0),
        ],
        current_kwh=1.0,
        usable_kwh=1.0,
        max_charge_kwh=1.0,
        max_discharge_kwh=1.0,
        replacement_price=3.0,
    )
    assert refill_first[0].grid_export_kwh == pytest.approx(1.0, abs=0.001)
    assert refill_first[1].batteries_charged_kwh == pytest.approx(1.0, abs=0.001)
    assert refill_first[2].grid_import_kwh == pytest.approx(0.0, abs=0.001)


def test_powmr_sbu_is_removed_from_primary_local_load_before_attribution() -> None:
    """Attribute exactly 1.600 AC kWh to Huawei export during PowMr SBU.

    In one 60-minute slot the house sensor contains 0.500 kWh, including the
    PowMr's dedicated 0.100 kWh load.  SBU supplies that 0.100 itself, leaving
    0.400 kWh eligible for Huawei self-consumption.  A 2.000 DC/AC kWh Huawei
    discharge therefore exports 2.000 - 0.400 = 1.600 kWh.  PowMr is never an
    export source and PV export is exactly zero.
    """
    secondary = SecondaryStorageConfig(
        enabled=True,
        capacity_kwh=15.0,
        current_soc_pct=80.0,
        min_soc_pct=20.0,
        max_soc_pct=100.0,
        nominal_voltage_v=24.0,
        load_power_w=100.0,
        max_charge_current_a=60.0,
        min_charge_current_a=10.0,
        charge_current_step_a=10.0,
        charge_efficiency_pct=100.0,
        discharge_efficiency_pct=100.0,
        inverter_standby_power_w=0.0,
        cycle_cost_per_kwh=0.0,
        replacement_price_per_kwh=0.0,
        base_load_includes_dedicated_load=True,
    )
    planned, _diagnostics = _solve(
        [_slot(0, import_price=10.0, export_price=5.0, house_kwh=0.5)],
        current_kwh=2.0,
        usable_kwh=2.0,
        max_charge_kwh=2.0,
        max_discharge_kwh=2.0,
        replacement_price=None,
        secondary=secondary,
    )

    slot = planned[0]
    assert slot.secondary_storage_mode == SECONDARY_MODE_SBU
    assert slot.secondary_storage_discharged_kwh == pytest.approx(0.100, abs=0.001)
    assert slot.batteries_discharged_kwh == pytest.approx(2.000, abs=0.001)
    assert slot.primary_battery_export_kwh == pytest.approx(1.600, abs=0.001)
    assert slot.pv_export_kwh == pytest.approx(0.0, abs=0.001)
    assert slot.grid_export_kwh == pytest.approx(1.600, abs=0.001)
    _assert_export_source_balance(planned)


def test_forecast_only_replacement_value_is_not_an_export_price_floor() -> None:
    """A forecast-derived R=5 may coexist with export at 2 after free refill.

    The first two 60-minute slots have published prices; the third is the
    unpublished tail.  Its only 5.000 forecast point creates R=5.000 without
    becoming actionable.  Selling 1.000 AC kWh at 2.000 and refilling 1.000
    DC kWh from free PV leaves terminal SoC unchanged, so the sale remains valid.
    """
    slots = [
        _slot(0, import_price=3.0, export_price=2.0),
        _slot(1, import_price=0.2, export_price=0.0, pv_kwh=1.0),
        _slot(2, import_price=0.0, export_price=0.0, price_actionable=False),
    ]
    forecast = PriceForecast(
        points=(ForecastPricePoint(start=slots[2].start, value=5.0),),
        mae=0.0,
        margin=0.0,
        enabled=True,
    )
    replacement_price = replacement_price_from_next_discharge(
        slots,
        _NOW,
        top_n=1,
        interval_minutes=60,
        forecast=forecast,
    )
    assert replacement_price == pytest.approx(5.0)
    assert replacement_price is not None

    planned, _diagnostics = _solve(
        slots,
        current_kwh=1.0,
        usable_kwh=1.0,
        max_charge_kwh=1.0,
        max_discharge_kwh=1.0,
        replacement_price=replacement_price,
    )

    assert planned[0].price.export_price < replacement_price
    assert planned[0].grid_export_kwh == pytest.approx(1.0, abs=0.001)
    assert planned[1].batteries_charged_kwh == pytest.approx(1.0, abs=0.001)
    assert planned[2].batteries_charged_kwh == pytest.approx(0.0, abs=0.001)
    assert planned[2].batteries_discharged_kwh == pytest.approx(0.0, abs=0.001)


def test_issue_694_exports_peak_pv_and_charges_later() -> None:
    """Preserve #694 same-slot PV routing over two 60-minute slots.

    With an empty 1 kWh battery and R=2, slot 0's 1 kWh PV is worth 2.4 on
    export and must all be exported.  Slot 1's 1 kWh PV is worth only 0.08 and
    must instead charge 1 kWh.  No Huawei-origin export exists in either slot.
    """
    planned, _diagnostics = _solve(
        [
            _slot(0, import_price=3.0, export_price=2.4, pv_kwh=1.0),
            _slot(1, import_price=0.1, export_price=0.08, pv_kwh=1.0),
        ],
        current_kwh=0.0,
        usable_kwh=1.0,
        max_charge_kwh=1.0,
        max_discharge_kwh=1.0,
        replacement_price=2.0,
        no_export=True,
    )

    assert planned[0].batteries_charged_kwh == pytest.approx(0.0, abs=0.001)
    assert planned[0].grid_export_kwh == pytest.approx(1.0, abs=0.001)
    assert planned[0].pv_export_kwh == pytest.approx(1.0, abs=0.001)
    assert planned[1].batteries_charged_kwh == pytest.approx(1.0, abs=0.001)
    assert planned[1].grid_export_kwh == pytest.approx(0.0, abs=0.001)
    assert all(s.primary_battery_export_kwh == 0.0 for s in planned)
    _assert_export_source_balance(planned)


def test_issue_592_defers_charge_to_inevitable_cheaper_surplus() -> None:
    """Preserve #592 deferred-surplus outcome over two 60-minute slots.

    A 1 kWh battery starts at 0.8, so it has 0.2 kWh headroom.  Both slots
    contain 1 kWh PV.  The plan exports 1.0 at 1.2 now, then stores 0.2 and
    exports 0.8 at 0.2 later.  Charging 0.2 in the dear slot would forfeit
    0.2 currency with no terminal-SoC benefit.
    """
    planned, _diagnostics = _solve(
        [
            _slot(0, import_price=1.2, export_price=1.2, pv_kwh=1.0),
            _slot(1, import_price=0.2, export_price=0.2, pv_kwh=1.0),
        ],
        current_kwh=0.8,
        usable_kwh=1.0,
        max_charge_kwh=0.5,
        max_discharge_kwh=0.5,
        replacement_price=1.7,
        no_export=True,
    )

    assert planned[0].batteries_charged_kwh == pytest.approx(0.0, abs=0.001)
    assert planned[0].pv_export_kwh == pytest.approx(1.0, abs=0.001)
    assert planned[1].batteries_charged_kwh == pytest.approx(0.2, abs=0.001)
    assert planned[1].pv_export_kwh == pytest.approx(0.8, abs=0.001)
    _assert_export_source_balance(planned)


def test_no_export_binds_battery_source_but_not_pv_export() -> None:
    """Keep no_export structural over two 60-minute source-separated slots.

    Slot 0 has 0.4 kWh load and no PV: exactly 0.4 DC kWh may serve the house,
    with zero export.  Slot 1 has 1.0 PV and 0.2 load: battery discharge and
    battery-origin export remain zero while the 0.8 AC kWh PV surplus exports.
    """
    planned, _diagnostics = _solve(
        [
            _slot(0, import_price=3.0, export_price=2.9, house_kwh=0.4),
            _slot(
                1,
                import_price=3.0,
                export_price=2.9,
                house_kwh=0.2,
                pv_kwh=1.0,
            ),
        ],
        current_kwh=1.0,
        usable_kwh=1.0,
        max_charge_kwh=1.0,
        max_discharge_kwh=1.0,
        replacement_price=None,
        no_export=True,
    )

    assert planned[0].batteries_discharged_kwh == pytest.approx(0.4, abs=0.001)
    assert planned[0].grid_export_kwh == pytest.approx(0.0, abs=0.001)
    assert planned[0].primary_battery_export_kwh == pytest.approx(0.0, abs=0.001)
    assert planned[1].batteries_discharged_kwh == pytest.approx(0.0, abs=0.001)
    assert planned[1].primary_battery_export_kwh == pytest.approx(0.0, abs=0.001)
    assert planned[1].pv_export_kwh == pytest.approx(0.8, abs=0.001)
    assert planned[1].grid_export_kwh == pytest.approx(0.8, abs=0.001)
    _assert_export_source_balance(planned)


def test_non_actionable_tail_has_no_primary_charge_or_discharge() -> None:
    """Keep battery control at zero in an unpublished 60-minute tail slot.

    The tail advertises 5 kWh PV and numeric 9/8 prices only as diagnostic
    values.  It may physically export PV, but ec, ed, and Huawei-origin export
    must all remain exactly 0 kWh; all aggregate export is non-battery PV.
    """
    planned, _diagnostics = _solve(
        [
            _slot(0, import_price=1.0, export_price=0.0, house_kwh=0.5),
            _slot(
                1,
                import_price=9.0,
                export_price=8.0,
                pv_kwh=5.0,
                price_actionable=False,
            ),
        ],
        current_kwh=0.5,
        usable_kwh=1.0,
        max_charge_kwh=1.0,
        max_discharge_kwh=1.0,
        replacement_price=10.0,
    )

    tail = planned[1]
    assert tail.batteries_charged_kwh == pytest.approx(0.0, abs=0.001)
    assert tail.batteries_discharged_kwh == pytest.approx(0.0, abs=0.001)
    assert tail.primary_battery_export_kwh == pytest.approx(0.0, abs=0.001)
    assert tail.pv_export_kwh == pytest.approx(tail.grid_export_kwh, abs=0.001)
    _assert_export_source_balance(planned)


def test_model_layout_declares_every_actual_solver_column() -> None:
    """Require one contiguous declared block map for every LP/MILP column.

    This two-slot solve has two DC export-attribution columns and two PV-source
    columns in addition to the pre-existing blocks.  Summed declared widths,
    objective width, equality/inequality matrix widths, and bounds count must
    all equal the runtime model column count.
    """
    _planned, diagnostics = _solve(
        [_slot(0, export_price=2.0), _slot(1, pv_kwh=1.0)],
        current_kwh=1.0,
        usable_kwh=1.0,
        max_charge_kwh=1.0,
        max_discharge_kwh=1.0,
        replacement_price=3.0,
    )
    blocks = diagnostics["model_variable_blocks"]
    for name in (
        "primary_battery_export",
        "pv_export",
        "export_source_mode",
        "primary_action_mode",
        "grid_flow_mode",
    ):
        assert name in blocks

    cursor = 0
    for metadata in blocks.values():
        assert metadata["offset"] == cursor
        assert metadata["width"] > 0
        assert isinstance(metadata["per_slot"], bool)
        cursor += metadata["width"]

    column_count = diagnostics["model_column_count"]
    assert cursor == column_count
    assert diagnostics["model_objective_column_count"] == column_count
    assert diagnostics["model_equality_column_count"] == column_count
    assert diagnostics["model_inequality_column_count"] == column_count
    assert diagnostics["model_bounds_count"] == column_count
    integral_blocks = diagnostics["model_integral_blocks"]
    required_integral = {
        "export_source_mode",
        "primary_action_mode",
        "grid_flow_mode",
    }
    assert required_integral <= set(integral_blocks)
    assert diagnostics["model_integrality_count"] >= sum(
        int(blocks[name]["width"]) for name in required_integral
    )


def test_mixed_export_sources_sum_to_aggregate_for_every_slot() -> None:
    """Publish exact AC source accounting when PV and Huawei export together.

    In one 60-minute slot, 0.5 kWh PV covers 0.2 kWh load and leaves 0.3 kWh
    PV export.  A full 1.0 kWh Huawei discharge is also exported.  Aggregate
    export is therefore 1.3 = 1.0 primary + 0.3 PV, in AC kWh.
    """
    planned, diagnostics = _solve(
        [
            _slot(
                0,
                import_price=5.0,
                export_price=4.0,
                house_kwh=0.2,
                pv_kwh=0.5,
            )
        ],
        current_kwh=1.0,
        usable_kwh=1.0,
        max_charge_kwh=1.0,
        max_discharge_kwh=1.0,
        replacement_price=None,
    )

    slot = planned[0]
    assert slot.primary_battery_export_kwh == pytest.approx(1.0, abs=0.001)
    assert slot.pv_export_kwh == pytest.approx(0.3, abs=0.001)
    assert slot.grid_export_kwh == pytest.approx(1.3, abs=0.001)
    assert diagnostics["export_source_balance_max_error_kwh"] <= 0.001
    _assert_export_source_balance(planned)


def test_diagnostics_serializes_primary_and_pv_export_sources() -> None:
    """Expose both AC-kWh source fields in the public diagnostics slot.

    The serializer rounds to three decimals: primary 0.4567 -> 0.457, PV
    0.1234 -> 0.123, and aggregate grid export is the published 0.580 kWh.
    """
    slot = _slot(0)
    slot.primary_battery_export_kwh = 0.4567
    slot.pv_export_kwh = 0.1234
    slot.grid_export_kwh = 0.5801
    dump = build_diagnostics_dump(
        PlannerInput(),
        PlannerOutput(slots=[slot]),
        integration_version="6.2.2-powmr.36",
    )
    serialized = dump["planner_output"]["slots"][0]

    assert serialized["primary_battery_export_kwh"] == pytest.approx(0.457)
    assert serialized["pv_export_kwh"] == pytest.approx(0.123)
    assert serialized["grid_export_kwh"] == pytest.approx(0.580)


def test_winner_slots_and_scores_match_final_and_direct_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep winner identity and every score view aligned on the 60-min replay.

    The selected candidate's slot list must be the final output list.  Its
    selector score, final score, and a direct ``score_plan`` call using the
    captured currency/kWh replacement value must match to 1e-6.  This prevents
    source-field post-processing from changing costs or flows after selection.
    """
    planner_input = _established_terminal_export_input()
    captured_replacement_prices: list[float | None] = []
    original_generate_candidates = engine_core.generate_candidates

    def _capture_generate_candidates(*args: Any, **kwargs: Any) -> Any:
        captured_replacement_prices.append(kwargs.get("replacement_price_per_kwh"))
        return original_generate_candidates(*args, **kwargs)

    monkeypatch.setattr(
        engine_core,
        "generate_candidates",
        _capture_generate_candidates,
    )
    output = engine_core.run_planner(planner_input)
    assert len(captured_replacement_prices) == 1
    replacement_price = captured_replacement_prices[0]
    assert replacement_price is not None

    winner = next(c for c in output.candidates if c.name == output.winner_name)
    assert winner._cost is not None
    assert output.plan_cost is not None
    assert output.slots is winner.slots
    assert output.plan_cost.score == pytest.approx(winner._cost.score, abs=1e-6)
    assert output.plan_cost.total_cost == pytest.approx(
        winner._cost.total_cost,
        abs=1e-6,
    )

    direct = score_plan(
        output.slots,
        CostWeights(
            min_soc_pct=0.0,
            max_soc_pct=100.0,
            cycle_cost_per_kwh=0.0,
            charge_efficiency_pct=100.0,
            discharge_efficiency_pct=100.0,
            battery_usable_capacity_kwh=10.0,
            max_charge_per_slot_kwh=5.0,
        ),
        slot_duration_hours=1.0,
        now=datetime.fromisoformat(planner_input.now_iso),
        initial_battery_kwh=10.0,
        replacement_price_per_kwh=replacement_price,
    )
    assert direct.score == pytest.approx(output.plan_cost.score, abs=1e-6)
    assert direct.total_cost == pytest.approx(output.plan_cost.total_cost, abs=1e-6)
    _assert_export_source_balance(output.slots)
