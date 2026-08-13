"""Regression tests for executable MILP battery-charge recommendations."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from custom_components.hsem.models.ev_config import EVConfig
from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.planner.milp._write_results import (
    _write_milp_results_to_slots,
)
from custom_components.hsem.utils.prices import SlotPrice
from custom_components.hsem.utils.recommendations import Recommendations

_NOW = datetime(2026, 8, 14, 18, 0, tzinfo=ZoneInfo("Europe/Stockholm"))


def _write_charge(
    *,
    charge_kwh: float,
    solar_surplus_kwh: float,
    ev_charge_kwh: float = 0.0,
    session_slot: bool = False,
    curtailed_kwh: float = 0.0,
) -> PlannedSlot:
    """Return one MILP slot after writing the requested charge allocation."""
    slot = PlannedSlot(
        start=_NOW,
        end=_NOW + timedelta(minutes=15),
        price=SlotPrice(import_price=1.731, export_price=0.725),
    )
    active_evs: list[EVConfig] = []
    ev_var_offsets: list[int] = []
    result_x = np.array([0.0])
    if ev_charge_kwh > 0.0:
        active_evs = [
            EVConfig(
                enabled=True,
                capacity_kwh=64.0,
                max_charge_per_slot=3.0,
                charger_efficiency=0.9,
                charger_min_power_w=0.0,
            )
        ]
        ev_var_offsets = [1]
        result_x = np.array([0.0, ev_charge_kwh])

    result = _write_milp_results_to_slots(
        slots=[slot],
        future_idx=[0],
        now=_NOW,
        ec_sol=np.array([charge_kwh]),
        ed_sol=np.array([0.0]),
        result_x=result_x,
        m=1,
        ge_off=0,
        active_evs=active_evs,
        ev_var_offsets=ev_var_offsets,
        pv_avail=np.array([solar_surplus_kwh]),
        base_load=np.array([0.0]),
        charge_eff=0.97,
        discharge_eff=0.97,
        p_exp=np.array([0.725]),
        min_export_price=0.0,
        _has_session_demand=session_slot,
        session_slots_set={0} if session_slot else set(),
        current_kwh=10.0,
        usable_kwh=28.5,
        curt_sol_full=np.array([curtailed_kwh]),
    )
    return result[0]


def _write_discharge(
    *,
    discharge_kwh: float,
    house_load_kwh: float,
    raw_export_kwh: float = 0.0,
) -> PlannedSlot:
    """Return one MILP slot after writing a discharge allocation."""
    slot = PlannedSlot(
        start=_NOW,
        end=_NOW + timedelta(minutes=15),
        price=SlotPrice(import_price=1.731, export_price=0.725),
    )
    return _write_milp_results_to_slots(
        slots=[slot],
        future_idx=[0],
        now=_NOW,
        ec_sol=np.array([0.0]),
        ed_sol=np.array([discharge_kwh]),
        result_x=np.array([raw_export_kwh]),
        m=1,
        ge_off=0,
        active_evs=[],
        ev_var_offsets=[],
        pv_avail=np.array([0.0]),
        base_load=np.array([house_load_kwh]),
        charge_eff=0.97,
        discharge_eff=0.97,
        p_exp=np.array([0.725]),
        min_export_price=0.0,
        _has_session_demand=False,
        session_slots_set=set(),
        current_kwh=10.0,
        usable_kwh=28.5,
        curt_sol_full=np.array([0.0]),
    )[0]


def test_mixed_solar_and_grid_charge_uses_grid_recommendation() -> None:
    """A small PV surplus must not hide a materially grid-funded charge."""
    slot = _write_charge(charge_kwh=2.45, solar_surplus_kwh=0.255)

    assert slot.recommendation == Recommendations.BatteriesChargeGrid.value
    assert slot.batteries_charged_kwh == 2.45
    assert slot.grid_import_kwh > 2.0


def test_solar_surplus_covering_full_charge_uses_solar_recommendation() -> None:
    """Keep solar-charge mode when PV covers the allocation after losses."""
    slot = _write_charge(charge_kwh=0.24, solar_surplus_kwh=0.255)

    assert slot.recommendation == Recommendations.BatteriesChargeSolar.value
    assert slot.batteries_charged_kwh == 0.24
    assert slot.grid_import_kwh == 0.0


def test_cooptimised_ev_leaves_enough_solar_for_battery() -> None:
    """An EV allocation does not force grid mode when residual PV covers both."""
    slot = _write_charge(
        charge_kwh=0.45,
        solar_surplus_kwh=1.1,
        ev_charge_kwh=0.45,
    )

    assert slot.recommendation == Recommendations.BatteriesChargeSolar.value
    assert slot.grid_import_kwh == 0.0


def test_cooptimised_ev_consumes_required_battery_solar() -> None:
    """Use grid mode when EV demand leaves too little PV for battery charging."""
    slot = _write_charge(
        charge_kwh=0.45,
        solar_surplus_kwh=0.9,
        ev_charge_kwh=0.45,
    )

    assert slot.recommendation == Recommendations.BatteriesChargeGrid.value
    assert slot.grid_import_kwh > 0.0


def test_session_ev_with_residual_solar_keeps_solar_recommendation() -> None:
    """Session demand may coexist with a fully solar-funded battery charge."""
    slot = _write_charge(
        charge_kwh=0.45,
        solar_surplus_kwh=1.1,
        ev_charge_kwh=0.45,
        session_slot=True,
    )

    assert slot.recommendation == Recommendations.BatteriesChargeSolar.value
    assert slot.grid_import_kwh == 0.0


def test_grid_funded_session_charge_fails_closed() -> None:
    """Never retain an impossible grid-funded battery charge in a session slot."""
    slot = _write_charge(
        charge_kwh=0.45,
        solar_surplus_kwh=0.9,
        ev_charge_kwh=0.45,
        session_slot=True,
    )

    assert slot.recommendation == Recommendations.BatteriesWaitMode.value
    assert slot.primary_battery_hold is True
    assert slot.batteries_charged_kwh == 0.0
    assert slot.grid_import_kwh == 0.0


def test_curtailed_pv_cannot_fund_a_solar_charge_label() -> None:
    """PV allocated to curtailment is unavailable to the primary battery."""
    slot = _write_charge(
        charge_kwh=0.776,
        solar_surplus_kwh=1.0,
        curtailed_kwh=0.5,
    )

    assert slot.recommendation == Recommendations.BatteriesChargeGrid.value
    assert slot.grid_import_kwh > 0.0


def test_solar_label_never_rounds_to_nonzero_grid_import() -> None:
    """A displayed 0.001 kWh grid shortfall must select grid-charge mode."""
    slot = _write_charge(
        charge_kwh=0.248,
        solar_surplus_kwh=0.255,
    )

    assert slot.recommendation == Recommendations.BatteriesChargeGrid.value
    assert slot.grid_import_kwh == 0.001


def test_sub_millikwh_solver_residue_does_not_create_charge_action() -> None:
    """Energy that rounds to zero must not emit an executable recommendation."""
    slot = _write_charge(charge_kwh=0.0004, solar_surplus_kwh=0.0)

    assert slot.recommendation == Recommendations.BatteriesWaitMode.value
    assert slot.primary_battery_hold is True
    assert slot.batteries_charged_kwh == 0.0
    assert slot.grid_import_kwh == 0.0


def test_idle_pv_export_is_completed_as_label_only_battery_hold() -> None:
    """The writer labels an idle export without consuming its solved PV flow."""
    slot = _write_charge(charge_kwh=0.0, solar_surplus_kwh=1.0)

    assert slot.recommendation == Recommendations.BatteriesWaitMode.value
    assert slot.primary_battery_hold is True
    assert slot.batteries_charged_kwh == 0.0
    assert slot.batteries_discharged_kwh == 0.0
    assert slot.grid_import_kwh == 0.0
    assert slot.grid_export_kwh == 1.0


def test_sub_millikwh_discharge_residue_becomes_battery_hold() -> None:
    """Rounded-zero discharge must not enable real hardware discharge."""
    slot = _write_discharge(discharge_kwh=0.0004, house_load_kwh=0.5)

    assert slot.recommendation == Recommendations.BatteriesWaitMode.value
    assert slot.primary_battery_hold is True
    assert slot.batteries_discharged_kwh == 0.0
    assert slot.grid_import_kwh == 0.5


def test_discharge_mode_uses_final_export_not_raw_solver_residue() -> None:
    """A raw export residue cannot turn house-serving discharge into export mode."""
    slot = _write_discharge(
        discharge_kwh=0.25,
        house_load_kwh=0.5,
        raw_export_kwh=0.0004,
    )

    assert slot.recommendation == Recommendations.BatteriesDischargeMode.value
    assert slot.grid_export_kwh == 0.0
