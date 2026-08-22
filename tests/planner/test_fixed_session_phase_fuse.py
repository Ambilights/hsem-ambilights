"""Phase-fuse safety for measured EV sessions outside HSEM control."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from custom_components.hsem.models.ev_config import EVConfig
from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.planner.milp_optimizer import is_scipy_available, solve_milp
from custom_components.hsem.utils.prices import SlotPrice

_TZ = ZoneInfo("Europe/Copenhagen")
_SLOT_START = datetime(2024, 6, 15, 14, 0, tzinfo=_TZ)


@pytest.mark.skipif(
    not is_scipy_available(), reason="scipy not available in this environment"
)
def test_unmanaged_partial_session_relaxes_only_its_unavoidable_phase_load() -> None:
    """A fixed live session stays feasible and cannot fund extra charging."""
    slots = [
        PlannedSlot(
            start=_SLOT_START + timedelta(hours=offset),
            end=_SLOT_START + timedelta(hours=offset + 1),
            price=SlotPrice(import_price=-1.0, export_price=0.0),
        )
        for offset in range(3)
    ]
    ev = EVConfig(
        enabled=True,
        initial_soc_kwh=0.0,
        target_kwh=0.0,
        capacity_kwh=2.0,
        max_charge_per_slot=1.0,
        charger_efficiency=1.0,
        charger_min_power_w=0.0,
        base_load_includes_ev=True,
        session_charge_kw=1.0,
        fixed_session_only=True,
        current_session_removed_from_base=True,
    )

    result = solve_milp(
        slots,
        _SLOT_START + timedelta(minutes=45),
        current_kwh=0.0,
        usable_kwh=10.0,
        max_charge_per_slot=1.0,
        max_discharge_per_slot=0.0,
        charge_efficiency_pct=100.0,
        discharge_efficiency_pct=100.0,
        ev_configs=[ev],
        main_fuse_amps=1.0,
        main_fuse_phases=3,
        phase_power_imbalance_w=(0.0, 0.0, 0.0),
        no_export=True,
    )

    assert result is not None
    planned, diagnostics = result
    assert [slot.ev_total_planned_load_kwh for slot in planned] == pytest.approx(
        [0.25, 1.0, 1.0]
    )
    assert planned[0].ev_planned_load_kwh == pytest.approx(0.25)
    assert planned[0].ev_accounted_load_kwh == pytest.approx(0.0)
    assert [slot.ev_accounted_load_kwh for slot in planned[1:]] == [1.0, 1.0]
    assert [slot.ev_planned_load_kwh for slot in planned[1:]] == [0.0, 0.0]
    assert [slot.ev_charger_calculated_power for slot in planned] == [0.0, 0.0, 0.0]
    assert [slot.batteries_charged_kwh for slot in planned] == [0.0, 0.0, 0.0]
    # Topology is unknown, so every phase envelope admits exactly the existing
    # unavoidable 1 kW session and no additional controllable demand.
    assert diagnostics["max_phase_import_kwh"] == pytest.approx(1.0)
