"""End-to-end regressions for fuse-safe, selection-stable plan publication."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from custom_components.hsem.models.hourly_consumption_average import (
    HourlyConsumptionAverage,
)
from custom_components.hsem.models.planner_input import PlannerInput
from custom_components.hsem.models.price_forecast import (
    ForecastPricePoint,
    PriceForecast,
)
from custom_components.hsem.models.price_point import PricePoint
from custom_components.hsem.models.solcast_slot import SolcastSlot
from custom_components.hsem.planner import run_planner
from custom_components.hsem.planner.candidate_generator import CANDIDATE_PASSIVE
from custom_components.hsem.planner.milp_optimizer import (
    CANDIDATE_MILP,
    is_scipy_available,
)

_NOW = datetime(2026, 8, 20, tzinfo=UTC)
_FUSE_AMPS = 2.0
_FUSE_KWH = _FUSE_AMPS * 3 * 230.0 / 1000.0

pytestmark = pytest.mark.skipif(
    not is_scipy_available(), reason="scipy not available in this environment"
)


def _averages(
    *, default: float, overrides: dict[int, float] | None = None
) -> list[HourlyConsumptionAverage]:
    values = overrides or {}
    return [
        HourlyConsumptionAverage(
            hour=hour,
            avg_1d=values.get(hour, default),
            avg_3d=values.get(hour, default),
            avg_7d=values.get(hour, default),
            avg_14d=values.get(hour, default),
        )
        for hour in range(24)
    ]


def _prices(*, hours: range, import_price: float) -> list[PricePoint]:
    return [
        PricePoint(hour=hour, import_price=import_price, export_price=0.0)
        for hour in hours
    ]


def _base_input(**overrides: Any) -> PlannerInput:
    values: dict[str, Any] = {
        "now_iso": _NOW.isoformat(),
        "interval_minutes": 60,
        "interval_length_hours": 24,
        "battery_soc_pct": 0.0,
        "battery_rated_capacity_kwh": 10.0,
        "battery_end_of_discharge_soc_pct": 0.0,
        "battery_max_soc_pct": 100.0,
        "battery_max_charge_power_w": 1000.0,
        "battery_max_discharge_power_w": 1000.0,
        "battery_charge_efficiency_pct": 100.0,
        "battery_discharge_efficiency_pct": 100.0,
        "battery_purchase_price": 0.0,
        "battery_cycle_cost_per_kwh": 0.0,
        "consumption_averages": _averages(default=0.0),
        "price_points": _prices(hours=range(24), import_price=0.10),
        "solcast_slots": [SolcastSlot(hour=hour) for hour in range(24)],
        "main_fuse_amps": _FUSE_AMPS,
        "main_fuse_phases": 3,
        "planner_hysteresis_enabled": False,
        "is_read_only": True,
    }
    values.update(overrides)
    return PlannerInput(**values)


def _winner(output):
    winner = next(c for c in output.candidates if c.name == output.winner_name)
    assert winner.name == CANDIDATE_MILP
    assert winner.slots is output.slots
    assert winner._cost == output.plan_cost
    return winner


def test_ev_minimum_power_cannot_reintroduce_a_fuse_violation() -> None:
    """Sub-minimum continuous EV allocations are dropped, not over-fused.

    House demand consumes 1.0 kWh of a 1.38 kWh hourly fuse allowance.  The
    remaining 380 W cannot start a 1380 W charger in any deadline slot.
    """
    output = run_planner(
        _base_input(
            consumption_averages=_averages(default=1.0),
            ev_planned_load_enabled=True,
            ev_planned_load_connected=True,
            ev_planned_load_smart_charging_enabled=True,
            ev_planned_load_current_soc_pct=0.0,
            ev_planned_load_target_soc_pct=20.0,
            ev_planned_load_battery_capacity_kwh=10.0,
            ev_planned_load_charger_power_kw=2.0,
            ev_planned_load_charger_efficiency_pct=100.0,
            ev_planned_load_charger_min_power_w=1380.0,
            ev_planned_load_deadline=_NOW + timedelta(hours=4),
            ev_planned_load_base_load_includes_ev=False,
        )
    )

    winner = _winner(output)
    assert output.ev_charging_plan is not None
    assert output.ev_charging_plan.state == "waiting"
    assert output.ev_charging_plan.charging_slots == []
    assert output.ev_charging_plan.data_quality["unmet_target_kwh"] == pytest.approx(
        2.0
    )

    for slot in output.slots:
        assert slot.grid_import_kwh <= _FUSE_KWH + 1e-9
        assert slot.ev_charger_calculated_power == pytest.approx(0.0)
        assert slot.ev_total_planned_load_kwh == pytest.approx(0.0)

    ev_diagnostics = winner.diagnostics["ev"]["ev0"]
    assert ev_diagnostics["total_dc_kwh"] == pytest.approx(0.0)
    assert ev_diagnostics["deadline_penalty_kwh"] == pytest.approx(2.0)
    assert ev_diagnostics["deadline_met"] is False
    assert ev_diagnostics["unplaceable_dc_kwh"] > 0.0
    assert winner.diagnostics["total_fuse_violation_kwh"] == pytest.approx(0.0)


def test_terminal_inventory_cannot_charge_past_the_fuse_or_stale_soc() -> None:
    """A high terminal value is bounded in-model and SoC follows the charge.

    Only the first price is published.  The next slot contains a forecast-valued
    10 kWh load, making retained battery energy extremely valuable.  A 1 A
    three-phase fuse still limits the only actionable charge to 0.69 kWh.
    """
    one_amp_kwh = 3 * 230.0 / 1000.0
    output = run_planner(
        _base_input(
            battery_max_charge_power_w=10_000.0,
            battery_max_discharge_power_w=10_000.0,
            consumption_averages=_averages(default=0.0, overrides={1: 10.0}),
            price_points=_prices(hours=range(1), import_price=0.01),
            price_forecast=PriceForecast(
                points=(
                    ForecastPricePoint(start=_NOW + timedelta(hours=1), value=100.0),
                ),
                enabled=True,
            ),
            main_fuse_amps=1.0,
        )
    )

    winner = _winner(output)
    first = output.slots[0]
    assert first.batteries_charged_kwh == pytest.approx(one_amp_kwh, abs=0.001)
    assert first.grid_import_kwh == pytest.approx(one_amp_kwh, abs=0.001)
    assert first.estimated_battery_capacity_kwh == pytest.approx(one_amp_kwh, abs=0.001)
    assert first.estimated_battery_soc_pct == pytest.approx(6.9, abs=0.02)
    assert all(
        slot.batteries_discharged_kwh == pytest.approx(0.0) for slot in output.slots
    )
    assert all(
        slot.estimated_battery_capacity_kwh == pytest.approx(one_amp_kwh, abs=0.001)
        for slot in output.slots
    )
    # The 10 kWh non-actionable house load is unavoidably above the fuse.  It
    # remains feasible and visible, but controllable charge never worsens it.
    assert winner.diagnostics["total_fuse_violation_kwh"] == pytest.approx(9.31)
    assert all(
        slot.grid_import_kwh <= max(one_amp_kwh, slot.avg_house_consumption_kwh) + 1e-9
        for slot in output.slots
    )
    assert (
        output.explanation.terminal_cost_to_go_final_valued_quantity_kwh
        == pytest.approx(one_amp_kwh, abs=0.001)
    )


def test_valid_milp_is_the_only_executable_candidate_under_signed_prices() -> None:
    """A valid MILP exclusively owns EV output under signed prices."""
    prices = [
        PricePoint(
            hour=hour,
            import_price=-1.0 if hour == 0 else 1.0,
            export_price=0.0,
        )
        for hour in range(24)
    ]
    output = run_planner(
        _base_input(
            battery_soc_pct=100.0,
            price_points=prices,
            ev_planned_load_enabled=True,
            ev_planned_load_connected=True,
            ev_planned_load_smart_charging_enabled=True,
            ev_planned_load_current_soc_pct=0.0,
            ev_planned_load_target_soc_pct=20.0,
            ev_planned_load_battery_capacity_kwh=10.0,
            ev_planned_load_charger_power_kw=2.0,
            ev_planned_load_charger_efficiency_pct=100.0,
            ev_planned_load_charger_min_power_w=0.0,
            ev_planned_load_deadline=_NOW + timedelta(hours=2),
            ev_planned_load_base_load_includes_ev=False,
        )
    )

    winner = _winner(output)
    passive = next(
        candidate
        for candidate in output.candidates
        if candidate.name == CANDIDATE_PASSIVE
    )
    assert passive._cost is not None
    assert winner._cost is not None
    assert all(
        slot.ev_total_planned_load_kwh == pytest.approx(0.0) for slot in passive.slots
    )
    assert output.slots[0].ev_charger_calculated_power <= 1380.0 + 1e-9
    assert sum(
        slot.ev_total_planned_load_kwh for slot in output.slots
    ) == pytest.approx(
        2.0,
        abs=0.001,
    )
    assert all(slot.grid_import_kwh <= _FUSE_KWH + 1e-9 for slot in output.slots)
    assert winner.slots is output.slots
    assert winner._cost == output.plan_cost


def test_solver_failure_fallback_disables_unconstrained_ev_demand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passive fallback turns EVs off before scoring when MILP has no result."""
    monkeypatch.setattr(
        "custom_components.hsem.planner.candidate_generator.solve_milp",
        lambda *args, **kwargs: None,
    )
    output = run_planner(
        _base_input(
            consumption_averages=_averages(default=1.0),
            ev_planned_load_enabled=True,
            ev_planned_load_connected=True,
            ev_planned_load_smart_charging_enabled=True,
            ev_planned_load_current_soc_pct=0.0,
            ev_planned_load_target_soc_pct=20.0,
            ev_planned_load_battery_capacity_kwh=10.0,
            ev_planned_load_charger_power_kw=2.0,
            ev_planned_load_charger_efficiency_pct=100.0,
            ev_planned_load_charger_min_power_w=1380.0,
            ev_planned_load_deadline=_NOW + timedelta(hours=4),
            ev_planned_load_base_load_includes_ev=False,
        )
    )

    assert output.winner_name == CANDIDATE_PASSIVE
    winner = next(
        candidate
        for candidate in output.candidates
        if candidate.name == output.winner_name
    )
    assert winner.slots is output.slots
    assert winner._cost == output.plan_cost
    assert output.ev_charging_plan is not None
    assert output.ev_charging_plan.state == "waiting"
    assert output.ev_charging_plan.charging_slots == []
    assert output.ev_charging_plan.data_quality["unmet_target_kwh"] == pytest.approx(
        2.0
    )
    for slot in output.slots:
        assert slot.grid_import_kwh <= _FUSE_KWH + 1e-9
        assert slot.ev_charger_calculated_power == pytest.approx(0.0)
        assert slot.ev_total_planned_load_kwh == pytest.approx(0.0)


def test_invalid_milp_fallback_disables_unconstrained_ev_demand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected solver candidate cannot expose passive greedy EV demand."""
    from custom_components.hsem.planner.candidate_validation import validate_candidate
    from custom_components.hsem.planner.cost_helpers import (
        slot_grid_cash_flow_cost,
    )

    def _reject_milp(candidate, floor, *, secondary_storage=None):  # noqa: ANN001, ANN202
        if candidate.name == CANDIDATE_MILP:
            return False, "synthetic post-solve validation failure"
        return validate_candidate(
            candidate,
            floor,
            secondary_storage=secondary_storage,
        )

    monkeypatch.setattr(
        "custom_components.hsem.planner.candidate_selector._validate_candidate",
        _reject_milp,
    )
    output = run_planner(
        _base_input(
            battery_soc_pct=50.0,
            battery_end_of_discharge_soc_pct=20.0,
            consumption_averages=_averages(default=1.0),
            ev_planned_load_enabled=True,
            ev_planned_load_connected=True,
            ev_planned_load_smart_charging_enabled=True,
            ev_planned_load_current_soc_pct=0.0,
            ev_planned_load_target_soc_pct=20.0,
            ev_planned_load_battery_capacity_kwh=10.0,
            ev_planned_load_charger_power_kw=2.0,
            ev_planned_load_charger_efficiency_pct=100.0,
            ev_planned_load_charger_min_power_w=1380.0,
            ev_planned_load_deadline=_NOW + timedelta(hours=4),
            ev_planned_load_base_load_includes_ev=False,
        )
    )

    assert output.winner_name == CANDIDATE_PASSIVE
    winner = next(
        candidate
        for candidate in output.candidates
        if candidate.name == output.winner_name
    )
    assert winner.slots is output.slots
    assert winner._cost == output.plan_cost
    assert output.ev_charging_plan is not None
    assert output.ev_charging_plan.state == "waiting"
    for slot in output.slots:
        assert slot.grid_import_kwh <= _FUSE_KWH + 1e-9
        assert slot.ev_charger_calculated_power == pytest.approx(0.0)
        assert slot.ev_total_planned_load_kwh == pytest.approx(0.0)
        assert slot.estimated_cost_currency == pytest.approx(
            slot_grid_cash_flow_cost(slot)
        )


@pytest.mark.parametrize("fallback_mode", ["no_result", "rejected"])
def test_passive_fallback_preserves_uncontrollable_live_session(
    monkeypatch: pytest.MonkeyPatch,
    fallback_mode: str,
) -> None:
    """Solver fallback keeps physical session load but emits no EV command."""
    if fallback_mode == "no_result":
        monkeypatch.setattr(
            "custom_components.hsem.planner.candidate_generator.solve_milp",
            lambda *args, **kwargs: None,
        )
    else:
        from custom_components.hsem.planner.candidate_validation import (
            validate_candidate,
        )

        def _reject_milp(candidate, floor, *, secondary_storage=None):  # noqa: ANN001, ANN202
            if candidate.name == CANDIDATE_MILP:
                return False, "synthetic post-solve validation failure"
            return validate_candidate(
                candidate,
                floor,
                secondary_storage=secondary_storage,
            )

        monkeypatch.setattr(
            "custom_components.hsem.planner.candidate_selector._validate_candidate",
            _reject_milp,
        )

    output = run_planner(
        _base_input(
            consumption_averages=_averages(default=1.0),
            main_fuse_amps=1.0,
            ev_planned_load_enabled=False,
            ev_planned_load_connected=False,
            ev_planned_load_smart_charging_enabled=False,
            ev_planned_load_battery_capacity_kwh=0.0,
            ev_planned_load_charger_power_kw=0.0,
            ev_planned_load_charger_efficiency_pct=100.0,
            ev_planned_load_base_load_includes_ev=False,
            ev_session_charge_kw=6.0,
        )
    )

    assert output.winner_name == CANDIDATE_PASSIVE
    winner = next(
        candidate
        for candidate in output.candidates
        if candidate.name == output.winner_name
    )
    assert winner.slots is output.slots
    assert winner._cost == output.plan_cost
    assert output.ev_charging_plan is None

    for slot in output.slots[:2]:
        assert slot.ev_total_planned_load_kwh == pytest.approx(6.0)
        assert slot.ev_planned_load_kwh == pytest.approx(6.0)
        assert slot.ev_accounted_load_kwh == pytest.approx(0.0)
        assert slot.ev_charger_calculated_power == pytest.approx(0.0)
        assert slot.ev_second_charger_calculated_power == pytest.approx(0.0)
        assert slot.batteries_charged_kwh == pytest.approx(0.0)
        assert slot.batteries_discharged_kwh == pytest.approx(0.0)
        assert slot.grid_import_kwh == pytest.approx(7.0)
        assert slot.estimated_cost_currency == pytest.approx(0.7)

    assert output.slots[2].ev_total_planned_load_kwh == pytest.approx(0.0)
    assert output.slots[2].grid_import_kwh == pytest.approx(1.0)


def test_passive_fixed_session_uses_pure_live_base_then_accounted_forecast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Current live pure-house data injects EV; future inclusive forecast accounts it."""
    monkeypatch.setattr(
        "custom_components.hsem.planner.candidate_generator.solve_milp",
        lambda *args, **kwargs: None,
    )
    output = run_planner(
        _base_input(
            consumption_averages=_averages(default=7.0),
            live_house_consumption_w=1000.0,
            house_power_includes_ev=False,
            ev_planned_load_enabled=False,
            ev_planned_load_connected=False,
            ev_planned_load_smart_charging_enabled=False,
            ev_planned_load_battery_capacity_kwh=0.0,
            ev_planned_load_charger_power_kw=0.0,
            ev_planned_load_charger_efficiency_pct=100.0,
            ev_planned_load_base_load_includes_ev=True,
            ev_session_charge_kw=6.0,
        )
    )

    assert output.winner_name == CANDIDATE_PASSIVE
    current, future = output.slots[:2]
    assert current.avg_house_consumption_kwh == pytest.approx(1.0)
    assert current.ev_planned_load_kwh == pytest.approx(6.0)
    assert current.ev_accounted_load_kwh == pytest.approx(0.0)
    assert current.grid_import_kwh == pytest.approx(7.0)
    assert future.avg_house_consumption_kwh == pytest.approx(7.0)
    assert future.ev_planned_load_kwh == pytest.approx(0.0)
    assert future.ev_accounted_load_kwh == pytest.approx(6.0)
    assert future.grid_import_kwh == pytest.approx(7.0)
    assert current.ev_charger_calculated_power == pytest.approx(0.0)
    assert future.ev_charger_calculated_power == pytest.approx(0.0)
