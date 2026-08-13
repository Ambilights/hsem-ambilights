"""Tests for configurable MILP time limits and safe incumbent reuse."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.planner.milp._incumbent import validate_incumbent
from custom_components.hsem.utils.prices import SlotPrice

_TZ = ZoneInfo("Europe/Copenhagen")
_NOW = datetime(2024, 6, 15, 0, 0, tzinfo=_TZ)


def _milp_modules():
    """Import native solver modules lazily after pytest collection."""
    try:
        from scipy import optimize as scipy_optimize

        from custom_components.hsem.planner.milp_optimizer import solve_milp
    except Exception as exc:
        pytest.skip(f"scipy/HA not available in test environment: {exc}")
    return solve_milp, scipy_optimize


def _validation_kwargs() -> dict:
    """Return a tiny complete integer model with one feasible solution."""
    return {
        "n_vars": 2,
        "slot_count": 2,
        "future_idx": [0, 1],
        "m": 2,
        "variable_blocks": {"decision": (0, 2)},
        "a_eq": [[1.0, 1.0]],
        "b_eq": [1.0],
        "a_ub": [[1.0, 0.0], [0.0, 1.0]],
        "b_ub": [1.0, 1.0],
        "bounds": [(0.0, 1.0), (0.0, 1.0)],
        "integrality": [1, 1],
    }


def _make_slots() -> list[PlannedSlot]:
    """Build a small but economically non-degenerate horizon."""
    slots: list[PlannedSlot] = []
    for hour, import_price in enumerate((0.5, 0.5, 3.0, 3.0)):
        start = _NOW + timedelta(hours=hour)
        slot = PlannedSlot(
            start=start,
            end=start + timedelta(hours=1),
            price=SlotPrice(
                import_price=import_price,
                export_price=import_price * 0.8,
            ),
        )
        slot.avg_house_consumption_kwh = 0.5
        slot.estimated_net_consumption_kwh = 0.5
        slots.append(slot)
    return slots


def test_validate_incumbent_accepts_complete_feasible_vector() -> None:
    """A complete finite vector satisfying every model row is accepted."""
    _milp_modules()
    result = validate_incumbent([1.0, 0.0], **_validation_kwargs())

    assert result.valid is True
    assert result.reason == "feasible"


@pytest.mark.parametrize(
    ("solution", "updates", "expected_reason"),
    [
        (None, {}, "missing_solution_vector"),
        ([1.0], {}, "solution_vector_length_1_expected_2"),
        ([float("nan"), 0.0], {}, "solution_vector_not_finite"),
        (
            [1.5, -0.5],
            {
                "bounds": [(None, None), (None, None)],
                "integrality": None,
            },
            "inequality_constraint_violation",
        ),
        (
            [0.5, 0.5],
            {},
            "integrality_violation",
        ),
        (
            [1.0, 0.0],
            {"future_idx": [1, 0]},
            "future_horizon_not_strictly_increasing",
        ),
    ],
)
def test_validate_incumbent_rejects_unsafe_results(
    solution: list[float] | None,
    updates: dict,
    expected_reason: str,
) -> None:
    """Incomplete, infeasible, fractional, or misaligned vectors are rejected."""
    _milp_modules()
    kwargs = _validation_kwargs()
    kwargs.update(updates)

    result = validate_incumbent(solution, **kwargs)

    assert result.valid is False
    assert result.reason == expected_reason


def test_time_limit_uses_validated_feasible_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A feasible time-limit incumbent remains a normal ``milp`` result."""
    solve_milp, scipy_optimize = _milp_modules()
    real_linprog = scipy_optimize.linprog
    captured_options: dict = {}

    def timed_linprog(*args, **kwargs):
        captured_options.update(kwargs["options"])
        result = real_linprog(*args, **kwargs)
        assert result.success
        result.success = False
        result.status = 1
        result.message = "Time limit reached. (HiGHS Status 13)"
        result.mip_gap = 0.0125
        return result

    monkeypatch.setattr(scipy_optimize, "linprog", timed_linprog)
    attempt: dict = {}

    solved = solve_milp(
        _make_slots(),
        _NOW,
        current_kwh=0.0,
        usable_kwh=4.0,
        max_charge_per_slot=2.0,
        max_discharge_per_slot=2.0,
        solver_time_limit_seconds=7.0,
        attempt_diagnostics=attempt,
    )

    assert solved is not None
    _slots, diagnostics = solved
    assert captured_options["time_limit"] == pytest.approx(7.0)
    assert diagnostics["solver_status"] == "time_limit_feasible_incumbent"
    assert diagnostics["solver_optimal"] is False
    assert diagnostics["incumbent_used"] is True
    assert diagnostics["incumbent_validation"] == "feasible"
    assert diagnostics["fallback_reason"] == ""
    assert attempt["solver_status"] == "time_limit_feasible_incumbent"


def test_time_limit_without_incumbent_falls_back_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeout before any feasible integer vector is never decoded."""
    solve_milp, scipy_optimize = _milp_modules()

    def timed_without_solution(*_args, **_kwargs):
        return scipy_optimize.OptimizeResult(
            {
                "success": False,
                "status": 1,
                "message": "Time limit reached. (HiGHS Status 13)",
                "x": None,
                "fun": None,
                "mip_gap": None,
            }
        )

    monkeypatch.setattr(scipy_optimize, "linprog", timed_without_solution)
    attempt: dict = {}

    solved = solve_milp(
        _make_slots(),
        _NOW,
        current_kwh=0.0,
        usable_kwh=4.0,
        max_charge_per_slot=2.0,
        max_discharge_per_slot=2.0,
        attempt_diagnostics=attempt,
    )

    assert solved is None
    assert attempt["solver_status"] == "time_limit_no_incumbent"
    assert attempt["incumbent_validation"] == "missing_solution_vector"
    assert attempt["fallback_reason"] == "time_limit_no_incumbent"


def test_time_limit_with_invalid_incumbent_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A returned vector that violates the model is rejected fail-closed."""
    solve_milp, scipy_optimize = _milp_modules()
    real_linprog = scipy_optimize.linprog

    def timed_with_invalid_solution(*args, **kwargs):
        result = real_linprog(*args, **kwargs)
        assert result.success
        result.x[0] = float("nan")
        result.success = False
        result.status = 1
        result.message = "Time limit reached. (HiGHS Status 13)"
        return result

    monkeypatch.setattr(scipy_optimize, "linprog", timed_with_invalid_solution)
    attempt: dict = {}

    solved = solve_milp(
        _make_slots(),
        _NOW,
        current_kwh=0.0,
        usable_kwh=4.0,
        max_charge_per_slot=2.0,
        max_discharge_per_slot=2.0,
        attempt_diagnostics=attempt,
    )

    assert solved is None
    assert attempt["solver_status"] == "time_limit_invalid_incumbent"
    assert attempt["incumbent_validation"] == "solution_vector_not_finite"
    assert attempt["fallback_reason"] == "time_limit_invalid_incumbent"


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (0.0, 1.0),
        (7.5, 7.5),
        (100.0, 60.0),
        (float("nan"), 15.0),
    ],
)
def test_solver_timeout_is_clamped(
    monkeypatch: pytest.MonkeyPatch, configured: float, expected: float
) -> None:
    """The HiGHS time limit always stays in the supported safe range."""
    solve_milp, scipy_optimize = _milp_modules()
    captured_options: dict = {}

    def timed_without_solution(*_args, **kwargs):
        captured_options.update(kwargs["options"])
        return scipy_optimize.OptimizeResult(
            success=False,
            status=1,
            message="Time limit reached. (HiGHS Status 13)",
            x=None,
        )

    monkeypatch.setattr(scipy_optimize, "linprog", timed_without_solution)
    solve_milp(
        _make_slots(), _NOW, 0.0, 4.0, 2.0, 2.0, solver_time_limit_seconds=configured
    )
    assert captured_options["time_limit"] == pytest.approx(expected)
