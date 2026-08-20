"""Regression tests for named MILP variable-bound construction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from custom_components.hsem.models.ev_config import EVConfig
from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.secondary_storage_config import (
    SecondaryStorageConfig,
)
from custom_components.hsem.models.terminal_cost_to_go import (
    TerminalCostToGo,
    TerminalValueTier,
)
from custom_components.hsem.planner.milp._layout import (
    ColumnBlock,
    MilpBound,
    MilpBoundsBuilder,
    MilpColumnLayout,
)
from custom_components.hsem.utils.prices import SlotPrice

_TZ = ZoneInfo("Europe/Copenhagen")
_NOW = datetime(2024, 6, 15, 0, 0, tzinfo=_TZ)


def _assert_bound_sequences(
    actual: Sequence[MilpBound],
    expected: Sequence[MilpBound],
) -> None:
    """Compare bound endpoints while respecting optional infinities."""
    assert len(actual) == len(expected)
    for actual_bound, expected_bound in zip(actual, expected, strict=True):
        actual_lower, actual_upper = actual_bound
        expected_lower, expected_upper = expected_bound
        if expected_lower is None:
            assert actual_lower is None
        else:
            assert actual_lower == pytest.approx(expected_lower)
        if expected_upper is None:
            assert actual_upper is None
        else:
            assert actual_upper == pytest.approx(expected_upper)


def _layout_with_two_equal_blocks() -> MilpColumnLayout:
    """Declare equal-width blocks whose domains must not follow write order."""
    layout = MilpColumnLayout(slot_count=2)
    layout.add("fixed", 2)
    layout.add("flexible", 2)
    return layout


def _scipy_optimize() -> Any:
    """Import SciPy lazily after collection, or skip in minimal environments."""
    try:
        from scipy import optimize as scipy_optimize
    except Exception as exc:
        pytest.skip(f"scipy is unavailable in this test environment: {exc}")
    return scipy_optimize


def test_bounds_builder_reverse_writes_preserve_layout_and_solution() -> None:
    """Reverse writes produce the same bounds and optimum as declaration order."""
    scipy_optimize = _scipy_optimize()
    expected = [(0.0, 0.0), (0.0, 0.0), (0.0, 1.0), (0.0, 1.0)]

    ordered = MilpBoundsBuilder(_layout_with_two_equal_blocks())
    ordered.fill("fixed", (0.0, 0.0))
    ordered.fill("flexible", (0.0, 1.0))

    reversed_writes = MilpBoundsBuilder(_layout_with_two_equal_blocks())
    reversed_writes.fill("flexible", (0.0, 1.0))
    reversed_writes.fill("fixed", (0.0, 0.0))

    ordered_bounds = ordered.finalize()
    reversed_bounds = reversed_writes.finalize()
    _assert_bound_sequences(ordered_bounds, expected)
    _assert_bound_sequences(reversed_bounds, expected)

    objective = [0.0, 0.0, -1.0, -2.0]
    ordered_result = scipy_optimize.linprog(objective, bounds=ordered_bounds)
    reversed_result = scipy_optimize.linprog(objective, bounds=reversed_bounds)

    assert ordered_result.success is True
    assert reversed_result.success is True
    assert reversed_result.fun == pytest.approx(ordered_result.fun)
    assert reversed_result.x == pytest.approx(ordered_result.x)
    assert reversed_result.x == pytest.approx([0.0, 0.0, 1.0, 1.0])


def test_bounds_builder_reports_every_missing_block_and_absolute_index() -> None:
    """Finalization identifies unassigned names and their absolute columns."""
    layout = MilpColumnLayout(slot_count=2)
    layout.add("assigned", 2)
    layout.add("missing_middle", 2)
    layout.add("missing_tail", 2)
    builder = MilpBoundsBuilder(layout)
    builder.fill("assigned", (0.0, 1.0))

    with pytest.raises(ValueError) as exc_info:
        builder.finalize()

    message = str(exc_info.value)
    assert "missing_middle" in message
    assert "missing_tail" in message
    for absolute_index in (2, 3, 4, 5):
        assert str(absolute_index) in message


@pytest.mark.parametrize("width", [1, 3])
def test_bounds_builder_rejects_wrong_block_width(width: int) -> None:
    """A named block accepts neither a truncated nor an oversized sequence."""
    builder = MilpBoundsBuilder(_layout_with_two_equal_blocks())

    with pytest.raises(ValueError, match=r"fixed.*width|width.*fixed"):
        builder.set("fixed", [(0.0, 1.0)] * width)


def test_bounds_builder_rejects_duplicate_assignment() -> None:
    """Writing one named block twice fails even when both domains are valid."""
    builder = MilpBoundsBuilder(_layout_with_two_equal_blocks())
    builder.fill("fixed", (0.0, 0.0))

    with pytest.raises(ValueError, match=r"duplicate.*fixed|fixed.*already"):
        builder.set("fixed", [(0.0, 1.0), (0.0, 1.0)])


def test_bounds_builder_rejects_overlapping_layout_declarations() -> None:
    """A malformed externally constructed layout cannot alias two block slices."""
    layout = MilpColumnLayout(
        slot_count=2,
        blocks={
            "left": ColumnBlock(offset=0, width=2),
            "right": ColumnBlock(offset=1, width=2),
        },
        column_count=3,
    )

    with pytest.raises(ValueError) as exc_info:
        MilpBoundsBuilder(layout)

    message = str(exc_info.value)
    assert "left" in message
    assert "right" in message
    assert "1" in message


def test_bounds_builder_rejects_unknown_block() -> None:
    """A typo cannot silently create or append an undeclared variable block."""
    builder = MilpBoundsBuilder(_layout_with_two_equal_blocks())

    with pytest.raises(ValueError, match=r"unknown.*typo|typo.*block"):
        builder.fill("typo", (0.0, 1.0))


@pytest.mark.parametrize(
    "invalid_bound",
    [
        (2.0, 1.0),
        (float("nan"), 1.0),
        (0.0, float("inf")),
        (float("-inf"), None),
    ],
)
def test_bounds_builder_rejects_inverted_or_nonfinite_domain(
    invalid_bound: MilpBound,
) -> None:
    """Inverted and non-finite endpoints fail before reaching SciPy."""
    builder = MilpBoundsBuilder(_layout_with_two_equal_blocks())

    with pytest.raises(ValueError):
        builder.fill("fixed", invalid_bound)


def test_bounds_builder_accepts_none_upper_bound_for_scipy() -> None:
    """An unbounded-above domain finalizes as an ordinary SciPy pair."""
    layout = MilpColumnLayout(slot_count=1)
    layout.add("slack", 1)
    builder = MilpBoundsBuilder(layout)
    builder.fill("slack", (0.0, None))

    bounds = builder.finalize()

    assert isinstance(bounds, list)
    assert isinstance(bounds[0], tuple)
    assert bounds[0][0] == pytest.approx(0.0)
    assert bounds[0][1] is None


def test_bounds_builder_rejects_layout_mutation_after_construction() -> None:
    """A builder cannot be reused after its layout declaration changes."""
    layout = _layout_with_two_equal_blocks()
    builder = MilpBoundsBuilder(layout)
    layout.add("late", 2)

    with pytest.raises(ValueError, match=r"layout.*chang|chang.*layout"):
        builder.fill("fixed", (0.0, 0.0))


def _full_feature_slots() -> list[PlannedSlot]:
    """Build three actionable one-hour slots with explicit physical units."""
    slots: list[PlannedSlot] = []
    for hour, import_price in enumerate((0.10, 0.60, 1.20)):
        start = _NOW + timedelta(hours=hour)
        slot = PlannedSlot(
            start=start,
            end=start + timedelta(hours=1),
            price=SlotPrice(
                import_price=import_price,
                export_price=import_price * 0.8,
            ),
        )
        slot.avg_house_consumption_kwh = 0.4
        slot.estimated_net_consumption_kwh = 0.4
        slots.append(slot)
    return slots


def _full_feature_terminal_model() -> TerminalCostToGo:
    """Build two bounded post-horizon primary-inventory tiers."""
    return TerminalCostToGo(
        tiers=(
            TerminalValueTier(
                start=_NOW + timedelta(hours=4),
                quantity_kwh=0.25,
                value_per_kwh=0.9,
                forecast_price_per_kwh=1.0,
            ),
            TerminalValueTier(
                start=_NOW + timedelta(hours=5),
                quantity_kwh=0.75,
                value_per_kwh=0.5,
                forecast_price_per_kwh=0.6,
            ),
        ),
        source="forecast",
        boundary=_NOW + timedelta(hours=3),
    )


def _full_feature_secondary() -> SecondaryStorageConfig:
    """Build a valid one-phase dedicated-load battery configuration."""
    return SecondaryStorageConfig(
        enabled=True,
        capacity_kwh=4.0,
        current_soc_pct=50.0,
        min_soc_pct=20.0,
        max_soc_pct=100.0,
        nominal_voltage_v=24.0,
        load_power_w=100.0,
        min_charge_current_a=10.0,
        max_charge_current_a=20.0,
        charge_current_step_a=10.0,
        charge_efficiency_pct=100.0,
        discharge_efficiency_pct=100.0,
        inverter_standby_power_w=55.0,
        replacement_price_per_kwh=0.8,
        base_load_includes_dedicated_load=True,
        grid_phase=3,
    )


def test_production_model_assigns_every_named_bound_slice_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All optional production blocks reach ``linprog`` in their named domains.

    The three slots are one hour each. House and dedicated-load inputs are AC
    kWh/W; primary and secondary charge/discharge domains are battery-side kWh.
    The solve enables one EV, aggregate and phase-aware fuse constraints,
    conditional export reserve, two primary terminal tiers, and PowMr storage.
    """
    scipy_optimize = _scipy_optimize()
    from custom_components.hsem.planner.milp_optimizer import solve_milp

    assignment_names: list[str] = []
    captured: dict[str, Any] = {}
    original_set = MilpBoundsBuilder.set
    original_fill = MilpBoundsBuilder.fill
    real_linprog = scipy_optimize.linprog

    def recording_set(
        self: MilpBoundsBuilder,
        name: str,
        values: Sequence[MilpBound],
    ) -> None:
        assignment_names.append(name)
        original_set(self, name, values)

    def recording_fill(
        self: MilpBoundsBuilder,
        name: str,
        bound: MilpBound,
    ) -> None:
        assignment_names.append(name)
        original_fill(self, name, bound)

    def capturing_linprog(*args: Any, **kwargs: Any) -> Any:
        captured["bounds"] = list(kwargs["bounds"])
        return real_linprog(*args, **kwargs)

    monkeypatch.setattr(MilpBoundsBuilder, "set", recording_set)
    monkeypatch.setattr(MilpBoundsBuilder, "fill", recording_fill)
    monkeypatch.setattr(scipy_optimize, "linprog", capturing_linprog)

    result = solve_milp(
        _full_feature_slots(),
        _NOW,
        current_kwh=1.0,
        usable_kwh=2.0,
        max_charge_per_slot=1.0,
        max_discharge_per_slot=1.0,
        charge_efficiency_pct=100.0,
        discharge_efficiency_pct=100.0,
        terminal_cost_to_go=_full_feature_terminal_model(),
        ev_configs=[
            EVConfig(
                enabled=True,
                initial_soc_kwh=0.0,
                target_kwh=0.5,
                capacity_kwh=2.0,
                max_charge_per_slot=0.5,
                charger_efficiency=1.0,
                deadline_slot=2,
            )
        ],
        main_fuse_amps=16.0,
        main_fuse_phases=3,
        phase_power_imbalance_w=(0.0, 0.0, 0.0),
        excess_export_discharge_buffer_pct=20.0,
        secondary_storage=_full_feature_secondary(),
    )

    assert result is not None
    _planned, diagnostics = result
    blocks = diagnostics["model_variable_blocks"]
    counts = Counter(assignment_names)
    assert set(counts) == set(blocks)
    assert all(count == 1 for count in counts.values())

    cursor = 0
    covered_indices: list[int] = []
    for metadata in blocks.values():
        offset = int(metadata["offset"])
        width = int(metadata["width"])
        assert offset == cursor
        covered_indices.extend(range(offset, offset + width))
        cursor += width
    assert covered_indices == list(range(diagnostics["model_column_count"]))

    m = 3
    expected_domains: dict[str, list[MilpBound]] = {
        "primary_charge": [(0.0, 1.0)] * m,
        "primary_discharge": [(0.0, 1.0)] * m,
        "grid_import": [(0.0, 2.48)] * m,
        "grid_export": [(0.0, 1.1)] * m,
        "pv": [(0.0, 0.0)] * m,
        "primary_throughput": [(0.0, None)] * m,
        "soc_max_penalty": [(0.0, None)] * m,
        "soc_min_penalty": [(0.0, None)] * m,
        "curtailment": [(0.0, 0.0)] * m,
        "ev_0_charge": [(0.0, 0.5)] * m,
        "ev_0_target_penalty": [(0.0, None)],
        "grid_import_penalty": [(0.0, None)] * m,
        "primary_battery_export": [(0.0, 1.0)] * m,
        "pv_export": [(0.0, 0.1)] * m,
        "export_source_mode": [(0.0, 1.0)] * m,
        "primary_action_mode": [(0.0, 1.0)] * m,
        "grid_flow_mode": [(0.0, 1.0)] * m,
        "battery_export_mode": [(0.0, 1.0)] * m,
        "primary_terminal_inventory": [(0.0, 0.25), (0.0, 0.75)],
        "secondary_charge": [(0.0, 0.48)] * m,
        "secondary_discharge": [(0.0, 3.2)] * m,
        "secondary_throughput": [(0.0, None)] * m,
        "secondary_charge_mode": [(0.0, 1.0)] * m,
        "secondary_sbu_mode": [(0.0, 1.0)] * m,
        "secondary_charge_steps": [(0.0, 2.0)] * m,
    }
    assert set(expected_domains) == set(blocks)

    captured_bounds = captured["bounds"]
    for name, expected in expected_domains.items():
        metadata = blocks[name]
        offset = int(metadata["offset"])
        width = int(metadata["width"])
        _assert_bound_sequences(
            captured_bounds[offset : offset + width],
            expected,
        )
