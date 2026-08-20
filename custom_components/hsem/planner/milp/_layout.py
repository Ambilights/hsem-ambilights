"""Single source of truth for MILP decision-vector column layout."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

type MilpBound = tuple[float | None, float | None]

_UNSET_BOUND = object()


@dataclass(frozen=True)
class ColumnBlock:
    """One contiguous named block in the MILP decision vector."""

    offset: int
    width: int
    per_slot: bool = True


@dataclass
class MilpColumnLayout:
    """Allocate named column blocks and validate every matrix consumer."""

    slot_count: int
    blocks: dict[str, ColumnBlock] = field(default_factory=dict)
    column_count: int = 0

    def add(self, name: str, width: int, *, per_slot: bool = True) -> int:
        """Append *width* columns and return the new block's offset."""
        if name in self.blocks:
            raise ValueError(f"duplicate MILP column block: {name}")
        if width < 0:
            raise ValueError(f"negative MILP block width for {name}: {width}")
        if per_slot and width != self.slot_count:
            raise ValueError(
                f"per-slot MILP block {name} has width {width}, "
                f"expected {self.slot_count}"
            )
        offset = self.column_count
        self.blocks[name] = ColumnBlock(offset, width, per_slot)
        self.column_count += width
        return offset

    def offset(self, name: str) -> int:
        """Return the declared offset for *name*."""
        return self.blocks[name].offset

    def variable_blocks(self) -> dict[str, tuple[int, int]]:
        """Return per-slot blocks for incumbent structural validation."""
        return {
            name: (block.offset, block.width)
            for name, block in self.blocks.items()
            if block.per_slot
        }

    def as_dict(self) -> dict[str, dict[str, int | bool]]:
        """Return a diagnostics-safe representation of the declaration."""
        return {
            name: {
                "offset": block.offset,
                "width": block.width,
                "per_slot": block.per_slot,
            }
            for name, block in self.blocks.items()
        }

    def assert_model_width(
        self,
        *,
        objective: Any | None = None,
        a_eq: Any | None = None,
        a_ub: Any | None = None,
        bounds: Sequence[MilpBound] | None = None,
    ) -> None:
        """Fail before solving when any hand-built consumer has drifted."""
        expected = self.column_count
        actuals: dict[str, int] = {}
        if objective is not None:
            actuals["objective"] = len(objective)
        if a_eq is not None:
            actuals["A_eq"] = int(a_eq.shape[1])
        if a_ub is not None:
            actuals["A_ub"] = int(a_ub.shape[1])
        if bounds is not None:
            actuals["bounds"] = len(bounds)
        mismatched = {
            name: width for name, width in actuals.items() if width != expected
        }
        if mismatched:
            rendered = ", ".join(
                f"{name}={width}" for name, width in mismatched.items()
            )
            raise ValueError(
                f"MILP column-layout mismatch: declared={expected}, {rendered}"
            )


class MilpBoundsBuilder:
    """Populate solver bounds by declared block name, never call order."""

    def __init__(self, layout: MilpColumnLayout) -> None:
        """Snapshot *layout* and preallocate one sentinel per solver column."""
        self._layout = layout
        self._blocks = dict(layout.blocks)
        self._column_count = layout.column_count
        self._entries: list[MilpBound | object] = [_UNSET_BOUND] * self._column_count
        self._assigned_blocks: set[str] = set()
        self._validate_layout_snapshot()

    def set(self, name: str, values: Sequence[MilpBound]) -> None:
        """Write one named block using exactly its declared number of bounds."""
        self._assert_layout_unchanged()
        block = self._resolve_block(name)
        if name in self._assigned_blocks:
            raise ValueError(f"duplicate MILP bounds block assignment: {name}")
        if len(values) != block.width:
            raise ValueError(
                f"MILP bounds block {name} has width {len(values)}, "
                f"expected {block.width}"
            )

        normalised = [
            self._normalise_bound(name, block.offset + index, bound)
            for index, bound in enumerate(values)
        ]
        self._write_block(name, block, normalised)

    def fill(self, name: str, bound: MilpBound) -> None:
        """Fill one named block with a single validated bound."""
        self._assert_layout_unchanged()
        block = self._resolve_block(name)
        if name in self._assigned_blocks:
            raise ValueError(f"duplicate MILP bounds block assignment: {name}")
        normalised = self._normalise_bound(name, block.offset, bound)
        self._write_block(name, block, [normalised] * block.width)

    def finalize(self) -> list[MilpBound]:
        """Return complete SciPy tuple bounds or fail with missing block ranges."""
        self._assert_layout_unchanged()
        missing: list[str] = []
        for name, block in self._blocks.items():
            unset_indices = [
                index
                for index in range(block.offset, block.offset + block.width)
                if self._entries[index] is _UNSET_BOUND
            ]
            if name not in self._assigned_blocks or unset_indices:
                rendered = _render_index_ranges(
                    unset_indices
                    if unset_indices
                    else range(block.offset, block.offset + block.width)
                )
                missing.append(f"{name}[{rendered}]")
        if missing:
            raise ValueError(f"missing MILP bounds blocks: {', '.join(missing)}")

        return [cast(MilpBound, entry) for entry in self._entries]

    def _resolve_block(self, name: str) -> ColumnBlock:
        """Return one snapshotted block or reject an unknown name."""
        block = self._blocks.get(name)
        if block is None:
            raise ValueError(f"unknown MILP bounds block: {name}")
        return block

    def _write_block(
        self,
        name: str,
        block: ColumnBlock,
        values: Sequence[MilpBound],
    ) -> None:
        """Write one already-validated block and reject occupied columns."""
        occupied = [
            index
            for index in range(block.offset, block.offset + block.width)
            if self._entries[index] is not _UNSET_BOUND
        ]
        if occupied:
            raise ValueError(
                f"overlapping MILP bounds block {name} at columns "
                f"{_render_index_ranges(occupied)}"
            )
        self._entries[block.offset : block.offset + block.width] = values
        self._assigned_blocks.add(name)

    def _validate_layout_snapshot(self) -> None:
        """Reject malformed or overlapping declared column ranges."""
        owners: list[str | None] = [None] * self._column_count
        for name, block in self._blocks.items():
            if (
                block.offset < 0
                or block.width < 0
                or block.offset + block.width > self._column_count
            ):
                raise ValueError(
                    f"MILP bounds block {name} range "
                    f"[{block.offset}:{block.offset + block.width}] exceeds "
                    f"column count {self._column_count}"
                )
            for index in range(block.offset, block.offset + block.width):
                owner = owners[index]
                if owner is not None:
                    raise ValueError(
                        f"overlapping MILP bounds blocks {owner} and {name} "
                        f"at column {index}"
                    )
                owners[index] = name

        undeclared = [index for index, owner in enumerate(owners) if owner is None]
        if undeclared:
            raise ValueError(
                "MILP column layout has undeclared bounds columns: "
                f"{_render_index_ranges(undeclared)}"
            )

    def _assert_layout_unchanged(self) -> None:
        """Reject additions or mutations after builder preallocation."""
        if (
            self._layout.column_count != self._column_count
            or self._layout.blocks != self._blocks
        ):
            raise ValueError("MILP column layout changed after bounds preallocation")

    @staticmethod
    def _normalise_bound(
        name: str,
        column: int,
        bound: MilpBound,
    ) -> MilpBound:
        """Validate and normalize one SciPy lower/upper tuple."""
        if not isinstance(bound, tuple) or len(bound) != 2:
            raise ValueError(
                f"invalid MILP bound for {name}[{column}]: expected (lower, upper)"
            )
        lower = MilpBoundsBuilder._normalise_endpoint(
            name,
            column,
            "lower",
            bound[0],
        )
        upper = MilpBoundsBuilder._normalise_endpoint(
            name,
            column,
            "upper",
            bound[1],
        )
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(
                f"invalid MILP bound for {name}[{column}]: "
                f"lower {lower} exceeds upper {upper}"
            )
        return (lower, upper)

    @staticmethod
    def _normalise_endpoint(
        name: str,
        column: int,
        endpoint_name: str,
        value: float | None,
    ) -> float | None:
        """Return one finite float endpoint, retaining ``None`` as unbounded."""
        if value is None:
            return None
        try:
            finite = math.isfinite(value)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"invalid MILP bound for {name}[{column}]: "
                f"{endpoint_name} endpoint is not numeric"
            ) from err
        if not finite:
            raise ValueError(
                f"invalid MILP bound for {name}[{column}]: "
                f"{endpoint_name} endpoint must be finite or None"
            )
        return float(value)


def _render_index_ranges(indices: Sequence[int] | range) -> str:
    """Render sorted column indices compactly for validation errors."""
    values = list(indices)
    if not values:
        return ""
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)
