"""Single source of truth for MILP decision-vector column layout."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
        bounds: list[tuple[float, float | None]] | None = None,
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
