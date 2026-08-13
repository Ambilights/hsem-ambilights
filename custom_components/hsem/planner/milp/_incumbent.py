"""Validation for time-limited HiGHS MILP incumbents.

HiGHS may return a feasible integer solution when it reaches its time limit,
but ``scipy.optimize.linprog`` marks that result as unsuccessful because
optimality was not proven.  HSEM may use such a solution only after checking
the complete model that was passed to the solver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

_FEASIBILITY_TOLERANCE = 1e-5
_INTEGRALITY_TOLERANCE = 1e-5


@dataclass(frozen=True)
class IncumbentValidation:
    """Result of validating one solver decision vector."""

    valid: bool
    reason: str
    max_equality_residual: float = 0.0
    max_inequality_violation: float = 0.0
    max_bound_violation: float = 0.0
    max_integrality_violation: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe fields for planner diagnostics."""
        return asdict(self)


def _invalid(
    reason: str,
    *,
    max_equality_residual: float = 0.0,
    max_inequality_violation: float = 0.0,
    max_bound_violation: float = 0.0,
    max_integrality_violation: float = 0.0,
) -> IncumbentValidation:
    """Build a failed validation result."""
    return IncumbentValidation(
        valid=False,
        reason=reason,
        max_equality_residual=max_equality_residual,
        max_inequality_violation=max_inequality_violation,
        max_bound_violation=max_bound_violation,
        max_integrality_violation=max_integrality_violation,
    )


def validate_incumbent(  # NOSONAR
    result_x: Any,
    *,
    n_vars: int,
    slot_count: int,
    future_idx: list[int],
    m: int,
    variable_blocks: dict[str, tuple[int, int]],
    a_eq: Any,
    b_eq: Any,
    a_ub: Any,
    b_ub: Any,
    bounds: list[tuple[float | None, float | None]],
    integrality: Any,
) -> IncumbentValidation:
    """Validate a time-limited decision vector against the complete model.

    The matrix checks include every primary battery, export-reserve, fuse,
    phase-aware, EV, and secondary-storage constraint because the final
    extended matrices are supplied here immediately before result decoding.
    """
    import numpy as np

    if result_x is None:
        return _invalid("missing_solution_vector")

    try:
        x = np.asarray(result_x, dtype=float)
    except TypeError, ValueError:
        return _invalid("invalid_solution_vector")

    if x.ndim != 1:
        return _invalid("solution_vector_not_one_dimensional")
    if x.size != n_vars:
        return _invalid(f"solution_vector_length_{x.size}_expected_{n_vars}")
    if not np.all(np.isfinite(x)):
        return _invalid("solution_vector_not_finite")

    if m <= 0 or len(future_idx) != m:
        return _invalid("future_horizon_length_mismatch")
    if any(index < 0 or index >= slot_count for index in future_idx):
        return _invalid("future_horizon_index_out_of_range")
    if any(right <= left for left, right in zip(future_idx, future_idx[1:])):
        return _invalid("future_horizon_not_strictly_increasing")

    for name, (offset, length) in variable_blocks.items():
        if length != m:
            return _invalid(f"variable_block_{name}_length_{length}_expected_{m}")
        if offset < 0 or offset + length > n_vars:
            return _invalid(f"variable_block_{name}_out_of_range")

    if len(bounds) != n_vars:
        return _invalid(f"bounds_length_{len(bounds)}_expected_{n_vars}")

    max_bound_violation = 0.0
    for value, (lower, upper) in zip(x, bounds, strict=True):
        if lower is not None:
            max_bound_violation = max(max_bound_violation, float(lower) - value)
        if upper is not None:
            max_bound_violation = max(max_bound_violation, value - float(upper))
    max_bound_violation = max(max_bound_violation, 0.0)
    if max_bound_violation > _FEASIBILITY_TOLERANCE:
        return _invalid(
            "bound_violation",
            max_bound_violation=max_bound_violation,
        )

    eq_matrix = np.asarray(a_eq, dtype=float)
    eq_rhs = np.asarray(b_eq, dtype=float)
    if eq_matrix.ndim != 2 or eq_matrix.shape[1] != n_vars:
        return _invalid("equality_matrix_shape_mismatch")
    if eq_rhs.ndim != 1 or eq_matrix.shape[0] != eq_rhs.size:
        return _invalid("equality_rhs_shape_mismatch")
    eq_residual = eq_matrix @ x - eq_rhs
    max_equality_residual = (
        float(np.max(np.abs(eq_residual))) if eq_residual.size else 0.0
    )
    if max_equality_residual > _FEASIBILITY_TOLERANCE:
        return _invalid(
            "equality_constraint_violation",
            max_equality_residual=max_equality_residual,
            max_bound_violation=max_bound_violation,
        )

    ub_matrix = np.asarray(a_ub, dtype=float)
    ub_rhs = np.asarray(b_ub, dtype=float)
    if ub_matrix.ndim != 2 or ub_matrix.shape[1] != n_vars:
        return _invalid("inequality_matrix_shape_mismatch")
    if ub_rhs.ndim != 1 or ub_matrix.shape[0] != ub_rhs.size:
        return _invalid("inequality_rhs_shape_mismatch")
    ub_residual = ub_matrix @ x - ub_rhs
    max_inequality_violation = (
        max(float(np.max(ub_residual)), 0.0) if ub_residual.size else 0.0
    )
    if max_inequality_violation > _FEASIBILITY_TOLERANCE:
        return _invalid(
            "inequality_constraint_violation",
            max_equality_residual=max_equality_residual,
            max_inequality_violation=max_inequality_violation,
            max_bound_violation=max_bound_violation,
        )

    max_integrality_violation = 0.0
    if integrality is not None:
        integral_flags = np.asarray(integrality, dtype=int)
        if integral_flags.ndim != 1 or integral_flags.size != n_vars:
            return _invalid("integrality_vector_shape_mismatch")
        integral_values = x[integral_flags != 0]
        if integral_values.size:
            max_integrality_violation = float(
                np.max(np.abs(integral_values - np.rint(integral_values)))
            )
        if max_integrality_violation > _INTEGRALITY_TOLERANCE:
            return _invalid(
                "integrality_violation",
                max_equality_residual=max_equality_residual,
                max_inequality_violation=max_inequality_violation,
                max_bound_violation=max_bound_violation,
                max_integrality_violation=max_integrality_violation,
            )

    return IncumbentValidation(
        valid=True,
        reason="feasible",
        max_equality_residual=max_equality_residual,
        max_inequality_violation=max_inequality_violation,
        max_bound_violation=max_bound_violation,
        max_integrality_violation=max_integrality_violation,
    )
