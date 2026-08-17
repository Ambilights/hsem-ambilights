"""Conditional primary-battery reserve for intentional grid export."""

from __future__ import annotations

from typing import Any

import numpy as np

_EPSILON_KWH = 1e-9


def _next_solar_refill_checkpoints(pv_avail: np.ndarray) -> np.ndarray:
    """Return the end-of-demand checkpoint following every active slot.

    A checkpoint is the slot immediately before the next forecast PV-surplus
    slot. When no later surplus exists, the horizon end is used. Requiring a
    reserve at that point protects the upcoming demand window while still
    allowing a later grid-charge decision to restore the reserve.
    """
    m = len(pv_avail)
    checkpoints = np.zeros(m, dtype=int)
    next_surplus: int | None = None

    for t in range(m - 1, -1, -1):
        if t + 1 < m and float(pv_avail[t + 1]) > _EPSILON_KWH:
            next_surplus = t + 1
        checkpoints[t] = m - 1 if next_surplus is None else max(t, next_surplus - 1)

    return checkpoints


def _add_battery_export_reserve_constraints(
    constraints: dict[str, Any],
    *,
    n_vars: int,
    m: int,
    ec_off: int,
    ed_off: int,
    export_mode_off: int,
    current_kwh: float,
    usable_kwh: float,
    discharge_eff: float,
    max_discharge_kwh: float,
    residual_house_load: np.ndarray,
    checkpoints: np.ndarray,
    reserve_kwh: float,
    primary_export_off: int | None = None,
) -> dict[str, Any]:
    """Append battery-export detection and conditional reserve constraints.

    export_mode[t] is forced to one whenever primary-battery discharge
    exceeds the house load left after forecast PV in slot t. If it is one,
    primary SoC at the following solar-refill checkpoint must be at least
    reserve_kwh.

    The reserve is deliberately conditional: ordinary self-consumption may
    use the full battery, while a deliberate grid-export trade must either
    retain the buffer or replenish it before the next solar opportunity.
    """
    old_a_ub = constraints["A_ub"]
    old_b_ub = constraints["b_ub"]
    old_rows = old_a_ub.shape[0]
    a_ub = np.zeros((old_rows + 2 * m, n_vars))
    b_ub = np.zeros(old_rows + 2 * m)
    a_ub[:old_rows, : old_a_ub.shape[1]] = old_a_ub
    b_ub[:old_rows] = old_b_ub

    # Big-M values are physical battery bounds, not arbitrary constants.
    max_delivered_kwh = max(max_discharge_kwh * discharge_eff, _EPSILON_KWH)
    max_export_dc_kwh = max(max_discharge_kwh, _EPSILON_KWH)
    soc_big_m_kwh = max(usable_kwh, _EPSILON_KWH)

    for t in range(m):
        if primary_export_off is not None:
            # Production models expose the destination directly: any positive
            # battery-origin DC export forces export mode on.
            a_ub[old_rows + t, primary_export_off + t] = 1.0
            a_ub[old_rows + t, export_mode_off + t] = -max_export_dc_kwh
        else:
            # Backward-compatible row for direct helper callers.
            a_ub[old_rows + t, ed_off + t] = discharge_eff
            a_ub[old_rows + t, export_mode_off + t] = -max_delivered_kwh
            b_ub[old_rows + t] = max(
                float(residual_house_load[t]),
                0.0,
            )

        # If z[t] == 1, preserve the configured reserve at the checkpoint:
        #   current + sum(ec-ed)[0:checkpoint] >= reserve
        # If z[t] == 0, usable_kwh relaxes the row to the normal zero-SoC floor.
        checkpoint = int(checkpoints[t])
        reserve_row = old_rows + m + t
        for k in range(checkpoint + 1):
            a_ub[reserve_row, ec_off + k] = -1.0
            a_ub[reserve_row, ed_off + k] = 1.0
        a_ub[reserve_row, export_mode_off + t] = soc_big_m_kwh
        b_ub[reserve_row] = current_kwh + soc_big_m_kwh - reserve_kwh

    constraints["A_ub"] = a_ub
    constraints["b_ub"] = b_ub
    constraints["bounds"] += [(0.0, 1.0)] * m
    return constraints
