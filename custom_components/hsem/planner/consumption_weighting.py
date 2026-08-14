"""Consumption-window weighting helpers for the HSEM planner.

The functions in this module are pure arithmetic.  They are kept separate
from slot population so the time-series transformation module remains small
and focused.
"""

from __future__ import annotations

from custom_components.hsem.const import (
    BASELINE_7D_SHARE,
    BASELINE_14D_SHARE,
    CAP7_DOWN,
    CAP7_UP,
    CAP14_DOWN,
    CAP14_UP,
    CHANGE3_LIMIT_DOWN_FACTOR,
    CHANGE3_LIMIT_UP_FACTOR,
    CHANGE_LIMIT_DOWN_FACTOR,
    CHANGE_LIMIT_UP_FACTOR,
    IQR_OUTLIER_MULTIPLIER,
)
from custom_components.hsem.utils.logger import log_planner

# A single window may pull the blend at most this far above/below the
# median of the other three windows (issue #592).  Prevents one stale or
# polluted window (e.g. a 14d average still holding pre-fix EV-charging
# nights) from dominating when the other three windows agree.
WINDOW_PEER_CLAMP_FACTOR: float = 3.0
# Absolute floor for the peer band so a near-zero house baseline (night
# slots) still allows the clamp to bite — a pure ratio band is degenerate
# when the peers sit at ~0.03 kWh.
WINDOW_PEER_CLAMP_FLOOR_KWH: float = 0.15


def clamp_window_to_peer_median(
    values: list[float],
    *,
    factor: float = WINDOW_PEER_CLAMP_FACTOR,
    floor: float = WINDOW_PEER_CLAMP_FLOOR_KWH,
) -> list[float]:
    """Clamp each window into a sanity band around the median of its peers.

    For each value, the allowed band is ``[0, max(median_others × factor,
    floor)]`` — an absolute *floor* keeps the band meaningful when the
    peers are near zero (e.g. night slots at ~0.03 kWh).

    Only the upward side is clamped: a window reading *below* its peers is
    never inflated — a genuine drop in consumption must flow through
    immediately, while a stale/polluted window can only ever overstate.
    This catches the classic pollution pattern — three windows agreeing
    low, one stale window 10–100× higher — without punishing legitimate
    gradual trends (where all four windows move together).

    Args:
        values: The 4 window values (1d, 3d, 7d, 14d), kWh/hour.
        factor: Ratio band half-width.  Default 3.0.
        floor: Absolute minimum band width (kWh/hour).  Default 0.15.

    Returns:
        New list with each value clamped into its peer band.
    """
    n = len(values)
    if n < 2:
        return list(values)
    out: list[float] = []
    for i, v in enumerate(values):
        others = [values[j] for j in range(n) if j != i]
        srt = sorted(others)
        m = len(srt)
        median = srt[m // 2] if m % 2 == 1 else (srt[m // 2 - 1] + srt[m // 2]) / 2.0
        hi = max(median * factor, floor)
        out.append(min(v, hi))
    return out


def detect_outliers_iqr(
    values: list[float],
    multiplier: float = IQR_OUTLIER_MULTIPLIER,
) -> list[bool]:
    """Return a boolean mask flagging outlier values via median-ratio detection.

    With only 4 data points (1d, 3d, 7d, 14d), the classic IQR Tukey fence
    produces wide bounds that rarely flag anything.  Instead we use a
    median-ratio approach: a value is an outlier when its ratio to the
    median of all 4 values exceeds ``multiplier`` (for upward outliers) or
    falls below ``1/multiplier`` (for downward outliers).

    This detects both upward spikes (e.g. 10.0 vs 1.0) and downward anomalies
    (e.g. 0.188 vs 0.708) while allowing gradual trends (e.g. 2.0, 1.9, 1.8, 1.0).

    When all values are identical (median = 0), no value is flagged.

    Args:
        values: List of 4 float values (typically 4: 1d, 3d, 7d, 14d).
        multiplier: Ratio threshold.  Defaults to
            :data:`IQR_OUTLIER_MULTIPLIER` (1.5).

    Returns:
        List of booleans the same length as *values*, where ``True`` means
        the corresponding value is an outlier.
    """
    n = len(values)
    if n < 4:
        return [False] * n

    sorted_vals = sorted(values)
    # Median of 4 values = average of the two middle values
    median = (sorted_vals[1] + sorted_vals[2]) / 2.0

    if abs(median) < 1e-12:
        return [False] * n  # all near-zero — no outliers

    upper_ratio = multiplier
    lower_ratio = 1.0 / multiplier

    return [v / median > upper_ratio or v / median < lower_ratio for v in values]


def weighted_avg_consumption(
    value_1d: float,
    value_3d: float,
    value_7d: float,
    value_14d: float,
    w1: int,
    w3: int,
    w7: int,
    w14: int,
) -> tuple[float, list[bool]]:
    """Apply outlier-aware dynamic reweighting and return the weighted average.

    Outliers are detected across the four consumption windows.  An outlier's
    weight is redistributed proportionally to the non-outlier windows.  The
    7d/14d and baseline caps remain as a final safety net.

    Returns:
        ``(weighted_average, outlier_mask)`` where the mask is ordered 1d,
        3d, 7d, 14d.
    """
    log_planner(
        "debug",
        "[pop] weighted_avg_consumption  vals=%.3f/%.3f/%.3f/%.3f  weights=%d/%d/%d/%d",
        value_1d,
        value_3d,
        value_7d,
        value_14d,
        w1,
        w3,
        w7,
        w14,
    )
    w_total_config = w1 + w3 + w7 + w14
    if w_total_config == 0:
        return 0.0, [False, False, False, False]

    # Detect on raw values so capping cannot hide extreme spikes.
    raw = [value_1d, value_3d, value_7d, value_14d]
    outlier_mask = detect_outliers_iqr(raw)

    value_1d, value_3d, value_7d, value_14d = clamp_window_to_peer_median(raw)

    value_7d_eff = max(CAP7_DOWN * value_14d, min(value_7d, CAP7_UP * value_14d))
    value_14d_eff = max(
        CAP14_DOWN * value_7d_eff, min(value_14d, CAP14_UP * value_7d_eff)
    )

    baseline = BASELINE_7D_SHARE * value_7d_eff + BASELINE_14D_SHARE * value_14d_eff
    value_1d_eff = max(
        baseline * CHANGE_LIMIT_DOWN_FACTOR,
        min(value_1d, baseline * CHANGE_LIMIT_UP_FACTOR),
    )
    value_3d_eff = max(
        baseline * CHANGE3_LIMIT_DOWN_FACTOR,
        min(value_3d, baseline * CHANGE3_LIMIT_UP_FACTOR),
    )

    weights = [float(w1), float(w3), float(w7), float(w14)]
    non_outlier_weight = sum(
        w for w, is_out in zip(weights, outlier_mask) if not is_out
    )

    if non_outlier_weight > 1e-9 and any(outlier_mask):
        scale = w_total_config / non_outlier_weight
        w1_eff = weights[0] * scale if not outlier_mask[0] else 0.0
        w3_eff = weights[1] * scale if not outlier_mask[1] else 0.0
        w7_eff = weights[2] * scale if not outlier_mask[2] else 0.0
        w14_eff = weights[3] * scale if not outlier_mask[3] else 0.0
    else:
        w1_eff, w3_eff, w7_eff, w14_eff = weights

    result = round(
        value_1d_eff * (w1_eff / 100)
        + value_3d_eff * (w3_eff / 100)
        + value_7d_eff * (w7_eff / 100)
        + value_14d_eff * (w14_eff / 100),
        3,
    )
    return result, outlier_mask
