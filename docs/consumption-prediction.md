# HSEM Consumption Prediction

This document explains how HSEM predicts house load (consumption) for the
planning horizon — the weighted-average model, outlier detection, spike
suppression, and reliability weighting.

---

## Overview

HSEM predicts house load using a **multi-window weighted average** of historical
consumption data. The prediction feeds the planner's net consumption calculation:

$$ net\\_consumption[t] = load\\_forecast[t] + ev\\_load[t] - pv\\_forecast[t] $$

Accurate load prediction is critical: over-prediction leads to unnecessary grid
imports; under-prediction leads to insufficient battery charging for peak hours.

HSEM supports two prediction modes, toggled via ``hsem_ml_consumption_enabled``:

- **Legacy** (default): Four-window weighted average using HSEM custom sensors
- **ML** (optional): Ridge regression on recorder history with local-calendar
  features, seasonality, and optional outdoor temperature

---

## ML mode (ridge regression)

When enabled, the ML predictor queries the HA recorder directly for historical
energy data from the configured energy sensor.  No custom sensor entities are required.

### Calendar features and physical slot identity

Recorder timestamps are UTC instants. HSEM converts every aware timestamp to
Home Assistant's configured timezone at the recorder boundary, then derives
day-of-week, day-of-year, local date, and wall-clock slot features from that
local value. Ordering, elapsed age, adjacency, and dictionary identity remain
UTC-based physical instants.

This separation is required at daylight-saving transitions. At 15-minute
resolution a complete local civil day contains 92, 96, or 100 physical slots.
Both occurrences of an autumn repeated hour remain distinct observations even
though they intentionally share the same wall-clock model feature. A spring
gap has no fabricated observations.

For cumulative energy sensors, HSEM keeps the last reading in each physical
slot. The delta from the previous slot's last reading to the current slot's
last reading is energy consumed in the **current** slot and is labelled with
that slot. Only adjacent physical slots are paired; recorder gaps, accumulator
resets, non-positive deltas, and implausibly large deltas are rejected instead
of being shifted or concentrated into another slot.

### Model formulation

Weighted ridge regression solves:

$$
\hat{\beta} = (X^\mathsf{T} W X + \alpha I)^{-1} X^\mathsf{T} W y
$$

where:

- $X$ is the $n \times k$ design matrix ($n$ observations, $k$ features)
- $W = \mathrm{diag}(w_1, \ldots, w_n)$ weights recent observations higher
- $\alpha = 1.0$ is the L2 regularization strength
- $y$ is the vector of observed per-slot energy (kWh)

Time-decay sample weights:

$$
w_i = \exp\left(-\frac{a_i}{\tau}\right)
$$

where $a_i$ is the age of observation $i$ in days and $\tau$ is the decay
half-life (default: half the configured history window).

### Feature layout

For 15-minute slots ($S = 96$):

| Index range | Count | Feature | Description |
|---|---|---|---|
| $0 \ldots 7S-1$ | 672 | one-hot $(\text{DOW}, \text{slot})$ | Day-of-week × 15-min wall slot |
| $7S$, $7S+1$ | 2 | $\sin(\text{DOY}), \cos(\text{DOY})$ | Day-of-year seasonality |
| $7S+2$ | 1 | $T$ | Outdoor temperature (°C), optional |
| $7S+2$ without temperature; $7S+3$ with it | 1 | $E_{t-1}$ | Previous slot energy (sequential mode only) |

Total: 674 without either optional feature, 675 with temperature or sequential
mode, and 676 with both.

### Prediction (independent mode)

For a target slot $(\text{DOW} = d, \text{slot} = s)$ on day-of-year $\delta$:

$$
\hat{E}_{d,s} = \beta_{d,s} + \beta_{\sin} \cdot \sin\left(\frac{2\pi\delta}{365}\right)
+ \beta_{\cos} \cdot \cos\left(\frac{2\pi\delta}{365}\right)
+ \beta_T \cdot T + \beta_{\text{lag}} \cdot E_{t-1}
$$

where $\beta_{d,s}$ is the fitted coefficient for that (DOW, slot) pair.
The intercept term is zero (absorbed by the one-hot encoding).

### Prediction (sequential mode)

When sequential prediction is enabled, slots within a day are predicted in
order and each output is fed as the lag input to the next slot:

$$
\begin{aligned}
\hat{E}_0 &= f(d, 0, \delta, T_0, 0) \\
\hat{E}_1 &= f(d, 1, \delta, T_1, \hat{E}_0) \\
\hat{E}_2 &= f(d, 2, \delta, T_2, \hat{E}_1) \\
&\;\vdots
\end{aligned}
$$

where $f(\cdot)$ is the prediction function above and $E_{-1} = 0$.
This captures intra-day momentum — a cooking spike at 08:00 naturally
elevates 08:15's prediction.
Sequential training resets the lag whenever two samples are not adjacent in
physical UTC time. Inference follows the real recommendation instants in
physical UTC order, skipping nonexistent spring wall slots and preserving both
autumn folds without inventing a synthetic 96-slot civil day.


### Fitting

Fitting is two-stage. Time-decay weighted `(day-of-week, slot)` group means are
first shrunk toward the same wall slot's all-weekday mean. Weighted ridge
regression then fits seasonality and any enabled temperature or lag feature to
the remaining residual. This preserves the slot profile when individual
weekday groups are sparse.

A retrain gate compares valid sample fingerprints rather than only the rolling
sample count. The matrix solve runs after at least four unseen or revised
fingerprints, so corrected recent readings retrain the model even if the window
count stays constant. The fit runs in Home Assistant's executor pool so it does
not block the event loop.

### Adaptive safety buffer

Each slot gets a per-slot safety margin based on the weighted standard
deviation of its (DOW, slot) group:

$$
\sigma_{d,s} = \sqrt{
    \frac{\sum_i w_i (E_i - \bar{E}_w)^2}{\sum_i w_i}
}
$$

where $\bar{E}_w$ is the time-decay weighted mean.  The buffer multiplies
$\sigma$ by an adaptive factor based on relative uncertainty:

| $\sigma / \mu$ | Safety factor | Meaning |
|---|---|---|
| $< 0.1$ | $0.0$ | Prediction is reliable — trust it |
| $< 0.3$ | $0.5$ | Moderate uncertainty — small buffer |
| $\geq 0.3$ | $1.0$ | Sparse or variable data — full buffer |

The MILP receives $\mu + f \cdot \sigma$ as the consumption estimate,
naturally building headroom in uncertain slots.  As history accumulates
and $\sigma$ shrinks, the buffer converges to zero automatically.

### Today's actuals

For completed slots on the current Home Assistant-local date, the predictor
uses actual meter deltas instead of predictions. The query starts one physical
slot before local midnight so the first slot retains its left accumulator
boundary. Actuals are keyed by canonical UTC slot start, so both folds of an
autumn repeated hour survive independently and an in-progress slot is never
treated as complete.

When net-consumption mode is enabled, import and export actuals are joined only
when both meters contain the same physical slot. A missing export observation
is unknown, not zero.

### Recorder caches and fail-safe fallback

- The configured history minimum defaults to 14 physical days; recorder reads
  are capped at 90 days.
- Processed history is cached for one physical hour. The cache key includes
  the Home Assistant instance, import and export entities, net/gross mode,
  slot cadence, and required history span. Temperature history has its own
  source-keyed cache.
- Changing a source, cadence, history requirement, net/gross mode, temperature
  availability, or sequential setting replaces the predictor. Coefficients
  from one configuration cannot pass the new-context retrain gate.
- Net mode requires an export entity and uses the physical intersection of
  import and export history. If either source is unavailable or the aligned
  result no longer spans the required history, HSEM falls back to legacy mode.
- A configured temperature feature is used only when enough temperature
  history exists. Training matches physical timestamps; inference uses the
  newest nearby observed temperature and holds it across the horizon because
  this input is not a future weather forecast. If history is insufficient,
  HSEM fits without temperature.
- If history cannot train a model, HSEM reports ML population failure and the
  coordinator uses the legacy averages. It never publishes an all-zero load
  horizon from an untrained predictor.

### Advantages over legacy mode

- **15-min resolution**: matches Nord Pool spot market
- **Day-of-week awareness**: Monday ≠ Saturday
- **Seasonality**: winter mornings get higher predictions than summer
- **Temperature**: optional observed outdoor temperature can explain
  weather-driven load
- **DST-safe identity**: repeated-hour observations and cache age use physical
  UTC time while model calendars remain local
- **No custom sensors**: reads directly from recorder database

---

## Legacy mode (weighted average)

Four overlapping historical windows are maintained per clock-hour (0–23):

| Window | Span | Default weight | Purpose |
|---|---|---|---|
| **1-day** | Last 24 hours | 25 % | Captures yesterday's pattern (weather, routine) |
| **3-day** | Last 72 hours | 30 % | Short-term trend (weekday pattern) |
| **7-day** | Last 168 hours | 30 % | Weekly rhythm (same weekday last week) |
| **14-day** | Last 336 hours | 15 % | Long-term baseline (weather-independent) |

Default weights sum to 100 %. Configurable via the options flow.

### Hourly averages

Each `HourlyConsumptionAverage` carries per-window averages for one clock-hour:

```python
@dataclass
class HourlyConsumptionAverage:
    hour: int           # 0-23
    avg_1d: float       # kWh average over the last 24 h for this hour
    avg_3d: float       # kWh average over the last 72 h
    avg_7d: float       # kWh average over the last 168 h
    avg_14d: float      # kWh average over the last 336 h
    day_offset: int     # 0 = today, 1 = tomorrow, ...
```

### Forecast computation

The raw forecast for hour `h` is:

$$ \mathrm{forecast}[h] = \frac{w_1 \cdot avg_1 + w_3 \cdot avg_3 + w_7 \cdot avg_7 + w_{14} \cdot avg_{14}}{w_1 + w_3 + w_7 + w_{14}} $$

Before this weighted average, the weights undergo three transformations:

1. **IQR outlier detection** — flag anomalous windows
2. **Spike detection and redistribution** — suppress sudden jumps
3. **Reliability weighting** — down-weight windows that disagree

And the raw window values first pass through one clamp:

0. **Peer-median clamp** (issue #592) — no window may exceed
   `max(3 × median of the other three windows, 0.15 kWh/h)`.  This catches
   the pattern the IQR mask structurally misses: with only four points,
   three windows agreeing low and one stale window 10–100× higher puts the
   median *between* the clusters, flagging both ends and zeroing the clean
   recent windows too.  Only the upward side is clamped — a genuine
   consumption drop flows through immediately.

---

## IQR outlier detection (issue #301)

Replaces the old ratio-based spike detection with the standard Tukey fence.

### Method

For each clock-hour, the four window values form a set of four data points.
The interquartile range (IQR) is computed, and values outside

$$ [Q_1 - k \cdot \mathrm{IQR}, Q_3 + k \cdot \mathrm{IQR}] $$

are flagged as outliers, where $k = 1.5$ (standard Tukey fence).

### Weight redistribution

When a window is flagged as an outlier, its weight is redistributed to the
remaining non-outlier windows **proportionally**. If ALL windows are outliers
(degenerate case), no redistribution occurs — all weights are kept unchanged.

---

## Spike detection caps

Even after IQR filtering, the planner applies additional capping to prevent
short-term spikes from dominating the forecast.

### Caps between 7-day and 14-day

| Cap | Value | Meaning |
|---|---|---|
| `CAP7_DOWN` | 0.85 | 7-day avg cannot be < 85 % of 14-day avg |
| `CAP7_UP` | 1.15 | 7-day avg cannot be > 115 % of 14-day avg |
| `CAP14_DOWN` | 0.90 | 14-day avg cannot be < 90 % of 7-day effective avg |
| `CAP14_UP` | 1.10 | 14-day avg cannot be > 110 % of 7-day effective avg |

### Spike detection (ratio-based)

When a short window is significantly higher than a longer window, it is
flagged as a spike:

| Comparison | Ratio range | Max weight reduction | Redistribution |
|---|---|---|---|
| 1d vs 7d | 1.30 – 2.00 | 50 % of 1d weight | 20 % → 3d, 55 % → 7d, 25 % → 14d |
| 3d vs 7d | 1.20 – 1.80 | 30 % of 3d weight | 60 % → 7d, 40 % → 14d |
| 7d vs 14d | 1.20 – 1.60 | 20 % of 7d weight | 100 % → 14d |
| 14d vs 7d | 1.15 – 1.50 | 15 % of 14d weight | 100 % → 7d |

**Severity scaling:** The fraction of weight actually removed interpolates
between 0 at the `_MIN` ratio and the maximum at the `_MAX` ratio:

$$ reduced\\_fraction = \frac{\mathrm{ratio} - ratio\\_min}{ratio\\_max - ratio\\_min} \cdot max\\_reduction $$

### Baseline capping

Short windows (1-day, 3-day) are also capped against a blended baseline:

$$ \mathrm{baseline} = 0.70 \cdot avg_7 + 0.30 \cdot avg_{14} $$

$$ capped\\_value = \mathrm{clamp}(value, 0.80 \cdot \mathrm{baseline}, 1.20 \cdot \mathrm{baseline}) $$

The 3-day uses slightly looser bounds (0.85 – 1.15) to avoid removing legitimate
multi-day trends.

---

## Reliability weighting

After spike suppression, each window's weight is further scaled by its
agreement with the other windows:

$$ w_i' = w_i \cdot \frac{1}{\epsilon + |avg_i - \mathrm{median}|} $$

Where $\epsilon = 0.05$ kWh (prevents division by zero and over-sensitivity).

The scale strength is configurable via `RELIABILITY_SCALE_STRENGTH` (default 1.0).
Setting it to 0 disables reliability weighting entirely.

Weights are normalised after scaling so they still sum to the original total.

---

## Assumptions and limitations

### ML mode limitations

1. **Linear relationship**: The model assumes linear relationships between
   features and consumption.  Non-linear effects (e.g. U-shaped heating/cooling
   curve vs temperature) are approximated but not fully captured.

2. **Minimum history**: Requires at least 14 days of recorder history for the
   energy sensor.  Falls back to legacy mode below this threshold.

3. **Meter topology**: The model starts from one configured accumulator. Net
   mode can subtract a physically aligned export accumulator, but it cannot
   reconstruct behind-the-meter loads that neither source measures.

4. **No external events**: Holidays, parties, or unusual appliance usage are
   not explicitly modeled — they appear as unexplained variance.

5. **No future weather feed**: Temperature inference holds the latest observed
   value across the horizon; it does not forecast a temperature change.

### Legacy mode limitations

1. **Stationarity**: The model assumes consumption patterns are relatively stable
   over the 14-day window.  Major lifestyle changes (new EV, heat pump, home
   renovation) require 14 days to be fully reflected.

2. **Weather dependence**: The model has no weather inputs.  Weather-driven
   consumption (AC, heating) appears as unexplained variance unless correlated
   with the same-day-previous-week pattern (7-day window).

3. **No day-of-week distinction**: All windows are rolling and do not distinguish
   weekdays from weekends.  A Monday forecast uses the same weights as a Saturday
   forecast, relying on the 7-day window to capture the weekly rhythm.

4. **Zero-consumption hours**: Hours with consistently zero consumption (e.g.
   night-time) produce zero forecasts, which is correct for most installations.

5. **Outlier detection limitations**: With only four data points (1d, 3d, 7d, 14d)
   per hour, the IQR method has limited statistical power.  The spike caps act as
   a second line of defence.

---

## Future improvements

The ML mode (ridge regression) addresses several legacy limitations and is the
forward path.  Remaining improvements include:

- **Future weather input**: Replace held observed temperature with an
  authoritative forecast when one is available
- **Multi-modal decomposition**: Separate models for weather-driven, EV-driven,
  and baseline load
- **Configurable alpha and decay**: Expose regularization and time-decay as
  user-tunable parameters
