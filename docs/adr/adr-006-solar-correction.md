# ADR-006: Solar Forecast Auto-Correction

**Status:** Accepted

**Date:** 2026-06-26

**Deciders:** Project maintainers

---

## Context

ADR-005 established a **diagnostic-only, non-adaptive** approach to forecast
confidence.  PV forecast errors were tracked and reported but never fed back
into the planner.  The rationale was stability: adaptive confidence creates
feedback loops, and non-stationary errors mean a system tuned for one season
fails in another.

Real-world operation of HSEM revealed a problem with this approach.  Solcast
PV forecasts exhibit **systematic per-hour bias patterns** — for example,
consistently over-forecasting morning production by 15 % while under-forecasting
midday production by 5 %.  The tracker reported these biases but offered no
mechanism to compensate for them.  Users with persistent forecast errors saw
the planner make suboptimal decisions based on systematically wrong PV values,
with no automated correction path.

The key questions were:

- Can we apply adaptive PV correction without introducing instability?
- How do we prevent the correction from over-fitting to short-term noise?
- Should the correction use the same data as the diagnostic tracker?

---

## Decision

We introduce **adaptive per-hour PV forecast correction** via the
`SolarForecastCorrector` (`utils/solar_corrector.py`). The corrector learns
from historical actual-vs-forecast PV ratios and applies corrections at
slot-population time — **before** the planner runs. Raw Solcast data is
never mutated.

This partially supersedes ADR-005's "non-adaptive" stance for PV data.
Load forecasts and price data remain non-adaptive. Learning is accepted only
from the final pre-slot baseline and fully covered actual energy; live-injected
current-slot values and incomplete telemetry are ineligible.

### Per-hour accuracy factors

For each clock-hour (0–23), the corrector maintains a rolling window of the
four most recent eligible **actual / raw forecast** PV ratios for that hour.
At 15-minute cadence, these are normally the four most recent physical
quarter-hour slots sharing that local clock-hour. This is a sample-count window,
not a four-day window. The mean ratio is clamped to **[0.3, 1.5]**.

```
factor[h] = clamp(mean(actual_pv / raw_pv for recent eligible samples at h), 0.3, 1.5)
corrected_pv[t] = raw_pv[t] × factor[hour_of(t)]
```

Each hour is learned independently. A near-zero raw forecast is skipped rather
than producing an unstable ratio.

### Intra-hour residual correction (2h linear decay)

In addition to the per-hour factor, a **residual correction** is applied to
the current and next slots. The residual is the mean of the four most recent
eligible **actual / corrected forecast** ratios. It decays linearly to 1.0
over eight slots (2 hours at 15-minute granularity).

```
residual = mean(actual/corrected_forecast for recent eligible closed slots)
decay[t] = 1.0 + (residual - 1.0) × max(0, 1 - slots_ahead / 8)
final_pv[t] = corrected_pv[t] × decay[t]
```

The `slots_ahead` value is a physical UTC distance from the current planning
instant, not an ordinal since midnight. The current in-progress slot maps to
zero, future boundaries advance by cadence, and both folds of an autumn
repeated hour remain distinct. This keeps the residual active throughout the
day and makes its decay DST-safe.

### Confidence scaling

The internal confidence value defaults to 0.50. Values below 0.50 damp the
learned hour factor toward 1.0; values at or above 0.50 apply the learned
factor at full strength. It does not select a historical percentile.
`sensor.hsem_solar_confidence_sensor` exposes the current factors,
confidence, and residual count as diagnostics.

### Integration point: slot population

Corrections are applied in `planner/slot_population.py` → `populate_solcast()`
**before** the planner runs. The raw Solcast values from the HA sensor are
never modified — only the per-slot copies in `PlannedSlot` receive the
correction.  This keeps the data pipeline auditable: the original forecast
is always available for comparison.

### Why this avoids the ADR-005 stability concerns

| ADR-005 concern | How ADR-006 addresses it |
|---|---|
| Feedback loops | Learning uses the frozen raw/corrected pre-slot baseline and measured energy, never a forecast rewritten after slot start. |
| Missing or stale telemetry | Only complete trusted physical-slot coverage is eligible; gaps are not treated as zero production. |
| Non-stationary errors | The rolling sample window adapts to change, while the [0.3, 1.5] clamp bounds both hour and residual ratios. |
| User transparency | The per-hour factors, confidence, and residual count are exposed as sensor attributes; corrected and raw values remain separate. |
| Minimal benefit | The correction is demonstrably beneficial: systematic per-hour bias of ±15 % over a full day translates to ±2–3 kWh of PV energy, which is enough to change the planner's charge/discharge decision. |

---

## Consequences

### Positive

- **Systematic PV bias is compensated automatically:** A Solcast over-forecast
  of 3 kWh/day no longer causes the planner to over-allocate solar surplus.
- **Per-hour granularity captures intra-day patterns:** Morning fog bias is
  corrected independently from midday clear-sky bias.
- **Short-term weather transitions are handled:** The residual correction
  adjusts after trusted closed-slot observations arrive.
- **Stable baselines:** Replanning after a slot starts cannot rewrite the
  forecast being evaluated or used for learning.
- **No raw data mutation:** Solcast API values are preserved for audit.
- **No new dependencies:** Pure Python, zero additional HA imports.

### Negative

- **Added complexity:** The planner pipeline now has an additional correction
  step that must be understood when debugging PV-related planning issues.
- **Cold start:** An hour with no eligible samples uses neutral factor 1.0.
  Residual correction is also neutral until an eligible closed slot exists.
- **Incomplete intervals learn nothing:** A restart or telemetry gap can delay
  learning, intentionally preferring no update over a false zero-actual sample.
- **Divergence from ADR-005:** The original "non-adaptive" principle is now
  qualified. Future contributors must understand that PV correction is adaptive
  while load and price treatment remain fixed.

### Mitigations

- Cold start: The corrector defaults to factor 1.0 (no correction) for hours
  with no history. The planner behaves identically to the pre-correction
  behaviour until data accumulates.
- Confidence scaling is bounded and never amplifies beyond the learned factor.
- Debug logging: Corrected vs raw PV is logged at debug level for every slot.
- The diagnostic `ForecastTracker` reports corrected forecast error while
  retaining raw PV separately for hour-factor learning.
- State is schema-versioned. v7.1.2 discards pre-v3 learned factors,
  history, and residuals because they may contain live-rewritten baselines;
  the confidence value is retained and learning restarts from eligible slots.
  Valid v3 state restores the exact bounded per-hour and residual buffers with
  a UTC processed-through watermark, preventing duplicate learning after a
  restart. Malformed, non-finite, or future-dated watermark state is rejected
  atomically.

---

## Alternatives Considered

### A. Keep non-adaptive (status quo from ADR-005)

Continue reporting forecast errors via the tracker but never correct them.

**Rejected because:** Systematic per-hour bias is a real, measurable problem.
Users with a consistent 15 % Solcast over-forecast see the planner make
suboptimal export and charge decisions every day.  "Report but don't fix"
is insufficient when the fix is straightforward and low-risk.

### B. Full machine-learning correction model

Train a model (e.g., gradient boosting) on weather features + historical
forecast errors to predict per-slot correction factors.

**Rejected because:**
- Massive complexity increase for marginal gain over the per-hour rolling mean.
- Requires weather data integration (cloud cover, irradiance) that HSEM does
  not currently collect.
- Training overhead and model persistence add infrastructure burden.

### C. Apply correction inside the MILP objective

Rather than pre-correcting PV values, add a correction term to the MILP
objective function.

**Rejected because:**
- Increases MILP variable count and solve time.
- Makes the correction opaque — the MILP output no longer reflects the
  physical PV forecast.
- Harder to audit: "Why did the MILP choose this plan?" is harder to answer
  when correction is inside the black box.

---

## Related

- ADR-005: Forecast Confidence (partially superseded by this ADR)
- `docs/forecast-accuracy-tracking.md` — Forecast accuracy technical guide
- `docs/planner-spec.md` — Solar correction invariant (§ Multi-day planning horizon)
- `docs/planner-guide.md` — Solar forecast auto-correction (§ Planning inputs → PV forecast)
- `utils/solar_corrector.py` — Implementation
- `planner/slot_population.py` — `populate_solcast()` integration point
- Issue #602 — Solar forecast accuracy auto-correction
