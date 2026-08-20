# ADR-003: Cost Scoring Architecture

**Status:** Accepted

**Date:** 2026-05-11

**Deciders:** Project maintainers

---

## Context

The HSEM planner needs to evaluate and compare candidate battery charge/discharge plans. The evaluation function must achieve two goals:

1. **Financial accuracy** — produce an auditable cost figure that corresponds to real money (what the user will actually pay or save).
2. **Plan selection** — provide a ranking metric that the candidate selector can use to pick the best plan, including trade-offs that have no direct monetary value (e.g., battery wear, safety constraints, future opportunity cost).

A naive single-aggregate approach (one number for everything) conflates these two purposes. A selector score that mixes real costs with synthetic penalties makes the "total cost" reported to the user un-auditable and misleading. Conversely, a pure monetary cost cannot express soft constraints like "don't drain the battery to zero just because the horizon is short."

We needed an architecture that cleanly separates financial reporting from plan selection without duplicating calculation logic.

---

## Decision

We split the cost function into **two distinct aggregates** returned for every candidate plan:

### 1. `total_cost` — real-money terms only

```
total_cost = grid_import_cost − export_revenue + cycle_cost + conversion_loss_cost
```

- Every term in `total_cost` corresponds to a real monetary flow.
- **No synthetic penalties** enter this aggregate.
- Suitable for auditing, bill comparison, and user-facing display.
- The value is comparable to the user's actual electricity bill for the horizon period.

### 2. `score` — selector objective

```
score = total_cost + soc_penalties + grid_limit_penalty + terminal_soc_value
        + primary_action_tiebreak
```

- Starts from `total_cost` (always includes real money).
- Adds **synthetic penalties** (quadratic SoC guard and grid power limit).
- Adds the **terminal inventory value** — a path-independent, bounded primary
  value plus the independent uniform secondary value at the actionable
  boundary.
- Adds a separately named **primary-action structural tiebreak** so a true
  economic tie prefers battery-to-house discharge without rewarding
  charge/discharge or export/refill cycles.
- The selector always picks the candidate with the **lowest** `score`, not the lowest `total_cost`.

### Why two aggregates instead of a single weighted sum

| Concern | `total_cost` | `score` |
|---|---|---|
| Auditable money | ✅ Yes | ✅ (as subset) |
| Values justified post-boundary reserve | ❌ No | ✅ (via bounded terminal value) |
| Avoids SoC bound violations | ❌ No | ✅ (via quadratic guard) |
| Picks cheapest plan | ✅ If penalties=0 | ✅ Always |

A single number cannot serve both purposes without one of them being wrong.

### Terminal inventory value formulation

Terminal inventory value is the sum of independent primary and secondary
battery terms. Primary storage uses a bounded piecewise value function;
secondary storage retains its uniform scalar:

```text
V_primary(E)
    = Σ_i v_i × min(q_i, max(E − Σ_{j<i} q_j, 0))

primary_terminal
    = V_primary(E_primary_initial) − V_primary(E_primary_final)

secondary_terminal
    = (E_secondary_initial − E_secondary_final) × p_secondary

terminal_soc_value = primary_terminal + secondary_terminal
```

Each positive primary tier represents only the battery-side residual house
demand of one exactly aligned, non-actionable slot beyond the contiguous
published-price boundary. Quantity is bounded by load after PV/accounted EV,
discharge efficiency and power, usable capacity, and a global capacity cap.
Marginal value starts from the Unagi point after MAE and operator-margin
haircut, then subtracts discharge conversion loss and cycle wear. Duplicate
points use the lower value; invalid data creates no tier.

This replaces the old primary `max(published, forecast)` scalar, which could
assign one expensive forecast price to the whole battery and keep a newly
published peak reserved as the boundary rolled. When a price becomes official,
its slot leaves the terminal tiers and is evaluated normally inside the
actionable horizon. With no valid tier, the explicit
`hardware_floor_only` model has zero primary terminal value; the effective
hardware discharge floor is the only reserve.

Unagi remains structurally non-actionable: it does not become a slot price,
extend price authority, create revenue, or authorise a storage action. Both
battery terms remain path-independent because they depend only on final
inventory. An export discharge followed by equal recharge restores the same
final inventory and has zero net terminal value; explicit prices,
efficiencies, wear, capacity, headroom, and power constraints decide the
cycle.

### Primary-action structural tiebreak

The selector-only term is:

```
epsilon = 0.00001 currency / DC kWh
battery_export_DC[t] =
    primary_battery_export_kwh[t] / discharge_efficiency
local_discharge_DC[t] = discharge[t] − battery_export_DC[t]
primary_action_tiebreak
    = epsilon × Σ(charge[t] + discharge[t])
      − 1.5 × epsilon × Σ(local_discharge_DC[t])
```

The per-kWh perturbation is `+epsilon` for charge, `-0.5*epsilon` for local
discharge, and `+epsilon` for battery-origin export. A charge/local-discharge
cycle costs `0.5*epsilon` and an export/refill cycle costs `2*epsilon`. On a
true lossless/economic tie, local discharge wins, preserving the intent of
issues #638/#655. A 97%-efficient discharge at a flat tariff is not an
economic tie and is deliberately no longer forced.

This is a weighted structural tiebreak, not a mathematically lexicographic
objective. It may affect economics below epsilon, and the configured 0.5% MIP
gap does not guarantee proof of an epsilon-sized distinction. Under ordinary
48-hour/10 kW primary-battery bounds the total perturbation remains below
about 0.01 currency.
The MILP identifies
`battery_export_DC[t]` explicitly; its AC counterpart is
`discharge_efficiency × battery_export_DC[t]`, and the remaining aggregate
grid export is attributed to PV. A binary source branch enforces
`battery_export_DC[t] = min(discharge[t], grid_export[t]/discharge_efficiency)`;
this causal attribution cannot be changed by the objective. The non-battery
source is physically capped by available PV (plus only PowMr SBU-revealed PV),
and flexible EV load is not a battery-eligible local sink. A separate binary
primary action mode prevents simultaneous charge and discharge. A third binary
grid-flow mode, using finite physical import/export bounds, prevents a
simultaneous meter wash from manufacturing an export source that disappears
when final net flow is written.

### Quadratic SoC guard

```python
penalty = weight * (soc - bound)**2  # if soc outside [min, max]
```

Quadratic form heavily penalises large violations while tolerating tiny numerical rounding errors.

### Past-slot exclusion

Slots marked `time_passed` are excluded from SoC penalty calculation because the SoC simulator writes `estimated_battery_soc = 0.0` as a sentinel, which would generate a false penalty of `weight * min_soc²` per past slot — identical across all candidates but log-misleading.

---

## Consequences

### Positive

- `total_cost` is auditable: every term maps to a real money flow. Users can compare it to their electricity bill.
- The selector can express preferences that have no monetary value (e.g., "don't violate SoC bounds") without corrupting the financial aggregate.
- Clear separation of concerns: cost function returns two numbers; the selector uses one, diagnostics expose both.
- Adding a new penalty (e.g., carbon intensity) adds it to `score` only, leaving `total_cost` untouched.

### Negative

- Callers must be aware of which aggregate to use. The wrong choice (using `total_cost` for selection, or `score` for billing) produces incorrect results.
- Slightly more complex API surface: every evaluation returns two floats instead of one.
- Terminal inventory value and the primary-action structural tiebreak
  are synthetic selector terms — neither is money the user will actually pay
  or receive.

### Trade-offs considered

- **Single weighted-sum approach** was rejected because it conflates monetary and non-monetary terms, making the "total cost" neither auditable nor a pure selector score.
- **Lexicographic ordering** (first minimise cost, then minimise penalties) was rejected because it cannot express trade-offs between cost and safety (e.g., paying slightly more to avoid draining the battery).
- **Post-selection re-costing** (compute pure cost after picking by score) was rejected because it introduces a possible mismatch between the selected plan and the reported cost, violating the `winner.cost == final_output.cost` invariant.

### Invariant

For every planner run:

- `winner.score == final_output.score` (no post-selection mutation)
- `winner.total_cost == final_output.total_cost`
- `score == total_cost + penalties + terminal_soc_value + primary_action_tiebreak`
- `score` may be above or below `total_cost`: penalties are non-negative,
  while terminal inventory and the structural tiebreak may have either sign
- When all penalties and selector-only inventory/tiebreak terms are zero:
  `score == total_cost`
