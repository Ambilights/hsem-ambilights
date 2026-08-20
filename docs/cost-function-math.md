# HSEM Cost Function — Mathematical Reference

This document provides the complete mathematical formulation of the HSEM cost function
(`planner/cost_function.py`). It is the source of truth for all cost calculations.

---

## Two-aggregate architecture

The cost function returns two distinct aggregates for every plan:

| Aggregate | Symbol | Contents | Used for |
|---|---|---|---|
| **total_cost** | $C_{total}$ | Real money terms only | Auditing, bill comparison |
| **score** | $S$ | $C_{total}$ + synthetic penalties + terminal inventory value + primary-action structural tiebreak | Candidate selection |

The selector picks the plan with the **lowest score**, not the lowest money cost.

---

## Total cost (money terms)

$$ C_{total} = C_{import} - R_{export} + C_{cycle} + C_{loss} $$

### Grid import cost

$$ C_{import} = \sum_{t \in slots} gi[t] \cdot p_{imp}[t] $$

Where $gi[t]$ is the actual grid import in kWh and $p_{imp}[t]$ is the import
price in currency/kWh.

> The cost function prices **actual grid energy drawn**, not stored energy.
> If the battery stores $x$ kWh and charge efficiency is $e$, the grid import
> is $x/e$, which includes conversion losses implicitly.

### Export revenue

$$ R_{export} = \sum_{t \in slots} ge[t] \cdot p_{exp}[t] $$

Where $ge[t]$ is the grid export in kWh and $p_{exp}[t]$ is the export price.

> Revenue is subtracted from total cost. Negative export prices (curtailment
> penalties) increase total cost.

### Battery cycle cost (depreciation)

$$ C_{cycle} = \sum_{t \in slots} \max(charge[t], discharge[t]) \cdot c_{cycle} $$

Where $c_{cycle}$ is the cycle cost per kWh throughput.

The cycle cost counts the **maximum** of charge and discharge per slot, not
their sum. This matches the MILP formulation where $m[t] = \max(ec[t], ed[t])$
and the 2× denominator in the cycle cost formula:

$$ c_{cycle} = \frac{purchase\_price}{2 \cdot usable\_kwh \cdot expected\_cycles} $$

The 2× denominator accounts for one full round-trip (charge + discharge = 2 ×
usable_kwh throughput per cycle). With this factor, charging $x$ kWh and
discharging $x$ kWh costs $c_{cycle} \cdot \max(x, x) = c_{cycle} \cdot x$,
which equals $\frac{purchase\_price \cdot x}{2 \cdot usable \cdot cycles}$ —
matching the expected wear for moving $x$ kWh through the battery in one
direction.

### Conversion loss cost

$$ C_{loss} = \sum_{t \in slots} \frac{charge[t] + discharge[t]}{2} \cdot \frac{\eta_{loss}}{100} \cdot \frac{p_{imp}[t] + p_{exp}[t]}{2} $$

Where $\eta_{loss}$ is the round-trip conversion loss percentage.

The conversion loss term prices the energy lost as heat during charge/discharge
at the average of import and export price — an opportunity-cost proxy.

When separate charge/discharge efficiencies are configured:

$$ \eta_{loss} = (1 - \eta_{chg} \cdot \eta_{dis}) \times 100 $$

Where $\eta_{chg}$ and $\eta_{dis}$ are efficiency fractions (e.g. 0.97).

## Score (selector objective)

$$ S = C_{total} + P_{soc} + P_{grid} + V_{terminal} + T_{action} $$

### SoC penalties (quadratic guard)

$$ P_{soc} = \sum_{t \in slots} \begin{cases}
w_{low} \cdot (soc_{min} - soc[t])^2 & \mathrm{if } soc[t] < soc_{min} \\
w_{high} \cdot (soc[t] - soc_{max})^2 & \mathrm{if } soc[t] > soc_{max} \\
0 & \mathrm{otherwise}
\end{cases} $$

These are **soft guards** — the SoC simulation already hard-clamps at hardware
limits, so violations are rare. The quadratic form heavily penalises large
deviations while tolerating tiny numerical rounding errors.

**Past-slot exclusion:** Slots with `time_passed` recommendation are excluded
from SoC penalty calculation. The SoC simulator writes `estimated_battery_soc = 0.0`
as a sentinel on past slots, which would otherwise generate a false penalty of
$w_{low} \cdot soc_{min}^2$ per past slot — identical across all candidates
but log-misleading.

### Grid limit penalty

$$ P_{grid} = \sum_{t \in slots} \max(0, \frac{|gi[t] - ge[t]|}{\Delta t} - L_{grid}) \cdot \Delta t \cdot w_{grid} $$

Where $\Delta t$ is slot duration in hours, $L_{grid}$ is the configured grid
power limit in kW, and $w_{grid}$ is the penalty weight per excess kWh.

### Terminal inventory value (opportunity cost)

Primary storage uses a bounded piecewise value function, while secondary
storage retains its uniform replacement coefficient:

$$
\mathcal{V}_p(E)
= \sum_i v_i
  \min\!\left(q_i,\max\!\left(E-\sum_{j<i}q_j,0\right)\right)
$$

$$ V_p = \mathcal{V}_p(E_{p,initial})
         - \mathcal{V}_p(E_{p,final}) $$

$$ V_s = (E_{s,initial} - E_{s,final}) \cdot p_{s,replacement} $$

$$ V_{terminal} = V_p + V_s $$

Where:

- $E_{p,initial}$ and $E_{p,final}$ = primary stored energy above the discharge
  floor at the start and end of the horizon (kWh)
- $E_{s,initial}$ and $E_{s,final}$ = secondary stored energy above its reserve
  at the start and end of the horizon (kWh)
- $(q_i,v_i)$ = positive primary battery-side quantity and marginal-value
  tiers, ordered by decreasing $v_i$
- $p_{s,replacement}$ = the independent non-negative secondary replacement
  price; missing or non-finite values resolve to zero

**Sign convention:**

$$\begin{aligned}
\Delta E_i &< 0 \mathrm{ (more energy at end)} \rightarrow V_i < 0 \mathrm{ (credit)} \\
\Delta E_i &> 0 \mathrm{ (less energy at end)} \rightarrow V_i > 0 \mathrm{ (penalty)}
\end{aligned}$$

Here $i \in \{p,s\}$. For primary storage these signs apply only to inventory
movement through a valued tier; movement wholly above total tier quantity has
zero terminal value.
For secondary storage the uniform coefficient applies across its usable
inventory.

Both components depend only on final inventory. Equal discharge and recharge
therefore cancel exactly in $V_{terminal}$ regardless of slot positions.
Primary inventory above $\sum_i q_i$ has no synthetic salvage value.

The primary tiers come only from exactly aligned, future non-actionable Unagi
points at or after the end of the contiguous published-price prefix. For each
eligible physical slot:

$$
p_i^* = \max(p_{Unagi,i}-\max(MAE,0)-\max(margin,0),0)
$$

$$
q_i = \min\!\left(
  \frac{\max(load_i-PV_i-EV_{accounted,i},0)}{\eta_{dis}},
  discharge\_limit_i,
  usable\_capacity
\right)
$$

$$
v_i = p_i^*\eta_{dis}
      - p_i^*(1-\eta_{dis})
      - cycle\_wear
$$

Only finite positive tiers survive, duplicate points use the lower effective
price, and total tier quantity is capped by usable capacity. If nothing
survives, the explicit `hardware_floor_only` model has
$\mathcal{V}_p(E)=0$; the effective hardware discharge floor is the only
terminal reserve. When publication advances, the newly official slot leaves
the tiers and is priced normally in-horizon.

Unagi never becomes a slot price, changes price actionability, or authorises an
action beyond the boundary. `resolve_secondary_terminal_price()` remains a
separate uniform mean-of-window model for secondary storage.

### Primary-action structural tiebreak (selector only)

The terminal terms are paired with a tiny weighted tiebreak:

$$ \epsilon = 0.00001\ \mathrm{currency/DC\ kWh} $$

$$ local[t] = discharge[t] - battery\_export_{DC}[t] $$

$$ T_{action}
 = \epsilon\sum_t(charge[t]+discharge[t])
 - 1.5\epsilon\sum_t local[t] $$

`PlanCostBreakdown.primary_action_tiebreak` stores $T_{action}$ separately
from `terminal_soc_value`. It contributes to `score` but not to auditable
`total_cost`, and diagnostics expose the same field name.

The per-DC-kWh perturbation is $+\epsilon$ for charge,
$-0.5\epsilon$ for local discharge, and $+\epsilon$ for battery-origin
export. Charge/local-discharge and export/refill cycles therefore add
$0.5\epsilon$ and $2\epsilon$ respectively. On a true lossless/economic tie,
local discharge wins, preserving the intent of issues #638/#655. A 97%
efficient discharge at one flat tariff is not an economic tie and is
deliberately no longer forced.

This is a structural weighted tiebreak, not a mathematically lexicographic
objective. It can decide economics below $\epsilon$, and the 0.5% MIP gap does
not promise proof of an $\epsilon$-sized distinction. At ordinary
48-hour/10 kW primary-battery bounds the total perturbation stays below about
0.01 currency.

Explicit export-source attribution makes the distinction exact:

$$ battery\_export_{DC}[t]
   = \min\left(discharge[t],\frac{grid\_export[t]}{\eta_{dis}}\right) $$

$$ battery\_export_{AC}[t] = \eta_{dis} \cdot battery\_export_{DC}[t] $$

$$ pv\_export[t] = grid\_export[t] - battery\_export_{AC}[t] $$

Both source fields are non-negative. Raw solver values satisfy the identity
within solver tolerance; the published 0.001 kWh fields are reconciled so the
identity is exact at display precision.
The MILP enforces the minimum with binary `export_source_mode[t]`; source
attribution is not selected by a cost tiebreak. Its `pv_export[t]` upper bound
is physical forecast surplus plus only PV exposed by a PowMr SBU transition.
Active flexible EV demand is excluded from battery-eligible local sinks, so
concurrent PV, EV load, battery discharge, and export cannot hide
battery-origin export as PV export.

Binary `primary_action_mode[t]` also enforces exact charge-or-discharge
eligibility in every slot. Both actions may be zero, but they cannot be
positive together.
A third binary, `grid_flow_mode[t]`, similarly enforces
`grid_import[t] <= M_import[t]*grid_flow_mode[t]` and
`grid_export[t] <= M_export[t]*(1-grid_flow_mode[t])`. Both meter flows may
be zero, but they cannot be positive together. The finite $M$ values are
physical per-slot upper bounds, not arbitrary constants.

For backwards compatibility, `score_plan()` accepts older aggregate-only
slots. When their supplied source sum misses aggregate export by more than
0.002 kWh, it bounds battery-origin AC export by
`min(grid_export, discharge × discharge_efficiency)` and assigns the
remainder to PV. Production MILP and passive candidates publish explicit
sources and do not use this fallback.

---

## Past-slot exclusion rules

The cost function **skips** any slot whose recommendation is `time_passed`:

- All energy-flow fields (`grid_import_kwh`, `batteries_charged`, etc.) are zero
  on past slots
- Including them would only affect the SoC penalty (bogus $w_{low} \cdot soc_{min}^2$)
- Skipping past slots does not change the winner (the bogus penalty is identical
  across candidates) but keeps the reported cost clean

---

## Cost invariants (test assertions)

For every planner run:

1. $C_{total} = C_{import} - R_{export} + C_{cycle} + C_{loss}$ (exact)
2. No synthetic penalty enters $C_{total}$
3. $S = C_{total} + P_{soc} + P_{grid} + P_{override} + V_{terminal} + T_{action}$ (exact)
4. When all penalties and selector-only inventory/tiebreak terms are zero: $S = C_{total}$
5. Selector picks minimum $S$, not minimum $C_{total}$
6. $score_{winner} = score_{final\_output}$ (no post-selection mutation)
7. Within an active primary tier, or anywhere under the secondary uniform
   coefficient, an otherwise identical plan ending with more valued energy has
   lower $V_{terminal}$ and lower $S$
8. Equal primary or secondary battery discharge and recharge contribute zero
   net terminal value, regardless of which slots contain those actions
9. Primary inventory above total tier quantity has zero synthetic salvage
   value; an empty `hardware_floor_only` model contributes zero
10. $battery\_export_{AC}[t] + pv\_export[t] = grid\_export[t]$ within solver
   tolerance before publication and exactly at 0.001 kWh published precision
11. $T_{action}$ exactly matches
    $\epsilon\sum(charge+discharge)-1.5\epsilon\sum local$ after export-source
    reconciliation
12. Raw MILP source attribution satisfies
    $battery\_export_{DC}[t]=\min(discharge[t],grid\_export[t]/\eta_{dis})$,
    and raw charge/discharge are never simultaneously positive
13. Raw grid import and export are never simultaneously positive
