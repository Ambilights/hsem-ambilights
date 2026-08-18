# HSEM Candidate Generation — Deep Dive

This document explains how candidate plans are generated, how assumptions drive
the candidate set, and the mathematical models behind each strategy.

---

## Table of Contents

1. [Why multiple candidates?](#why-multiple-candidates)
2. [Candidate list](#candidate-list)
3. [Assumptions behind each candidate](#assumptions-behind-each-candidate)
4. [MILP global optimisation](#milp-global-optimisation)
5. [Hysteresis](#hysteresis)

---

## Why multiple candidates?

Battery scheduling is a sequential decision problem under uncertainty because
prices, PV, and load are forecasts. HSEM uses one dynamic MILP for active
optimisation, a passive executable fallback for solver failure, and a
`no_action` diagnostic comparator. This structure captures full-horizon value,
degrades safely, and keeps plan comparisons auditable.

---

## Candidate list

The generator (`planner/candidate_generator.py`) operates in **MILP-only mode**.
The MILP finds the globally optimal solution; heuristic candidates are disabled
because the MILP consistently dominates them. Only diagnostic baselines remain
alongside the MILP:

| # | Name | Strategy |
|---|---|---|
| 1 | `no_action` | Battery completely idle — no forced charge or discharge |
| 2 | `passive` | Solar charging only where PV surplus exists; no grid charge or forced discharge |
| 3 | `milp` | Globally-optimal LP solution (when scipy is available) |

## Assumptions behind each candidate

### `no_action` (diagnostic floor)

**Assumption:** The inverter can be left in its default self-consumption mode
(no external scheduling). PV surplus is exported; battery only moves according
to its native operating logic.

**Purpose:** Provides an auditable diagnostic comparator. It is never eligible
to win; the passive candidate is the executable fallback.

**Mathematical model:**
- All recommendations cleared to `None`
- No grid charge, no forced discharge, no force export
- SoC simulation still runs: PV charges battery if native logic would do so
- Terminal SoC is still accounted for

### `passive` (inverter default)

**Assumption:** The inverter's default behaviour is solar-following — it charges
from PV surplus when available and stays idle otherwise. No grid price arbitrage.

**Purpose:** Models what happens if HSEM only sets PV-charge recommendations
without any grid-based scheduling.

**Mathematical model:**
- Solar surplus slots (where `estimated_net_consumption < 0`) get `batteries_charge_solar`
- All other charge/discharge/export recommendations cleared
- Battery fills from PV, never from grid
- Battery never discharges unless native inverter logic does so

---

## MILP global optimisation

The MILP solver (`planner/milp_optimizer.py`) uses scipy's HiGHS to find the
globally optimal charge/discharge schedule. This is the **primary planner** —
the MILP solution is preferred over all heuristic candidates.

See [MILP Optimization](milp-optimization.md) for the full LP formulation,
variable layout, constraints, solver pipeline, and post-processing flow.

### EV co-optimisation

When one or more active EVs are provided, the MILP co-optimises EV charging
alongside primary- and optional secondary-battery flows. EV variables are part
of the same LP vector and energy-balance constraints.

### Fallback

If `scipy` is unavailable or the solver fails without a validated incumbent,
the MILP candidate is dropped and `passive` is used. `no_action` remains
available only for diagnostics.

---

## Hysteresis

### Plan-level hysteresis (issue #372)

Prevents the planner from switching strategies for tiny cost improvements.
When active:

1. The previously active plan is re-evaluated with current data
2. If its score improvement over the best new candidate is below both thresholds,
   the previous plan is kept

| Threshold | Default | Behaviour |
|---|---|---|
| Absolute | 0.0 currency | New plan must be cheaper by at least this amount |
| Percentage | 5.0 % | New plan must be cheaper by at least this % of previous score |

### Window-level hysteresis (issue #315)

Suppresses only command-equivalent `batteries_charge_solar` ↔ unrestricted
`batteries_discharge_mode` label changes. Transitions that change Huawei mode,
EV control, or a flow-derived partial-discharge cap pass through immediately.
The hold time is configured by `planner_window_hysteresis_minutes`
(default: 10).
