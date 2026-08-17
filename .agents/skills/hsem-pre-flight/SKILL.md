---
name: hsem-pre-flight
description: Run before starting any HSEM code change. Checkout main, pull latest, read repository rules and relevant docs, create a feature branch.
---

# HSEM Pre-Flight — Start Any Code Change

Activate this skill **before writing any code** — when the user asks to fix a bug, implement a feature, or make any change to the HSEM codebase.

## Step 1: Checkout Main and Pull Latest

```bash
git checkout main
git pull
```

## Step 2: Read Repository Rules

Read `AGENTS.md` completely. It is the authoritative source for safety rules,
canonical references, branch discipline, code standards, and quality gates. Do
not rely on a cached MILP layout or module map; inspect the current source and
`docs/planner-spec.md` for the task at hand.

## Step 3: Read Any Issue Being Solved

If this is issue-driven work, read the full GitHub issue before touching any code.

## Step 4: Create a Feature Branch

Format: `<type>/<issue-number>-<slug>`

| Type | Use for |
|------|---------|
| `feat` | New features |
| `fix` | Bug fixes |
| `chore` | Repository/code chores |
| `docs` | Documentation updates |
| `refactor` | Code refactoring |
| `perf` | Performance improvements |
| `test` | Test additions/updates |
| `ci` | CI/CD changes |

Examples: `fix/444-milp-cycle-cost`, `feat/123-add-solar-forecast`

All branches MUST be based on main unless the user explicitly instructs otherwise.

## Step 5: Identify Relevant Documentation

Based on the change type, read these docs before touching code:

| Change touches | Must read |
|---------------|-----------|
| Planner engine, cost function, SoC simulation, candidate generation, slot population, safety gates | `docs/planner-spec.md` |
| Huawei Solar sensors | `docs/huawei_entities.md` |
| Config/options flow | `docs/config-flow-reference.md` |
| EV charging | `docs/ev-charge-plan-setup.md` |
| Planner inputs/outputs | `docs/planner-guide.md` |

## Step 6: Understand the Affected Code

Search and read the relevant source files. Do not guess file paths — use `grep` and `find_path` to locate them.

## Reminder: One Issue Per Branch

Solve **one issue only** per branch and PR. Do not combine multiple issues. Do not refactor unrelated code.
