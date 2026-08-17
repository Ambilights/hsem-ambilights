# Claude Code instructions

Read [`AGENTS.md`](AGENTS.md) completely before changing this repository. It is
the authoritative source for branch discipline, hardware safety requirements,
code standards, documentation responsibilities, security constraints, and
required quality gates.

Use these task-specific canonical references:

- Read [`docs/planner-spec.md`](docs/planner-spec.md) completely before any
  planner, cost, SoC, MILP, candidate, slot, or safety-gate change. Update the
  specification and invariant tests whenever planner semantics change.
- Read [`docs/huawei_entities.md`](docs/huawei_entities.md) before consuming or
  writing a Huawei Solar entity. Never guess an entity ID or replace live
  hardware data with a numeric constant.
- Read [`docs/config-flow-reference.md`](docs/config-flow-reference.md) for
  config/options flow work and keep `translations/en.json` synchronized.
- Follow [`CODE_QUALITY_STANDARDS.md`](CODE_QUALITY_STANDARDS.md) and run the
  lint, typing, quality, and test gates defined in `AGENTS.md` before a PR.

Preserve the `hsem` Home Assistant domain and every hardware safety gate. Keep
changes focused, deterministic, and covered by regression tests. Do not commit
credentials, diagnostics containing identifiers, or Home Assistant secrets.
