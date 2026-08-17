# Contributing

HSEM Ambilights is developed for one specific Huawei/PowMr installation. Small,
well-evidenced fixes are welcome, but changes that generalize the project or add
new hardware are not automatically in scope.

Before changing code:

1. Read [`AGENTS.md`](AGENTS.md) and the task-specific canonical documents it
   identifies.
2. Search the issue tracker and open a focused issue when behaviour or scope
   needs agreement.
3. Create a dedicated branch using the naming convention in `AGENTS.md`.
4. Keep planner semantics, documentation, and regression tests synchronized.

Before submitting a pull request, run:

```bash
./scripts/quality.sh lint
./scripts/quality.sh typing
./scripts/quality.sh quality
./scripts/quality.sh test
```

The pull request must explain the change, safety/economic impact, tests, known
limitations, and any required configuration changes. Never include credentials,
tokens, or unredacted Home Assistant diagnostics.

Contributions are licensed under the repository's AGPL-3.0 license.
