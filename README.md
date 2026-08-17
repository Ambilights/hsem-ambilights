# HSEM Ambilights

Personal home energy planner. Plans a Huawei battery and a PowMr inverter against
Nord Pool SE4 prices, using solar and consumption forecasts, EV charging, and
per-phase fuse protection.

Built for one specific installation. Not general-purpose, not supported, and
changed without notice.

## Installation

Add `https://github.com/Ambilights/hsem-ambilights` to HACS as a custom
**Integration** repository, install **HSEM Ambilights**, then restart Home
Assistant. Back up the Home Assistant configuration before upgrading; releases
may change planner behaviour or configuration assumptions.

The active integration version is published in
[`custom_components/hsem/manifest.json`](custom_components/hsem/manifest.json).
Release packages are available from the
[GitHub releases page](https://github.com/Ambilights/hsem-ambilights/releases).

## Documentation

- [Documentation index](docs/index.md)
- [Configuration reference](docs/config-flow-reference.md)
- [Planner specification](docs/planner-spec.md)
- [Troubleshooting and diagnostics](docs/troubleshooting-guide.md)

Bug reports must use the repository's
[issue tracker](https://github.com/Ambilights/hsem-ambilights/issues) and include
the information requested by the issue template. See
[CONTRIBUTING.md](CONTRIBUTING.md) before proposing a code change.

## Provenance

Derived from **[HSEM by woopstar](https://github.com/woopstar/hsem)** — the MILP
planner, cost function, solar correction and EV co-optimisation are upstream work,
modified since 2026-08-11. If you find HSEM useful,
[support the author](https://www.buymeacoffee.com/woopstar).

Licensed under [AGPL-3.0](LICENSE), inherited from upstream.
