# HSEM-PowMr

[![GitHub Release][releases-shield]][releases]
[![License][license-shield]][license]

A fork of [woopstar/hsem][upstream] that adds a **PowMr inverter as a second battery** alongside the Huawei system, and plans both together.

> **Modified fork.** Forked from [woopstar/hsem][upstream] on 2026-08-11 and modified since. Upstream is not responsible for anything in this repository. Licensed AGPL-3.0, same as upstream.

---

## What this fork adds

- **PowMr as secondary storage** — SoC, charge/discharge power, and source-priority selects are modelled as a second battery in the MILP, not bolted on afterwards
- **Huawei-first, PowMr-second charge allocation** — the optimiser fills the primary battery first and uses the PowMr for what's left
- **Phase-aware charging** — hard per-phase limits derived from the live meter, so grid charging respects the main fuse on each leg rather than just in aggregate
- **Quarter-hourly spot prices** — 15-minute Nord Pool points survive to the planner instead of being collapsed to hourly averages
- **Forecast-driven seasonal fill** — idle slots are filled from the PV forecast rather than a fixed month table

Everything else — the MILP planner, cost function, solar correction, EV co-optimisation, financial sensors — is upstream's work.

## Status

This is a **personal fork** running against live hardware. Releases are cut when something is worth keeping, sometimes as prereleases that have only been tested on one system. There is no support commitment. If you want the stable, widely-tested integration, use [upstream][upstream].

## Requirements

- [Huawei Solar integration by wlcrs](https://github.com/wlcrs/huawei_solar) — **1.5.0a1 or later**
- [Solcast integration](https://github.com/BJReplay/ha-solcast-solar)
- An electricity price integration — [Nordpool](https://github.com/custom-components/nordpool), [Energi Data Service](https://github.com/MTrab/energidataservice), [Amber](https://amber.com.au), etc.
- A PowMr inverter exposed to Home Assistant as entities — SoC, battery net power, load power, and the output/charger source-priority selects. This fork is developed against an [ESPHome](https://esphome.io) node wired to the inverter's serial port, but HSEM only needs the entity IDs, so any route works. Required only for the secondary-storage features.

`sensor.inverter_active_power_control` and `sensor.batteries_rated_capacity` are disabled by default in the Huawei integration. Enable both on the inverter and battery devices, or HSEM will report missing entities.

## Installation

**HACS** — add `https://github.com/Ambilights/hsem-PowMr` as a custom repository with category **Integration**, install, then restart Home Assistant.

**Manual** — copy `custom_components/hsem/` into your Home Assistant `custom_components/` directory and restart.

Then add the integration from Settings → Devices & Services.

## Documentation

The feature set is upstream's, so upstream's documentation applies:

- [HSEM Wiki](https://github.com/woopstar/hsem/wiki) — working modes, battery schedules, FAQ
- [`docs/`](docs/) — planner specification, cost function, sensor reference

Anything PowMr- or fork-specific is in this repository's commit history and release notes.

## Credits

Built on [HSEM by woopstar][upstream] — the planner, optimiser, and the overwhelming majority of the code are theirs. If you find this useful, [support the upstream author](https://www.buymeacoffee.com/woopstar).

Licensed under [AGPL-3.0](LICENSE), inherited from upstream.

[upstream]: https://github.com/woopstar/hsem
[releases-shield]: https://img.shields.io/github/v/release/Ambilights/hsem-PowMr?style=for-the-badge
[releases]: https://github.com/Ambilights/hsem-PowMr/releases
[license-shield]: https://img.shields.io/github/license/Ambilights/hsem-PowMr?style=for-the-badge
[license]: https://github.com/Ambilights/hsem-PowMr/blob/main/LICENSE
