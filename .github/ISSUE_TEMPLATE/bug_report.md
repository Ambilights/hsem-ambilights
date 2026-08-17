---
name: Bug report
about: Report a reproducible HSEM Ambilights problem
title: "[Bug]: "
labels: ""
assignees: ""
---

## Before submitting

- Search existing issues and test the latest release where practical.
- Remove tokens, credentials, serial numbers, entity IDs, and other private
  installation details from logs and screenshots.
- For unsafe hardware behaviour, enable HSEM Read-Only mode before reproducing.

## Problem

Describe what happened and when it started.

## Expected behaviour

Describe the result you expected.

## Reproduction

List the smallest sequence that reproduces the problem, including relevant
HSEM settings or working-mode overrides.

## Environment

- HSEM Ambilights version:
- Home Assistant version:
- Installation method (HACS/manual):
- Huawei Solar integration version:
- Relevant inverter and storage topology:
- Problem started after an upgrade? If so, from which version:

## Diagnostics and logs

Attach the Home Assistant config-entry diagnostics for HSEM and the relevant
HSEM log excerpt. Redact private identifiers and secrets first.

For planner decisions, also include:

- The affected local date and time slots, with timezone.
- Import/export prices and whether each price was actionable.
- Battery SoC, planned charge/discharge/export, and available PV.
- The candidate/solver explanation when available.

## Additional context

Add screenshots or other evidence that helps distinguish a planning problem
from an inverter write, telemetry, or configuration problem.
