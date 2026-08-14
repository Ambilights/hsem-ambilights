"""Dataclass for a forecast PV production estimate for a single time slot."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SolcastSlot:
    """Forecast PV production estimate for a single time slot.

    Attributes:
        hour:
            0-based calendar hour (0-23).
        pv_estimate:
            PV energy estimate in kWh for the full slot duration.
        pv_estimate_available:
            Whether ``pv_estimate`` came from an authoritative forecast point.
            Defaults to ``True`` for backward compatibility with manual planner
            inputs; a genuine published zero remains available.
        day_offset:
            Number of whole calendar days from the planning midnight (0 = today,
            1 = tomorrow, …).  Defaults to 0 for backward compatibility with
            callers that only pass 24 single-day entries.
        slot_in_day:
            Optional elapsed-time slot ordinal within the local calendar day.
            When present, this distinguishes the two occurrences of a repeated
            autumn DST hour.  ``None`` keeps the legacy hour-granular behaviour.
    """

    hour: int  # 0-23
    pv_estimate: float = 0.0
    day_offset: int = 0
    slot_in_day: int | None = None
    pv_estimate_available: bool = True
