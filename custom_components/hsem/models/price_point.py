"""Dataclass for an import or export electricity price for a single time slot."""

from __future__ import annotations

from dataclasses import dataclass

from custom_components.hsem.models.price_source import PriceSource


@dataclass
class PricePoint:
    """An import or export electricity price for a single time slot.

    Attributes:
        hour:
            0-based calendar hour (0-23).
        import_price:
            Import price in local currency/kWh (e.g. DKK/kWh).
        export_price:
            Export price in local currency/kWh.
        day_offset:
            Number of whole calendar days from the planning midnight (0 = today,
            1 = tomorrow, …).  Defaults to 0 for backward compatibility with
            callers that only pass 24 single-day entries.
        slot_in_day:
            Optional 0-based elapsed-time ordinal of the sub-hourly slot within
            its local calendar day.  For 15-minute slots a spring-transition,
            ordinary, or autumn-transition civil day has 92, 96, or 100
            ordinals respectively.
            ``None`` (default) means the point is hour-granular — existing
            hourly callers are unaffected.  When set, the planner keys the
            point by ``(day_offset, slot_in_day)`` so quarter-hourly prices
            (e.g. Nord Pool 15-min MTUs) survive to the MILP instead of being
            collapsed to one price per hour (issue #720).
        import_price_available:
            Whether ``import_price`` came from the configured source. This
            keeps an unpublished price distinct from a genuine zero price.
        export_price_available:
            Whether ``export_price`` came from the configured source.
        import_price_source / export_price_source:
            Optional provenance for the corresponding channel (``primary``,
            ``entsoe``, or ``forecast``).
    """

    hour: int  # 0-23
    import_price: float = 0.0
    export_price: float = 0.0
    day_offset: int = 0
    slot_in_day: int | None = None
    import_price_available: bool = True
    export_price_available: bool = True
    import_price_source: PriceSource | None = None
    export_price_source: PriceSource | None = None
