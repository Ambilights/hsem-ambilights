"""Daily plan-vs-actual tracker that accumulates metrics and manages 90-day JSON history.

This is a pure-Python tracker stored on the coordinator.  It accumulates plan and
actual values each cycle, triggers midnight persistence, and resets counters for the
new day.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

from custom_components.hsem.models.daily_metrics import DailyMetrics
from custom_components.hsem.models.daily_record import DailyRecord
from custom_components.hsem.models.day_rollover_result import DayRolloverResult

_ENERGY_EPSILON_KWH = 1e-9


@dataclass(frozen=True)
class ActualPriceInterval:
    """Published prices covering one physical interval.

    ``start`` and ``end`` are compared as UTC instants.  This preserves both
    occurrences of an autumn DST fold even though they share the same local
    wall-clock label.
    """

    start: datetime
    end: datetime
    import_price: float
    export_price: float
    import_price_available: bool = True
    export_price_available: bool = True


@dataclass
class DailyPlanVsActualTracker:
    """Accumulates daily plan-vs-actual metrics and manages 90-day JSON history.

    This is a pure-Python tracker stored on the coordinator.  It accumulates
    plan and actual values each cycle, triggers midnight persistence, and
    resets counters for the new day.

    Attributes:
        history_file: Full path to ``hsem_daily_history.json``.
        max_history_days: Rolling window size in days (default 90).
        today: Today's date (set on initialisation and checked each cycle).
        actual: Cumulative actual metrics since midnight.
        plan: Cumulative planned metrics since midnight.
        last_soc_pct: Previous battery SoC reading for cycle tracking (or None).
        history: List of persisted :class:`DailyRecord` objects (up to 90).
    """

    history_file: str = ""
    max_history_days: int = 90
    today: str = ""  # ISO-format date string
    actual: DailyMetrics = field(default_factory=DailyMetrics)
    plan: DailyMetrics = field(default_factory=DailyMetrics)
    last_soc_pct: float | None = None
    history: list[DailyRecord] = field(default_factory=list)

    # Last-seen cumulative meter readings for delta calculation.
    _last_import_energy_kwh: float | None = field(default=None, repr=False)
    _last_export_energy_kwh: float | None = field(default=None, repr=False)
    _last_pv_energy_kwh: float | None = field(default=None, repr=False)
    _last_import_sample_at: datetime | None = field(default=None, repr=False)
    _last_export_sample_at: datetime | None = field(default=None, repr=False)
    _last_pv_sample_at: datetime | None = field(default=None, repr=False)
    _last_soc_sample_at: datetime | None = field(default=None, repr=False)
    _last_import_price: float | None = field(default=None, repr=False)
    _last_export_price: float | None = field(default=None, repr=False)
    _last_import_price_available: bool = field(default=False, repr=False)
    _last_export_price_available: bool = field(default=False, repr=False)
    _pending_actual_by_date: dict[str, DailyMetrics] = field(
        default_factory=dict,
        repr=False,
    )
    last_actual_delta: DailyMetrics = field(default_factory=DailyMetrics, repr=False)
    last_actual_delta_by_date: dict[str, DailyMetrics] = field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Set today's date if not already set.

        History loading is deferred — call :meth:`load_history` explicitly
        to avoid blocking I/O during dataclass construction.
        """
        # ``today`` is intentionally injected by ``check_day_rollover(now)``.
        # A process-local calendar date may differ from Home Assistant's
        # configured timezone around midnight.

    # ------------------------------------------------------------------
    # Midday counter reset
    # ------------------------------------------------------------------

    async def check_day_rollover(self, now: datetime) -> DayRolloverResult | None:
        """Finalise the prior HA-local day without discarding meter baselines.

        Meter samples are accumulated before this method is called. Any
        cross-midnight delta is split on the UTC timeline and staged by local
        calendar date, so the old-day share is persisted before the new-day
        share becomes the active accumulator.
        """
        today_str = now.date().isoformat()
        if not self.today:
            self.today = today_str
            self.actual = self._pending_actual_by_date.pop(
                today_str,
                DailyMetrics(),
            )
            return None
        if today_str == self.today:
            return None

        prior_record = self._build_today_record()
        saved = await self._save_record_to_history(prior_record)

        # A long coordinator outage can span more than one local date. Keep
        # each staged date distinct rather than merging it into the restart day.
        for day_str in sorted(self._pending_actual_by_date):
            if day_str >= today_str:
                continue
            intermediate = DailyRecord(
                date=day_str,
                actual=self._pending_actual_by_date.pop(day_str),
            )
            intermediate.compute_diff()
            saved = await self._save_record_to_history(intermediate) and saved

        self.today = today_str
        self.actual = self._pending_actual_by_date.pop(today_str, DailyMetrics())
        self.plan = DailyMetrics()

        # Deliberately preserve the cumulative-meter, timestamp, and SoC
        # baselines. They are required to account for the first physical
        # interval after midnight (including daily meters that reset to zero).
        return DayRolloverResult(record=prior_record, saved=saved)

    # ------------------------------------------------------------------
    # Accumulation helpers
    # ------------------------------------------------------------------

    def accumulate_actual(
        self,
        grid_import_energy_kwh: float | None = None,
        grid_export_energy_kwh: float | None = None,
        pv_energy_kwh: float | None = None,
        soc_pct: float | None = None,
        rated_capacity_kwh: float = 0.0,
        import_price: float = 0.0,
        export_price: float = 0.0,
        import_price_available: bool = True,
        export_price_available: bool = True,
        *,
        max_gap_seconds: float,
        now: datetime | None = None,
        price_intervals: Sequence[ActualPriceInterval] = (),
    ) -> DailyMetrics:
        """Accumulate measured deltas on the canonical physical timeline.

        Cumulative meter energy is assumed to be distributed uniformly between
        consecutive finite samples. The interval is split at published-price,
        planner-slot, local-midnight, and DST boundaries using UTC instants.
        Energy remains counted when a price is unavailable, while money is
        omitted.

        Missing or non-finite telemetry resets that channel's baseline. Invalid,
        nonpositive, reversed, and over-limit physical intervals are rejected
        after the new sample has advanced the baseline, preventing recovery from
        bridging stale data. A decreasing cumulative meter is treated as a daily
        reset, whose current reading belongs to the new local day.
        """
        self.last_actual_delta = DailyMetrics()
        self.last_actual_delta_by_date = {}

        if now is not None and not self.today:
            self.today = now.date().isoformat()

        import_sample = self._finite_sample(grid_import_energy_kwh)
        if import_sample is not None:
            self._accumulate_meter(
                current=import_sample,
                previous=self._last_import_energy_kwh,
                previous_at=self._last_import_sample_at,
                now=now,
                max_gap_seconds=max_gap_seconds,
                price_intervals=price_intervals,
                energy_field="grid_import_kwh",
                money_field="grid_import_cost",
                prior_price=self._last_import_price,
                prior_price_available=self._last_import_price_available,
                price_kind="import",
            )
        self._last_import_energy_kwh = import_sample
        self._last_import_sample_at = now

        export_sample = self._finite_sample(grid_export_energy_kwh)
        if export_sample is not None:
            self._accumulate_meter(
                current=export_sample,
                previous=self._last_export_energy_kwh,
                previous_at=self._last_export_sample_at,
                now=now,
                max_gap_seconds=max_gap_seconds,
                price_intervals=price_intervals,
                energy_field="grid_export_kwh",
                money_field="grid_export_rev",
                prior_price=self._last_export_price,
                prior_price_available=self._last_export_price_available,
                price_kind="export",
            )
        self._last_export_energy_kwh = export_sample
        self._last_export_sample_at = now

        finite_import_price = self._finite_sample(import_price)
        finite_export_price = self._finite_sample(export_price)
        self._last_import_price = finite_import_price
        self._last_export_price = finite_export_price
        self._last_import_price_available = (
            import_price_available and finite_import_price is not None
        )
        self._last_export_price_available = (
            export_price_available and finite_export_price is not None
        )

        pv_sample = self._finite_sample(pv_energy_kwh)
        if pv_sample is not None:
            self._accumulate_meter(
                current=pv_sample,
                previous=self._last_pv_energy_kwh,
                previous_at=self._last_pv_sample_at,
                now=now,
                max_gap_seconds=max_gap_seconds,
                price_intervals=price_intervals,
                energy_field="pv_produced_kwh",
            )
        self._last_pv_energy_kwh = pv_sample
        self._last_pv_sample_at = now

        soc_sample = self._finite_sample(soc_pct)
        capacity_kwh = self._finite_sample(rated_capacity_kwh)
        previous_soc = self._finite_sample(self.last_soc_pct)
        if (
            soc_sample is not None
            and previous_soc is not None
            and capacity_kwh is not None
            and capacity_kwh > 0
        ):
            cycled_kwh = abs(soc_sample - previous_soc) * capacity_kwh / 100.0
            self._allocate_delta(
                delta=cycled_kwh,
                start=self._last_soc_sample_at,
                end=now,
                max_gap_seconds=max_gap_seconds,
                price_intervals=price_intervals,
                energy_field="battery_cycled_kwh",
            )
        self.last_soc_pct = soc_sample
        self._last_soc_sample_at = now

        return self.last_actual_delta

    @staticmethod
    def _finite_sample(value: float | None) -> float | None:
        """Return a finite float sample, or None for missing/invalid data."""
        if value is None or not math.isfinite(value):
            return None
        return float(value)

    def _accumulate_meter(
        self,
        *,
        current: float,
        previous: float | None,
        previous_at: datetime | None,
        now: datetime | None,
        max_gap_seconds: float,
        price_intervals: Sequence[ActualPriceInterval],
        energy_field: str,
        money_field: str | None = None,
        prior_price: float | None = None,
        prior_price_available: bool = False,
        price_kind: str | None = None,
    ) -> None:
        """Calculate one finite cumulative-meter delta and allocate it."""
        if (
            previous is None
            or not math.isfinite(current)
            or not math.isfinite(previous)
        ):
            return

        delta = current - previous
        effective_start = previous_at
        if delta < -_ENERGY_EPSILON_KWH:
            # Daily utility meters commonly reset at local midnight. Their
            # first positive reading belongs to the new day rather than being
            # a new baseline that should be silently discarded.
            delta = max(current, 0.0)
            if (
                now is not None
                and now.tzinfo is not None
                and previous_at is not None
                and previous_at.tzinfo is not None
            ):
                local_midnight = datetime.combine(
                    now.date(),
                    time.min,
                    tzinfo=now.tzinfo,
                )
                if local_midnight.astimezone(UTC) > previous_at.astimezone(UTC):
                    effective_start = local_midnight
        elif delta < _ENERGY_EPSILON_KWH:
            return

        self._allocate_delta(
            delta=delta,
            start=effective_start,
            end=now,
            max_gap_seconds=max_gap_seconds,
            price_intervals=price_intervals,
            energy_field=energy_field,
            money_field=money_field,
            prior_price=prior_price,
            prior_price_available=prior_price_available,
            price_kind=price_kind,
        )

    def _allocate_delta(
        self,
        *,
        delta: float,
        start: datetime | None,
        end: datetime | None,
        max_gap_seconds: float,
        price_intervals: Sequence[ActualPriceInterval],
        energy_field: str,
        money_field: str | None = None,
        prior_price: float | None = None,
        prior_price_available: bool = False,
        price_kind: str | None = None,
    ) -> None:
        """Allocate one finite delta across a bounded physical interval."""
        if (
            not math.isfinite(delta)
            or delta <= _ENERGY_EPSILON_KWH
            or not self._valid_physical_interval(start, end, max_gap_seconds)
        ):
            return

        segments = self._physical_segments(start, end, price_intervals)
        if not segments:
            return

        total_seconds = sum(
            (segment_end - segment_start).total_seconds()
            for segment_start, segment_end, _, _ in segments
        )
        if total_seconds <= 0:
            return

        for segment_start, segment_end, day_str, interval in segments:
            share = (segment_end - segment_start).total_seconds() / total_seconds
            segment_price: float | None = None
            if interval is not None and price_kind == "import":
                if interval.import_price_available:
                    segment_price = interval.import_price
            elif interval is not None and price_kind == "export":
                if interval.export_price_available:
                    segment_price = interval.export_price
            elif (
                interval is None
                and prior_price_available
                and prior_price is not None
                and math.isfinite(prior_price)
            ):
                segment_price = prior_price

            self._record_actual_delta(
                day_str,
                energy_field,
                delta * share,
                money_field,
                segment_price,
            )

    @staticmethod
    def _valid_physical_interval(
        start: datetime | None,
        end: datetime | None,
        max_gap_seconds: float,
    ) -> bool:
        """Return whether two samples define an accepted UTC interval."""
        if (
            start is None
            or end is None
            or start.tzinfo is None
            or end.tzinfo is None
            or not math.isfinite(max_gap_seconds)
            or max_gap_seconds <= 0
        ):
            return False
        elapsed_seconds = (end.astimezone(UTC) - start.astimezone(UTC)).total_seconds()
        return 0 < elapsed_seconds <= max_gap_seconds

    def _record_actual_delta(
        self,
        day_str: str,
        energy_field: str,
        energy: float,
        money_field: str | None,
        price: float | None,
    ) -> None:
        """Record an allocated delta in daily and per-cycle accumulators."""
        daily = (
            self.actual
            if day_str == self.today
            else self._pending_actual_by_date.setdefault(day_str, DailyMetrics())
        )
        cycle_day = self.last_actual_delta_by_date.setdefault(day_str, DailyMetrics())
        for metrics in (daily, self.last_actual_delta, cycle_day):
            setattr(metrics, energy_field, getattr(metrics, energy_field) + energy)
            if money_field is not None and price is not None and math.isfinite(price):
                setattr(
                    metrics,
                    money_field,
                    getattr(metrics, money_field) + energy * price,
                )

    @staticmethod
    def _physical_segments(
        start: datetime | None,
        end: datetime | None,
        price_intervals: Sequence[ActualPriceInterval],
    ) -> list[tuple[datetime, datetime, str, ActualPriceInterval | None]]:
        """Return UTC segments split by price boundaries and local midnight."""
        if start is None or end is None or start.tzinfo is None or end.tzinfo is None:
            return []

        start_utc = start.astimezone(UTC)
        end_utc = end.astimezone(UTC)
        if end_utc <= start_utc:
            return []

        valid_intervals: list[tuple[datetime, datetime, ActualPriceInterval]] = []
        boundaries = {start_utc, end_utc}
        for candidate_interval in price_intervals:
            if (
                candidate_interval.start.tzinfo is None
                or candidate_interval.end.tzinfo is None
            ):
                continue
            interval_start = candidate_interval.start.astimezone(UTC)
            interval_end = candidate_interval.end.astimezone(UTC)
            if interval_end <= interval_start:
                continue
            valid_intervals.append((interval_start, interval_end, candidate_interval))
            if start_utc < interval_start < end_utc:
                boundaries.add(interval_start)
            if start_utc < interval_end < end_utc:
                boundaries.add(interval_end)

        local_tz = end.tzinfo
        next_date = start_utc.astimezone(local_tz).date() + timedelta(days=1)
        final_date = end_utc.astimezone(local_tz).date()
        while next_date <= final_date:
            midnight = datetime.combine(next_date, time.min, tzinfo=local_tz)
            midnight_utc = midnight.astimezone(UTC)
            if start_utc < midnight_utc < end_utc:
                boundaries.add(midnight_utc)
            next_date += timedelta(days=1)

        ordered = sorted(boundaries)
        segments: list[tuple[datetime, datetime, str, ActualPriceInterval | None]] = []
        for segment_start, segment_end in pairwise(ordered):
            midpoint = segment_start + (segment_end - segment_start) / 2
            interval = next(
                (
                    item
                    for interval_start, interval_end, item in valid_intervals
                    if interval_start <= midpoint < interval_end
                ),
                None,
            )
            day_str = midpoint.astimezone(local_tz).date().isoformat()
            segments.append((segment_start, segment_end, day_str, interval))
        return segments

    def accumulate_plan(
        self,
        grid_import_kwh: float = 0.0,
        grid_export_kwh: float = 0.0,
        cycle_kwh: float = 0.0,
        pv_kwh: float = 0.0,
        import_price: float = 0.0,
        export_price: float = 0.0,
        import_price_available: bool = True,
        export_price_available: bool = True,
    ) -> None:
        """Accumulate planned energy values from a single time slot.

        Args:
            grid_import_kwh: Planned grid import for the slot (kWh).
            grid_export_kwh: Planned grid export for the slot (kWh).
            cycle_kwh: Planned battery cycle energy for the slot (kWh).
            pv_kwh: Planned PV production for the slot (kWh).
            import_price: Spot import price (currency/kWh).
            export_price: Spot export price (currency/kWh).
            import_price_available: Whether the import price is authoritative.
            export_price_available: Whether the export price is authoritative.
        """
        self.plan.grid_import_kwh += grid_import_kwh
        if import_price_available and math.isfinite(import_price):
            self.plan.grid_import_cost += grid_import_kwh * import_price
        self.plan.grid_export_kwh += grid_export_kwh
        if export_price_available and math.isfinite(export_price):
            self.plan.grid_export_rev += grid_export_kwh * export_price
        self.plan.battery_cycled_kwh += cycle_kwh
        self.plan.pv_produced_kwh += pv_kwh

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    def _build_today_record(self) -> DailyRecord:
        """Build a :class:`DailyRecord` from current accumulators."""
        record = DailyRecord(
            date=self.today,
            actual=self.actual,
            plan=self.plan,
        )
        record.compute_diff()
        return record

    def get_today_record(self) -> DailyRecord:
        """Return today's record with computed diff."""
        record = self._build_today_record()
        return record

    def get_yesterday_record(self) -> DailyRecord | None:
        """Return the record before the injected HA-local date, if present."""
        if not self.today:
            return None
        yesterday_str = (date.fromisoformat(self.today) - timedelta(days=1)).isoformat()
        for record in reversed(self.history):
            if record.date == yesterday_str:
                return record
        return None

    # ------------------------------------------------------------------
    # JSON persistence
    # ------------------------------------------------------------------

    async def load_history(self) -> None:
        """Load history from the JSON file, if it exists.

        Handles corrupted files gracefully by starting with an empty history.
        """
        path = Path(self.history_file)
        if not path.exists():
            return

        data = await asyncio.to_thread(self._read_history_file, path)
        if data is None:
            return

        days = data.get("days", [])
        if isinstance(days, list):
            self.history = [DailyRecord.from_dict(d) for d in days]
            self._prune_history()

    @staticmethod
    def _read_history_file(path: Path) -> dict[str, Any] | None:
        """Read and parse the history JSON file (sync, offloaded to thread)."""
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)  # type: ignore[no-any-return]
        except json.JSONDecodeError, OSError:
            return None

    async def _save_record_to_history(self, record: DailyRecord) -> bool:
        """Upsert a daily record and persist the history atomically.

        Midnight may write a provisional crash-safe snapshot before the first
        post-midnight meter sample arrives. The completed rollover replaces
        that same date rather than appending a duplicate.
        """
        for index, existing in enumerate(self.history):
            if existing.date == record.date:
                self.history[index] = record
                break
        else:
            self.history.append(record)
        self.history.sort(key=lambda item: item.date)
        self._prune_history()

        return await self._write_history_file()

    def _prune_history(self) -> None:
        """Keep only the most recent ``max_history_days`` records."""
        if len(self.history) > self.max_history_days:
            self.history = self.history[-self.max_history_days :]

    async def _write_history_file(self) -> bool:
        """Write the history list to disk atomically.

        Returns:
            ``True`` on success, ``False`` on I/O error.
        """
        if not self.history_file:
            return False

        data = {
            "updated": datetime.now().isoformat(),
            "days": [r.as_dict() for r in self.history],
        }

        path = Path(self.history_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        return await asyncio.to_thread(self._write_history_file_sync, data, path)

    @staticmethod
    def _write_history_file_sync(data: dict[str, Any], path: Path) -> bool:
        """Write the history list to disk atomically (sync, offloaded to thread)."""
        try:
            # Write to a temp file in the same directory, then atomically rename.
            fd, tmp_path = tempfile.mkstemp(
                suffix=".json",
                prefix=".hsem_daily_history_",
                dir=str(path.parent),
                text=True,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, str(path))
            except Exception:
                # Clean up temp file on failure.
                with suppress(OSError):
                    os.unlink(tmp_path)
                raise
            return True
        except OSError:
            return False

    # ------------------------------------------------------------------
    # Attribute export for the sensor
    # ------------------------------------------------------------------

    def as_sensor_attributes(self) -> dict[str, Any]:
        """Return all data needed by the sensor as a flat dict.

        The sensor state is ``net_cost_actual``.

        Note: History is limited to 7 days in attributes to stay under
        Home Assistant's 16KB state attribute limit. Full 90-day history
        is available in the JSON file.
        """
        today_record = self.get_today_record()
        yesterday_record = self.get_yesterday_record()

        attrs: dict[str, Any] = {
            "today": today_record.as_dict(),
            "yesterday": yesterday_record.as_dict() if yesterday_record else None,
            "history": [r.as_dict() for r in self.history[-7:]],
            "history_file": self.history_file,
            "history_days": self.max_history_days,
            "history_total_days": len(self.history),
        }
        return attrs
