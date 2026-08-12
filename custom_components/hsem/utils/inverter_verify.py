"""Write-and-verify helper for inverter and battery hardware writes.

Single responsibility: wrap a hardware write with a read-back verification
loop so that HSEM can confirm each setting was accepted by the inverter before
marking the apply cycle as successful.

Design
------
- Write the desired value via a caller-supplied coroutine.
- Wait a configurable settle time so the inverter has time to persist the value.
- Read the current value back via a caller-supplied reader callable.
- Accept the write if the read-back value matches within the specified tolerance.
- Retry up to ``max_retries`` times on mismatch or transient error.
- Return an :class:`ApplyResult` that the caller can log and surface to the
  status sensor.

The helpers in this module are intentionally free of Home Assistant dependencies
so that they can be unit-tested without a running HA instance.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from time import monotonic
from typing import Any

from custom_components.hsem.utils.logger import HSEM_LOGGER as _LOGGER

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

_LOG_FMT = "%s — %s"

#: Default seconds to wait between write and read-back.
#: The inverter is polled every 5-10 s by HA, so the settle time must be
#: long enough for at least one full poll cycle to complete.
DEFAULT_SETTLE_SECONDS: float = 10.0

#: Default maximum number of write+verify attempts.
DEFAULT_MAX_RETRIES: int = 3

#: Absolute tolerance for numeric (float/int) comparisons.
DEFAULT_NUMERIC_TOLERANCE: float = 1.0

#: Initial cooldown after a target fails a complete write-and-verify cycle.
DEFAULT_BACKOFF_INITIAL_SECONDS: float = 30.0

#: Maximum cooldown for repeated failures of the same target and desired value.
DEFAULT_BACKOFF_MAX_SECONDS: float = 300.0


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class ApplyStatus(StrEnum):
    """Outcome of a single write-and-verify operation."""

    #: The read-back value matched the desired value within tolerance.
    OK = "ok"

    #: The write was accepted but the read-back timed out or returned None.
    UNVERIFIED = "unverified"

    #: All retries exhausted; the inverter did not accept the value.
    FAILED = "failed"

    #: The write was skipped because the current value already matched.
    SKIPPED = "skipped"


class NonRetryableWriteError(Exception):
    """Hardware write failure that must enter backoff without local retries."""


@dataclass
class ApplyResult:
    """Detailed outcome of a :func:`async_write_and_verify` call.

    Attributes:
        entity_id: The HA entity that was written.
        desired: The value that was written.
        actual: The last read-back value (``None`` if unreadable).
        status: Outcome enum.
        attempts: How many write+verify rounds were performed.
        error_message: Human-readable reason for failure (empty on success).
    """

    entity_id: str
    desired: Any
    actual: Any
    status: ApplyStatus
    attempts: int = 0
    error_message: str = ""


@dataclass
class CycleApplySummary:
    """Aggregated results for one full apply cycle (all writes in one coordinator
    tick).

    Attributes:
        results: Individual :class:`ApplyResult` per write operation.
        last_updated: ISO-format timestamp set by the caller after the cycle.
    """

    results: list[ApplyResult] = field(default_factory=list)
    last_updated: str | None = None

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def overall_status(self) -> ApplyStatus:
        """Worst-case status across all results.

        Priority: FAILED > UNVERIFIED > OK > SKIPPED.
        If the results list is empty, returns ``ApplyStatus.SKIPPED``.
        """
        if not self.results:
            return ApplyStatus.SKIPPED
        priority = (
            ApplyStatus.FAILED,
            ApplyStatus.UNVERIFIED,
            ApplyStatus.OK,
            ApplyStatus.SKIPPED,
        )
        for status in priority:
            if any(r.status == status for r in self.results):
                return status
        return ApplyStatus.SKIPPED

    @property
    def failed_entities(self) -> list[str]:
        """Entity IDs whose last write ultimately failed verification."""
        return [r.entity_id for r in self.results if r.status == ApplyStatus.FAILED]

    @property
    def unverified_entities(self) -> list[str]:
        """Entity IDs whose write could not be verified (reader returned None)."""
        return [r.entity_id for r in self.results if r.status == ApplyStatus.UNVERIFIED]


@dataclass
class _WriteBackoffEntry:
    """Failure state for one entity and desired value."""

    desired: Any
    failures: int
    retry_at: float


class WriteFailureBackoff:
    """Per-target exponential cooldown for failed hardware writes.

    A changed desired value is never blocked. This is important for safety
    transitions: a failed optimisation write must not delay a later command
    that asks the inverter to enter a different mode.
    """

    def __init__(
        self,
        *,
        initial_seconds: float = DEFAULT_BACKOFF_INITIAL_SECONDS,
        max_seconds: float = DEFAULT_BACKOFF_MAX_SECONDS,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        """Initialise the cooldown registry."""
        if initial_seconds <= 0:
            raise ValueError("initial_seconds must be > 0")
        if max_seconds < initial_seconds:
            raise ValueError("max_seconds must be >= initial_seconds")

        self._initial_seconds = initial_seconds
        self._max_seconds = max_seconds
        self._clock = clock
        self._entries: dict[str, _WriteBackoffEntry] = {}

    def remaining_seconds(self, entity_id: str, desired: Any) -> float:
        """Return the active cooldown, or zero when a write may proceed."""
        entry = self._entries.get(entity_id)
        if entry is None:
            return 0.0
        if not _same_desired_value(entry.desired, desired):
            self._entries.pop(entity_id, None)
            return 0.0

        remaining = entry.retry_at - self._clock()
        return max(0.0, remaining)

    def record_failure(self, entity_id: str, desired: Any) -> float:
        """Record a failed cycle and return its newly scheduled cooldown."""
        previous = self._entries.get(entity_id)
        failures = (
            previous.failures + 1
            if previous is not None and _same_desired_value(previous.desired, desired)
            else 1
        )
        delay = min(
            self._initial_seconds * (2.0 ** (failures - 1)),
            self._max_seconds,
        )
        self._entries[entity_id] = _WriteBackoffEntry(
            desired=desired,
            failures=failures,
            retry_at=self._clock() + delay,
        )
        return delay

    def record_success(self, entity_id: str) -> None:
        """Clear any failure history after a successful or unnecessary write."""
        self._entries.pop(entity_id, None)

    def clear(self) -> None:
        """Clear all cooldowns during entity unload."""
        self._entries.clear()


def get_write_failure_backoff(owner: object) -> WriteFailureBackoff | None:
    """Return an owner's backoff registry without accepting loose test mocks."""
    candidate = getattr(owner, "_write_failure_backoff", None)
    return candidate if isinstance(candidate, WriteFailureBackoff) else None


# ---------------------------------------------------------------------------
# Core write-and-verify primitive
# ---------------------------------------------------------------------------


async def async_write_and_verify(
    entity_id: str,
    desired: Any,
    writer: Callable[[], Awaitable[None]],
    reader: Callable[[], Any],
    *,
    tolerance: float = DEFAULT_NUMERIC_TOLERANCE,
    settle_seconds: float = DEFAULT_SETTLE_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    skip_if_equal: bool = True,
    backoff: WriteFailureBackoff | None = None,
) -> ApplyResult:
    """Write *desired* to an inverter entity and verify the value was accepted.

    Args:
        entity_id: HA entity that is written (used only for logging/reporting).
        desired: The value to write.
        writer: Zero-argument coroutine that performs the actual hardware write.
        reader: Zero-argument callable that returns the current entity value
                (may be a regular function or a coroutine).  Returns ``None``
                when the entity is unavailable.
        tolerance: Accepted absolute difference for numeric comparisons.
                   String comparisons use exact equality regardless.
        settle_seconds: Seconds to wait after writing before reading back.
        max_retries: Maximum number of write+verify attempts.
        skip_if_equal: When ``True``, skip the write entirely if the current
                       value already matches *desired* within tolerance.
        backoff: Optional per-target cooldown registry. A completed failure
                 activates the cooldown; success clears it.

    Returns:
        :class:`ApplyResult` describing the outcome.
    """
    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")

    # ------------------------------------------------------------------
    # Pre-flight: read current value
    # ------------------------------------------------------------------
    try:
        current = await reader() if inspect.iscoroutinefunction(reader) else reader()
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("Could not read current value of %s: %s", entity_id, exc)
        current = None

    if (
        skip_if_equal
        and current is not None
        and _values_match(current, desired, tolerance)
    ):
        if backoff is not None:
            backoff.record_success(entity_id)
        return ApplyResult(
            entity_id=entity_id,
            desired=desired,
            actual=current,
            status=ApplyStatus.SKIPPED,
            attempts=0,
        )

    if backoff is not None:
        remaining = backoff.remaining_seconds(entity_id, desired)
        if remaining > 0:
            return ApplyResult(
                entity_id=entity_id,
                desired=desired,
                actual=current,
                status=ApplyStatus.FAILED,
                attempts=0,
                error_message=(
                    f"Hardware write in backoff for {remaining:.0f} more seconds"
                ),
            )

    # ------------------------------------------------------------------
    # Retry loop
    # ------------------------------------------------------------------
    last_actual: Any = current
    last_error = ""
    attempts_performed = 0

    for attempt in range(1, max_retries + 1):
        attempts_performed = attempt
        try:
            await writer()
        except NonRetryableWriteError as exc:
            last_error = f"Non-retryable write error: {exc}"
            break
        except Exception as exc:  # noqa: BLE001
            last_error = f"Write error on attempt {attempt}: {exc}"
            _LOGGER.warning(_LOG_FMT, entity_id, last_error)
            # Wait before retrying even after a write error (device may recover).
            if attempt < max_retries:
                await asyncio.sleep(settle_seconds)
            continue

        # Wait for the inverter to settle before reading back.
        await asyncio.sleep(settle_seconds)

        try:
            readback = (
                await reader() if inspect.iscoroutinefunction(reader) else reader()
            )
        except Exception as exc:  # noqa: BLE001
            last_error = f"Read-back error on attempt {attempt}: {exc}"
            _LOGGER.warning(_LOG_FMT, entity_id, last_error)
            last_actual = None
            continue

        last_actual = readback

        if readback is None:
            last_error = f"Read-back returned None on attempt {attempt}"
            _LOGGER.warning("%s — %s", entity_id, last_error)
            continue

        if _values_match(readback, desired, tolerance):
            _LOGGER.debug(
                "%s verified after %d attempt(s): desired=%s, actual=%s",
                entity_id,
                attempt,
                desired,
                readback,
            )
            result = ApplyResult(
                entity_id=entity_id,
                desired=desired,
                actual=readback,
                status=ApplyStatus.OK,
                attempts=attempt,
            )
            if backoff is not None:
                backoff.record_success(entity_id)
            return result

        last_error = (
            f"Mismatch on attempt {attempt}: desired={desired}, actual={readback}"
        )
        _LOGGER.warning(_LOG_FMT, entity_id, last_error)

    # ------------------------------------------------------------------
    # All retries exhausted
    # ------------------------------------------------------------------
    final_status = ApplyStatus.UNVERIFIED if last_actual is None else ApplyStatus.FAILED
    backoff_seconds = (
        backoff.record_failure(entity_id, desired) if backoff is not None else 0.0
    )
    _LOGGER.error(
        "%s write-and-verify FAILED after %d attempt(s). Last error: %s%s",
        entity_id,
        attempts_performed,
        last_error,
        (
            f". Hardware write backoff active for {backoff_seconds:.0f} seconds"
            if backoff_seconds > 0
            else ""
        ),
    )
    result = ApplyResult(
        entity_id=entity_id,
        desired=desired,
        actual=last_actual,
        status=final_status,
        attempts=attempts_performed,
        error_message=last_error,
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _values_match(actual: Any, desired: Any, tolerance: float) -> bool:
    """Return True when *actual* is close enough to *desired*.

    Numeric types use an absolute tolerance comparison.
    Strings and other types use exact equality.

    Args:
        actual: The read-back value.
        desired: The intended value.
        tolerance: Maximum allowed absolute difference for numeric types.

    Returns:
        True if the values are considered equal within tolerance.
    """
    if isinstance(desired, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(actual) - float(desired)) <= tolerance
    return str(actual).lower().strip() == str(desired).lower().strip()


def _same_desired_value(first: Any, second: Any) -> bool:
    """Compare desired values exactly while normalising string casing."""
    if isinstance(first, str) or isinstance(second, str):
        return str(first).lower().strip() == str(second).lower().strip()
    return bool(first == second)
