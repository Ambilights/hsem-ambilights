"""Window-level hysteresis — prevent rapid recommendation toggles."""

from __future__ import annotations

from datetime import UTC, datetime

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.utils.datetime_utils import slot_contains
from custom_components.hsem.utils.logger import log_planner
from custom_components.hsem.utils.recommendations import Recommendations
from custom_components.hsem.utils.units import is_material_planned_energy_kwh

_MSC_ALIAS_RECOMMENDATIONS = frozenset(
    {
        Recommendations.BatteriesChargeSolar.value,
        Recommendations.BatteriesDischargeMode.value,
    }
)

# ---------------------------------------------------------------------------
# Window-level hysteresis — prevent rapid recommendation toggles
# ---------------------------------------------------------------------------


def apply_window_hysteresis(
    slots: list[PlannedSlot],
    now: datetime,
    *,
    window_hysteresis_minutes: int,
    previous_current_recommendation: str | None,
    previous_current_slot_start: datetime | None,
) -> tuple[str | None, datetime | None]:
    """Hold only command-equivalent MSC label changes for a minimum time.

    ``batteries_charge_solar`` and unrestricted
    ``batteries_discharge_mode`` both execute Huawei maximise-self-consumption
    with the normal discharge cap, so their display-label oscillation can be
    held safely. Every transition that changes a hardware mode, EV command,
    or discharge cap is accepted immediately. In particular, a partial BDM
    allocation with material discharge *and* grid import must retain its
    plan-derived cap and is never relabelled as solar charge.

    Args:
        slots:
            Ordered list of planned slots (mutated in place).
        now:
            Timezone-aware current datetime.
        window_hysteresis_minutes:
            Minimum hold time in minutes.  0 disables the feature entirely.
        previous_current_recommendation:
            Recommendation that was active on the current slot during the
            previous planner run.  ``None`` on first run.
        previous_current_slot_start:
            Time when ``previous_current_recommendation`` became active.
            ``None`` on first run. The legacy parameter name is retained for
            API compatibility.

    Returns:
        A ``(updated_recommendation, current_slot_start)`` tuple.
        ``updated_recommendation`` is the (possibly held) recommendation
        for the current slot, and ``current_slot_start`` is the activation
        time of that recommendation (for persisting across cycles).  The
        legacy tuple-field name is retained for API compatibility.
    """
    if window_hysteresis_minutes <= 0:
        # Feature disabled — find and return current recommendation unchanged
        for s in slots:
            if slot_contains(s.start, s.end, now):
                return s.recommendation, now
        return None, None

    # Find the current slot
    current_slot: PlannedSlot | None = None
    for s in slots:
        if slot_contains(s.start, s.end, now):
            current_slot = s
            break

    if current_slot is None:
        return None, None

    new_rec = current_slot.recommendation

    # Missing/unpublished prices revoke authority for price-driven actions.
    # Never restore an older grid-charge or forced-export label onto the
    # planner's passive non-actionable allocation.
    if not current_slot.price_actionable:
        activated_at = (
            previous_current_slot_start
            if new_rec == previous_current_recommendation
            else now
        )
        return new_rec, activated_at

    # A validated MILP idle allocation is an explicit zero-energy control
    # decision, not merely a display label. Holding an older charge/discharge
    # string here would make hardware execute energy absent from the solved
    # and scored flow fields. Let this transition through immediately.
    if current_slot.primary_battery_hold:
        activated_at = (
            previous_current_slot_start
            if new_rec == previous_current_recommendation
            else now
        )
        return new_rec, activated_at

    # No previous state — first run, no hysteresis to apply
    if previous_current_recommendation is None or previous_current_slot_start is None:
        return new_rec, now

    # If the recommendation hasn't changed at all, no hold needed
    if new_rec == previous_current_recommendation:
        return new_rec, previous_current_slot_start

    command_equivalent_alias = (
        frozenset({new_rec, previous_current_recommendation})
        == _MSC_ALIAS_RECOMMENDATIONS
    )
    partial_bdm = (
        new_rec == Recommendations.BatteriesDischargeMode.value
        and is_material_planned_energy_kwh(current_slot.batteries_discharged_kwh)
        and is_material_planned_energy_kwh(current_slot.grid_import_kwh)
    )
    if not command_equivalent_alias or partial_bdm:
        log_planner(
            "debug",
            "[window_hysteresis] Allowing command-changing transition '%s' → "
            "'%s' immediately (partial_bdm=%s).",
            previous_current_recommendation,
            new_rec,
            partial_bdm,
        )
        return new_rec, now

    # Recommendation changed — check hold time
    elapsed_minutes = (
        now.astimezone(UTC) - previous_current_slot_start.astimezone(UTC)
    ).total_seconds() / 60.0
    if elapsed_minutes < window_hysteresis_minutes:
        # Hold the previous recommendation
        log_planner(
            "debug",
            "[window_hysteresis] Holding previous recommendation '%s' on current "
            "slot (elapsed=%.1f min < hold=%d min). New '%s' suppressed.",
            previous_current_recommendation,
            elapsed_minutes,
            window_hysteresis_minutes,
            new_rec,
        )
        current_slot.recommendation = previous_current_recommendation
        return previous_current_recommendation, previous_current_slot_start

    # Enough time has passed — allow the switch
    log_planner(
        "debug",
        "[window_hysteresis] Allowing transition '%s' → '%s' on current slot "
        "(elapsed=%.1f min >= hold=%d min).",
        previous_current_recommendation,
        new_rec,
        elapsed_minutes,
        window_hysteresis_minutes,
    )
    return new_rec, now
