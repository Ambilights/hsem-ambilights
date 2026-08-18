"""Recommendation resolver for HSEMWorkingModeSensor.

Single responsibility: apply post-planner adjustments to the **current**
time-slot recommendation based on real-time state that the planner engine
cannot observe (for example, live EV charging status).

This module is purely decisional — no I/O, no hardware writes.
"""

from __future__ import annotations

from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.models.live_state import LiveState
from custom_components.hsem.utils.conversion import convert_to_float
from custom_components.hsem.utils.logger import HSEM_LOGGER
from custom_components.hsem.utils.recommendations import Recommendations


def _fmt_live_w(power_w: float | None) -> str:
    """Format a live power reading for log lines (``None`` → ``n/a``)."""
    if power_w is None:
        return "n/a"
    return f"{int(power_w)}W"


def resolve_current_recommendation(
    rec: HourlyRecommendation,
    live: LiveState,
) -> None:
    """Adjust the current-interval recommendation based on live runtime state.

    The planner engine produces recommendations using static forecasts and
    cannot know, for example, whether a car just plugged in.  This function
    applies the final layer of real-time overrides in priority order:

    1. **Published negative import price** → force export available PV.
    2. **Grid charge active** → grid charging takes priority over EV smart charge.
    3. **EV actively charging** → switch to EV smart charging mode.

    The recommendation is modified **in-place** on ``rec``.

    Args:
        rec: The :class:`HourlyRecommendation` for the current time slot.
        live: Live state snapshot at call time.
    """
    if rec is None:
        return

    original_recommendation = rec.recommendation

    # 1. A published, actionable negative import price → force export.
    # The numeric live-state fallback remains 0.0 when its source is missing,
    # and a cached/stale negative value must not restore price-driven control
    # after the current planner slot has become non-actionable.
    import_price = convert_to_float(live.import_electricity_price)
    if (
        rec.price_actionable
        and live.import_electricity_price_available
        and import_price is not None
        and import_price < 0
    ):
        rec.recommendation = Recommendations.ForceExport.value
        HSEM_LOGGER.debug(
            "[resolver] negative import price (%.4f) → overriding %s to force_export",
            import_price,
            original_recommendation,
        )
        return

    # 2. Grid charging in progress → preserve, do not override
    if rec.recommendation == Recommendations.BatteriesChargeGrid.value:
        HSEM_LOGGER.debug(
            "[resolver] batteries_charge_grid active → keeping recommendation unchanged"
        )
        return

    # 3. Any EV is actively charging AND the planner allocated EV load for
    #    this slot → override with EV smart charging.
    #
    # The planner's ``ev_charger_calculated_power`` is HSEM's *command* to the
    # charger, not a reflection of what the charger is doing.  If the planner
    # set it to 0, that means "stop charging" (e.g. target SoC reached, no
    # surplus PV, expensive grid power).  In that case we must NOT override
    # the recommendation to ``ev_smart_charging`` — the planner's original
    # recommendation (e.g. ``batteries_wait_mode``) should stand.
    #
    # We only override when the planner actually allocated EV load for this
    # slot (``ev_charger_calculated_power > 0`` or ``ev_total_planned_load_kwh > 0``).
    planner_allocated_ev = (
        rec.ev_charger_calculated_power > 1e-9
        or rec.ev_second_charger_calculated_power > 1e-9
        or rec.ev_total_planned_load_kwh > 1e-9
    )
    ev_actively_charging = live.ev.is_charging or live.ev_second.is_charging

    if ev_actively_charging and planner_allocated_ev:
        rec.recommendation = Recommendations.EVSmartCharging.value
        HSEM_LOGGER.debug(
            "[resolver] EV actively charging + planner_allocated_ev=True "
            "(planned_ev_power=%dW planned_ev2_power=%dW ev_total_load=%.3fkWh "
            "live_ev_power=%s live_ev2_power=%s) "
            "→ overriding %s to ev_smart_charging",
            rec.ev_charger_calculated_power,
            rec.ev_second_charger_calculated_power,
            rec.ev_total_planned_load_kwh,
            _fmt_live_w(live.ev.power_w),
            _fmt_live_w(live.ev_second.power_w),
            original_recommendation,
        )
        return

    if ev_actively_charging and not planner_allocated_ev:
        HSEM_LOGGER.debug(
            "[resolver] EV actively charging but planner_allocated_ev=False "
            "(planned_ev_power=%dW planned_ev2_power=%dW ev_total_load=%.3fkWh "
            "live_ev_power=%s live_ev2_power=%s) "
            "→ keeping original recommendation %s",
            rec.ev_charger_calculated_power,
            rec.ev_second_charger_calculated_power,
            rec.ev_total_planned_load_kwh,
            _fmt_live_w(live.ev.power_w),
            _fmt_live_w(live.ev_second.power_w),
            original_recommendation,
        )
