"""Tests for custom_sensors/recommendation_resolver.py.

All remaining priority branches of :func:`resolve_current_recommendation` are
tested with plain dataclasses — no Home Assistant required.
"""

from __future__ import annotations

from datetime import UTC

from custom_components.hsem.custom_sensors.recommendation_resolver import (
    resolve_current_recommendation,
)
from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.models.live_state import EVLiveState, LiveState
from custom_components.hsem.utils.recommendations import Recommendations

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rec(recommendation: str | None = None) -> HourlyRecommendation:
    """Return a minimal HourlyRecommendation with a given recommendation value."""
    from datetime import datetime

    now = datetime.now(tz=UTC)
    return HourlyRecommendation(
        avg_house_consumption_kwh=0.5,
        avg_house_consumption_1d_kwh=0.5,
        avg_house_consumption_3d_kwh=0.5,
        avg_house_consumption_7d_kwh=0.5,
        avg_house_consumption_14d_kwh=0.5,
        batteries_charged_kwh=0.0,
        batteries_discharged_kwh=0.0,
        end=now,
        estimated_battery_capacity_kwh=5.0,
        estimated_battery_soc_pct=50,
        estimated_cost_currency=0.1,
        estimated_net_consumption_kwh=0.3,
        export_price=0.5,
        grid_export_kwh=0.0,
        grid_import_kwh=0.0,
        import_price=0.8,
        recommendation=recommendation,
        solcast_pv_estimate_kwh=0.0,
        start=now,
        import_price_available=True,
        export_price_available=True,
        price_actionable=True,
    )


def _make_live(
    import_price: float = 0.5,
    ev_charging: bool = False,
    ev2_charging: bool = False,
    battery_kwh: float = 5.0,
    ev_power_w: float | None = None,
    ev2_power_w: float | None = None,
    import_price_available: bool = True,
) -> LiveState:
    live = LiveState()
    live.import_electricity_price = import_price
    live.import_electricity_price_available = import_price_available
    live.ev = EVLiveState(is_charging=ev_charging, power_w=ev_power_w)
    live.ev_second = EVLiveState(is_charging=ev2_charging, power_w=ev2_power_w)
    live.battery_current_capacity_kwh = battery_kwh
    return live


# ---------------------------------------------------------------------------
# Priority 1: Negative import price → ForceExport
# ---------------------------------------------------------------------------


class TestNegativeImportPrice:
    def test_negative_price_overrides_any_recommendation(self):
        rec = _make_rec(recommendation=Recommendations.BatteriesDischargeMode.value)
        resolve_current_recommendation(rec, _make_live(import_price=-0.01))
        assert rec.recommendation == Recommendations.ForceExport.value

    def test_zero_price_does_not_force_export(self):
        rec = _make_rec(recommendation=Recommendations.BatteriesWaitMode.value)
        resolve_current_recommendation(rec, _make_live(import_price=0.0))
        assert rec.recommendation == Recommendations.BatteriesWaitMode.value

    def test_positive_price_does_not_force_export(self):
        rec = _make_rec(recommendation=Recommendations.BatteriesWaitMode.value)
        resolve_current_recommendation(rec, _make_live(import_price=0.5))
        assert rec.recommendation == Recommendations.BatteriesWaitMode.value

    def test_unavailable_negative_live_price_does_not_force_export(self):
        """A stale numeric value has no authority after its source disappears."""
        rec = _make_rec(recommendation=Recommendations.BatteriesWaitMode.value)
        live = _make_live(import_price=-0.01, import_price_available=False)

        resolve_current_recommendation(rec, live)

        assert rec.recommendation == Recommendations.BatteriesWaitMode.value

    def test_nonactionable_slot_does_not_force_export(self):
        """A live negative price cannot revive price control in an unknown slot."""
        rec = _make_rec(recommendation=Recommendations.BatteriesWaitMode.value)
        rec.price_actionable = False

        resolve_current_recommendation(rec, _make_live(import_price=-0.01))

        assert rec.recommendation == Recommendations.BatteriesWaitMode.value


# ---------------------------------------------------------------------------
# Priority 2: Grid charge → preserved
# ---------------------------------------------------------------------------


class TestGridChargePreserved:
    def test_grid_charge_not_overridden_by_ev(self):
        rec = _make_rec(recommendation=Recommendations.BatteriesChargeGrid.value)
        live = _make_live(import_price=0.5, ev_charging=True)
        resolve_current_recommendation(rec, live)
        assert rec.recommendation == Recommendations.BatteriesChargeGrid.value

    def test_grid_charge_not_overridden_by_negative_price(self):
        rec = _make_rec(recommendation=Recommendations.BatteriesChargeGrid.value)
        live = _make_live(import_price=-0.05)
        # Negative price is priority 1, so it DOES override grid charge
        resolve_current_recommendation(rec, live)
        assert rec.recommendation == Recommendations.ForceExport.value


# ---------------------------------------------------------------------------
# Priority 3: Active EV → EVSmartCharging
# ---------------------------------------------------------------------------


class TestEVSmartCharging:
    def test_ev1_charging_triggers_ev_mode(self):
        rec = _make_rec(recommendation=Recommendations.BatteriesDischargeMode.value)
        rec.ev_charger_calculated_power = 7500.0  # Planner allocated power
        live = _make_live(import_price=0.5, ev_charging=True)
        resolve_current_recommendation(rec, live)
        assert rec.recommendation == Recommendations.EVSmartCharging.value

    def test_ev2_charging_triggers_ev_mode(self):
        rec = _make_rec(recommendation=Recommendations.BatteriesWaitMode.value)
        rec.ev_second_charger_calculated_power = 11000.0  # Planner allocated power
        live = _make_live(import_price=0.5, ev2_charging=True)
        resolve_current_recommendation(rec, live)
        assert rec.recommendation == Recommendations.EVSmartCharging.value

    def test_no_ev_charging_no_override(self):
        rec = _make_rec(recommendation=Recommendations.BatteriesWaitMode.value)
        live = _make_live(ev_charging=False, ev2_charging=False)
        resolve_current_recommendation(rec, live)
        assert rec.recommendation == Recommendations.BatteriesWaitMode.value

    def test_ev1_charging_but_planner_zero_power_no_override(self):
        """EV is charging but planner set power to 0 → do NOT override."""
        rec = _make_rec(recommendation=Recommendations.BatteriesWaitMode.value)
        # Planner explicitly set power to 0 (stop charging command)
        rec.ev_charger_calculated_power = 0.0
        rec.ev_total_planned_load_kwh = 0.0
        live = _make_live(ev_charging=True)
        resolve_current_recommendation(rec, live)
        # Should keep original WaitMode because planner said stop
        assert rec.recommendation == Recommendations.BatteriesWaitMode.value

    def test_ev1_charging_with_positive_power_overrides(self):
        """EV is charging AND planner allocated positive power → override."""
        rec = _make_rec(recommendation=Recommendations.BatteriesWaitMode.value)
        rec.ev_charger_calculated_power = 7500.0  # Planner allocated power
        live = _make_live(ev_charging=True)
        resolve_current_recommendation(rec, live)
        assert rec.recommendation == Recommendations.EVSmartCharging.value

    def test_ev2_charging_with_positive_power_overrides(self):
        """Second EV charging AND planner allocated positive power → override."""
        rec = _make_rec(recommendation=Recommendations.BatteriesWaitMode.value)
        rec.ev_second_charger_calculated_power = 11000.0  # Planner allocated power
        live = _make_live(ev2_charging=True)
        resolve_current_recommendation(rec, live)
        assert rec.recommendation == Recommendations.EVSmartCharging.value

    def test_ev_relabel_preserves_primary_battery_hold(self) -> None:
        """An active allocated EV may relabel, but cannot erase strict Hold."""
        rec = _make_rec(recommendation=Recommendations.BatteriesWaitMode.value)
        rec.ev_charger_calculated_power = 7500.0
        rec.primary_battery_hold = True
        live = _make_live(import_price=0.5, ev_charging=True)

        resolve_current_recommendation(rec, live)

        assert rec.recommendation == Recommendations.EVSmartCharging.value
        assert rec.primary_battery_hold is True


# ---------------------------------------------------------------------------
# None rec safety
# ---------------------------------------------------------------------------


class TestNoneRec:
    def test_none_rec_does_not_raise(self):
        live = _make_live()
        resolve_current_recommendation(None, live)  # type: ignore[arg-type]
        # No exception = pass
