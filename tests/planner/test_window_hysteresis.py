"""Tests for window-level hysteresis (issue #315).

Window-level hysteresis prevents display-label flapping only when both labels
map to the same executable maximise-self-consumption command. Hardware-mode,
EV-command, and flow-derived cap changes pass through immediately.

Acceptance criteria
-------------------
1. Only unrestricted batteries_charge_solar ↔ batteries_discharge_mode
   aliases are held within the hold window.
2. Minimum hold time is configurable.
3. Neutral recommendations (wait_mode, time_passed, None) do not trigger hold.
4. Feature disabled (0 min) always allows the switch.
5. First run (no previous state) always accepts the new recommendation.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from custom_components.hsem.planner.charge_scheduler import apply_window_hysteresis
from custom_components.hsem.utils.prices import SlotPrice
from custom_components.hsem.utils.recommendations import Recommendations

_TZ = ZoneInfo("Europe/Copenhagen")
_NOW = datetime(2024, 6, 15, 12, 0, tzinfo=_TZ)


def _make_slots(
    *recommendations: str | None,
) -> list:
    """Build a list of sequential PlannedSlots with the given recommendations."""
    from custom_components.hsem.models.planned_slot import PlannedSlot

    slots: list[PlannedSlot] = []
    for i, rec in enumerate(recommendations):
        start = _NOW + timedelta(hours=i)
        end = start + timedelta(hours=1)
        slots.append(
            PlannedSlot(
                start=start,
                end=end,
                price=SlotPrice(import_price=0.20, export_price=0.05),
                recommendation=rec,
            )
        )
    return slots


class TestWindowHysteresis:
    """Window-level hysteresis acceptance tests."""

    # ------------------------------------------------------------------
    # First-run behaviour
    # ------------------------------------------------------------------

    def test_no_previous_state_first_run(self):
        """When there is no previous state, hysteresis is inactive."""
        slots = _make_slots(
            Recommendations.BatteriesChargeGrid.value,
        )
        rec, _ = apply_window_hysteresis(
            slots,
            _NOW,
            window_hysteresis_minutes=30,
            previous_current_recommendation=None,
            previous_current_slot_start=None,
        )
        assert rec == Recommendations.BatteriesChargeGrid.value, (
            "First run must accept the new recommendation"
        )

    # ------------------------------------------------------------------
    # Command-changing transitions (must pass through)
    # ------------------------------------------------------------------

    def test_charge_to_charge_within_hold(self):
        """Grid-charge → solar-charge changes hardware and cannot be held."""
        slots = _make_slots(
            Recommendations.BatteriesChargeSolar.value,
        )
        rec, _ = apply_window_hysteresis(
            slots,
            _NOW,
            window_hysteresis_minutes=30,
            previous_current_recommendation=Recommendations.BatteriesChargeGrid.value,
            previous_current_slot_start=_NOW - timedelta(minutes=5),
        )
        assert rec == Recommendations.BatteriesChargeSolar.value
        assert slots[0].recommendation == Recommendations.BatteriesChargeSolar.value

    def test_discharge_to_discharge_within_hold(self):
        """MSC → fully-fed discharge changes hardware and cannot be held."""
        slots = _make_slots(
            Recommendations.ForceBatteriesDischarge.value,
        )
        rec, _ = apply_window_hysteresis(
            slots,
            _NOW,
            window_hysteresis_minutes=30,
            previous_current_recommendation=Recommendations.BatteriesDischargeMode.value,
            previous_current_slot_start=_NOW - timedelta(minutes=5),
        )
        assert rec == Recommendations.ForceBatteriesDischarge.value
        assert slots[0].recommendation == Recommendations.ForceBatteriesDischarge.value

    def test_ev_smart_charging_to_solar_within_hold(self):
        """EV control → solar MSC changes commands and cannot be held."""
        slots = _make_slots(
            Recommendations.BatteriesChargeSolar.value,
        )
        rec, _ = apply_window_hysteresis(
            slots,
            _NOW,
            window_hysteresis_minutes=10,
            previous_current_recommendation=Recommendations.EVSmartCharging.value,
            previous_current_slot_start=_NOW - timedelta(minutes=2),
        )
        assert rec == Recommendations.BatteriesChargeSolar.value

    def test_solar_to_unrestricted_bdm_is_held(self):
        """The two unrestricted MSC labels may be coalesced safely."""
        slots = _make_slots(Recommendations.BatteriesDischargeMode.value)

        rec, started_at = apply_window_hysteresis(
            slots,
            _NOW,
            window_hysteresis_minutes=10,
            previous_current_recommendation=Recommendations.BatteriesChargeSolar.value,
            previous_current_slot_start=_NOW - timedelta(minutes=2),
        )

        assert rec == Recommendations.BatteriesChargeSolar.value
        assert started_at == _NOW - timedelta(minutes=2)
        assert slots[0].recommendation == Recommendations.BatteriesChargeSolar.value

    def test_unrestricted_bdm_to_solar_is_held(self):
        """The safe MSC alias is symmetric."""
        slots = _make_slots(Recommendations.BatteriesChargeSolar.value)

        rec, _ = apply_window_hysteresis(
            slots,
            _NOW,
            window_hysteresis_minutes=10,
            previous_current_recommendation=Recommendations.BatteriesDischargeMode.value,
            previous_current_slot_start=_NOW - timedelta(minutes=2),
        )

        assert rec == Recommendations.BatteriesDischargeMode.value

    def test_partial_bdm_bypasses_alias_hold(self):
        """A 0.002 kWh grid share makes the BDM cap command-significant."""
        slots = _make_slots(Recommendations.BatteriesDischargeMode.value)
        slots[0].batteries_discharged_kwh = 0.2
        slots[0].grid_import_kwh = 0.002

        rec, started_at = apply_window_hysteresis(
            slots,
            _NOW,
            window_hysteresis_minutes=10,
            previous_current_recommendation=Recommendations.BatteriesChargeSolar.value,
            previous_current_slot_start=_NOW - timedelta(minutes=2),
        )

        assert rec == Recommendations.BatteriesDischargeMode.value
        assert started_at == _NOW
        assert slots[0].recommendation == Recommendations.BatteriesDischargeMode.value

    def test_grid_import_rounding_residue_keeps_alias_hold(self):
        """Exactly 0.001 kWh is publication residue, not a partial BDM cap."""
        slots = _make_slots(Recommendations.BatteriesDischargeMode.value)
        slots[0].batteries_discharged_kwh = 0.2
        slots[0].grid_import_kwh = 0.001

        rec, started_at = apply_window_hysteresis(
            slots,
            _NOW,
            window_hysteresis_minutes=10,
            previous_current_recommendation=Recommendations.BatteriesChargeSolar.value,
            previous_current_slot_start=_NOW - timedelta(minutes=2),
        )

        assert rec == Recommendations.BatteriesChargeSolar.value
        assert started_at == _NOW - timedelta(minutes=2)
        assert slots[0].recommendation == Recommendations.BatteriesChargeSolar.value

    def test_optimizer_hold_bypasses_previous_actionable_label(self):
        """Label hysteresis must not add energy to a solved MILP hold slot."""
        slots = _make_slots(Recommendations.EVSmartCharging.value)
        slots[0].primary_battery_hold = True

        rec, start = apply_window_hysteresis(
            slots,
            _NOW,
            window_hysteresis_minutes=30,
            previous_current_recommendation=Recommendations.BatteriesChargeGrid.value,
            previous_current_slot_start=_NOW - timedelta(minutes=2),
        )

        assert rec == Recommendations.EVSmartCharging.value
        assert start == _NOW
        assert slots[0].recommendation == Recommendations.EVSmartCharging.value
        assert slots[0].primary_battery_hold is True

    def test_unpublished_slot_never_restores_price_driven_action(self):
        """Hysteresis cannot revive grid charge or forced battery export."""
        for previous in (
            Recommendations.BatteriesChargeGrid.value,
            Recommendations.ForceBatteriesDischarge.value,
        ):
            slots = _make_slots(Recommendations.BatteriesDischargeMode.value)
            slots[0].price_actionable = False

            rec, start = apply_window_hysteresis(
                slots,
                _NOW,
                window_hysteresis_minutes=30,
                previous_current_recommendation=previous,
                previous_current_slot_start=_NOW - timedelta(minutes=2),
            )

            assert rec == Recommendations.BatteriesDischargeMode.value
            assert start == _NOW
            assert (
                slots[0].recommendation == Recommendations.BatteriesDischargeMode.value
            )

    # ------------------------------------------------------------------
    # Within-category transitions after hold time expires
    # ------------------------------------------------------------------

    def test_alias_transition_after_hold(self):
        """A command-equivalent alias change is released at hold expiry."""
        slots = _make_slots(
            Recommendations.BatteriesDischargeMode.value,
        )
        rec, _ = apply_window_hysteresis(
            slots,
            _NOW,
            window_hysteresis_minutes=5,
            previous_current_recommendation=Recommendations.BatteriesChargeSolar.value,
            previous_current_slot_start=_NOW - timedelta(minutes=10),
        )
        assert rec == Recommendations.BatteriesDischargeMode.value, (
            "MSC alias change after hold time must be allowed"
        )

    # ------------------------------------------------------------------
    # Cross-category transitions within hold time
    # ------------------------------------------------------------------

    def test_charge_to_discharge_within_hold(self):
        """Grid charge → BDM changes hardware and must pass through."""
        slots = _make_slots(
            Recommendations.BatteriesDischargeMode.value,
        )
        rec, _ = apply_window_hysteresis(
            slots,
            _NOW,
            window_hysteresis_minutes=30,
            previous_current_recommendation=Recommendations.BatteriesChargeGrid.value,
            previous_current_slot_start=_NOW - timedelta(minutes=5),
        )
        assert rec == Recommendations.BatteriesDischargeMode.value
        assert slots[0].recommendation == Recommendations.BatteriesDischargeMode.value

    def test_discharge_to_charge_within_hold(self):
        """BDM → grid charge changes hardware and must pass through."""
        slots = _make_slots(
            Recommendations.BatteriesChargeGrid.value,
        )
        rec, _ = apply_window_hysteresis(
            slots,
            _NOW,
            window_hysteresis_minutes=30,
            previous_current_recommendation=Recommendations.BatteriesDischargeMode.value,
            previous_current_slot_start=_NOW - timedelta(minutes=5),
        )
        assert rec == Recommendations.BatteriesChargeGrid.value
        assert slots[0].recommendation == Recommendations.BatteriesChargeGrid.value

    def test_charge_to_force_export_within_hold(self):
        """EV charge → force export changes hardware and must pass through."""
        slots = _make_slots(
            Recommendations.ForceExport.value,
        )
        rec, _ = apply_window_hysteresis(
            slots,
            _NOW,
            window_hysteresis_minutes=15,
            previous_current_recommendation=Recommendations.EVSmartCharging.value,
            previous_current_slot_start=_NOW - timedelta(minutes=2),
        )
        assert rec == Recommendations.ForceExport.value

    # ------------------------------------------------------------------
    # Cross-category transitions after hold time expires
    # ------------------------------------------------------------------

    def test_charge_to_discharge_after_hold(self):
        """Charge→discharge after hold time must allow the switch."""
        slots = _make_slots(
            Recommendations.BatteriesDischargeMode.value,
        )
        rec, _ = apply_window_hysteresis(
            slots,
            _NOW,
            window_hysteresis_minutes=10,
            previous_current_recommendation=Recommendations.BatteriesChargeGrid.value,
            previous_current_slot_start=_NOW - timedelta(minutes=15),
        )
        assert rec == Recommendations.BatteriesDischargeMode.value, (
            "Charge→discharge after hold time must be allowed"
        )

    def test_discharge_to_charge_after_hold(self):
        """Discharge→charge after hold time must allow the switch."""
        slots = _make_slots(
            Recommendations.BatteriesChargeGrid.value,
        )
        rec, _ = apply_window_hysteresis(
            slots,
            _NOW,
            window_hysteresis_minutes=5,
            previous_current_recommendation=Recommendations.BatteriesDischargeMode.value,
            previous_current_slot_start=_NOW - timedelta(minutes=10),
        )
        assert rec == Recommendations.BatteriesChargeGrid.value, (
            "Discharge→charge after hold time must be allowed"
        )

    # ------------------------------------------------------------------
    # Neutral recommendations
    # ------------------------------------------------------------------

    def test_charge_to_neutral_no_hold(self):
        """Charge→neutral must not hold."""
        slots = _make_slots(
            Recommendations.BatteriesWaitMode.value,
        )
        rec, _ = apply_window_hysteresis(
            slots,
            _NOW,
            window_hysteresis_minutes=30,
            previous_current_recommendation=Recommendations.BatteriesChargeGrid.value,
            previous_current_slot_start=_NOW - timedelta(minutes=2),
        )
        assert rec == Recommendations.BatteriesWaitMode.value, (
            "Charge→neutral must not be held"
        )

    def test_discharge_to_neutral_no_hold(self):
        """Discharge→neutral must not hold."""
        slots = _make_slots(
            None,
        )
        rec, _ = apply_window_hysteresis(
            slots,
            _NOW,
            window_hysteresis_minutes=30,
            previous_current_recommendation=Recommendations.BatteriesDischargeMode.value,
            previous_current_slot_start=_NOW - timedelta(minutes=2),
        )
        assert rec is None, "Discharge→neutral must not be held"

    def test_neutral_to_charge_no_hold(self):
        """Neutral→charge must not hold."""
        slots = _make_slots(
            Recommendations.BatteriesChargeGrid.value,
        )
        rec, _ = apply_window_hysteresis(
            slots,
            _NOW,
            window_hysteresis_minutes=30,
            previous_current_recommendation=Recommendations.TimePassed.value,
            previous_current_slot_start=_NOW - timedelta(minutes=2),
        )
        assert rec == Recommendations.BatteriesChargeGrid.value, (
            "Neutral→charge must not be held"
        )

    # ------------------------------------------------------------------
    # Feature disabled
    # ------------------------------------------------------------------

    def test_feature_disabled_always_allows_switch(self):
        """When window_hysteresis_minutes is 0, all transitions are allowed."""
        slots = _make_slots(
            Recommendations.BatteriesDischargeMode.value,
        )
        rec, _ = apply_window_hysteresis(
            slots,
            _NOW,
            window_hysteresis_minutes=0,
            previous_current_recommendation=Recommendations.BatteriesChargeGrid.value,
            previous_current_slot_start=_NOW - timedelta(minutes=2),
        )
        assert rec == Recommendations.BatteriesDischargeMode.value, (
            "Transition must be allowed when feature is disabled"
        )

    # ------------------------------------------------------------------
    # Edge cases — exact boundary
    # ------------------------------------------------------------------

    def test_exactly_at_hold_time_boundary(self):
        """Transition exactly at hold time boundary must be allowed."""
        slots = _make_slots(
            Recommendations.BatteriesDischargeMode.value,
        )
        rec, _ = apply_window_hysteresis(
            slots,
            _NOW,
            window_hysteresis_minutes=10,
            previous_current_recommendation=Recommendations.BatteriesChargeSolar.value,
            previous_current_slot_start=_NOW - timedelta(minutes=10),
        )
        assert rec == Recommendations.BatteriesDischargeMode.value, (
            "Transition exactly at hold time boundary must be allowed (>=)"
        )

    def test_one_second_before_boundary(self):
        """Transition just before hold time boundary must be held."""
        slots = _make_slots(
            Recommendations.BatteriesDischargeMode.value,
        )
        rec, _ = apply_window_hysteresis(
            slots,
            _NOW,
            window_hysteresis_minutes=10,
            previous_current_recommendation=Recommendations.BatteriesChargeSolar.value,
            previous_current_slot_start=_NOW - timedelta(minutes=9, seconds=59),
        )
        assert rec == Recommendations.BatteriesChargeSolar.value, (
            "Transition just before hold time boundary must be held"
        )

    # ------------------------------------------------------------------
    # Return value semantics
    # ------------------------------------------------------------------

    def test_returns_updated_start_time_on_switch(self):
        """An accepted switch starts a fresh hold period at transition time."""
        slots = _make_slots(
            Recommendations.BatteriesDischargeMode.value,
        )
        switch_time = _NOW + timedelta(minutes=11)
        _, start = apply_window_hysteresis(
            slots,
            switch_time,
            window_hysteresis_minutes=5,
            previous_current_recommendation=Recommendations.BatteriesChargeSolar.value,
            previous_current_slot_start=_NOW - timedelta(minutes=15),
        )
        assert start == switch_time

    def test_accepted_switch_gets_a_new_hold_before_flip_back(self):
        """A late in-slot transition cannot immediately flap to its old label."""
        switched_slots = _make_slots(Recommendations.BatteriesDischargeMode.value)
        switch_time = _NOW + timedelta(minutes=11)
        switched_rec, switched_at = apply_window_hysteresis(
            switched_slots,
            switch_time,
            window_hysteresis_minutes=10,
            previous_current_recommendation=Recommendations.BatteriesChargeSolar.value,
            previous_current_slot_start=_NOW,
        )
        assert switched_rec == Recommendations.BatteriesDischargeMode.value
        assert switched_at == switch_time

        flip_slots = _make_slots(Recommendations.BatteriesChargeSolar.value)
        flip_rec, flip_started_at = apply_window_hysteresis(
            flip_slots,
            switch_time + timedelta(minutes=1),
            window_hysteresis_minutes=10,
            previous_current_recommendation=switched_rec,
            previous_current_slot_start=switched_at,
        )
        assert flip_rec == Recommendations.BatteriesDischargeMode.value
        assert flip_started_at == switch_time

    def test_fallback_fold_uses_elapsed_real_time(self):
        """The repeated autumn hour cannot extend a hold by another hour."""
        stockholm = ZoneInfo("Europe/Stockholm")
        activated = datetime(2026, 10, 25, 2, 55, tzinfo=stockholm, fold=0)
        now = datetime(2026, 10, 25, 2, 5, tzinfo=stockholm, fold=1)
        slots = _make_slots(Recommendations.BatteriesDischargeMode.value)
        slots[0].start = datetime(2026, 10, 25, 2, 0, tzinfo=stockholm, fold=0)
        slots[0].end = datetime(2026, 10, 25, 3, 0, tzinfo=stockholm)

        rec, started_at = apply_window_hysteresis(
            slots,
            now,
            window_hysteresis_minutes=10,
            previous_current_recommendation=Recommendations.BatteriesChargeSolar.value,
            previous_current_slot_start=activated,
        )

        assert rec == Recommendations.BatteriesDischargeMode.value
        assert started_at == now

    def test_returns_previous_start_time_on_hold(self):
        """When held, the returned start time must be the previous slot start."""
        slots = _make_slots(
            Recommendations.BatteriesDischargeMode.value,
        )
        prev_start = _NOW - timedelta(minutes=2)
        _, start = apply_window_hysteresis(
            slots,
            _NOW,
            window_hysteresis_minutes=10,
            previous_current_recommendation=Recommendations.BatteriesChargeSolar.value,
            previous_current_slot_start=prev_start,
        )
        assert start == prev_start, (
            "Returned start time must be the previous slot start when held"
        )
