"""Battery export must beat what the same stored energy is worth kept.

``apply_excess_export`` had two branches.  The one covering slots with net
house demand compared correctly — ``export_price >= import_price + wear``,
i.e. never sell below what you would pay to buy it back.  The other branch,
for slots where PV already covers the house, was commented "pure export
profit" and tested only a static price floor.

That reasoning holds for surplus *solar*, which has no alternative use.  It
does not hold for stored *battery* energy, which is what gets discharged.

Observed live on a real 48-hour plan: three ``force_batteries_discharge``
slots sold 7.12 kWh at an average of 0.657 SEK/kWh during hours when PV
covered the house, and the same plan then bought 3.77 kWh back at an average
of 0.963 to serve the evening — a 1.15 SEK loss — before grid-charging again
at 23:45 to store for the next day's peak.  Only a manually configured
``battery_export_min_price`` floor could suppress it.

The branch now compares against the best import price the stored energy could
still displace, so the trade is declined on its own economics.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.planner.discharge_scheduler import (
    apply_excess_export,
    best_alternative_import_price,
)
from custom_components.hsem.utils.prices import SlotPrice
from custom_components.hsem.utils.recommendations import Recommendations

_NOW = datetime(2026, 8, 15, 17, 0, tzinfo=UTC)


def _slot(
    offset_hours: float,
    *,
    import_price: float,
    export_price: float,
    net_kwh: float,
    price_actionable: bool = True,
) -> PlannedSlot:
    """Build a one-hour slot at ``_NOW + offset_hours``."""
    start = _NOW + timedelta(hours=offset_hours)
    slot = PlannedSlot(
        start=start,
        end=start + timedelta(hours=1),
        price=SlotPrice(import_price=import_price, export_price=export_price),
        solcast_pv_estimate_kwh=max(-net_kwh, 0.0),
        avg_house_consumption_kwh=0.5,
        estimated_net_consumption_kwh=net_kwh,
        recommendation=None,
    )
    slot.price_actionable = price_actionable
    return slot


def _run(slots: list[PlannedSlot], *, floor: float = 0.0) -> list[str | None]:
    warnings: list[str] = []
    apply_excess_export(
        slots,
        _NOW,
        current_capacity=10.0,
        required_capacity=2.0,
        export_price_threshold=0.0,
        warnings=warnings,
        export_min_price=0.0,
        recommended_threshold=0.0,
        battery_export_min_price=floor,
    )
    return [s.recommendation for s in slots]


class TestBestAlternativeImportPrice:
    """What a kWh is worth if kept rather than sold."""

    def test_highest_later_import_price_wins(self) -> None:
        slots = [
            _slot(0, import_price=1.685, export_price=0.688, net_kwh=-0.2),
            _slot(1, import_price=1.100, export_price=0.300, net_kwh=0.6),
            _slot(2, import_price=1.646, export_price=0.658, net_kwh=0.6),
        ]

        assert best_alternative_import_price(slots, _NOW, 5.0) == 1.646

    def test_earlier_slots_are_ignored(self) -> None:
        slots = [
            _slot(0, import_price=9.99, export_price=0.10, net_kwh=0.6),
            _slot(1, import_price=1.100, export_price=0.300, net_kwh=0.6),
        ]

        assert best_alternative_import_price(slots, slots[0].start, 5.0) == 1.100

    def test_scan_stops_once_solar_would_refill_the_export(self) -> None:
        """Beyond a real refill, held energy only displaces free PV."""
        slots = [
            _slot(1, import_price=0.50, export_price=0.20, net_kwh=-3.0),
            _slot(2, import_price=9.99, export_price=0.20, net_kwh=0.6),
        ]

        assert best_alternative_import_price(slots, _NOW, 2.0) == 0.0

    def test_a_single_sunny_slot_does_not_end_the_scan(self) -> None:
        """The old "stop at first surplus" rule undervalued every later hour."""
        slots = [
            _slot(1, import_price=0.50, export_price=0.20, net_kwh=-0.2),
            _slot(2, import_price=1.646, export_price=0.658, net_kwh=0.6),
        ]

        assert best_alternative_import_price(slots, _NOW, 5.0) == 1.646

    def test_no_later_demand_means_nothing_is_forgone(self) -> None:
        slots = [_slot(0, import_price=1.0, export_price=0.9, net_kwh=-0.2)]

        assert best_alternative_import_price(slots, _NOW, 5.0) == 0.0

    def test_unpriced_slots_are_skipped_not_treated_as_worthless(self) -> None:
        slots = [
            _slot(
                1,
                import_price=0.0,
                export_price=0.0,
                net_kwh=0.6,
                price_actionable=False,
            ),
            _slot(2, import_price=1.500, export_price=0.400, net_kwh=0.6),
        ]

        assert best_alternative_import_price(slots, _NOW, 5.0) == 1.500


class TestLiveScenario:
    """The plan that motivated this, reduced to its essentials."""

    @staticmethod
    def _evening() -> list[PlannedSlot]:
        # 17:00 PV still covers the house at the day's best export price;
        # the evening that follows needs the battery at ~0.96-1.65.
        return [
            _slot(0, import_price=1.685, export_price=0.688, net_kwh=-0.22),
            _slot(2, import_price=1.646, export_price=0.658, net_kwh=0.10),
            _slot(3, import_price=1.628, export_price=0.643, net_kwh=0.60),
            _slot(5, import_price=0.963, export_price=0.143, net_kwh=0.60),
            _slot(6, import_price=0.922, export_price=0.078, net_kwh=0.42),
        ]

    def test_selling_below_the_evening_import_price_is_declined(self) -> None:
        """0.688 must not be taken when the same kWh saves 1.646 later."""
        slots = self._evening()

        recs = _run(slots)

        assert recs[0] != Recommendations.ForceBatteriesDischarge.value

    def test_no_static_floor_is_needed_to_decline_it(self) -> None:
        """The whole point: economics, not a configured threshold."""
        slots = self._evening()

        recs = _run(slots, floor=0.0)

        assert Recommendations.ForceBatteriesDischarge.value not in recs

    def test_a_genuinely_high_export_price_is_still_taken(self) -> None:
        """Above every later import price, selling really is best."""
        slots = self._evening()
        slots[0].price = SlotPrice(import_price=1.685, export_price=2.000)

        recs = _run(slots)

        assert recs[0] == Recommendations.ForceBatteriesDischarge.value


class TestUnchangedBehaviour:
    """The branch that was already correct must not move."""

    def test_net_demand_slot_still_requires_export_above_import(self) -> None:
        slots = [
            _slot(0, import_price=0.30, export_price=0.90, net_kwh=0.6),
            _slot(1, import_price=0.30, export_price=0.10, net_kwh=0.6),
        ]

        recs = _run(slots)

        assert recs[0] == Recommendations.ForceBatteriesDischarge.value
        assert recs[1] != Recommendations.ForceBatteriesDischarge.value

    def test_the_configured_floor_still_applies_on_top(self) -> None:
        """Raising the floor must still suppress an otherwise-eligible slot."""
        eligible = [_slot(0, import_price=0.30, export_price=0.90, net_kwh=0.6)]
        suppressed = [_slot(0, import_price=0.30, export_price=0.90, net_kwh=0.6)]

        assert _run(eligible, floor=0.0)[0] == (
            Recommendations.ForceBatteriesDischarge.value
        )
        assert _run(suppressed, floor=1.5)[0] != (
            Recommendations.ForceBatteriesDischarge.value
        )
