"""Dataclass representing a single planning slot recommendation.

Each :class:`HourlyRecommendation` captures the planner's decision for one
time slot: the recommended working mode, all energy flows (consumption,
production, battery, EV, grid), and the resulting cost estimate.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class HourlyRecommendation:
    """A single time-slot planning decision produced by the HSEM planner.

    All numeric energy fields are in kWh for the slot duration.

    Attributes:
        start: Timezone-aware start of the slot.
        end: Timezone-aware end of the slot.
        recommendation: Working-mode recommendation string (or None).
        import_price: Spot import price (local currency/kWh).
        export_price: Spot export price (local currency/kWh).
        import_price_available / export_price_available: Whether each numeric
            price came from its configured source. A genuine zero is available.
        price_actionable: Whether price-driven control is allowed in the slot.
        avg_house_consumption_kwh: Weighted spike-aware consumption estimate (kWh).
        historical_avg_house_consumption_kwh: Weighted estimate before the
            current slot is replaced with live power. Runtime EV protection
            uses this stable history rather than the live-resolved value.
        avg_house_consumption_1d_kwh: 1-day window contribution (kWh).
        avg_house_consumption_3d_kwh: 3-day window contribution (kWh).
        avg_house_consumption_7d_kwh: 7-day window contribution (kWh).
        avg_house_consumption_14d_kwh: 14-day window contribution (kWh).
        solcast_pv_estimate_kwh: Forecast PV production (kWh).
        solcast_pv_estimate_available: Whether the numeric PV estimate came
            from a finite, matched forecast point. A published zero is available.
        estimated_net_consumption_kwh: avg_consumption + ev_planned_load_kwh - pv_estimate (kWh).
        ev_planned_load_kwh: Extra EV AC load added to net consumption (kWh, ≥ 0).
            Combined injected load from primary and second EV.  Zero when EV
            planned load integration is disabled, the EV is not scheduled to
            charge, or ``base_load_includes_ev=True`` (EV already in base load).
        ev_accounted_load_kwh: EV AC load already included in the house
            consumption sensor (kWh, ≥ 0).  Non-zero only when
            ``base_load_includes_ev=True``.  Not added to net consumption.
        ev_total_planned_load_kwh: Total EV AC load planned for this slot
            (kWh, ≥ 0).  Equals ``ev_planned_load_kwh + ev_accounted_load_kwh``.
            Use this for diagnostics and UI — it is non-zero whenever EV
            charging is planned regardless of the ``base_load_includes_ev`` flag.
        ev_charger_calculated_power: Target AC power (W) for the primary EV
            charger during this slot.  Zero when no charging is planned.
        ev_second_charger_calculated_power: Target AC power (W) for the
            second EV charger during this slot.  Zero when no charging is planned.
        estimated_cost_currency: Estimated grid cost for the slot (local currency).
        batteries_charged_kwh: Energy scheduled to be charged into battery (kWh).
        batteries_discharged_kwh: Energy drawn from battery by the SoC simulation (kWh).
        estimated_battery_capacity_kwh: Remaining usable battery energy above the
            discharge floor at the end of the slot (kWh).
        estimated_battery_soc_pct: Simulated absolute battery SoC (0-100 %) at the
            end of the slot, relative to the rated capacity.  Populated by
            :func:`~planner.soc_simulation.simulate_soc` and suitable for
            plotting in an Apex chart time-series.
        grid_import_kwh: Energy imported from the grid during this slot (kWh).
        grid_export_kwh: Energy exported to the grid during this slot (kWh).
        primary_battery_hold: Whether the optimiser explicitly requires the
            primary battery to hold during this slot. This survives display
            relabelling such as ``EVSmartCharging``.
    """

    start: datetime
    end: datetime
    avg_house_consumption_kwh: float
    avg_house_consumption_1d_kwh: float
    avg_house_consumption_3d_kwh: float
    avg_house_consumption_7d_kwh: float
    avg_house_consumption_14d_kwh: float
    batteries_charged_kwh: float
    batteries_discharged_kwh: float
    estimated_battery_capacity_kwh: float
    estimated_battery_soc_pct: float
    estimated_cost_currency: float
    estimated_net_consumption_kwh: float
    export_price: float
    grid_export_kwh: float
    grid_import_kwh: float
    import_price: float
    recommendation: Any | None
    solcast_pv_estimate_kwh: float
    historical_avg_house_consumption_kwh: float = 0.0
    ev_planned_load_kwh: float = 0.0
    ev_accounted_load_kwh: float = 0.0
    ev_total_planned_load_kwh: float = 0.0
    ev_charger_calculated_power: float = 0.0
    ev_second_charger_calculated_power: float = 0.0
    secondary_storage_load_kwh: float = 0.0
    secondary_storage_charged_kwh: float = 0.0
    secondary_storage_discharged_kwh: float = 0.0
    secondary_storage_grid_import_kwh: float = 0.0
    secondary_storage_estimated_capacity_kwh: float = 0.0
    secondary_storage_estimated_soc_pct: float = 0.0
    secondary_storage_charge_current_a: float = 0.0
    secondary_storage_mode: str | None = None
    primary_battery_hold: bool = False
    import_price_available: bool = False
    export_price_available: bool = False
    price_actionable: bool = False
    solcast_pv_estimate_available: bool = False
