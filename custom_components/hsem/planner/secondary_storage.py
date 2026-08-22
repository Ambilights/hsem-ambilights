"""Pure helpers for dedicated-load secondary-storage planning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from custom_components.hsem.models.planned_slot import PlannedSlot
from custom_components.hsem.models.price_forecast import PriceForecast
from custom_components.hsem.models.secondary_storage_config import (
    SecondaryStorageConfig,
)
from custom_components.hsem.planner.cost_helpers import slot_grid_cash_flow_cost
from custom_components.hsem.planner.future_value import forecast_effective_prices
from custom_components.hsem.utils.datetime_utils import utc_key
from custom_components.hsem.utils.misc import clamp_efficiency
from custom_components.hsem.utils.units import slot_duration_hours

SECONDARY_MODE_CHARGE = "charge"
SECONDARY_MODE_SBU = "sbu"
SECONDARY_MODE_UTILITY = "utility"

type SecondaryTerminalPriceSource = Literal[
    "configured",
    "published",
    "forecast",
    "none",
]


@dataclass(frozen=True)
class SecondaryTerminalPriceResolution:
    """Resolved PowMr terminal inventory price and its authoritative source."""

    price_per_kwh: float | None
    source: SecondaryTerminalPriceSource


def secondary_charge_limits_kwh(
    config: SecondaryStorageConfig,
    slot_hours: float,
) -> tuple[float, float]:
    """Return minimum and maximum battery-side charge energy per slot."""
    volts = max(config.nominal_voltage_v, 0.0)
    minimum = volts * max(config.min_charge_current_a, 0.0) * slot_hours / 1000.0
    maximum = volts * max(config.max_charge_current_a, 0.0) * slot_hours / 1000.0
    return min(minimum, maximum), maximum


def secondary_slot_duration_hours(slot: PlannedSlot, now: datetime) -> float:
    """Return usable PowMr duration for a future or partially elapsed slot."""
    if utc_key(slot.end) <= utc_key(now):
        return 0.0
    if utc_key(slot.start) < utc_key(now):
        return slot_duration_hours(now, slot.end)
    return slot_duration_hours(slot.start, slot.end)


def populate_secondary_storage_load(
    slots: list[PlannedSlot],
    config: SecondaryStorageConfig,
    now: datetime,
) -> None:
    """Populate the dedicated-load energy forecast on every slot."""
    if not config.valid:
        return
    for slot in slots:
        hours = secondary_slot_duration_hours(slot, now)
        slot.secondary_storage_load_kwh = round(
            max(config.load_power_w, 0.0) * hours / 1000.0,
            6,
        )


def secondary_site_load_offset_kwh(
    slot: PlannedSlot,
    config: SecondaryStorageConfig,
) -> float:
    """Return the dedicated load represented in the site-load forecast."""
    load_kwh = max(slot.secondary_storage_load_kwh, 0.0)
    if not config.base_load_includes_dedicated_load:
        return load_kwh
    # Mixed Utility/SBU history can contain less than a full dedicated load.
    # Subtracting more than the gross house forecast would model impossible
    # PowMr backfeed, so value only the portion demonstrably present.
    return min(load_kwh, max(slot.avg_house_consumption_kwh, 0.0))


def apply_secondary_utility_bypass(
    slots: list[PlannedSlot],
    config: SecondaryStorageConfig,
    now: datetime,
    *,
    export_min_price: float = 0.0,
) -> None:
    """Apply a physically valid utility-bypass fallback to non-MILP plans."""
    if not config.valid:
        return

    populate_secondary_storage_load(slots, config, now)
    current_capacity = config.current_usable_kwh
    for slot in slots:
        if utc_key(slot.end) <= utc_key(now):
            continue

        load_kwh = slot.secondary_storage_load_kwh
        slot.secondary_storage_charged_kwh = 0.0
        slot.secondary_storage_discharged_kwh = 0.0
        slot.secondary_storage_grid_import_kwh = round(load_kwh, 3)
        slot.secondary_storage_estimated_capacity_kwh = round(current_capacity, 3)
        slot.secondary_storage_estimated_soc_pct = round(
            config.current_soc_pct,
            2,
        )
        slot.secondary_storage_charge_current_a = 0.0
        slot.secondary_storage_mode = SECONDARY_MODE_UTILITY

        site_delta = 0.0 if config.base_load_includes_dedicated_load else load_kwh
        net_grid = slot.grid_import_kwh - slot.grid_export_kwh + site_delta
        slot.grid_import_kwh = round(max(net_grid, 0.0), 3)
        slot.grid_export_kwh = round(max(-net_grid, 0.0), 3)
        slot.estimated_cost_currency = round(
            slot_grid_cash_flow_cost(slot, export_min_price=export_min_price), 4
        )


def resolve_secondary_terminal_price(
    slots: list[PlannedSlot],
    config: SecondaryStorageConfig,
    now: datetime,
    forecast: PriceForecast | None = None,
) -> float | None:
    """Return the configured or horizon-tail value of stored secondary energy.

    With ``forecast`` supplied, only its points beyond the published prefix are
    eligible — a prediction never competes with a real price — and those are
    valued by the same rule this function already uses for published prices:
    the mean of the window, discounted for discharge efficiency.  The higher of
    the two then wins, so a prediction can only raise the worth of stored
    energy.  The filter matters more here than on the primary side, because a
    mean taken over overlapping published and predicted estimates of the same
    hours is arbitrary rather than merely conservative.

    The mean is deliberate rather than a peak: the PowMr serves its dedicated
    load continuously, so stored energy is spent across the window rather than
    concentrated on its dearest hour. That differs from the retained legacy
    primary scalar helper; production primary planning uses bounded
    residual-demand tiers instead.
    """
    return resolve_secondary_terminal_price_details(
        slots,
        config,
        now,
        forecast=forecast,
    ).price_per_kwh


def resolve_secondary_terminal_price_details(
    slots: list[PlannedSlot],
    config: SecondaryStorageConfig,
    now: datetime,
    forecast: PriceForecast | None = None,
) -> SecondaryTerminalPriceResolution:
    """Return the PowMr terminal price together with its winning source."""
    if not config.valid:
        return SecondaryTerminalPriceResolution(None, "none")
    if config.replacement_price_per_kwh is not None:
        return SecondaryTerminalPriceResolution(
            max(config.replacement_price_per_kwh, 0.0),
            "configured",
        )

    discharge_eff = clamp_efficiency(config.discharge_efficiency_pct)
    predicted = forecast_effective_prices(forecast, now, slots)
    predicted_value = (
        (sum(predicted) / len(predicted)) * discharge_eff if predicted else None
    )
    published_value = _published_secondary_terminal_price(slots, discharge_eff, now)
    if published_value is None:
        return SecondaryTerminalPriceResolution(
            predicted_value,
            "forecast" if predicted_value is not None else "none",
        )
    if predicted_value is None:
        return SecondaryTerminalPriceResolution(published_value, "published")
    if predicted_value > published_value:
        return SecondaryTerminalPriceResolution(predicted_value, "forecast")
    return SecondaryTerminalPriceResolution(published_value, "published")


def _published_secondary_terminal_price(
    slots: list[PlannedSlot],
    discharge_eff: float,
    now: datetime,
) -> float | None:
    """Original published-price-only horizon-tail valuation, unchanged."""
    future = [
        slot
        for slot in slots
        if (
            utc_key(slot.end) > utc_key(now)
            and slot.price_actionable
            and math.isfinite(slot.price.import_price)
        )
    ]
    if not future:
        return None

    horizon_end = max(utc_key(slot.end) for slot in future)
    tail_start = horizon_end - timedelta(hours=24)
    tail_prices = [
        max(slot.price.import_price, 0.0)
        for slot in future
        if utc_key(slot.start) >= tail_start
    ]
    if not tail_prices:
        return None

    # Battery-side energy can offset only discharge-efficiency-adjusted load.
    return (sum(tail_prices) / len(tail_prices)) * discharge_eff
