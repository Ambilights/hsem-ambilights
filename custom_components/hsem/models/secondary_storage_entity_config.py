"""Home Assistant configuration for optional secondary storage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SecondaryStorageEntityConfig:
    """Configuration and entity IDs for a dedicated-load secondary battery."""

    enabled: bool = False
    control_enabled: bool = False

    soc_entity: str | None = None
    battery_net_power_entity: str | None = None
    load_power_entity: str | None = None
    output_source_priority_entity: str | None = None
    charger_source_priority_entity: str | None = None
    max_charge_current_entity: str | None = None

    capacity_kwh: float = 15.0
    min_soc_pct: float = 20.0
    max_soc_pct: float = 100.0
    nominal_voltage_v: float = 24.0
    min_charge_current_a: float = 10.0
    max_charge_current_a: float = 60.0
    charge_efficiency_pct: float = 93.0
    discharge_efficiency_pct: float = 93.0
    inverter_standby_power_w: float = 55.0
    cycle_cost_per_kwh: float = 0.0
    base_load_includes_dedicated_load: bool = True
    allow_primary_battery_transfer: bool = False
    grid_phase: int = 3
