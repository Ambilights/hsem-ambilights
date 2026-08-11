"""Configuration for an optional dedicated-load secondary battery."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SecondaryStorageConfig:
    """Describe a non-exporting battery that supplies a dedicated AC load.

    The secondary inverter is connected to the site AC bus for utility bypass
    and charging, but its battery output can serve only ``load_power_w``. It
    cannot export to the site bus or grid.
    """

    enabled: bool = False
    capacity_kwh: float = 0.0
    current_soc_pct: float = 0.0
    min_soc_pct: float = 20.0
    max_soc_pct: float = 100.0
    nominal_voltage_v: float = 24.0
    load_power_w: float = 0.0
    max_charge_current_a: float = 0.0
    min_charge_current_a: float = 0.0
    charge_current_step_a: float = 10.0
    charge_efficiency_pct: float = 93.0
    discharge_efficiency_pct: float = 93.0
    inverter_standby_power_w: float = 0.0
    cycle_cost_per_kwh: float = 0.0
    replacement_price_per_kwh: float | None = None
    base_load_includes_dedicated_load: bool = False
    allow_primary_battery_transfer: bool = False

    @property
    def current_usable_kwh(self) -> float:
        """Return current energy above the configured reserve."""
        bounded_soc = min(max(self.current_soc_pct, self.min_soc_pct), self.max_soc_pct)
        return self.capacity_kwh * (bounded_soc - self.min_soc_pct) / 100.0

    @property
    def usable_kwh(self) -> float:
        """Return energy available between minimum and maximum SoC."""
        return self.capacity_kwh * max(self.max_soc_pct - self.min_soc_pct, 0.0) / 100.0

    @property
    def valid(self) -> bool:
        """Return whether the configuration can be optimised safely."""
        return (
            self.enabled
            and self.capacity_kwh > 1e-9
            and self.nominal_voltage_v > 1e-9
            and 0.0 <= self.current_soc_pct <= 100.0
            and 0.0 < self.min_charge_current_a <= self.max_charge_current_a
            and self.charge_current_step_a > 1e-9
            and self.load_power_w >= 0.0
            and 0.0 <= self.min_soc_pct < self.max_soc_pct <= 100.0
            and 0.0 < self.charge_efficiency_pct <= 100.0
            and 0.0 < self.discharge_efficiency_pct <= 100.0
        )
