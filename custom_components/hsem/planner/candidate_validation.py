"""Physical validation for fully simulated candidate plans."""

from __future__ import annotations

from custom_components.hsem.models.secondary_storage_config import (
    SecondaryStorageConfig,
)
from custom_components.hsem.planner.candidate_generator import CandidatePlan

_SOC_TOLERANCE_PCT = 0.5


def validate_candidate(
    candidate: CandidatePlan,
    end_of_discharge_soc_pct: float,
    *,
    secondary_storage: SecondaryStorageConfig | None = None,
) -> tuple[bool, str]:
    """Return whether primary and secondary SoC remain inside their floors."""
    floor = end_of_discharge_soc_pct - _SOC_TOLERANCE_PCT
    for slot in candidate.slots:
        soc = slot.estimated_battery_soc_pct
        if soc > 0 and soc < floor:
            return (
                False,
                (
                    f"SoC {soc:.1f}% dropped below floor "
                    f"{end_of_discharge_soc_pct:.1f}% "
                    f"at slot starting {slot.start.isoformat()}."
                ),
            )
        secondary_soc = slot.secondary_storage_estimated_soc_pct
        if (
            secondary_storage is not None
            and secondary_storage.valid
            and secondary_soc > 0
            and secondary_soc < secondary_storage.min_soc_pct - _SOC_TOLERANCE_PCT
        ):
            return (
                False,
                (
                    f"Secondary SoC {secondary_soc:.1f}% dropped below floor "
                    f"{secondary_storage.min_soc_pct:.1f}% "
                    f"at slot starting {slot.start.isoformat()}."
                ),
            )
    return True, ""
