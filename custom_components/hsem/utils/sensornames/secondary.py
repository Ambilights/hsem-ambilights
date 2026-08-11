"""Entity names for the optional secondary-storage diagnostics."""

from homeassistant.util import slugify

from custom_components.hsem.const import DOMAIN


def get_secondary_storage_plan_sensor_name() -> str:
    """Return the secondary-storage plan display name."""
    return "Secondary Storage Plan"


def get_secondary_storage_plan_sensor_unique_id(entry_id: str) -> str:
    """Return the entry-scoped secondary-storage plan unique ID."""
    return f"{DOMAIN}_{entry_id}_secondary_storage_plan"


def get_secondary_storage_plan_sensor_entity_id() -> str:
    """Return the default secondary-storage plan entity ID."""
    return f"sensor.{slugify(f'{DOMAIN}_secondary_storage_plan')}"
