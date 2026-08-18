"""Generic electricity price config/options flow step for HSEM.

Replaces the former ``energidataservice`` step with a provider-agnostic
prices step that supports:

* Import and export price sensors (required)
* Optional separate forecast sensors (e.g. Amber Electric)
* Optional paired ENTSO-E published-price backup sensors
* Configurable update interval (15, 30, or 60 minutes)
* A minimum export price slider

The naming convention uses ``electricity_price`` rather than a specific
provider name so that users of Energi Data Service, Nordpool, Amber Electric,
or any other price source share the same configuration step.
"""

from __future__ import annotations

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.selector import selector

from custom_components.hsem.utils.config_validator import (
    async_validate_entity_ids,
    merge_errors,
    validate_price,
)
from custom_components.hsem.utils.misc import get_config_value
from custom_components.hsem.utils.price_sources import (
    normalize_price_unit,
    parse_entsoe_price_attributes,
    validate_price_cadence,
)

_ENTSOE_IMPORT_FIELD = "hsem_import_electricity_price_entsoe_sensor"
_ENTSOE_EXPORT_FIELD = "hsem_export_electricity_price_entsoe_sensor"
ENTSOE_SENSOR_FIELDS = (_ENTSOE_IMPORT_FIELD, _ENTSOE_EXPORT_FIELD)


def _validate_entsoe_sensor_pair(
    hass: HomeAssistant,
    user_input: dict,
    entity_errors: dict[str, str],
) -> dict[str, str]:
    """Validate the optional, already-adjusted ENTSO-E sensor pair."""
    import_value = user_input.get(_ENTSOE_IMPORT_FIELD)
    export_value = user_input.get(_ENTSOE_EXPORT_FIELD)
    import_configured = bool(
        import_value.strip() if isinstance(import_value, str) else import_value
    )
    export_configured = bool(
        export_value.strip() if isinstance(export_value, str) else export_value
    )

    if import_configured != export_configured:
        missing_field = (
            _ENTSOE_EXPORT_FIELD if import_configured else _ENTSOE_IMPORT_FIELD
        )
        return {missing_field: "entsoe_sensor_pair_required"}
    if not import_configured:
        return {}
    if _ENTSOE_IMPORT_FIELD in entity_errors or _ENTSOE_EXPORT_FIELD in entity_errors:
        return {}

    import_state = hass.states.get(str(import_value).strip())
    export_state = hass.states.get(str(export_value).strip())
    import_attributes = getattr(import_state, "attributes", None)
    export_attributes = getattr(export_state, "attributes", None)
    import_prices, import_reason = parse_entsoe_price_attributes(import_attributes)
    export_prices, export_reason = parse_entsoe_price_attributes(export_attributes)
    errors: dict[str, str] = {}

    if import_reason is not None:
        errors[_ENTSOE_IMPORT_FIELD] = "entsoe_price_data_invalid"
    if export_reason is not None:
        errors[_ENTSOE_EXPORT_FIELD] = "entsoe_price_data_invalid"

    if (
        import_reason is None
        and export_reason is None
        and tuple(import_prices) != tuple(export_prices)
    ):
        errors.setdefault(_ENTSOE_EXPORT_FIELD, "entsoe_price_timestamps_misaligned")

    try:
        expected_minutes = int(
            str(user_input.get("hsem_electricity_price_update_interval"))
        )
    except ValueError, TypeError:
        expected_minutes = 0

    for field, prices, parse_reason in (
        (_ENTSOE_IMPORT_FIELD, import_prices, import_reason),
        (_ENTSOE_EXPORT_FIELD, export_prices, export_reason),
    ):
        if parse_reason is None and validate_price_cadence(prices, expected_minutes):
            errors.setdefault(field, "entsoe_price_cadence_mismatch")

    for entsoe_field, primary_field, entsoe_state in (
        (
            _ENTSOE_IMPORT_FIELD,
            "hsem_import_electricity_price_sensor",
            import_state,
        ),
        (
            _ENTSOE_EXPORT_FIELD,
            "hsem_export_electricity_price_sensor",
            export_state,
        ),
    ):
        primary_state = hass.states.get(
            str(user_input.get(primary_field) or "").strip()
        )
        primary_unit = normalize_price_unit(getattr(primary_state, "attributes", None))
        entsoe_unit = normalize_price_unit(getattr(entsoe_state, "attributes", None))
        if not primary_unit or not entsoe_unit or primary_unit != entsoe_unit:
            errors.setdefault(entsoe_field, "entsoe_price_unit_mismatch")

    return errors


async def get_prices_step_schema(
    config_entry: ConfigEntry | None,
) -> vol.Schema:  # NOSONAR
    """Return the data schema for the 'prices' step.

    Args:
        config_entry: A Home Assistant ``ConfigEntry`` (may be ``None``
            during initial config flow before the entry is created).

    Returns:
        A ``vol.Schema`` with entity selectors for import/export price
        sensors, optional forecast sensors, a minimum export price
        slider, and an update-interval dropdown.
    """
    entsoe_import_value = get_config_value(config_entry, _ENTSOE_IMPORT_FIELD)
    entsoe_export_value = get_config_value(config_entry, _ENTSOE_EXPORT_FIELD)
    entsoe_import_marker = vol.Optional(_ENTSOE_IMPORT_FIELD)
    entsoe_export_marker = vol.Optional(_ENTSOE_EXPORT_FIELD)
    if isinstance(entsoe_import_value, str) and entsoe_import_value.strip():
        entsoe_import_marker = vol.Optional(
            _ENTSOE_IMPORT_FIELD,
            description={"suggested_value": entsoe_import_value.strip()},
        )
    if isinstance(entsoe_export_value, str) and entsoe_export_value.strip():
        entsoe_export_marker = vol.Optional(
            _ENTSOE_EXPORT_FIELD,
            description={"suggested_value": entsoe_export_value.strip()},
        )

    return vol.Schema(
        {
            vol.Required(
                "hsem_import_electricity_price_sensor",
                default=get_config_value(
                    config_entry, "hsem_import_electricity_price_sensor"
                ),
            ): selector({"entity": {"domain": "sensor"}}),
            vol.Required(
                "hsem_export_electricity_price_sensor",
                default=get_config_value(
                    config_entry, "hsem_export_electricity_price_sensor"
                ),
            ): selector({"entity": {"domain": "sensor"}}),
            vol.Optional(
                "hsem_import_electricity_price_forecast_sensor",
                default=get_config_value(
                    config_entry,
                    "hsem_import_electricity_price_forecast_sensor",
                ),
            ): selector({"entity": {"domain": "sensor"}}),
            vol.Optional(
                "hsem_export_electricity_price_forecast_sensor",
                default=get_config_value(
                    config_entry,
                    "hsem_export_electricity_price_forecast_sensor",
                ),
            ): selector({"entity": {"domain": "sensor"}}),
            entsoe_import_marker: selector(
                {"entity": {"domain": ["sensor", "entsoe"]}}
            ),
            entsoe_export_marker: selector(
                {"entity": {"domain": ["sensor", "entsoe"]}}
            ),
            vol.Required(
                "hsem_price_forecast_valuation_enabled",
                default=get_config_value(
                    config_entry, "hsem_price_forecast_valuation_enabled"
                ),
            ): selector({"boolean": {}}),
            vol.Optional(
                "hsem_price_forecast_valuation_sensor",
                default=get_config_value(
                    config_entry, "hsem_price_forecast_valuation_sensor"
                ),
            ): selector({"entity": {"domain": "sensor"}}),
            vol.Required(
                "hsem_price_forecast_valuation_margin",
                default=get_config_value(
                    config_entry, "hsem_price_forecast_valuation_margin"
                ),
            ): selector(
                {
                    "number": {
                        "min": 0.00,
                        "max": 2.00,
                        "step": 0.01,
                        "mode": "slider",
                    }
                }
            ),
            vol.Required(
                "hsem_export_electricity_min_price",
                default=get_config_value(
                    config_entry, "hsem_export_electricity_min_price"
                ),
            ): selector(
                {
                    "number": {
                        "min": -2.00,
                        "max": 2.00,
                        "step": 0.01,
                        "mode": "slider",
                    }
                }
            ),
            vol.Required(
                "hsem_electricity_price_update_interval",
                default=str(
                    get_config_value(
                        config_entry, "hsem_electricity_price_update_interval"
                    )
                ),
            ): selector(
                {
                    "select": {
                        "multiple": False,
                        "translation_key": "update_interval_minutes",
                        "mode": "list",
                        "options": [
                            "15",
                            "30",
                            "60",
                        ],
                    }
                }
            ),
        }
    )


async def validate_prices_input(
    hass: HomeAssistant, user_input: dict
) -> dict[str, str]:
    """Validate user input for the 'prices' step.

    Args:
        hass: Home Assistant instance (used for entity existence checks).
        user_input: Raw user-supplied dict from the config/options form.

    Returns:
        A dict of field → error-key; empty dict when validation passes.
    """
    required_entity_fields = [
        "hsem_import_electricity_price_sensor",
        "hsem_export_electricity_price_sensor",
    ]
    if bool(user_input.get("hsem_price_forecast_valuation_enabled")):
        required_entity_fields.append("hsem_price_forecast_valuation_sensor")

    entity_errors = await async_validate_entity_ids(
        hass,
        user_input,
        required_fields=required_entity_fields,
        optional_fields=[_ENTSOE_IMPORT_FIELD, _ENTSOE_EXPORT_FIELD],
    )
    entsoe_errors = _validate_entsoe_sensor_pair(hass, user_input, entity_errors)
    price_errors = validate_price(
        user_input,
        "hsem_export_electricity_min_price",
        min_price=-2.0,
        max_price=2.0,
        allow_negative=True,
    )
    required_errors: dict[str, str] = {}
    for field in (
        "hsem_export_electricity_min_price",
        "hsem_electricity_price_update_interval",
    ):
        if field not in user_input:
            required_errors[field] = "required"
    return merge_errors(entity_errors, entsoe_errors, price_errors, required_errors)
