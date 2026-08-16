"""Price and Solcast PV population (async + snapshot).

Populates import/export price and Solcast PV estimate fields on
:class:`HourlyRecommendation` slots from HA sensor attributes (async)
or from a pre-collected :class:`StateSnapshot` (snapshot).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN

from custom_components.hsem.models.hourly_recommendation import HourlyRecommendation
from custom_components.hsem.models.sensor_config import SensorConfig
from custom_components.hsem.models.state_snapshot import StateSnapshot
from custom_components.hsem.utils.conversion import convert_to_boolean, convert_to_float
from custom_components.hsem.utils.datetime_utils import (
    normalize_datetime,
    normalize_slot_start,
    utc_key,
)
from custom_components.hsem.utils.logger import HSEM_LOGGER as _LOGGER

from . import _resolve_cached  # noqa: F401

_TOMORROW_ONLY_PRICE_ATTRIBUTES = frozenset({"raw_tomorrow", "prices_tomorrow"})


def _source_attributes_available(state: Any) -> bool:
    """Return whether a HA state can authoritatively publish its attributes."""
    raw_state = getattr(state, "state", None)
    return not (
        isinstance(raw_state, str)
        and raw_state.strip().lower() in {STATE_UNKNOWN, STATE_UNAVAILABLE}
    )


def _tomorrow_attribute_available(attributes: dict[str, Any], attr: str) -> bool:
    """Honor an explicit source withdrawal of tomorrow-only price arrays."""
    if attr not in _TOMORROW_ONLY_PRICE_ATTRIBUTES:
        return True
    if "tomorrow_valid" not in attributes:
        return True
    return convert_to_boolean(attributes["tomorrow_valid"])


async def async_populate_price_and_solcast(
    sensor: Any,  # NOSONAR -- HA internal type; circular import risk
    recommendations: list[HourlyRecommendation],
    cfg: SensorConfig,
) -> None:
    """Populate import/export prices and Solcast PV estimates into recommendation slots.

    Reads attribute arrays from the EDS and Solcast sensors, matches each data
    point to the corresponding :class:`HourlyRecommendation` by datetime, and
    writes the value into the appropriate field.

    Args:
        sensor: The ``HSEMWorkingModeSensor`` instance for HA access and logging.
        recommendations: Mutable list of recommendation slots to update.
        cfg: Current sensor configuration.
    """
    # ---------------------------------------------------------------------------
    # Price interval semantics
    # ---------------------------------------------------------------------------
    # Prices are a *rate* (currency/kWh), not an energy quantity, so they
    # must NOT be summed across slots.  However, the price sensor publishes
    # one value per update interval (15, 30, or 60 min) while HSEM may plan
    # at a finer resolution (e.g. 15-min slots inside a 60-min interval).
    #
    # `price_share` converts between the two resolutions:
    #
    #   price_share = electricity_price_update_interval / recommendation_interval_minutes
    #
    # In `_async_update_hourly_field` (below) each raw price value is divided by
    # `price_share` before writing to the per-slot recommendation object.  This
    # stores the price *scaled to one recommendation slot's share* of the price
    # update interval.
    #
    # The inverse multiply (`rec.import_price * price_share`) is applied later in
    # `coordinator_builder.build_planner_input` to recover the original price rate
    # before passing it to the planner engine.
    #
    # Common configurations and their price_share values:
    #   Price 60 min  / slots 15 min  →  price_share = 4.0
    #   Price 30 min  / slots 15 min  →  price_share = 2.0
    #   Price 15 min  / slots 15 min  →  price_share = 1.0  (no scaling)
    #   Price 60 min  / slots 60 min  →  price_share = 1.0  (no scaling)
    price_share = (
        cfg.electricity_price_update_interval / cfg.recommendation_interval_minutes
    )
    # Solcast forecasts are always given as hourly totals (Wh/h), so the share
    # factor is always relative to 60 minutes regardless of price configuration.
    solcast_share = 60.0 / cfg.recommendation_interval_minutes
    # Source-side window used to floor data-point timestamps before matching:
    # price sources publish at the configured price cadence, Solcast hourly.
    price_source_minutes = cfg.electricity_price_update_interval
    solcast_source_minutes = 60

    # Import price — read from primary sensor (may embed forecast attributes)
    import_matched = await _async_update_hourly_field(
        sensor,
        recommendations,
        cfg.import_electricity_price_sensor,
        "import_price",
        price_share,
        cfg.solcast_pv_forecast_forecast_likelihood,
        price_source_minutes,
    )
    # Import price — gap-fill only from the dedicated forecast sensor.  A
    # prediction must never displace a published price, so slots the primary
    # source already covered are left alone.
    if cfg.import_electricity_price_forecast_sensor:
        import_matched += await _async_update_hourly_field(
            sensor,
            recommendations,
            cfg.import_electricity_price_forecast_sensor,
            "import_price",
            price_share,
            cfg.solcast_pv_forecast_forecast_likelihood,
            price_source_minutes,
            only_if_missing=True,
        )
    if import_matched == 0:
        _LOGGER.warning(
            "No import price data matched from sensor(s) %s — "
            "slots retain 0.0 only as a display fallback; unavailable prices "
            "are non-actionable and automatic storage uses strict Hold. "
            "Check that the sensor is available and its attribute format is supported.",
            cfg.import_electricity_price_sensor,
        )
    # Export price — read from primary sensor
    export_matched = await _async_update_hourly_field(
        sensor,
        recommendations,
        cfg.export_electricity_price_sensor,
        "export_price",
        price_share,
        cfg.solcast_pv_forecast_forecast_likelihood,
        price_source_minutes,
    )
    # Export price — gap-fill only, same contract as the import channel.
    if cfg.export_electricity_price_forecast_sensor:
        export_matched += await _async_update_hourly_field(
            sensor,
            recommendations,
            cfg.export_electricity_price_forecast_sensor,
            "export_price",
            price_share,
            cfg.solcast_pv_forecast_forecast_likelihood,
            price_source_minutes,
            only_if_missing=True,
        )
    if export_matched == 0:
        _LOGGER.warning(
            "No export price data matched from sensor(s) %s — "
            "slots retain 0.0 only as a display fallback; unavailable prices "
            "are non-actionable and automatic storage uses strict Hold. "
            "Check that the sensor is available and its attribute format is supported.",
            cfg.export_electricity_price_sensor,
        )
    # Solcast today
    solcast_today_matched = await _async_update_hourly_field(
        sensor,
        recommendations,
        cfg.solcast_pv_forecast_forecast_today,
        "solcast_pv_estimate_kwh",
        solcast_share,
        cfg.solcast_pv_forecast_forecast_likelihood,
        solcast_source_minutes,
    )
    if solcast_today_matched == 0:
        _LOGGER.debug(
            "No Solcast today data matched from sensor %s — "
            "PV estimates will be 0.0 for today.",
            cfg.solcast_pv_forecast_forecast_today,
        )
    # Solcast tomorrow
    solcast_tomorrow_matched = await _async_update_hourly_field(
        sensor,
        recommendations,
        cfg.solcast_pv_forecast_forecast_tomorrow,
        "solcast_pv_estimate_kwh",
        solcast_share,
        cfg.solcast_pv_forecast_forecast_likelihood,
        solcast_source_minutes,
    )
    if solcast_tomorrow_matched == 0:
        _LOGGER.debug(
            "No Solcast tomorrow data matched from sensor %s — "
            "PV estimates will be 0.0 for tomorrow.",
            cfg.solcast_pv_forecast_forecast_tomorrow,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _availability_attr(field_name: str) -> str | None:
    """Return the availability flag paired with ``field_name``, if any.

    The PV estimate does not follow the ``<field>_available`` convention, so
    the mapping is explicit rather than derived.
    """
    if field_name in {"import_price", "export_price"}:
        return f"{field_name}_available"
    if field_name == "solcast_pv_estimate_kwh":
        return "solcast_pv_estimate_available"
    return None


async def _async_update_hourly_field(
    sensor: Any,  # NOSONAR -- HA internal type; circular import risk
    recommendations: list[HourlyRecommendation],
    sensor_id: str | None,
    field_name: str,
    share: float,
    solcast_likelihood_key: str,
    source_interval_minutes: int,
    only_if_missing: bool = False,
) -> int:
    """Match sensor attribute data to recommendation slots and write one field.

    Args:
        sensor: The ``HSEMWorkingModeSensor`` instance.
        recommendations: Mutable recommendation list.
        sensor_id: Entity ID to read attributes from, or None (no-op).
        field_name: Attribute name on :class:`HourlyRecommendation` to set.
        share: Divisor applied to each raw value (accounts for sub-hourly slots).
        solcast_likelihood_key: Attribute key for Solcast PV estimate field.
        only_if_missing: When True, leave slots whose channel is already
            available untouched.  Used for the dedicated forecast sensors so a
            prediction cannot overwrite a published price.

    Returns:
        Number of data points successfully written to at least one slot.
    """
    if sensor_id is None:
        return 0

    source_window = timedelta(minutes=source_interval_minutes)

    sensor_state = sensor.hass.states.get(sensor_id)
    if not sensor_state:
        _LOGGER.debug(f"Input sensor {sensor_id} was not found for data.")
        return 0
    if not _source_attributes_available(sensor_state):
        _LOGGER.debug(
            "Input sensor %s is unavailable; ignoring stale attributes.", sensor_id
        )
        return 0

    # Each source exposes a different attribute key / time-key / value-key
    data_sources: dict[str, list[dict[str, str]]] = {
        "forecast": [{"k": "hour", "v": "price"}],
        "raw_tomorrow": [
            {"k": "hour", "v": "price"},
            {"k": "start", "v": "value"},  # custom-components/nordpool
        ],
        "raw_today": [
            {"k": "hour", "v": "price"},
            {"k": "start", "v": "value"},  # custom-components/nordpool
        ],
        "prices": [{"k": "start", "v": "price"}],
        "prices_today": [
            {"k": "start", "v": "price"},
            {"k": "time", "v": "price"},
        ],
        "prices_tomorrow": [
            {"k": "start", "v": "price"},
            {"k": "time", "v": "price"},
        ],
        "detailedHourly": [{"k": "period_start", "v": solcast_likelihood_key}],
        "detailedForecast": [{"k": "period_start", "v": solcast_likelihood_key}],
        "data": [{"k": "start_time", "v": "price_per_kwh"}],
        # Amber Electric forecast sensor format: forecasts array on the forecast sensor
        "forecasts": [{"k": "start_time", "v": "per_kwh"}],
    }

    avail_attr = _availability_attr(field_name)
    matched = 0
    for attr, kv_list in data_sources.items():
        if not _tomorrow_attribute_available(sensor_state.attributes, attr):
            continue
        sensor_data = sensor_state.attributes.get(attr) or []
        if not sensor_data:
            continue

        _LOGGER.debug(f"Updating data for {field_name}...")

        for data in sensor_data:
            for kv in kv_list:
                raw_time = data.get(kv["k"])
                if not raw_time:
                    continue

                if isinstance(raw_time, datetime):
                    dt_key = raw_time
                else:
                    dt_key = datetime.fromisoformat(str(raw_time))

                try:
                    # Floor the data point to the start of its enclosing source
                    # interval (e.g. 15 min for quarter-hourly prices, 60 min for
                    # hourly Solcast forecasts).  Flooring to the *hour* here
                    # would collapse sub-hourly price points onto the same key
                    # and overwrite all quarter-hour slots of the hour with the
                    # last hourly value (issue #720).
                    dt_key = normalize_slot_start(dt_key, source_interval_minutes)
                except ValueError, OSError:  # noqa: TRY302
                    # Skip data points with unparseable or non-local timestamps
                    continue

                value = convert_to_float(data.get(kv["v"]))
                if value is None or not math.isfinite(value):
                    continue

                # Scale raw value down to one recommendation-slot's share of the
                # source update interval.
                #
                # For prices:   share = price_share (price interval / slot interval)
                #   Price 60 min / slot 15 min → share=4 → store price/4 per slot
                #   Price 15 min / slot 15 min → share=1 → store price unchanged
                #
                # For Solcast PV:   share = solcast_share (60 / slot interval)
                #   60-min hourly forecast / slot 15 min → share=4 → store Wh/4 per slot
                #   60-min hourly forecast / slot 60 min → share=1 → store Wh unchanged
                #
                # The coordinator's `build_planner_input` applies the inverse multiply
                # (×price_share) before handing prices/PV to the planner engine, so the
                # divide here and the multiply there cancel exactly and the planner always
                # receives the original hourly-equivalent rate or energy quantity.
                value = value / share

                window_start = utc_key(dt_key)
                window_end = window_start + source_window
                for obj in recommendations:
                    # A data point covers every slot whose start falls inside
                    # the source window that begins at dt_key:
                    #   - 15-min prices + 15-min slots → one point per slot
                    #   - hourly prices + 15-min slots → one point fans out
                    #     to all four quarter-hour slots of the hour
                    # Flooring dt_key to the *hour* (old behavior) collapsed
                    # all four quarter-hour prices onto one key (issue #720).
                    obj_start = utc_key(normalize_datetime(obj.start))
                    if window_start <= obj_start < window_end:
                        if (
                            only_if_missing
                            and avail_attr is not None
                            and getattr(obj, avail_attr, False)
                        ):
                            continue
                        setattr(obj, field_name, round(value, 5))
                        if avail_attr is not None:
                            setattr(obj, avail_attr, True)
                        matched += 1

    return matched


# ---------------------------------------------------------------------------
# Snapshot-based population (no HA state lookups)
# ---------------------------------------------------------------------------


def populate_price_and_solcast_from_snapshot(
    recommendations: list[HourlyRecommendation],
    snapshot: StateSnapshot,
    cfg: SensorConfig,
) -> None:
    """Populate prices and Solcast PV estimates using a pre-collected snapshot.

    Synchronous — no HA state lookups needed.  Uses :attr:`StateSnapshot.sensor_attributes`
    which was populated by :func:`~state_collector.async_collect_all_states`.

    Args:
        recommendations: Mutable list of recommendation slots to update.
        snapshot: Pre-collected state snapshot.
        cfg: Current sensor configuration.
    """
    price_share = (
        cfg.electricity_price_update_interval / cfg.recommendation_interval_minutes
    )
    solcast_share = 60.0 / cfg.recommendation_interval_minutes
    price_source_minutes = cfg.electricity_price_update_interval
    solcast_source_minutes = 60

    import_matched = _update_hourly_field_from_attrs(
        recommendations,
        snapshot.sensor_attributes.get(cfg.import_electricity_price_sensor or ""),
        "import_price",
        price_share,
        cfg.solcast_pv_forecast_forecast_likelihood,
        price_source_minutes,
    )
    # Gap-fill only — a prediction must never displace a published price.
    if cfg.import_electricity_price_forecast_sensor:
        import_matched += _update_hourly_field_from_attrs(
            recommendations,
            snapshot.sensor_attributes.get(
                cfg.import_electricity_price_forecast_sensor or ""
            ),
            "import_price",
            price_share,
            cfg.solcast_pv_forecast_forecast_likelihood,
            price_source_minutes,
            only_if_missing=True,
        )
    if import_matched == 0:
        _LOGGER.warning(
            "No import price data matched from sensor(s) %s — "
            "slots retain 0.0 only as a display fallback; unavailable prices "
            "are non-actionable and automatic storage uses strict Hold. "
            "Check that the sensor is available and its attribute format is supported.",
            cfg.import_electricity_price_sensor,
        )
    export_matched = _update_hourly_field_from_attrs(
        recommendations,
        snapshot.sensor_attributes.get(cfg.export_electricity_price_sensor or ""),
        "export_price",
        price_share,
        cfg.solcast_pv_forecast_forecast_likelihood,
        price_source_minutes,
    )
    if cfg.export_electricity_price_forecast_sensor:
        export_matched += _update_hourly_field_from_attrs(
            recommendations,
            snapshot.sensor_attributes.get(
                cfg.export_electricity_price_forecast_sensor or ""
            ),
            "export_price",
            price_share,
            cfg.solcast_pv_forecast_forecast_likelihood,
            price_source_minutes,
            only_if_missing=True,
        )
    if export_matched == 0:
        _LOGGER.warning(
            "No export price data matched from sensor(s) %s — "
            "slots retain 0.0 only as a display fallback; unavailable prices "
            "are non-actionable and automatic storage uses strict Hold. "
            "Check that the sensor is available and its attribute format is supported.",
            cfg.export_electricity_price_sensor,
        )
    solcast_today_matched = _update_hourly_field_from_attrs(
        recommendations,
        snapshot.sensor_attributes.get(cfg.solcast_pv_forecast_forecast_today or ""),
        "solcast_pv_estimate_kwh",
        solcast_share,
        cfg.solcast_pv_forecast_forecast_likelihood,
        solcast_source_minutes,
    )
    if solcast_today_matched == 0:
        _LOGGER.debug(
            "No Solcast today data matched from sensor %s — "
            "PV estimates will be 0.0 for today.",
            cfg.solcast_pv_forecast_forecast_today,
        )
    solcast_tomorrow_matched = _update_hourly_field_from_attrs(
        recommendations,
        snapshot.sensor_attributes.get(cfg.solcast_pv_forecast_forecast_tomorrow or ""),
        "solcast_pv_estimate_kwh",
        solcast_share,
        cfg.solcast_pv_forecast_forecast_likelihood,
        solcast_source_minutes,
    )
    if solcast_tomorrow_matched == 0:
        _LOGGER.debug(
            "No Solcast tomorrow data matched from sensor %s — "
            "PV estimates will be 0.0 for tomorrow.",
            cfg.solcast_pv_forecast_forecast_tomorrow,
        )


def _update_hourly_field_from_attrs(
    recommendations: list[HourlyRecommendation],
    attributes: dict[str, Any] | None,
    field_name: str,
    share: float,
    solcast_likelihood_key: str,
    source_interval_minutes: int,
    only_if_missing: bool = False,
) -> int:
    """Match pre-read sensor attribute data to recommendation slots.

    This is the snapshot-based counterpart of ``_async_update_hourly_field``.
    Instead of calling ``hass.states.get()``, it operates on the
    ``attributes`` dict that was pre-read during snapshot collection.

    Args:
        recommendations: Mutable recommendation list.
        attributes: The ``.attributes`` dict of the sensor, or ``None``.
        field_name: Attribute name on :class:`HourlyRecommendation` to set.
        share: Divisor applied to each raw value.
        solcast_likelihood_key: Attribute key for Solcast PV estimate field.
        only_if_missing: When True, leave slots whose channel is already
            available untouched.  Used for the dedicated forecast sensors so a
            prediction cannot overwrite a published price.

    Returns:
        Number of data points successfully written to at least one slot.
    """
    if attributes is None:
        return 0

    source_window = timedelta(minutes=source_interval_minutes)

    data_sources: dict[str, list[dict[str, str]]] = {
        "forecast": [{"k": "hour", "v": "price"}],
        "raw_tomorrow": [
            {"k": "hour", "v": "price"},
            {"k": "start", "v": "value"},  # custom-components/nordpool
        ],
        "raw_today": [
            {"k": "hour", "v": "price"},
            {"k": "start", "v": "value"},  # custom-components/nordpool
        ],
        "prices": [{"k": "start", "v": "price"}],
        "prices_today": [
            {"k": "start", "v": "price"},
            {"k": "time", "v": "price"},
        ],
        "prices_tomorrow": [
            {"k": "start", "v": "price"},
            {"k": "time", "v": "price"},
        ],
        "detailedHourly": [{"k": "period_start", "v": solcast_likelihood_key}],
        "detailedForecast": [{"k": "period_start", "v": solcast_likelihood_key}],
        "data": [{"k": "start_time", "v": "price_per_kwh"}],
        # Amber Electric forecast sensor format: forecasts array on the forecast sensor
        "forecasts": [{"k": "start_time", "v": "per_kwh"}],
    }

    avail_attr = _availability_attr(field_name)
    matched = 0
    for attr, kv_list in data_sources.items():
        if not _tomorrow_attribute_available(attributes, attr):
            continue
        sensor_data = attributes.get(attr) or []
        if not sensor_data:
            continue

        for data in sensor_data:
            for kv in kv_list:
                raw_time = data.get(kv["k"])
                if not raw_time:
                    continue

                if isinstance(raw_time, datetime):
                    dt_key = raw_time
                else:
                    try:
                        dt_key = datetime.fromisoformat(str(raw_time))
                    except ValueError, TypeError:
                        continue

                try:
                    # Same source-interval flooring as the async path (issue #720).
                    dt_key = normalize_slot_start(dt_key, source_interval_minutes)
                except ValueError, OSError:
                    continue

                value = convert_to_float(data.get(kv["v"]))
                if value is None or not math.isfinite(value):
                    continue

                value = value / share

                window_start = utc_key(dt_key)
                window_end = window_start + source_window
                for obj in recommendations:
                    obj_start = utc_key(normalize_datetime(obj.start))
                    if window_start <= obj_start < window_end:
                        if (
                            only_if_missing
                            and avail_attr is not None
                            and getattr(obj, avail_attr, False)
                        ):
                            continue
                        setattr(obj, field_name, round(value, 5))
                        if avail_attr is not None:
                            setattr(obj, avail_attr, True)
                        matched += 1

    return matched
