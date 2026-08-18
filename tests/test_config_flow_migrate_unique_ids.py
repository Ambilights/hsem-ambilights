"""Regression tests for HSEM config-entry and entity-registry migrations."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.hsem.config_flow import (
    _CHARGE_RATE_BUCKETS,
    _V3_DEPRECATED_KEYS,
    HSEMConfigFlow,
)

ENTRY_ID = "01JHBRS16N1VQM58YSEB88AC90"


def _make_entry(
    *,
    version: int,
    data: dict | None = None,
    options: dict | None = None,
) -> MagicMock:
    """Create a config-entry mock with concrete mutable mappings."""
    entry = MagicMock()
    entry.version = version
    entry.entry_id = ENTRY_ID
    entry.data = data or {}
    entry.options = options or {}
    return entry


def _registry_entry(
    domain: str,
    unique_id: str,
    entity_id: str,
    *,
    platform: str = "hsem",
) -> SimpleNamespace:
    """Create the registry attributes consumed by the migration."""
    return SimpleNamespace(
        domain=domain,
        platform=platform,
        unique_id=unique_id,
        entity_id=entity_id,
    )


def _current_retired_registry_entries() -> list[SimpleNamespace]:
    """Return one current-format registry row for each of the 20 retired entities."""
    rows: list[SimpleNamespace] = []
    index = 0

    for bucket in _CHARGE_RATE_BUCKETS:
        rows.append(
            _registry_entry(
                "number",
                f"hsem_{ENTRY_ID}_charge_rate_{bucket}",
                f"number.user_renamed_retired_{index}",
            )
        )
        index += 1

    for number in (1, 2, 3):
        schedule = f"batteries_enable_batteries_schedule_{number}"
        for domain, suffix in (
            ("switch", f"{schedule}_switch"),
            ("time", f"{schedule}_start_time"),
            ("time", f"{schedule}_end_time"),
        ):
            rows.append(
                _registry_entry(
                    domain,
                    f"hsem_{ENTRY_ID}_hsem_{suffix}",
                    f"{domain}.user_renamed_retired_{index}",
                )
            )
            index += 1

    for name in ("status", "power", "info", "sessions"):
        rows.append(
            _registry_entry(
                "sensor",
                f"hsem_{ENTRY_ID}_ocpp_charger_{name}_sensor",
                f"sensor.user_renamed_retired_{index}",
            )
        )
        index += 1

    assert len(rows) == 20
    return rows


@pytest.mark.asyncio
async def test_migrate_v2_to_v3_strips_config_and_removes_all_entities() -> None:
    """v2->v3 removes all retired values and registry rows, even renamed rows."""
    preserved_data = {
        "device_name": "Preserved",
        "hsem_ev_planned_load_enabled": True,
        "hsem_ev_target_soc": 82,
    }
    preserved_options = {
        "hsem_ev_second_enabled": True,
        "hsem_ev_second_target_soc": 77,
    }
    entry = _make_entry(
        version=2,
        data={**preserved_data, **dict.fromkeys(_V3_DEPRECATED_KEYS, "old")},
        options={**preserved_options, **dict.fromkeys(_V3_DEPRECATED_KEYS, "old")},
    )
    hass = MagicMock()
    registry = MagicMock()
    retired_rows = _current_retired_registry_entries()
    unrelated = _registry_entry(
        "switch",
        f"hsem_{ENTRY_ID}_hsem_ev_smart_charging_switch",
        "switch.hsem_ev_smart_charging",
    )

    with (
        patch(
            "custom_components.hsem.config_flow.er.async_get",
            return_value=registry,
        ),
        patch(
            "custom_components.hsem.config_flow.er.async_entries_for_config_entry",
            return_value=[*retired_rows, unrelated],
        ),
        patch(
            "custom_components.hsem.config_flow.er.async_migrate_entries",
            new=AsyncMock(),
        ) as migrate_entries,
    ):
        result = await HSEMConfigFlow().async_migrate_entry(hass, entry)

    assert result is True
    migrate_entries.assert_not_awaited()
    update = hass.config_entries.async_update_entry.call_args.kwargs
    assert update["version"] == 3
    assert update["data"] == preserved_data
    assert update["options"] == preserved_options
    assert len(_V3_DEPRECATED_KEYS) == 25

    removed = {call.args[0] for call in registry.async_remove.call_args_list}
    assert removed == {row.entity_id for row in retired_rows}
    assert unrelated.entity_id not in removed


@pytest.mark.asyncio
async def test_registry_cleanup_accepts_all_known_unique_id_shapes() -> None:
    """Legacy, intermediate, and current retired IDs are all removed."""
    schedule = "batteries_enable_batteries_schedule_1"
    rows = [
        _registry_entry(
            "number",
            "hsem_charge_rate_below_0",
            "number.legacy_charge_rate",
        ),
        _registry_entry(
            "switch",
            f"hsem_hsem_{schedule}_switch",
            "switch.legacy_schedule",
        ),
        _registry_entry(
            "time",
            f"hsem_{ENTRY_ID}_{schedule}_start_time",
            "time.intermediate_schedule",
        ),
        _registry_entry(
            "time",
            f"hsem_{ENTRY_ID}_hsem_{schedule}_end_time",
            "time.current_schedule",
        ),
        _registry_entry(
            "sensor",
            "hsem_ocpp_charger_status_sensor",
            "sensor.legacy_ocpp_status",
        ),
        _registry_entry(
            "time",
            f"hsem_{ENTRY_ID}_hsem_{schedule}_end_time",
            "time.foreign_platform",
            platform="other",
        ),
    ]
    entry = _make_entry(version=2)
    hass = MagicMock()
    registry = MagicMock()

    with (
        patch(
            "custom_components.hsem.config_flow.er.async_get",
            return_value=registry,
        ),
        patch(
            "custom_components.hsem.config_flow.er.async_entries_for_config_entry",
            return_value=rows,
        ),
    ):
        await HSEMConfigFlow().async_migrate_entry(hass, entry)

    assert {call.args[0] for call in registry.async_remove.call_args_list} == {
        "number.legacy_charge_rate",
        "switch.legacy_schedule",
        "time.intermediate_schedule",
        "time.current_schedule",
        "sensor.legacy_ocpp_status",
    }


@pytest.mark.asyncio
async def test_migrate_v2_to_v3_is_idempotent_when_targets_are_absent() -> None:
    """The migration succeeds when no deprecated values or entities exist."""
    entry = _make_entry(
        version=2,
        data={"device_name": "HSEM"},
        options={"hsem_ev_smart_charging": True},
    )
    hass = MagicMock()
    registry = MagicMock()

    with (
        patch(
            "custom_components.hsem.config_flow.er.async_get",
            return_value=registry,
        ),
        patch(
            "custom_components.hsem.config_flow.er.async_entries_for_config_entry",
            return_value=[],
        ),
    ):
        result = await HSEMConfigFlow().async_migrate_entry(hass, entry)

    assert result is True
    registry.async_remove.assert_not_called()
    update = hass.config_entries.async_update_entry.call_args.kwargs
    assert update["version"] == 3
    assert update["data"] == {"device_name": "HSEM"}
    assert update["options"] == {"hsem_ev_smart_charging": True}


@pytest.mark.asyncio
async def test_migrate_v1_chains_unique_id_remap_and_v3_cleanup() -> None:
    """A v1 entry reaches v3 in one pass without resurrecting retired values."""
    entry = _make_entry(
        version=1,
        data={
            "hsem_energi_data_service_import": "sensor.old_import",
            "hsem_batteries_enable_batteries_schedule_1": True,
            "hsem_ev_target_soc": 85,
        },
        options={
            "hsem_ocpp_enabled": True,
            "hsem_charge_rate_override_21_to_35": 4500,
            "hsem_ev_smart_charging": True,
        },
    )
    hass = MagicMock()
    registry = MagicMock()

    with (
        patch(
            "custom_components.hsem.config_flow.er.async_get",
            return_value=registry,
        ),
        patch(
            "custom_components.hsem.config_flow.er.async_entries_for_config_entry",
            return_value=[],
        ),
        patch(
            "custom_components.hsem.config_flow.er.async_migrate_entries",
            new=AsyncMock(),
        ) as migrate_entries,
    ):
        result = await HSEMConfigFlow().async_migrate_entry(hass, entry)

    assert result is True
    migrate_entries.assert_awaited_once()
    assert migrate_entries.await_args is not None
    _, migrated_entry_id, update_unique_id = migrate_entries.await_args.args
    assert migrated_entry_id == ENTRY_ID
    old = SimpleNamespace(unique_id="hsem_workingmode_sensor")
    assert update_unique_id(old) == {
        "new_unique_id": f"hsem_{ENTRY_ID}_workingmode_sensor"
    }

    hass.config_entries.async_update_entry.assert_called_once()
    update = hass.config_entries.async_update_entry.call_args.kwargs
    assert update["version"] == 3
    assert update["data"]["hsem_import_electricity_price_sensor"] == "sensor.old_import"
    assert update["data"]["hsem_ev_target_soc"] == 85
    assert "hsem_batteries_enable_batteries_schedule_1" not in update["data"]
    assert update["options"] == {"hsem_ev_smart_charging": True}
