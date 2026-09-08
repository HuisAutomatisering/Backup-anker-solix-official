"""Shared fakes for entity platform tests (switch/select/number).

Mirrors the lightweight-fake pattern of test_sensor.py / test_base_entity.py:
no real Modbus I/O, no running coordinator poll loop. The fake coordinator
exposes exactly the surface AnkerSolixBaseEntity and the control platforms
read, and records write_register / protection / selection calls so tests can
assert on hardware-write behaviour (value, sign, address, data type).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock


def make_write_result(
    success: bool = True,
    is_transient: bool = False,
    error_reason: str | None = None,
) -> SimpleNamespace:
    """Stand-in for modbus_manager.WriteResult."""
    return SimpleNamespace(
        success=success,
        is_transient=is_transient,
        error_reason=error_reason,
        raw_response=None,
        tx_frame=None,
    )


class FakeCoordinator:
    """Stand-in exposing only what the control platforms read."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.entry = SimpleNamespace(entry_id="test-entry")
        self.device_info = {"model": "Solarbank Max"}
        self.device_name = "Test Device"
        self.data: dict[str, Any] | None = data if data is not None else {}
        self.last_update_success = True
        self._protected: dict[str, Any] = {}
        self._user_selections: dict[str, Any] = {}
        self._addresses: dict[str, int] = {}
        self._unavailable_registers: set[int] = set()
        self._full_config_cache: dict[str, Any] | None = None

        self.device_logger = MagicMock()
        self.modbus_manager = MagicMock()
        self.write_register = AsyncMock(return_value=make_write_result())
        self.modbus_manager.write_register = self.write_register

        self.hass = MagicMock()
        self.hass.data = {}
        self.hass.services.async_call = AsyncMock()

    # --- connectivity / registers ---
    def is_connected(self) -> bool:
        return self.last_update_success

    def is_register_available(self, address: int) -> bool:
        return address not in self._unavailable_registers

    def get_data_point_address(self, entity_key: str) -> int | None:
        return self._addresses.get(entity_key)

    # --- write protection ---
    def get_protected_value(self, entity_key: str) -> tuple[bool, Any]:
        if entity_key in self._protected:
            return True, self._protected[entity_key]
        return False, None

    def set_write_protection(
        self, entity_key: str, value: Any, duration: float
    ) -> None:
        self._protected[entity_key] = value

    # --- user selections ---
    def get_user_selection(self, entity_key: str) -> Any:
        return self._user_selections.get(entity_key)

    def set_user_selection(self, entity_key: str, value: Any) -> None:
        self._user_selections[entity_key] = value

    def clear_user_selection(self, entity_key: str) -> None:
        self._user_selections.pop(entity_key, None)

    # --- coordinator listener (CoordinatorEntity.async_added_to_hass) ---
    def async_add_listener(self, listener, context=None):
        return lambda: None


def attach_entity_hass(entity, entity_id: str) -> None:
    """Give an entity the minimal hass surface used by control platforms.

    - async_create_task schedules the coroutine on the running loop (so
      delayed-revert / soft-warning coroutines actually execute in tests)
    - states.async_entity_ids contains the entity id (the revert coroutines
      bail out otherwise)
    - async_write_ha_state is stubbed (the real one needs a platform)
    """
    entity.hass = MagicMock()
    entity.hass.data = {}
    entity.hass.services.async_call = AsyncMock()
    entity.hass.async_create_task = lambda coro: asyncio.ensure_future(coro)
    entity.hass.states.async_entity_ids.return_value = [entity_id]
    entity.entity_id = entity_id
    entity.async_write_ha_state = MagicMock()
    entity._attr_name = entity_id
