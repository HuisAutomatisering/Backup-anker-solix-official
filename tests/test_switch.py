"""Unit tests for AnkerSolixSwitch (switch.py).

Covers state parsing (is_on with option mapping, write protection and the
separate read_entity_key path), availability gating, and the full
_async_set_state write path: success (protection + logbook), transient/hard
write failures, exceptions, missing/invalid address, and write_condition
blocking with UI state persistence.
"""

from __future__ import annotations

import pytest
from custom_components.anker_solix_official.switch import (
    AnkerSolixSwitch,
    _is_switch_entity,
)
from homeassistant.exceptions import ServiceValidationError

from ._fakes import FakeCoordinator, attach_entity_hass, make_write_result


@pytest.fixture
def coord() -> FakeCoordinator:
    return FakeCoordinator()


def _make_switch(coord, key="grid_switch", **config) -> AnkerSolixSwitch:
    cfg = {
        "address": 100,
        "data_type": "UINT16",
        "options": {"1": "enabled", "0": "disabled"},
        # HA 2026.2+ stores _attr_* in __attr_* descriptors: reading
        # self._attr_icon in switch.__init__ crashes if base never set it,
        # so production configs always carry an icon (all shipped YAMLs do).
        "icon": "mdi:toggle-switch-outline",
    }
    cfg.update(config)
    entity = AnkerSolixSwitch(coord, key, cfg)
    attach_entity_hass(entity, f"switch.{key}")
    return entity


class TestIsSwitchEntity:
    def test_matches_control_switch(self) -> None:
        assert _is_switch_entity(
            "k", {"data_type_category": "control", "control_type": "switch"}
        )

    @pytest.mark.parametrize(
        "config",
        [
            {"data_type_category": "control", "control_type": "select"},
            {"data_type_category": "read", "control_type": "switch"},
            {},
        ],
    )
    def test_rejects_non_switch(self, config) -> None:
        assert not _is_switch_entity("k", config)


class TestInit:
    def test_falsy_icon_replaced_by_default(self, coord) -> None:
        entity = _make_switch(coord, icon="")
        assert entity._attr_icon == "mdi:toggle-switch"

    def test_configured_icon_kept(self, coord) -> None:
        entity = _make_switch(coord, icon="mdi:power")
        assert entity._attr_icon == "mdi:power"

    def test_read_entity_key_and_write_address(self, coord) -> None:
        entity = _make_switch(coord, read_entity_key="status_reg", address=200)
        assert entity._read_entity_key == "status_reg"
        assert entity._write_address == 200


class TestIsOn:
    def test_on_when_raw_equals_enabled_value(self, coord) -> None:
        coord.data = {"grid_switch": 1}
        assert _make_switch(coord).is_on is True

    def test_off_when_raw_equals_disabled_value(self, coord) -> None:
        coord.data = {"grid_switch": 0}
        assert _make_switch(coord).is_on is False

    def test_custom_option_mapping(self, coord) -> None:
        coord.data = {"grid_switch": 2}
        entity = _make_switch(coord, options={"2": "enabled", "1": "disabled"})
        assert entity.is_on is True

    def test_none_when_key_missing(self, coord) -> None:
        coord.data = {}
        assert _make_switch(coord).is_on is None

    def test_none_and_warning_on_invalid_raw(self, coord) -> None:
        coord.data = {"grid_switch": "abc"}
        assert _make_switch(coord).is_on is None

    def test_protected_value_overrides_device_data(self, coord) -> None:
        coord.data = {"grid_switch": 0}
        coord._protected["grid_switch"] = 1
        assert _make_switch(coord).is_on is True

    def test_reads_from_read_entity_key(self, coord) -> None:
        coord.data = {"status_reg": 1, "grid_switch": 0}
        entity = _make_switch(coord, read_entity_key="status_reg")
        assert entity.is_on is True

    def test_read_entity_protection_wins(self, coord) -> None:
        coord.data = {"status_reg": 0}
        coord._protected["status_reg"] = 1
        entity = _make_switch(coord, read_entity_key="status_reg")
        assert entity.is_on is True

    def test_read_entity_key_without_data_returns_none(self, coord) -> None:
        coord.data = None
        entity = _make_switch(coord, read_entity_key="status_reg")
        assert entity.is_on is None

    def test_invalid_option_value_falls_back_to_default(self, coord) -> None:
        coord.data = {"grid_switch": 1}
        entity = _make_switch(coord, options={"bad": "enabled", "0": "disabled"})
        assert entity._get_option_value("enabled", 1) == 1
        assert entity.is_on is True


class TestAvailable:
    def test_unavailable_when_refresh_failed(self, coord) -> None:
        coord.last_update_success = False
        assert _make_switch(coord).available is False

    def test_available_by_default(self, coord) -> None:
        assert _make_switch(coord).available is True

    def test_capability_bit_unset_hides(self, coord) -> None:
        coord.data = {"mask": 0b011}
        coord._addresses["mask"] = 999
        entity = _make_switch(coord, capability_entity="mask", capability_bit=2)
        assert entity.available is False

    def test_capability_bit_set_shows(self, coord) -> None:
        coord.data = {"mask": 0b100}
        coord._addresses["mask"] = 999
        entity = _make_switch(coord, capability_entity="mask", capability_bit=2)
        assert entity.available is True

    def test_unreadable_status_register_does_not_hide(self, coord) -> None:
        coord.data = {}
        coord._addresses["status_reg"] = 555
        coord._unavailable_registers.add(555)
        entity = _make_switch(coord, read_entity_key="status_reg")
        assert entity.available is True


class TestSetState:
    async def test_turn_on_writes_enabled_value(self, coord) -> None:
        entity = _make_switch(coord)
        await entity.async_turn_on()
        coord.write_register.assert_awaited_once_with(100, 1, "UINT16")
        assert coord._protected["grid_switch"] == 1
        entity.async_write_ha_state.assert_called_once()
        call = coord.hass.services.async_call.await_args
        assert call.args[0] == "logbook"

    async def test_turn_off_writes_disabled_value(self, coord) -> None:
        entity = _make_switch(coord)
        await entity.async_turn_off()
        coord.write_register.assert_awaited_once_with(100, 0, "UINT16")
        assert coord._protected["grid_switch"] == 0

    async def test_read_entity_key_gets_protection_too(self, coord) -> None:
        entity = _make_switch(coord, read_entity_key="status_reg")
        await entity.async_turn_on()
        assert coord._protected["grid_switch"] == 1
        assert coord._protected["status_reg"] == 1

    async def test_transient_failure_logs_warning_no_protection(
        self, coord
    ) -> None:
        coord.write_register.return_value = make_write_result(
            success=False, is_transient=True, error_reason="timeout"
        )
        entity = _make_switch(coord)
        await entity.async_turn_on()
        assert "grid_switch" not in coord._protected
        coord.device_logger.warning.assert_called()
        coord.device_logger.error.assert_not_called()
        entity.async_write_ha_state.assert_not_called()

    async def test_hard_failure_logs_error(self, coord) -> None:
        coord.write_register.return_value = make_write_result(
            success=False, error_reason="illegal address"
        )
        await _make_switch(coord).async_turn_off()
        coord.device_logger.error.assert_called()

    async def test_exception_logs_error(self, coord) -> None:
        coord.write_register.side_effect = OSError("link down")
        await _make_switch(coord).async_turn_on()
        coord.device_logger.error.assert_called()

    async def test_missing_address_aborts_write(self, coord) -> None:
        entity = _make_switch(coord)
        entity._write_address = None
        await entity.async_turn_on()
        coord.write_register.assert_not_awaited()

    async def test_invalid_address_aborts_write(self, coord) -> None:
        entity = _make_switch(coord, address="not-an-int")
        await entity.async_turn_on()
        coord.write_register.assert_not_awaited()

    async def test_write_condition_blocks_and_persists_ui_value(
        self, coord
    ) -> None:
        coord.data = {"mode": 1}
        entity = _make_switch(
            coord,
            write_condition={"entity": "mode", "value": 2, "hint": "need_mode_2"},
        )
        with pytest.raises(ServiceValidationError) as exc:
            await entity.async_turn_on()
        assert exc.value.translation_key == "need_mode_2"
        coord.write_register.assert_not_awaited()
        assert coord._user_selections["grid_switch"] == "enabled"

    async def test_write_condition_passing_allows_write(self, coord) -> None:
        coord.data = {"mode": 2}
        entity = _make_switch(
            coord, write_condition={"entity": "mode", "value": 2}
        )
        await entity.async_turn_on()
        coord.write_register.assert_awaited_once()
