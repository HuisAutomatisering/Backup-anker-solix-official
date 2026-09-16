"""Unit tests for ModbusLocalDeviceSelect (select.py).

Covers capability-mask option filtering, visibility gating (bit and legacy
value modes), current_option resolution (direction-selector auto-fill with
one-shot logbook, write protection, numeric-to-translation mapping), the
async_select_option write path (success/transient/hard failure/exception,
unmapped option, missing/invalid address, write_condition blocking), and
_auto_rewrite_power_on_direction_change sign/gain/skip logic.
"""

from __future__ import annotations

import asyncio

import pytest
from custom_components.anker_solix_official.select import (
    ModbusLocalDeviceSelect,
    _is_select_entity,
)
from homeassistant.exceptions import ServiceValidationError

from ._fakes import FakeCoordinator, attach_entity_hass, make_write_result


@pytest.fixture
def coord() -> FakeCoordinator:
    return FakeCoordinator()


def _make_select(coord, key="work_mode", **config) -> ModbusLocalDeviceSelect:
    cfg = {
        "address": 100,
        "data_type": "UINT16",
        "options": {"1": "mode_a", "2": "mode_b"},
        # See test_switch.py: HA 2026.2+ requires icon to be set before
        # select.__init__ reads self._attr_icon.
        "icon": "mdi:cog",
    }
    cfg.update(config)
    entity = ModbusLocalDeviceSelect(coord, key, cfg)
    attach_entity_hass(entity, f"select.{key}")
    return entity


def _make_direction(coord, key="direction") -> ModbusLocalDeviceSelect:
    return _make_select(
        coord,
        key,
        is_direction_selector=True,
        options={"-1": "charge", "1": "discharge"},
    )


class TestIsSelectEntity:
    def test_matches_control_select(self) -> None:
        assert _is_select_entity(
            "k", {"data_type_category": "control", "display_type": "select"}
        )

    @pytest.mark.parametrize(
        "config",
        [
            {"data_type_category": "control", "display_type": "input"},
            {"data_type_category": "read", "display_type": "select"},
            {},
        ],
    )
    def test_rejects_non_select(self, config) -> None:
        assert not _is_select_entity("k", config)


class TestInit:
    def test_falsy_icon_replaced_by_default(self, coord) -> None:
        assert _make_select(coord, icon="")._attr_icon == "mdi:menu"

    def test_option_maps_built(self, coord) -> None:
        entity = _make_select(coord)
        assert entity._all_translation_keys == ["mode_a", "mode_b"]
        assert entity._options_map == {"mode_a": "1", "mode_b": "2"}


class TestCapabilityFiltering:
    def test_all_options_without_capability_config(self, coord) -> None:
        assert _make_select(coord).options == ["mode_a", "mode_b"]

    def test_all_options_when_mask_not_read_yet(self, coord) -> None:
        coord.data = {}
        entity = _make_select(
            coord, capability_entity="mask", option_capability_bits={"1": 0, "2": 1}
        )
        assert entity.options == ["mode_a", "mode_b"]

    def test_mask_filters_options_by_bit(self, coord) -> None:
        coord.data = {"mask": 0b01}
        entity = _make_select(
            coord, capability_entity="mask", option_capability_bits={"1": 0, "2": 1}
        )
        assert entity.options == ["mode_a"]

    def test_option_without_bit_requirement_always_included(self, coord) -> None:
        coord.data = {"mask": 0b00}
        entity = _make_select(
            coord, capability_entity="mask", option_capability_bits={"2": 1}
        )
        assert entity.options == ["mode_a"]

    def test_non_integer_mask_returns_all(self, coord) -> None:
        coord.data = {"mask": "garbage"}
        entity = _make_select(
            coord, capability_entity="mask", option_capability_bits={"1": 0}
        )
        assert entity.options == ["mode_a", "mode_b"]

    def test_mask_missing_from_data_returns_all(self, coord) -> None:
        coord.data = {"other": 1}
        entity = _make_select(
            coord, capability_entity="mask", option_capability_bits={"1": 0}
        )
        assert entity.options == ["mode_a", "mode_b"]


class TestAvailable:
    def test_unavailable_when_refresh_failed(self, coord) -> None:
        coord.last_update_success = False
        assert _make_select(coord).available is False

    def test_visibility_bit_set(self, coord) -> None:
        coord.data = {"vis": 0b100}
        entity = _make_select(coord, visibility_entity="vis", visibility_bit=2)
        assert entity.available is True

    def test_visibility_bit_unset(self, coord) -> None:
        coord.data = {"vis": 0b011}
        entity = _make_select(coord, visibility_entity="vis", visibility_bit=2)
        assert entity.available is False

    def test_visibility_bit_missing_value(self, coord) -> None:
        coord.data = {"other": 1}
        entity = _make_select(coord, visibility_entity="vis", visibility_bit=2)
        assert entity.available is False

    def test_visibility_bit_invalid_value(self, coord) -> None:
        coord.data = {"vis": "abc"}
        entity = _make_select(coord, visibility_entity="vis", visibility_bit=0)
        assert entity.available is False

    def test_visibility_bit_no_data(self, coord) -> None:
        coord.data = None
        entity = _make_select(coord, visibility_entity="vis", visibility_bit=0)
        assert entity.available is False

    def test_legacy_visibility_value_match(self, coord) -> None:
        coord.data = {"mode": 3}
        entity = _make_select(coord, visibility_entity="mode", visibility_value=3)
        assert entity.available is True

    def test_legacy_visibility_value_mismatch(self, coord) -> None:
        coord.data = {"mode": 1}
        entity = _make_select(coord, visibility_entity="mode", visibility_value=3)
        assert entity.available is False

    def test_legacy_visibility_invalid_value(self, coord) -> None:
        coord.data = {"mode": "x"}
        entity = _make_select(coord, visibility_entity="mode", visibility_value=3)
        assert entity.available is False

    def test_legacy_visibility_no_data(self, coord) -> None:
        coord.data = None
        entity = _make_select(coord, visibility_entity="mode", visibility_value=3)
        assert entity.available is False


class TestCurrentOption:
    def test_none_when_unavailable(self, coord) -> None:
        coord.last_update_success = False
        assert _make_select(coord).current_option is None

    def test_maps_device_value_to_translation_key(self, coord) -> None:
        coord.data = {"work_mode": 2}
        assert _make_select(coord).current_option == "mode_b"

    def test_none_when_key_missing(self, coord) -> None:
        coord.data = {}
        assert _make_select(coord).current_option is None

    def test_none_when_value_unmapped(self, coord) -> None:
        coord.data = {"work_mode": 99}
        assert _make_select(coord).current_option is None

    def test_protected_value_wins(self, coord) -> None:
        coord.data = {"work_mode": 1}
        coord._protected["work_mode"] = "mode_b"
        assert _make_select(coord).current_option == "mode_b"

    async def test_direction_selector_returns_user_selection(
        self, coord
    ) -> None:
        coord._user_selections["direction"] = "discharge"
        assert _make_direction(coord).current_option == "discharge"

    async def test_direction_selector_autofills_charge_once(
        self, coord
    ) -> None:
        entity = _make_direction(coord)
        assert entity.current_option == "charge"
        await asyncio.sleep(0)
        assert coord._user_selections["direction"] == "charge"
        entity.hass.services.async_call.assert_awaited_once()
        assert entity.current_option == "charge"
        await asyncio.sleep(0)
        entity.hass.services.async_call.assert_awaited_once()


class TestSelectOption:
    async def test_writes_mapped_value(self, coord) -> None:
        entity = _make_select(coord)
        await entity.async_select_option("mode_b")
        coord.write_register.assert_awaited_once_with(100, 2, "UINT16")
        assert coord._protected["work_mode"] == "mode_b"
        entity.async_write_ha_state.assert_called()
        assert coord.hass.services.async_call.await_args.args[0] == "logbook"

    async def test_custom_protection_duration(self, coord) -> None:
        entity = _make_select(coord, write_protection_duration=5.0)
        await entity.async_select_option("mode_a")
        assert coord._protected["work_mode"] == "mode_a"

    async def test_unmapped_option_aborts(self, coord) -> None:
        await _make_select(coord).async_select_option("nope")
        coord.write_register.assert_not_awaited()

    async def test_missing_address_aborts(self, coord) -> None:
        entity = _make_select(coord)
        entity._config.pop("address")
        await entity.async_select_option("mode_a")
        coord.write_register.assert_not_awaited()

    async def test_invalid_address_aborts(self, coord) -> None:
        await _make_select(coord, address="abc").async_select_option("mode_a")
        coord.write_register.assert_not_awaited()

    async def test_transient_failure_logs_warning(self, coord) -> None:
        coord.write_register.return_value = make_write_result(
            success=False, is_transient=True, error_reason="timeout"
        )
        await _make_select(coord).async_select_option("mode_a")
        coord.device_logger.warning.assert_called()
        assert "work_mode" not in coord._protected

    async def test_hard_failure_logs_error(self, coord) -> None:
        coord.write_register.return_value = make_write_result(success=False)
        await _make_select(coord).async_select_option("mode_a")
        coord.device_logger.error.assert_called()

    async def test_exception_logs_error(self, coord) -> None:
        coord.write_register.side_effect = OSError("link down")
        await _make_select(coord).async_select_option("mode_a")
        coord.device_logger.error.assert_called()

    async def test_write_condition_blocks_and_persists(self, coord) -> None:
        coord.data = {"mode": 1}
        entity = _make_select(
            coord,
            write_condition={"entity": "mode", "value": 2, "hint": "need_mode_2"},
        )
        with pytest.raises(ServiceValidationError) as exc:
            await entity.async_select_option("mode_a")
        assert exc.value.translation_key == "need_mode_2"
        coord.write_register.assert_not_awaited()
        assert coord._user_selections["work_mode"] == "mode_a"

    async def test_direction_selector_stores_selection_no_register_write(
        self, coord
    ) -> None:
        entity = _make_direction(coord)
        await entity.async_select_option("discharge")
        assert coord._user_selections["direction"] == "discharge"
        entity.async_write_ha_state.assert_called()
        coord.write_register.assert_not_awaited()
        assert coord.hass.services.async_call.await_args.args[0] == "logbook"


class TestAutoRewritePower:
    def _link_power(self, coord, **power_cfg) -> None:
        cfg = {
            "direction_entity": "direction",
            "gain": 1,
            "address": 200,
            "data_type": "INT32",
            "unit": "W",
        }
        cfg.update(power_cfg)
        coord._full_config_cache = {"control_items": {"power_setpoint": cfg}}

    async def test_charge_rewrites_negative(self, coord) -> None:
        self._link_power(coord)
        coord._user_selections["power_setpoint"] = 300
        await _make_direction(coord).async_select_option("charge")
        coord.write_register.assert_awaited_once_with(200, -300, "INT32")
        assert coord._protected["power_setpoint"] == 300

    async def test_discharge_rewrites_positive(self, coord) -> None:
        self._link_power(coord)
        coord._user_selections["power_setpoint"] = -300
        await _make_direction(coord).async_select_option("discharge")
        coord.write_register.assert_awaited_once_with(200, 300, "INT32")

    async def test_gain_applied(self, coord) -> None:
        self._link_power(coord, gain=10)
        coord._user_selections["power_setpoint"] = 30
        await _make_direction(coord).async_select_option("charge")
        coord.write_register.assert_awaited_once_with(200, -300, "INT32")

    async def test_no_config_cache_skips(self, coord) -> None:
        await _make_direction(coord).async_select_option("charge")
        coord.write_register.assert_not_awaited()

    async def test_no_linked_power_entity_skips(self, coord) -> None:
        coord._full_config_cache = {"control_items": {"other": {"gain": 1}}}
        await _make_direction(coord).async_select_option("charge")
        coord.write_register.assert_not_awaited()

    @pytest.mark.parametrize("power_selection", [None, 0])
    async def test_no_power_selection_skips(self, coord, power_selection) -> None:
        self._link_power(coord)
        if power_selection is not None:
            coord._user_selections["power_setpoint"] = power_selection
        await _make_direction(coord).async_select_option("charge")
        coord.write_register.assert_not_awaited()

    async def test_rewrite_failure_logs_warning(self, coord) -> None:
        self._link_power(coord)
        coord._user_selections["power_setpoint"] = 300
        coord.write_register.return_value = make_write_result(
            success=False, error_reason="busy"
        )
        await _make_direction(coord).async_select_option("charge")
        assert "power_setpoint" not in coord._protected

    async def test_rewrite_exception_swallowed(self, coord) -> None:
        self._link_power(coord, address="bad")
        coord._user_selections["power_setpoint"] = 300
        await _make_direction(coord).async_select_option("charge")
        coord.write_register.assert_not_awaited()


class TestExtraStateAttributes:
    def test_attributes(self, coord) -> None:
        entity = _make_select(coord, count=1)
        attrs = entity.extra_state_attributes
        assert attrs["modbus_address"] == 100
        assert attrs["data_type"] == "UINT16"
        assert attrs["register_count"] == 1
        assert attrs["available_options"] == {"1": "mode_a", "2": "mode_b"}
