"""Unit tests for ModbusLocalDeviceNumber (number.py).

Covers the highest-risk control logic in the integration: value parsing
(native_value with user selection / never_read / read_once / protection /
direction / read_mode branches), dynamic native_max_value, SOC constraint
validation, value_constraints (forbidden/warning ranges), and the full
async_set_native_value write path — direction sign, gain scaling, capacity
rejection, failure revert (issue #83 UI revert), mutual exclusion, and
dispatcher wiring via async_added_to_hass.
"""

from __future__ import annotations

import asyncio

import pytest
from custom_components.anker_solix_official.base_entity import AnkerSolixBaseEntity
from custom_components.anker_solix_official.number import (
    SIGNAL_MUTUAL_EXCLUSION_UPDATE,
    ModbusLocalDeviceNumber,
    _is_number_entity,
)
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers.dispatcher import async_dispatcher_send

from ._fakes import FakeCoordinator, attach_entity_hass, make_write_result

REVERT_DELAY = 0.1  # const.WRITE_CONDITION_REVERT_DELAY


@pytest.fixture
def coord() -> FakeCoordinator:
    return FakeCoordinator()


def _make_number(coord, key="power", **config) -> ModbusLocalDeviceNumber:
    cfg = {
        "address": 100,
        "data_type": "INT32",
        # HA 2026.2+: __init__ reads self._attr_icon and logs
        # self._attr_native_unit_of_measurement unconditionally; production
        # configs always carry icon and unit (all shipped YAMLs do).
        "icon": "mdi:flash",
        "unit": "W",
    }
    cfg.update(config)
    entity = ModbusLocalDeviceNumber(coord, key, cfg)
    attach_entity_hass(entity, f"number.{key}")
    return entity


async def _drain_revert() -> None:
    await asyncio.sleep(REVERT_DELAY + 0.05)


class TestIsNumberEntity:
    def test_matches_control_input(self) -> None:
        assert _is_number_entity(
            "k", {"data_type_category": "control", "display_type": "input"}
        )

    @pytest.mark.parametrize(
        "config",
        [
            {"data_type_category": "control", "display_type": "select"},
            {"data_type_category": "read", "display_type": "input"},
            {},
        ],
    )
    def test_rejects_non_number(self, config) -> None:
        assert not _is_number_entity("k", config)


class TestInit:
    def test_defaults(self, coord) -> None:
        entity = _make_number(coord)
        assert entity._attr_native_min_value == 0
        assert entity._config_max_value == 100
        assert entity._attr_native_step == 1
        assert entity._attr_mode.value == "box"

    def test_falsy_icon_replaced_by_default(self, coord) -> None:
        assert _make_number(coord, icon="")._attr_icon == "mdi:counter"

    def test_range_step_unit_from_config(self, coord) -> None:
        entity = _make_number(
            coord, min_value=10, max_value=800, step=5, unit="W"
        )
        assert entity._attr_native_min_value == 10
        assert entity._config_max_value == 800
        assert entity._attr_native_step == 5
        assert entity._attr_native_unit_of_measurement == "W"


class TestNativeMaxValue:
    def test_static_max_without_dynamic_entities(self, coord) -> None:
        assert _make_number(coord, max_value=500).native_max_value == 500

    def test_dynamic_config_without_direction_entity(self, coord) -> None:
        entity = _make_number(coord, max_value=500, max_charge_power_entity="mc")
        assert entity.native_max_value == 500

    def test_charge_direction_uses_device_max(self, coord) -> None:
        coord.data = {"mc": -800}
        coord._user_selections["dir"] = "charge"
        entity = _make_number(
            coord,
            max_value=500,
            min_value=100,
            direction_entity="dir",
            max_charge_power_entity="mc",
        )
        assert entity.native_max_value == 800

    def test_discharge_direction_uses_device_max(self, coord) -> None:
        coord.data = {"md": 600}
        coord._user_selections["dir"] = "discharge"
        entity = _make_number(
            coord,
            max_value=500,
            direction_entity="dir",
            max_discharge_power_entity="md",
        )
        assert entity.native_max_value == 600

    def test_device_max_clamped_to_min_value(self, coord) -> None:
        coord.data = {"mc": 50}
        coord._user_selections["dir"] = "charge"
        entity = _make_number(
            coord,
            min_value=100,
            direction_entity="dir",
            max_charge_power_entity="mc",
        )
        assert entity.native_max_value == 100

    @pytest.mark.parametrize("raw", [None, 0, "abc"])
    def test_invalid_device_max_falls_back(self, coord, raw) -> None:
        coord.data = {"mc": raw} if raw is not None else {}
        coord._user_selections["dir"] = "charge"
        entity = _make_number(
            coord,
            max_value=500,
            direction_entity="dir",
            max_charge_power_entity="mc",
        )
        assert entity.native_max_value == 500

    def test_no_direction_selected_falls_back(self, coord) -> None:
        coord.data = {"mc": 800}
        entity = _make_number(
            coord,
            max_value=500,
            direction_entity="dir",
            max_charge_power_entity="mc",
        )
        assert entity.native_max_value == 500


class TestAvailable:
    def test_unavailable_when_refresh_failed(self, coord) -> None:
        coord.last_update_success = False
        assert _make_number(coord).available is False

    def test_capability_bit_unset_hides(self, coord) -> None:
        coord.data = {"mask": 0b01}
        coord._addresses["mask"] = 999
        entity = _make_number(coord, capability_entity="mask", capability_bit=1)
        assert entity.available is False

    def test_visibility_bit_gates(self, coord) -> None:
        entity = _make_number(coord, visibility_entity="vis", visibility_bit=0)
        coord.data = {"vis": 0b1}
        assert entity.available is True
        coord.data = {"vis": 0b0}
        assert entity.available is False
        coord.data = {"vis": "x"}
        assert entity.available is False
        coord.data = {}
        assert entity.available is False

    def test_legacy_visibility_value_gates(self, coord) -> None:
        entity = _make_number(coord, visibility_entity="mode", visibility_value=2)
        coord.data = {"mode": 2}
        assert entity.available is True
        coord.data = {"mode": 1}
        assert entity.available is False
        coord.data = {"mode": "x"}
        assert entity.available is False
        coord.data = None
        assert entity.available is False


class TestNativeValue:
    def test_none_when_unavailable(self, coord) -> None:
        coord.last_update_success = False
        assert _make_number(coord).native_value is None

    def test_none_when_write_condition_fails(self, coord) -> None:
        coord.data = {"power": 100, "mode": 1}
        entity = _make_number(
            coord, write_condition={"entity": "mode", "value": 2}
        )
        assert entity.native_value is None

    def test_device_value_as_int_when_gain_1(self, coord) -> None:
        coord.data = {"power": 300}
        assert _make_number(coord).native_value == 300

    def test_device_value_as_float_when_gain_not_1(self, coord) -> None:
        coord.data = {"power": 1.5}
        assert _make_number(coord, gain=10).native_value == 1.5

    def test_none_when_key_missing(self, coord) -> None:
        coord.data = {}
        assert _make_number(coord).native_value is None

    def test_none_when_data_empty(self, coord) -> None:
        coord.data = None
        assert _make_number(coord).native_value is None

    def test_non_numeric_value_returns_none(self, coord) -> None:
        coord.data = {"power": "abc"}
        assert _make_number(coord).native_value is None

    def test_user_selection_wins_int(self, coord) -> None:
        coord.data = {"power": 100}
        coord._user_selections["power"] = 250
        assert _make_number(coord).native_value == 250

    def test_user_selection_wins_float_with_gain(self, coord) -> None:
        coord._user_selections["power"] = 2.5
        assert _make_number(coord, gain=10).native_value == 2.5

    def test_never_read_returns_default(self, coord) -> None:
        coord.data = {"power": 999}
        entity = _make_number(coord, never_read_device=True, default_value=50)
        assert entity.native_value == 50

    def test_never_read_default_float_with_gain(self, coord) -> None:
        entity = _make_number(
            coord, never_read_device=True, default_value=1.5, gain=10
        )
        assert entity.native_value == 1.5

    def test_read_once_caches_initial_device_value(self, coord) -> None:
        coord.data = {"power": -300}
        entity = _make_number(coord, read_once=True)
        assert entity.native_value == 300
        coord.data = {"power": 999}
        assert entity.native_value == 300

    def test_read_once_last_known_value_wins(self, coord) -> None:
        coord.data = {"power": 300}
        entity = _make_number(coord, read_once=True)
        entity._last_known_value = 42
        assert entity.native_value == 42

    def test_read_once_missing_device_value(self, coord) -> None:
        coord.data = {}
        assert _make_number(coord, read_once=True).native_value is None

    def test_read_once_non_numeric_device_value(self, coord) -> None:
        coord.data = {"power": "abc"}
        assert _make_number(coord, read_once=True).native_value is None

    def test_protected_value_returned_as_int(self, coord) -> None:
        coord.data = {"power": 100}
        coord._protected["power"] = 250
        assert _make_number(coord).native_value == 250

    def test_protected_value_float_with_gain(self, coord) -> None:
        coord._protected["power"] = 2.5
        assert _make_number(coord, gain=10).native_value == 2.5

    def test_invalid_protected_value_returns_none(self, coord) -> None:
        coord._protected["power"] = "abc"
        assert _make_number(coord).native_value is None

    def test_direction_entity_shows_absolute_value(self, coord) -> None:
        coord.data = {"power": -500}
        entity = _make_number(coord, direction_entity="dir")
        assert entity.native_value == 500

    def test_read_mode_positive_only(self, coord) -> None:
        entity = _make_number(coord, read_mode="positive_only")
        coord.data = {"power": 300}
        assert entity.native_value == 300
        coord.data = {"power": -300}
        assert entity.native_value == 0

    def test_read_mode_negative_only(self, coord) -> None:
        entity = _make_number(coord, read_mode="negative_only")
        coord.data = {"power": -300}
        assert entity.native_value == 300
        coord.data = {"power": 300}
        assert entity.native_value == 0


class TestGetSocEntityValue:
    def test_user_selection_first(self, coord) -> None:
        coord._user_selections["backup_soc"] = 30
        coord.data = {"backup_soc": 10}
        entity = _make_number(coord)
        assert entity._get_soc_entity_value("backup_soc") == 30.0

    def test_falls_back_to_device_data(self, coord) -> None:
        coord.data = {"backup_soc": 10}
        assert _make_number(coord)._get_soc_entity_value("backup_soc") == 10.0

    def test_invalid_data_returns_none(self, coord) -> None:
        coord.data = {"backup_soc": "abc"}
        assert _make_number(coord)._get_soc_entity_value("backup_soc") is None

    def test_missing_returns_none(self, coord) -> None:
        coord.data = {}
        assert _make_number(coord)._get_soc_entity_value("backup_soc") is None


class TestSocConstraints:
    def _entity(self, coord, **validation) -> ModbusLocalDeviceNumber:
        return _make_number(coord, soc_validation=validation)

    def test_greater_than_violation_raises(self, coord) -> None:
        coord.data = {"backup_soc": 20}
        entity = self._entity(coord, greater_than="backup_soc")
        with pytest.raises(ServiceValidationError) as exc:
            entity._validate_soc_constraints(10, entity._config["soc_validation"])
        assert exc.value.translation_key == "soc_must_be_greater_than"

    def test_greater_than_list_form(self, coord) -> None:
        coord.data = {"a": 5, "b": 15}
        entity = self._entity(coord, greater_than=["a", "b"])
        with pytest.raises(ServiceValidationError):
            entity._validate_soc_constraints(10, entity._config["soc_validation"])

    def test_greater_than_satisfied(self, coord) -> None:
        coord.data = {"backup_soc": 5}
        entity = self._entity(coord, greater_than="backup_soc")
        entity._validate_soc_constraints(10, entity._config["soc_validation"])

    def test_greater_than_or_equal_violation(self, coord) -> None:
        coord.data = {"backup_soc": 20}
        entity = self._entity(coord, greater_than_or_equal="backup_soc")
        with pytest.raises(ServiceValidationError) as exc:
            entity._validate_soc_constraints(10, entity._config["soc_validation"])
        assert exc.value.translation_key == "soc_must_be_greater_than_or_equal"

    def test_less_than_violation(self, coord) -> None:
        coord.data = {"limit": 20}
        entity = self._entity(coord, less_than="limit")
        with pytest.raises(ServiceValidationError) as exc:
            entity._validate_soc_constraints(30, entity._config["soc_validation"])
        assert exc.value.translation_key == "soc_must_be_less_than"

    def test_less_than_or_equal_violation(self, coord) -> None:
        coord.data = {"limit": 20}
        entity = self._entity(coord, less_than_or_equal="limit")
        with pytest.raises(ServiceValidationError) as exc:
            entity._validate_soc_constraints(30, entity._config["soc_validation"])
        assert exc.value.translation_key == "soc_must_be_less_than_or_equal"

    def test_missing_target_skips_check(self, coord) -> None:
        coord.data = {}
        entity = self._entity(coord, greater_than="backup_soc")
        entity._validate_soc_constraints(10, entity._config["soc_validation"])

    def test_condition_entity_match_enforces(self, coord) -> None:
        coord.data = {"mode": 1, "backup_soc": 20}
        entity = self._entity(
            coord, condition_entity="mode", condition_value=1, greater_than="backup_soc"
        )
        with pytest.raises(ServiceValidationError):
            entity._validate_soc_constraints(10, entity._config["soc_validation"])

    def test_condition_entity_mismatch_skips(self, coord) -> None:
        coord.data = {"mode": 2, "backup_soc": 20}
        entity = self._entity(
            coord, condition_entity="mode", condition_value=1, greater_than="backup_soc"
        )
        entity._validate_soc_constraints(10, entity._config["soc_validation"])

    def test_condition_entity_missing_skips(self, coord) -> None:
        coord.data = {"backup_soc": 20}
        entity = self._entity(
            coord, condition_entity="mode", condition_value=1, greater_than="backup_soc"
        )
        entity._validate_soc_constraints(10, entity._config["soc_validation"])

    def test_condition_entity_invalid_skips(self, coord) -> None:
        coord.data = {"mode": "abc", "backup_soc": 20}
        entity = self._entity(
            coord, condition_entity="mode", condition_value=1, greater_than="backup_soc"
        )
        entity._validate_soc_constraints(10, entity._config["soc_validation"])

    def test_condition_without_data_skips(self, coord) -> None:
        coord.data = None
        entity = self._entity(
            coord, condition_entity="mode", condition_value=1, greater_than="backup_soc"
        )
        entity._validate_soc_constraints(10, entity._config["soc_validation"])


class TestValueConstraints:
    def test_no_constraints_noop(self, coord) -> None:
        _make_number(coord)._validate_value_constraints(50)

    def test_forbidden_range_hit_raises(self, coord) -> None:
        entity = _make_number(
            coord,
            max_value=1000,
            value_constraints={
                "rules": [
                    {"type": "forbidden_range", "min": 10, "max": 20, "error_key": "forbidden_zone"}
                ]
            },
        )
        with pytest.raises(ServiceValidationError) as exc:
            entity._validate_value_constraints(15)
        assert exc.value.translation_key == "forbidden_zone"
        assert exc.value.translation_placeholders["allowed_min"] == "21"

    def test_forbidden_range_miss_passes(self, coord) -> None:
        entity = _make_number(
            coord,
            value_constraints={
                "rules": [{"type": "forbidden_range", "min": 10, "max": 20}]
            },
        )
        entity._validate_value_constraints(25)

    def test_forbidden_range_incomplete_rule_ignored(self, coord) -> None:
        entity = _make_number(
            coord,
            value_constraints={"rules": [{"type": "forbidden_range", "min": 10}]},
        )
        entity._validate_value_constraints(15)

    def test_unknown_rule_ignored(self, coord) -> None:
        entity = _make_number(
            coord, value_constraints={"rules": [{"type": "mystery"}]}
        )
        entity._validate_value_constraints(15)


class TestSetValue:
    async def test_writes_value(self, coord) -> None:
        entity = _make_number(coord)
        await entity.async_set_native_value(300)
        coord.write_register.assert_awaited_once_with(100, 300, "INT32")

    async def test_success_sets_protection_and_clears_selection(
        self, coord
    ) -> None:
        coord._user_selections["power"] = 999
        entity = _make_number(coord)
        await entity.async_set_native_value(300)
        assert coord._protected["power"] == 300
        assert "power" not in coord._user_selections
        assert entity._last_known_value == 300
        entity.async_write_ha_state.assert_called()
        assert coord.hass.services.async_call.await_args.args[0] == "logbook"

    async def test_never_read_stores_permanent_selection(self, coord) -> None:
        entity = _make_number(coord, never_read_device=True)
        await entity.async_set_native_value(300)
        assert coord._user_selections["power"] == 300
        assert "power" not in coord._protected

    async def test_gain_scales_write_value(self, coord) -> None:
        entity = _make_number(coord, gain=10)
        await entity.async_set_native_value(1.5)
        coord.write_register.assert_awaited_once_with(100, 15, "INT32")

    async def test_write_multiplier_applied(self, coord) -> None:
        entity = _make_number(coord, write_multiplier=-1)
        await entity.async_set_native_value(300)
        coord.write_register.assert_awaited_once_with(100, -300, "INT32")

    async def test_charge_direction_writes_negative(self, coord) -> None:
        coord._user_selections["dir"] = "charge"
        entity = _make_number(coord, direction_entity="dir")
        await entity.async_set_native_value(300)
        coord.write_register.assert_awaited_once_with(100, -300, "INT32")

    async def test_discharge_direction_writes_positive(self, coord) -> None:
        coord._user_selections["dir"] = "discharge"
        entity = _make_number(coord, direction_entity="dir")
        await entity.async_set_native_value(-300)
        coord.write_register.assert_awaited_once_with(100, 300, "INT32")

    async def test_direction_not_selected_rejects_write(self, coord) -> None:
        entity = _make_number(coord, direction_entity="dir")
        await entity.async_set_native_value(300)
        coord.write_register.assert_not_awaited()
        assert entity.hass.services.async_call.await_args.args[0] == "logbook"

    async def test_linked_entity_zeroed_and_signalled(self, coord) -> None:
        entity = _make_number(coord, linked_entity="other_power")
        await entity.async_set_native_value(300)
        assert coord._protected["other_power"] == 0

    async def test_missing_address_aborts(self, coord) -> None:
        entity = _make_number(coord)
        entity._config.pop("address")
        await entity.async_set_native_value(300)
        coord.write_register.assert_not_awaited()

    async def test_invalid_address_aborts(self, coord) -> None:
        entity = _make_number(coord, address="abc")
        await entity.async_set_native_value(300)
        coord.write_register.assert_not_awaited()

    async def test_write_condition_blocks_and_reverts(self, coord) -> None:
        coord.data = {"power": 100, "mode": 1}
        entity = _make_number(
            coord,
            write_condition={"entity": "mode", "value": 2, "hint": "need_mode_2"},
        )
        with pytest.raises(ServiceValidationError) as exc:
            await entity.async_set_native_value(300)
        assert exc.value.translation_key == "need_mode_2"
        coord.write_register.assert_not_awaited()
        await _drain_revert()
        assert "power" not in coord._user_selections

    async def test_forbidden_constraint_blocks_write(self, coord) -> None:
        entity = _make_number(
            coord,
            value_constraints={
                "rules": [{"type": "forbidden_range", "min": 10, "max": 20}]
            },
        )
        with pytest.raises(ServiceValidationError):
            await entity.async_set_native_value(15)
        coord.write_register.assert_not_awaited()

    async def test_soc_violation_reverts_and_raises(self, coord) -> None:
        coord.data = {"backup_soc": 20}
        entity = _make_number(coord, soc_validation={"greater_than": "backup_soc"})
        with pytest.raises(ServiceValidationError) as exc:
            await entity.async_set_native_value(10)
        assert exc.value.translation_key == "soc_must_be_greater_than"
        coord.write_register.assert_not_awaited()
        await _drain_revert()
        assert "power" not in coord._user_selections

    async def test_charge_capacity_too_low_rejects(self, coord) -> None:
        coord._user_selections["dir"] = "charge"
        coord.data = {"mc": 50}
        entity = _make_number(
            coord,
            min_value=100,
            direction_entity="dir",
            max_charge_power_entity="mc",
        )
        with pytest.raises(ServiceValidationError) as exc:
            await entity.async_set_native_value(200)
        assert exc.value.translation_key == "charge_power_too_low"
        coord.write_register.assert_not_awaited()

    async def test_discharge_capacity_too_low_rejects(self, coord) -> None:
        coord._user_selections["dir"] = "discharge"
        coord.data = {"md": 50}
        entity = _make_number(
            coord,
            min_value=100,
            direction_entity="dir",
            max_discharge_power_entity="md",
        )
        with pytest.raises(ServiceValidationError) as exc:
            await entity.async_set_native_value(200)
        assert exc.value.translation_key == "discharge_power_too_low"

    async def test_capacity_sufficient_allows_write(self, coord) -> None:
        coord._user_selections["dir"] = "charge"
        coord.data = {"mc": 800}
        entity = _make_number(
            coord,
            min_value=100,
            direction_entity="dir",
            max_charge_power_entity="mc",
        )
        await entity.async_set_native_value(200)
        coord.write_register.assert_awaited_once_with(100, -200, "INT32")

    async def test_capacity_raw_missing_allows_write(self, coord) -> None:
        coord._user_selections["dir"] = "charge"
        coord.data = {"other": 1}
        entity = _make_number(
            coord,
            min_value=100,
            direction_entity="dir",
            max_charge_power_entity="mc",
        )
        await entity.async_set_native_value(200)
        coord.write_register.assert_awaited_once()

    async def test_transient_failure_reverts(self, coord) -> None:
        coord.write_register.return_value = make_write_result(
            success=False, is_transient=True, error_reason="timeout"
        )
        entity = _make_number(coord)
        await entity.async_set_native_value(300)
        coord.device_logger.warning.assert_called()
        assert "power" not in coord._protected
        await _drain_revert()
        assert entity._last_known_value is None
        assert entity.async_write_ha_state.call_count >= 2

    async def test_hard_failure_reverts(self, coord) -> None:
        coord.write_register.return_value = make_write_result(success=False)
        entity = _make_number(coord)
        await entity.async_set_native_value(300)
        coord.device_logger.error.assert_called()
        await _drain_revert()

    async def test_exception_reverts(self, coord) -> None:
        coord.write_register.side_effect = OSError("link down")
        entity = _make_number(coord)
        await entity.async_set_native_value(300)
        coord.device_logger.error.assert_called()
        await _drain_revert()

    async def test_never_read_failure_restores_previous_selection(
        self, coord
    ) -> None:
        coord._user_selections["power"] = 250
        coord.write_register.return_value = make_write_result(success=False)
        entity = _make_number(coord, never_read_device=True)
        await entity.async_set_native_value(300)
        await _drain_revert()
        assert coord._user_selections["power"] == 250

    async def test_never_read_failure_without_previous_clears(self, coord) -> None:
        coord.write_register.return_value = make_write_result(success=False)
        entity = _make_number(coord, never_read_device=True, default_value=0)
        coord._user_selections["power"] = 0
        await entity.async_set_native_value(300)
        await _drain_revert()
        assert coord._user_selections.get("power") in (None, 0)


class TestMutualExclusion:
    def test_handler_updates_only_matching_entity(self, coord) -> None:
        entity = _make_number(coord, key="power")
        entity._handle_mutual_exclusion_update("other")
        entity.async_write_ha_state.assert_not_called()
        entity._handle_mutual_exclusion_update("power")
        entity.async_write_ha_state.assert_called_once()

    async def test_async_added_to_hass_wires_dispatcher(self, hass, coord) -> None:
        entity = _make_number(coord, key="power")
        entity.hass = hass
        await entity.async_added_to_hass()
        async_dispatcher_send(hass, SIGNAL_MUTUAL_EXCLUSION_UPDATE, "power")
        entity.async_write_ha_state.assert_called_once()
        async_dispatcher_send(hass, SIGNAL_MUTUAL_EXCLUSION_UPDATE, "other")
        entity.async_write_ha_state.assert_called_once()


class TestCoordinatorUpdate:
    def test_read_once_first_update_reads_value(self, coord) -> None:
        coord.data = {"power": 300}
        entity = _make_number(coord, read_once=True)
        entity._handle_coordinator_update()
        entity.async_write_ha_state.assert_called_once()
        assert entity._has_initial_read

    def test_read_once_later_updates_only_refresh_availability(
        self, coord
    ) -> None:
        coord.data = {"power": 300}
        entity = _make_number(coord, read_once=True)
        entity._has_initial_read = True
        entity._handle_coordinator_update()
        entity.async_write_ha_state.assert_called_once()

    def test_protected_keeps_last_known_value(self, coord) -> None:
        coord.data = {"power": 100}
        coord._protected["power"] = 300
        entity = _make_number(coord)
        entity._last_known_value = 300
        entity._handle_coordinator_update()
        assert entity._last_known_value == 300

    def test_unprotected_clears_last_known_value(self, coord) -> None:
        coord.data = {"power": 100}
        entity = _make_number(coord)
        entity._last_known_value = 300
        entity._handle_coordinator_update()
        assert entity._last_known_value is None


class TestExtraStateAttributes:
    def test_basic_attributes(self, coord) -> None:
        entity = _make_number(coord, count=2)
        attrs = entity.extra_state_attributes
        assert attrs["modbus_address"] == 100
        assert attrs["data_type"] == "INT32"
        assert attrs["register_count"] == 2
        assert attrs["last_set_time"] is None
        assert "gain" not in attrs

    def test_gain_included_when_not_1(self, coord) -> None:
        attrs = _make_number(coord, gain=10).extra_state_attributes
        assert attrs["gain"] == 10

    def test_dynamic_power_limits(self, coord) -> None:
        coord.data = {"mc": -800, "md": 600}
        coord._user_selections["dir"] = "charge"
        entity = _make_number(
            coord,
            min_value=100,
            max_value=500,
            direction_entity="dir",
            max_charge_power_entity="mc",
            max_discharge_power_entity="md",
        )
        attrs = entity.extra_state_attributes
        assert attrs["min_power"] == 100
        assert attrs["max_power"] == 800
        assert attrs["max_charge_power"] == 800
        assert attrs["max_discharge_power"] == 600
        assert attrs["current_direction"] == "charge"

    def test_dynamic_limits_without_data(self, coord) -> None:
        coord.data = None
        entity = _make_number(coord, max_charge_power_entity="mc")
        attrs = entity.extra_state_attributes
        assert attrs["max_charge_power"] == "N/A"
        assert "current_direction" not in attrs

    def test_direction_not_set(self, coord) -> None:
        coord.data = {}
        entity = _make_number(
            coord, direction_entity="dir", max_charge_power_entity="mc"
        )
        assert entity.extra_state_attributes["current_direction"] == "not set"


class TestEvaluateOperator:
    """base_entity._evaluate_operator — shared write_condition comparator."""

    @pytest.mark.parametrize(
        ("value", "operator", "target", "expected"),
        [
            (1.0, "eq", 1, True),
            (1.4, "eq", 1, True),
            (1.6, "eq", 1, False),
            (1.5, "eq", 1.5, True),
            (1.6, "eq", 1.5, False),
            (2.0, "ne", 1, True),
            (1.0, "ne", 1, False),
            (1.6, "ne", 1.5, True),
            (1.5, "ne", 1.5, False),
            (2.0, "gt", 1, True),
            (1.0, "gt", 1, False),
            (2.0, "gte", 2, True),
            (1.0, "gte", 2, False),
            (1.0, "lt", 2, True),
            (2.0, "lt", 2, False),
            (2.0, "lte", 2, True),
            (3.0, "lte", 2, False),
            (1.0, "in", [1, 2], True),
            (3.0, "in", [1, 2], False),
            (1.0, "in", 1, True),
            (3.0, "not_in", [1, 2], True),
            (1.0, "not_in", [1, 2], False),
            (2.0, "eq", "2", True),
            (1.0, "eq", None, True),
            (1.0, "eq", "abc", True),
            (1.0, "weird_op", 1, True),
        ],
    )
    def test_operators(self, value, operator, target, expected) -> None:
        assert (
            AnkerSolixBaseEntity._evaluate_operator(value, operator, target)
            is expected
        )
