"""SoC mapping and detection; device access is real unless explicitly replaced."""

from unittest.mock import Mock

import pytest


# Verify representative supported SoC names map to the expected hardware generations.
@pytest.mark.parametrize(
    "soc, expected",
    [("Ascend910B1", "A2"), ("Ascend910_9391", "A3"), ("Ascend950", "A5")],
)
def test_known_soc_maps_to_expected_generation(generation_module, soc, expected):
    assert (
        generation_module.generation_from_soc(soc)
        is generation_module.Generation[expected]
    )


# Verify unknown or invalid SoC identifiers raise an error instead of guessing a generation.
@pytest.mark.parametrize(
    "soc", ["Ascend910B-unknown", None, []], ids=["unknown-model", "none", "unhashable"]
)
def test_unknown_soc_is_not_guessed(generation_module, soc):
    with pytest.raises(generation_module.HardwareDetectionError) as caught:
        generation_module.generation_from_soc(soc)
    assert repr(soc) in str(caught.value)


# Verify checkout hardware detection identifies the current NPU and resolves its generation.
def test_real_npu_is_identified(npu_runtime, generation_module):
    hardware = generation_module.get_hardware()
    assert hardware.soc_name == npu_runtime.torch.npu.get_device_name()
    assert hardware.generation is generation_module.generation_from_soc(
        hardware.soc_name
    )
    assert hardware.soc_name == npu_runtime.soc_name


# Verify successful hardware detection is cached without repeating the device query.
def test_successful_detection_is_cached(npu_runtime, generation_module, monkeypatch):
    read_name = Mock(return_value="Ascend910B1")
    monkeypatch.setattr(npu_runtime.torch.npu, "get_device_name", read_name)

    first = generation_module.get_hardware()
    second = generation_module.get_hardware()

    assert first is second
    assert first.generation is generation_module.Generation.A2
    read_name.assert_called_once_with()


# Verify device query failures preserve their cause and allow a successful cached retry.
def test_device_read_failure_preserves_cause_and_can_retry(
    npu_runtime, generation_module, monkeypatch
):
    error = RuntimeError("device query failed")
    read_name = Mock(side_effect=[error, "Ascend910_9391"])
    monkeypatch.setattr(npu_runtime.torch.npu, "get_device_name", read_name)

    with pytest.raises(generation_module.HardwareDetectionError) as caught:
        generation_module.get_hardware()
    assert caught.value.__cause__ is error
    hardware = generation_module.get_hardware()
    assert hardware.generation is generation_module.Generation.A3
    assert generation_module.get_hardware() is hardware
    assert read_name.call_count == 2


# Verify an unknown SoC failure is not cached and detection can retry successfully.
def test_unknown_soc_is_not_cached(npu_runtime, generation_module, monkeypatch):
    read_name = Mock(side_effect=["unknown-soc", "Ascend950"])
    monkeypatch.setattr(npu_runtime.torch.npu, "get_device_name", read_name)

    with pytest.raises(generation_module.HardwareDetectionError):
        generation_module.get_hardware()
    assert (
        generation_module.get_hardware().generation is generation_module.Generation.A5
    )
    assert read_name.call_count == 2
