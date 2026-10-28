"""ACL architecture mapping and detection; device access is real unless replaced."""

from unittest.mock import Mock

import pytest


# Verify validated ACL architecture IDs map to the expected architecture generations.
@pytest.mark.parametrize("arch_id, expected", [(2201, "ARCH22"), (3510, "ARCH35")])
def test_known_arch_maps_to_expected_generation(generation_module, arch_id, expected):
    assert (
        generation_module.generation_from_arch(arch_id)
        is generation_module.Generation[expected]
    )


# Verify unknown or invalid architecture identifiers fail instead of guessing a generation.
@pytest.mark.parametrize(
    "arch_id", [-1, None, []], ids=["unknown-architecture", "none", "unhashable"]
)
def test_unknown_arch_is_not_guessed(generation_module, arch_id):
    with pytest.raises(generation_module.HardwareDetectionError) as caught:
        generation_module.generation_from_arch(arch_id)
    assert repr(arch_id) in str(caught.value)


# Verify checkout detection agrees with the real ACL query on visible device zero.
def test_real_npu_is_identified(npu_runtime, generation_module):
    hardware = generation_module.get_hardware()
    arch_id, status = npu_runtime.acl.rt.get_device_info(0, 601)
    assert status == 0
    assert hardware.arch_id == arch_id
    assert hardware.generation is generation_module.generation_from_arch(arch_id)


# Verify successful detection caches the ACL result and uses device zero with attribute 601.
def test_successful_detection_is_cached(npu_runtime, generation_module, monkeypatch):
    read_info = Mock(return_value=(2201, 0))
    monkeypatch.setattr(npu_runtime.acl.rt, "get_device_info", read_info)

    first = generation_module.get_hardware()
    second = generation_module.get_hardware()

    assert first is second
    assert first.arch_id == 2201
    assert first.generation is generation_module.Generation.ARCH22
    read_info.assert_called_once_with(0, 601)


# Verify ACL exceptions and error return codes are not cached and permit a successful retry.
@pytest.mark.parametrize("failure", ["exception", "return-code"])
def test_device_query_failure_can_retry_without_caching(
    npu_runtime, generation_module, monkeypatch, failure
):
    error = RuntimeError("device query failed")
    read_info = Mock(
        side_effect=[error if failure == "exception" else (0, 17), (3510, 0)]
    )
    monkeypatch.setattr(npu_runtime.acl.rt, "get_device_info", read_info)

    with pytest.raises(generation_module.HardwareDetectionError) as caught:
        generation_module.get_hardware()
    if failure == "exception":
        assert caught.value.__cause__ is error
    else:
        assert "device=0, error_code=17" in str(caught.value)
        assert caught.value.__cause__ is None
    hardware = generation_module.get_hardware()
    assert hardware.arch_id == 3510
    assert hardware.generation is generation_module.Generation.ARCH35
    assert generation_module.get_hardware() is hardware
    assert read_info.call_count == 2
    assert all(
        args == (0, 601) and not kwargs for args, kwargs in read_info.call_args_list
    )


# Verify unknown architecture failures are not cached and detection retries the ACL query.
def test_unknown_arch_is_not_cached(npu_runtime, generation_module, monkeypatch):
    read_info = Mock(side_effect=[(-1, 0), (3510, 0)])
    monkeypatch.setattr(npu_runtime.acl.rt, "get_device_info", read_info)

    with pytest.raises(generation_module.HardwareDetectionError):
        generation_module.get_hardware()
    assert (
        generation_module.get_hardware().generation
        is generation_module.Generation.ARCH35
    )
    assert read_info.call_count == 2
