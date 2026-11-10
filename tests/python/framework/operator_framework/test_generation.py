"""ACL architecture mapping and detection; device access is real unless replaced."""

from unittest.mock import Mock

import pytest


@pytest.fixture
def soc_query(monkeypatch):
    import torch_npu

    query = Mock()
    monkeypatch.setattr(torch_npu._C, "_npu_get_soc_version", query)
    return query


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


# Verify detection agrees with real ACL or its SoC fallback on visible device zero.
def test_real_npu_is_identified(npu_runtime, generation_module):
    import torch_npu

    hardware = generation_module.get_hardware()
    try:
        arch_id, status = npu_runtime.acl.rt.get_device_info(0, 601)
    except Exception:
        status = -1
    if status != 0:
        soc = torch_npu._C._npu_get_soc_version()
        arch_id = generation_module.SOC_TO_ARCH[soc // 10 * 10]
    assert hardware.arch_id == arch_id
    assert hardware.generation is generation_module.generation_from_arch(arch_id)


# Verify successful detection caches the ACL result and uses device zero with attribute 601.
def test_successful_detection_is_cached(
    npu_runtime, generation_module, monkeypatch, soc_query
):
    read_info = Mock(return_value=(2201, 0))
    monkeypatch.setattr(npu_runtime.acl.rt, "get_device_info", read_info)

    first = generation_module.get_hardware()
    second = generation_module.get_hardware()

    assert first is second
    assert first.arch_id == 2201
    assert first.generation is generation_module.Generation.ARCH22
    read_info.assert_called_once_with(0, 601)
    soc_query.assert_not_called()


# ACL failure is recoverable when the SoC query succeeds; cache that success.
@pytest.mark.parametrize("failure", ["exception", "return-code"])
@pytest.mark.parametrize("soc,arch_id", [(221, 2201), (251, 2201), (261, 3510)])
def test_device_query_failure_uses_soc_fallback(
    npu_runtime, generation_module, monkeypatch, soc_query, failure, soc, arch_id
):
    read_info = Mock(
        side_effect=(
            RuntimeError("device query failed") if failure == "exception" else None
        ),
        return_value=(0, 17),
    )
    monkeypatch.setattr(npu_runtime.acl.rt, "get_device_info", read_info)
    soc_query.return_value = soc

    hardware = generation_module.get_hardware()
    assert hardware.arch_id == arch_id
    assert hardware.generation is generation_module.generation_from_arch(arch_id)
    assert generation_module.get_hardware() is hardware
    read_info.assert_called_once_with(0, 601)
    soc_query.assert_called_once_with()


# Only failure of both queries should raise; that failure must permit a retry.
@pytest.mark.parametrize("failure", ["exception", "return-code"])
def test_device_query_failure_can_retry_without_caching(
    npu_runtime, generation_module, monkeypatch, soc_query, failure
):
    error = RuntimeError("device query failed")
    read_info = Mock(
        side_effect=[error if failure == "exception" else (0, 17), (3510, 0)]
    )
    monkeypatch.setattr(npu_runtime.acl.rt, "get_device_info", read_info)
    soc_error = RuntimeError("SoC query failed")
    soc_query.side_effect = soc_error

    with pytest.raises(
        generation_module.HardwareDetectionError, match="through torch_npu"
    ) as caught:
        generation_module.get_hardware()
    assert caught.value.__cause__ is soc_error
    assert generation_module.get_hardware.cache_info().currsize == 0
    hardware = generation_module.get_hardware()
    assert hardware.arch_id == 3510
    assert hardware.generation is generation_module.Generation.ARCH35
    assert generation_module.get_hardware() is hardware
    assert read_info.call_count == 2
    assert all(
        args == (0, 601) and not kwargs for args, kwargs in read_info.call_args_list
    )
    soc_query.assert_called_once_with()


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
