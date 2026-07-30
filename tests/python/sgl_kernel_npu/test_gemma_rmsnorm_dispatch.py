from unittest.mock import Mock

import pytest
import sgl_kernel_npu.norm.gemma_rmsnorm as gemma_rmsnorm
import torch
import torch_npu
from sgl_kernel_npu.norm.add_rmsnorm_bias import (
    add_gemma_rms_norm as compatibility_add_gemma_rms_norm,
)
from sgl_kernel_npu.norm.add_rmsnorm_bias import (
    gemma_rms_norm as compatibility_gemma_rms_norm,
)
from sgl_kernel_npu.utils.npu_device import (
    NpuDeviceFamily,
    _family_from_soc_version,
    get_npu_device_family,
)


@pytest.mark.parametrize(
    ("soc_version", "expected"),
    [
        (200, NpuDeviceFamily.UNKNOWN),
        (205, NpuDeviceFamily.UNKNOWN),
        (220, NpuDeviceFamily.A2),
        (225, NpuDeviceFamily.A2),
        (250, NpuDeviceFamily.A3),
        (255, NpuDeviceFamily.A3),
        (260, NpuDeviceFamily.ASCEND_950),
        (0, NpuDeviceFamily.UNKNOWN),
        (999, NpuDeviceFamily.UNKNOWN),
    ],
)
def test_family_from_soc_version(soc_version, expected):
    assert _family_from_soc_version(soc_version) is expected


def test_device_family_detection_is_cached(monkeypatch):
    get_soc_version = Mock(return_value=260)
    monkeypatch.setattr(torch_npu.npu, "get_soc_version", get_soc_version)
    get_npu_device_family.cache_clear()

    assert get_npu_device_family() is NpuDeviceFamily.ASCEND_950
    assert get_npu_device_family() is NpuDeviceFamily.ASCEND_950
    get_soc_version.assert_called_once_with()

    get_npu_device_family.cache_clear()


def test_device_family_detection_failure_is_unknown(monkeypatch):
    monkeypatch.setattr(
        torch_npu.npu,
        "get_soc_version",
        Mock(side_effect=RuntimeError("detection failed")),
    )
    get_npu_device_family.cache_clear()

    assert get_npu_device_family() is NpuDeviceFamily.UNKNOWN

    get_npu_device_family.cache_clear()


def test_invalid_soc_version_is_unknown(monkeypatch):
    monkeypatch.setattr(torch_npu.npu, "get_soc_version", Mock(return_value=None))
    get_npu_device_family.cache_clear()

    assert get_npu_device_family() is NpuDeviceFamily.UNKNOWN

    get_npu_device_family.cache_clear()


@pytest.mark.parametrize(
    ("family", "provider"),
    [
        (NpuDeviceFamily.A2, gemma_rmsnorm._native_gemma_rms_norm),
        (NpuDeviceFamily.A3, gemma_rmsnorm._native_gemma_rms_norm),
        (NpuDeviceFamily.ASCEND_950, gemma_rmsnorm._triton_gemma_rms_norm),
        (NpuDeviceFamily.UNKNOWN, gemma_rmsnorm._fallback_gemma_rms_norm),
    ],
)
def test_gemma_provider_table(family, provider):
    assert gemma_rmsnorm._GEMMA_RMS_NORM_PROVIDERS[family] is provider


@pytest.mark.parametrize(
    ("family", "provider"),
    [
        (NpuDeviceFamily.A2, gemma_rmsnorm._native_add_gemma_rms_norm),
        (NpuDeviceFamily.A3, gemma_rmsnorm._native_add_gemma_rms_norm),
        (
            NpuDeviceFamily.ASCEND_950,
            gemma_rmsnorm._triton_add_gemma_rms_norm,
        ),
        (NpuDeviceFamily.UNKNOWN, gemma_rmsnorm._fallback_add_gemma_rms_norm),
    ],
)
def test_add_gemma_provider_table(family, provider):
    assert gemma_rmsnorm._ADD_GEMMA_RMS_NORM_PROVIDERS[family] is provider


@pytest.mark.parametrize("family", list(NpuDeviceFamily))
def test_public_api_selects_cached_family_provider(monkeypatch, family):
    input = torch.ones(2, 4, dtype=torch.float16)
    weight = torch.ones(4, dtype=torch.float16)
    expected = torch.zeros_like(input)
    provider = Mock(return_value=expected)
    monkeypatch.setattr(gemma_rmsnorm, "_validate_inputs", Mock())
    monkeypatch.setattr(
        gemma_rmsnorm, "get_npu_device_family", Mock(return_value=family)
    )
    monkeypatch.setitem(gemma_rmsnorm._GEMMA_RMS_NORM_PROVIDERS, family, provider)

    output = gemma_rmsnorm.gemma_rms_norm(input, weight)

    assert output is expected
    provider.assert_called_once()
    provider_input, provider_weight, provider_eps = provider.call_args.args
    assert provider_input is input
    assert provider_weight is weight
    assert provider_eps == 1e-6


@pytest.mark.parametrize("family", list(NpuDeviceFamily))
def test_public_add_api_selects_cached_family_provider(monkeypatch, family):
    input = torch.ones(2, 4, dtype=torch.float16)
    weight = torch.ones(4, dtype=torch.float16)
    residual = torch.ones_like(input)
    expected = (torch.zeros_like(input), input + residual)
    provider = Mock(return_value=expected)
    monkeypatch.setattr(gemma_rmsnorm, "_validate_inputs", Mock())
    monkeypatch.setattr(
        gemma_rmsnorm, "get_npu_device_family", Mock(return_value=family)
    )
    monkeypatch.setitem(
        gemma_rmsnorm._ADD_GEMMA_RMS_NORM_PROVIDERS,
        family,
        provider,
    )

    output = gemma_rmsnorm.add_gemma_rms_norm(input, weight, residual)

    assert output is expected
    provider.assert_called_once()
    provider_input, provider_weight, provider_residual, provider_eps = (
        provider.call_args.args
    )
    assert provider_input is input
    assert provider_weight is weight
    assert provider_residual is residual
    assert provider_eps == 1e-6


def test_selected_provider_errors_are_not_swallowed(monkeypatch):
    input = torch.ones(2, 4, dtype=torch.float16)
    weight = torch.ones(4, dtype=torch.float16)
    monkeypatch.setattr(gemma_rmsnorm, "_validate_inputs", Mock())
    monkeypatch.setattr(
        gemma_rmsnorm,
        "get_npu_device_family",
        Mock(return_value=NpuDeviceFamily.ASCEND_950),
    )
    monkeypatch.setitem(
        gemma_rmsnorm._GEMMA_RMS_NORM_PROVIDERS,
        NpuDeviceFamily.ASCEND_950,
        Mock(side_effect=RuntimeError("kernel failed")),
    )

    with pytest.raises(RuntimeError, match="kernel failed"):
        gemma_rmsnorm.gemma_rms_norm(input, weight)


def test_legacy_module_reexports_stable_api():
    assert compatibility_gemma_rms_norm is gemma_rmsnorm.gemma_rms_norm
    assert compatibility_add_gemma_rms_norm is gemma_rmsnorm.add_gemma_rms_norm
