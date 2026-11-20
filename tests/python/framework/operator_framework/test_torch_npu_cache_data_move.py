"""Contract tests for the TorchNPU data-move, cache and tensor wrappers."""

import sys
from unittest.mock import Mock, patch, sentinel

import pytest


@pytest.mark.parametrize(
    "category, name, vendor_name, vendor_impl, args, kwargs, expected_args, expected_kwargs",
    [
        pytest.param(
            "data_move",
            "dtype_cast",
            "npu_dtype_cast",
            "ops-math",
            (sentinel.input, sentinel.dtype),
            {},
            (sentinel.input, sentinel.dtype),
            {},
            id="dtype_cast",
        ),
        pytest.param(
            "data_move",
            "dtype_cast",
            "npu_dtype_cast",
            "ops-math",
            (sentinel.input, sentinel.dtype),
            {"input_dtype": sentinel.packed},
            (sentinel.input, sentinel.dtype),
            {"input_dtype": sentinel.packed},
            id="dtype_cast-with-input-dtype",
        ),
        pytest.param(
            "data_move",
            "format_cast",
            "npu_format_cast",
            "ops-math",
            (sentinel.input, 29),
            {"customize_dtype": sentinel.dtype, "input_dtype": sentinel.packed},
            (sentinel.input, 29),
            {"customize_dtype": sentinel.dtype, "input_dtype": sentinel.packed},
            id="format_cast",
        ),
        pytest.param(
            "data_move",
            "prefetch",
            "npu_prefetch",
            "runtime",
            (sentinel.input, sentinel.dependency, 4096),
            {},
            (sentinel.input, sentinel.dependency, 4096),
            {},
            id="prefetch",
        ),
        pytest.param(
            "data_move",
            "prefetch",
            "npu_prefetch",
            "runtime",
            (sentinel.input, None, 4096, 64),
            {},
            (sentinel.input, None, 4096),
            {"offset": 64},
            id="prefetch-with-offset",
        ),
        pytest.param(
            "mem_cache",
            "reshape_and_cache",
            "_npu_reshape_and_cache",
            "ascend-transformer-boost",
            (),
            {
                "key": sentinel.key,
                "value": sentinel.value,
                "key_cache": sentinel.key_cache,
                "value_cache": sentinel.value_cache,
                "slot_indices": sentinel.slots,
            },
            (),
            {
                "key": sentinel.key,
                "value": sentinel.value,
                "key_cache": sentinel.key_cache,
                "value_cache": sentinel.value_cache,
                "slot_indices": sentinel.slots,
            },
            id="reshape_and_cache",
        ),
        pytest.param(
            "mem_cache",
            "scatter_pa_kv_cache",
            "npu_scatter_pa_kv_cache",
            "ops-transformer",
            (
                sentinel.key,
                sentinel.value,
                sentinel.key_cache,
                sentinel.value_cache,
                sentinel.slots,
            ),
            {"cache_mode": "Norm", "compress_lens": sentinel.lengths},
            (
                sentinel.key,
                sentinel.value,
                sentinel.key_cache,
                sentinel.value_cache,
                sentinel.slots,
            ),
            {
                "cache_mode": "Norm",
                "compress_lens": sentinel.lengths,
                "compress_seq_offsets": None,
                "seq_lens": None,
            },
            id="scatter_pa_kv_cache",
        ),
        pytest.param(
            "mem_cache",
            "kv_rmsnorm_rope_cache",
            "npu_kv_rmsnorm_rope_cache",
            "ops-transformer",
            (
                sentinel.kv,
                sentinel.gamma,
                sentinel.cos,
                sentinel.sin,
                sentinel.slots,
                sentinel.key_cache,
                sentinel.value_cache,
            ),
            {
                "cache_mode": "PA_BNSD",
                "c_kv_scale": sentinel.scale,
                "is_output_kv": True,
            },
            (
                sentinel.kv,
                sentinel.gamma,
                sentinel.cos,
                sentinel.sin,
                sentinel.slots,
                sentinel.key_cache,
                sentinel.value_cache,
            ),
            {
                "k_rope_scale": None,
                "c_kv_scale": sentinel.scale,
                "k_rope_offset": None,
                "c_kv_offset": None,
                "epsilon": 1e-5,
                "cache_mode": "PA_BNSD",
                "is_output_kv": True,
            },
            id="kv_rmsnorm_rope_cache",
        ),
        pytest.param(
            "mem_cache",
            "scatter_nd_update_",
            "npu_scatter_nd_update_",
            "ops-nn",
            (sentinel.input, sentinel.indices, sentinel.updates),
            {},
            (sentinel.input, sentinel.indices, sentinel.updates),
            {},
            id="scatter_nd_update_",
        ),
    ],
)
@pytest.mark.parametrize("explicit_impl", [False, True], ids=["auto", "explicit"])
def test_torch_npu_operator_forwards_to_vendor(
    registry_for_generation,
    category,
    name,
    vendor_name,
    vendor_impl,
    args,
    kwargs,
    expected_args,
    expected_kwargs,
    explicit_impl,
):
    modules = registry_for_generation("ARCH22")
    impl = vendor_impl if explicit_impl else "auto"
    vendor = Mock(return_value=sentinel.result)
    with patch.object(sys.modules["torch_npu"], vendor_name, vendor, create=True):
        exported = modules.import_module(f"operators.{category}")
        operator = getattr(exported, name)

        assert name in exported.__all__
        assert modules.registry.get_op_name(operator) == f"{category}.{name}"
        implementations = ["auto", vendor_impl]
        if name != "prefetch":
            implementations.append("torch_native")
        assert modules.public_api.list_impls(operator) == sorted(implementations)
        assert modules.public_api.get_impl(
            operator, "auto"
        ) is modules.public_api.get_impl(operator, vendor_impl)
        vendor.assert_not_called()
        assert operator(*args, **kwargs, impl=impl) is sentinel.result

    vendor.assert_called_once_with(*expected_args, **expected_kwargs)


@pytest.mark.parametrize("impl", ["auto", "runtime"])
def test_prefetch_rejects_arch35_before_calling_torch_npu(
    registry_for_generation, impl
):
    modules = registry_for_generation("ARCH35")
    prefetch = modules.import_module("operators.data_move").prefetch
    vendor = Mock()

    with patch.object(sys.modules["torch_npu"], "npu_prefetch", vendor, create=True):
        with pytest.raises(modules.registry.UnsupportedGenerationError):
            prefetch(sentinel.input, None, 4096, impl=impl)

    vendor.assert_not_called()


def test_scatter_nd_original_import_alias(operator_modules):
    mem_cache = operator_modules.import_module("operators.mem_cache")
    tensor_ops = operator_modules.import_module("operators.tensor_ops")
    assert tensor_ops.scatter_nd_update_ is mem_cache.scatter_nd_update_
