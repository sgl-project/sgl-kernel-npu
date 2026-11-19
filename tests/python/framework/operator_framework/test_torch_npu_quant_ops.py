"""Exercise the actual public quantization/matmul registry and TorchNPU boundary."""

from unittest.mock import Mock, sentinel

import pytest

CASES = (
    (
        "matmul",
        "dual_level_quant_matmul",
        6,
        {"bias": sentinel.bias, "output_dtype": sentinel.dtype},
        True,
    ),
    (
        "matmul",
        "quant_matmul",
        3,
        {
            "pertoken_scale": sentinel.scale,
            "scale_dtype": sentinel.dtype,
            "group_sizes": [1, 1, 32],
        },
        False,
    ),
    (
        "matmul",
        "transpose_quant_batchmatmul",
        3,
        {"perm_x1": (1, 0, 2), "x2_scale": sentinel.scale},
        True,
    ),
    (
        "matmul",
        "weight_quant_batchmatmul",
        4,
        {"antiquant_group_size": 128, "bias": sentinel.bias},
        False,
    ),
    (
        "quant",
        "dynamic_block_quant",
        1,
        {"dst_type": sentinel.dtype, "col_block_size": 128},
        False,
    ),
    (
        "quant",
        "dynamic_dual_level_mx_quant",
        1,
        {"smooth_scale": sentinel.scale, "round_mode": "rint"},
        True,
    ),
    ("quant", "dynamic_mx_quant", 1, {"dst_type": sentinel.dtype, "axis": -1}, True),
    (
        "quant",
        "dynamic_quant",
        1,
        {
            "smooth_scales": sentinel.scale,
            "group_index": sentinel.group,
            "dst_type": sentinel.dtype,
        },
        False,
    ),
)


@pytest.mark.parametrize(
    "category,name,nargs,kwargs,arch35_only", CASES, ids=[case[1] for case in CASES]
)
@pytest.mark.parametrize("generation", ["ARCH22", "ARCH35"])
def test_torch_npu_quant_operator(
    registry_for_generation,
    monkeypatch,
    category,
    name,
    nargs,
    kwargs,
    arch35_only,
    generation,
):
    import torch_npu

    modules = registry_for_generation(generation)
    api = modules.import_module(f"operators.{category}")
    public = getattr(api, name)
    backend = Mock(return_value=sentinel.output)
    monkeypatch.setattr(torch_npu, f"npu_{name}", backend, raising=False)
    args = tuple(getattr(sentinel, f"arg_{i}") for i in range(nargs))

    assert name in api.__all__
    assert modules.registry.get_op_name(public) == f"{category}.{name}"
    assert modules.public_api.list_impls(public) == ["auto", "torch_npu"]
    with pytest.raises(modules.registry.ImplementationNotFoundError):
        public(*args, impl="missing", **kwargs)
    backend.assert_not_called()

    if arch35_only and generation == "ARCH22":
        for impl in ("auto", "torch_npu"):
            with pytest.raises(modules.registry.UnsupportedGenerationError):
                public(*args, impl=impl, **kwargs)
        backend.assert_not_called()
        return

    assert public(*args, **kwargs) is sentinel.output
    backend.assert_called_once_with(*args, **kwargs)
    backend.reset_mock()
    assert public(*args, impl="torch_npu", **kwargs) is sentinel.output
    backend.assert_called_once_with(*args, **kwargs)

    backend.side_effect = RuntimeError("TorchNPU rejected the input")
    backend.reset_mock()
    with pytest.raises(RuntimeError, match="TorchNPU rejected the input"):
        public(*args, **kwargs)
    backend.assert_called_once_with(*args, **kwargs)
