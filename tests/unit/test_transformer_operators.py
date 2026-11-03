"""CPU contracts and numerical references, isolated from the NPU bootstrap.

Only device detection is simulated; all reference computations use real torch.
Run: python -m pytest tests/unit/test_transformer_operators.py
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[2] / "python/sgl_kernel_npu/sgl_kernel_npu"
OP_NAMES = (
    "sparse_flash_mla_metadata",
    "sparse_flash_mla",
    "mixed_quant_sparse_flash_mla_metadata",
    "mixed_quant_sparse_flash_mla",
    "quant_lightning_indexer_metadata",
    "quant_lightning_indexer",
    "kv_compress_epilog",
    "indexer_quant_cache",
    "mhc_pre",
    "mhc_post",
    "inplace_partial_rotary_mul",
    "compressor",
    "moe_gating_top_k",
)


@pytest.fixture(params=["ARCH22", "ARCH35"])
def package(request, monkeypatch):
    ns = "_dsv4_ut_" + uuid4().hex
    root = ModuleType(ns)
    root.__path__ = [str(SOURCE)]
    sys.modules[ns] = root
    name = ns + ".operators.utils.generation"
    spec = importlib.util.spec_from_file_location(
        name, SOURCE / "operators/utils/generation.py"
    )
    generation = importlib.util.module_from_spec(spec)
    sys.modules[name] = generation
    spec.loader.exec_module(generation)
    generation.get_hardware = lambda: generation.Hardware(
        3510 if request.param == "ARCH35" else 2201,
        generation.Generation[request.param],
    )
    # Some existing category exports use absolute package imports. Bind them
    # to this isolated namespace too, without loading the native bootstrap.
    for suffix in ("", ".operators", ".operators.utils", ".operators.utils.registry"):
        module = root if not suffix else importlib.import_module(ns + suffix)
        monkeypatch.setitem(sys.modules, "sgl_kernel_npu" + suffix, module)
    api = SimpleNamespace()
    categories = {
        "attention": [
            "sparse_flash_mla",
            "sparse_flash_mla_metadata",
            "mixed_quant_sparse_flash_mla",
            "mixed_quant_sparse_flash_mla_metadata",
            "quant_lightning_indexer",
            "quant_lightning_indexer_metadata",
            "compressor",
        ],
        "kvcacheio": ["kv_compress_epilog", "indexer_quant_cache", "NPU_KV_QUANT_PATH"],
        "norm": ["mhc_pre", "mhc_post"],
        "rope": ["inplace_partial_rotary_mul"],
        "moe": ["moe_gating_top_k"],
    }
    for category, names in categories.items():
        module = importlib.import_module(ns + ".operators." + category)
        for name in names:
            setattr(api, name, getattr(module, name))
    registry = importlib.import_module(ns + ".operators.utils.registry")
    yield SimpleNamespace(api=api, registry=registry, ns=ns, arch=request.param)
    for name in list(sys.modules):
        if name == ns or name.startswith(ns + "."):
            del sys.modules[name]


def test_implementation_names_and_default_alias(package):
    for name in OP_NAMES:
        fn = getattr(package.api, name)
        backend = "torch_npu" if name == "moe_gating_top_k" else "cann_ops_transformer"
        assert package.registry.list_impls(fn) == sorted(
            ["auto", backend, "torch_native"]
        )
        only_a5 = name.startswith("mixed_quant") or name == "kv_compress_epilog"
        if only_a5 and package.arch == "ARCH22":
            with pytest.raises(package.registry.UnsupportedGenerationError):
                package.registry.get_impl(fn, "auto")
        else:
            assert package.registry.get_impl(fn, "auto") is package.registry.get_impl(
                fn, backend
            )
            assert package.registry.get_impl(
                fn, "auto"
            ) is not package.registry.get_impl(fn, "torch_native")
    assert "cann_ops_transformer" not in sys.modules


def test_mhc_pre_and_post_known_uniform_projection(package):
    api = package.api
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.bfloat16)
    y, post, comb = api.mhc_pre(
        x,
        torch.zeros(8, 4),
        torch.ones(3),
        torch.zeros(8),
        2,
        4,
        1e-6,
        1e-6,
        impl="torch_native",
    )
    torch.testing.assert_close(y, x.float().sum(1).mul(0.5 + 1e-6).to(x.dtype))
    torch.testing.assert_close(post, torch.ones(1, 2))
    torch.testing.assert_close(comb.sum(-1), torch.ones(1, 2), atol=3e-6, rtol=0)
    result = api.mhc_post(y, x, post, comb, impl="torch_native")
    expected = torch.stack(
        [y.float()[0] + (x.float()[0] * comb[0, :, i, None]).sum(0) for i in range(2)]
    )[None]
    torch.testing.assert_close(result, expected.to(x.dtype))


def test_partial_rope_is_inplace_and_keeps_nope(package):
    x = torch.tensor([[[[9.0, 8.0, 1.0, 2.0, 3.0, 4.0]]]])
    ptr = x.data_ptr()
    package.api.inplace_partial_rotary_mul(
        x,
        torch.zeros(1, 1, 1, 4),
        torch.ones(1, 1, 1, 4),
        partial_slice=[2, 6],
        impl="torch_native",
    )
    assert x.data_ptr() == ptr
    torch.testing.assert_close(x, torch.tensor([[[[9.0, 8.0, -2.0, 1.0, -4.0, 3.0]]]]))


def test_bf16_paged_swa_and_sink(package):
    q = torch.ones(1, 1, 2)
    cache = torch.tensor([[[[1.0, 0.0]], [[0.0, 2.0]]], [[[9.0, 9.0]], [[9.0, 9.0]]]])
    out, lse = package.api.sparse_flash_mla(
        q,
        ori_kv=cache,
        ori_block_table=torch.tensor([[0]]),
        seqused_ori_kv=torch.tensor([2]),
        cu_seqlens_q=torch.tensor([0, 1]),
        sinks=torch.tensor([0.0]),
        return_softmax_lse=True,
        impl="torch_native",
    )
    logits = torch.tensor([1.0, 2.0, 0.0])
    torch.testing.assert_close(out[0, 0], logits.softmax(0)[:2] @ cache[0, :, 0])
    torch.testing.assert_close(lse[0, 0], logits.logsumexp(0))


def test_dspark_logical_indices_and_zero_length_padding(package):
    q = torch.zeros(3, 1, 2)
    cache = torch.arange(8.0).reshape(1, 4, 1, 2)
    indices = torch.tensor([[[1, 2, 3, -1]], [[1, 2, 3, -1]], [[0, -1, -1, -1]]])
    out, _ = package.api.sparse_flash_mla(
        q,
        ori_kv=cache,
        ori_block_table=torch.tensor([[0], [0]]),
        seqused_ori_kv=torch.tensor([2, 1]),
        cu_seqlens_q=torch.tensor([0, 2, 3]),
        ori_sparse_indices=indices,
        ori_topk_length=torch.tensor([[3], [3], [0]]),
        ori_mask_mode=0,
        impl="torch_native",
    )
    torch.testing.assert_close(out[:2, 0], cache[0, 1:, 0].mean(0).expand(2, -1))
    assert out[2].count_nonzero() == 0


def test_compressed_causal_boundary(package):
    q = torch.zeros(2, 1, 2)
    ori = torch.zeros(1, 4, 1, 2)
    cmp = torch.tensor([[[[4.0, 8.0]], [[100.0, 100.0]]]])
    out, _ = package.api.sparse_flash_mla(
        q,
        ori_kv=ori,
        cmp_kv=cmp,
        ori_block_table=torch.tensor([[0]]),
        cmp_block_table=torch.tensor([[0]]),
        seqused_ori_kv=torch.tensor([4]),
        seqused_cmp_kv=torch.tensor([1]),
        cmp_residual_kv=torch.tensor([0]),
        cu_seqlens_q=torch.tensor([0, 2]),
        cmp_ratio=4,
        cmp_mask_mode=3,
        impl="torch_native",
    )
    torch.testing.assert_close(out[0], torch.zeros(1, 2))
    torch.testing.assert_close(out[1, 0], torch.tensor([0.8, 1.6]))


def test_indexer_causal_topk_and_padding(package):
    q = torch.ones(2, 1, 2, dtype=torch.int8)
    k = torch.tensor([[[[1, 1]], [[3, 3]], [[2, 2]], [[0, 0]]]], dtype=torch.int8)
    ids, scores = package.api.quant_lightning_indexer(
        q,
        k,
        torch.ones(2, 1),
        torch.ones(2, 1),
        torch.ones(1, 4),
        4,
        2,
        cu_seqlens_q=torch.tensor([0, 2]),
        seqused_k=torch.tensor([3]),
        cmp_residual_k=torch.tensor([0]),
        block_table=torch.tensor([[0]]),
        cmp_ratio=4,
        return_value=1,
        impl="torch_native",
    )
    assert ids[0, 0].tolist() == [1, 0, -1, -1]
    assert ids[1, 0].tolist() == [1, 2, 0, -1]
    torch.testing.assert_close(scores[1, 0, :3], torch.tensor([6.0, 4.0, 2.0]))


def test_indexer_cache_skip_and_valid_slot_zero(package):
    keys = torch.zeros(1, 3, 1, 4, dtype=torch.float8_e4m3fn)
    scales = torch.zeros(1, 3)
    x = torch.tensor([[1.0, -2.0, 3.0, 4.0], [9.0, 9.0, 9.0, 9.0]])
    package.api.indexer_quant_cache(
        keys, scales, x, torch.tensor([0, -1]), impl="torch_native"
    )
    assert scales[0, 0] > 0 and scales[0, 1:].count_nonzero() == 0
    torch.testing.assert_close(
        keys[0, 0, 0].float() * scales[0, 0], x[0], atol=0.15, rtol=0.05
    )


def test_packed_kv_layout_and_skipped_slot(package):
    if package.arch != "ARCH35":
        pytest.skip("A5 layout")
    assert package.api.NPU_KV_QUANT_PATH.packed_dim(448, 64) == 608
    cache = torch.zeros(1, 3, 1, 608, dtype=torch.float8_e4m3fn)
    x = torch.zeros(2, 512, dtype=torch.bfloat16)
    x[0, :448] = 1
    x[0, 448:] = 2
    package.api.kv_compress_epilog(cache, x, torch.tensor([0, -1]), impl="torch_native")
    unpack = importlib.import_module(
        package.ns + ".implementation.torch_native.kvcacheio._quantization"
    ).unpack_kv
    torch.testing.assert_close(unpack(cache)[0], x[0].float())
    assert cache.view(torch.uint8)[0, 1:].count_nonzero() == 0
    # Validate writer bytes directly, independently of the reader.
    raw = cache.view(torch.uint8)[0, 0, 0]
    torch.testing.assert_close(
        raw[:128].view(torch.bfloat16), torch.full((64,), 2.0, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(
        raw[576:590].view(torch.bfloat16),
        torch.full((7,), 1 / 256, dtype=torch.bfloat16),
    )


@pytest.mark.parametrize("coff", [1, 2])
@pytest.mark.parametrize("cycle_table", [False, True])
def test_compressor_prefill_matches_incremental_state(package, coff, cycle_table):
    torch.manual_seed(4)
    x = torch.randn(4, 3)
    w = torch.randn(coff * 4, 3)
    wg = torch.zeros_like(w)
    ape = torch.zeros(2, coff * 4)
    state = torch.zeros(1, 8, 2 * coff * 4)
    table = (
        torch.tensor([0]) if cycle_table else torch.arange(8, dtype=torch.int32)[None]
    )

    # Explicit tables include history columns before the current input position.
    def run(inp, cache, start):
        explicit = (
            table
            if cycle_table
            else torch.arange(2 * coff + len(inp))[None].clamp(min=0) - 2 * coff + start
        )
        if not cycle_table:
            explicit = torch.where(explicit < 0, 7, explicit)
        return package.api.compressor(
            inp,
            w,
            wg,
            cache,
            ape,
            torch.ones(4),
            rope_sin=torch.zeros(4, 2),
            rope_cos=torch.ones(4, 2),
            rope_head_dim=2,
            cmp_ratio=2,
            state_block_table=explicit,
            cu_seqlens=torch.tensor([0, len(inp)]),
            start_pos=torch.tensor([start]),
            coff=coff,
            impl="torch_native",
        )

    full = run(x, state.clone(), 0)
    streamed_state = state.clone()
    a = run(x[:2], streamed_state, 0)
    b = run(x[2:], streamed_state, 2)
    torch.testing.assert_close(full[:2], torch.cat([a[:1], b[:1]]))


def test_cann_mhc_shape_dtype_and_argument_order(package):
    runtime = importlib.import_module(
        package.ns + ".operators.utils._transformer_compat"
    )
    backend = importlib.import_module(package.ns + ".operators.norm.mhc_pre")
    x = torch.ones(2, 2, 4, dtype=torch.bfloat16)
    fused = Mock(
        return_value=(torch.ones(2, 1, 4), torch.ones(2, 1, 2), torch.ones(2, 1, 2, 2))
    )
    with patch.object(
        backend,
        "select_family",
        return_value=runtime.OperatorFamily("cann_ops_transformer", (fused,), ()),
    ):
        y, post, comb = package.api.mhc_pre(
            x, object(), object(), object(), 2, 4, 1e-6, 1e-5
        )
    assert fused.call_args.args[0].shape == (2, 1, 2, 4)
    assert fused.call_args.args[-5:] == (2, 4, 1e-6, 1e-5, False)
    assert y.shape == (2, 4) and y.dtype == x.dtype
    op = Mock(return_value=x)
    post_backend = importlib.import_module(package.ns + ".operators.norm.mhc_post")
    with patch.object(
        post_backend,
        "select_family",
        return_value=runtime.OperatorFamily("cann_ops_transformer", (op,), ()),
    ):
        package.api.mhc_post(y, x, post, comb)
    assert op.call_args.args == (x, comb, y, post)


def test_vendor_guard_restores_environment_on_error(package, monkeypatch, tmp_path):
    runtime = importlib.import_module(
        package.ns + ".operators.utils._cann_ops_transformer"
    )
    (tmp_path / "libcust_opapi.so").touch()
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", "vendor")
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path) + os.pathsep + "/toolkit/lib")
    with pytest.raises(RuntimeError):
        with runtime._without_cust_opapi_vendors():
            assert "ASCEND_CUSTOM_OPP_PATH" not in os.environ
            assert os.environ["LD_LIBRARY_PATH"] == "/toolkit/lib"
            raise RuntimeError("first call failed")
    assert os.environ["ASCEND_CUSTOM_OPP_PATH"] == "vendor"
    assert os.environ["LD_LIBRARY_PATH"].startswith(str(tmp_path))


def test_guarded_packet_first_success_and_retry(package):
    runtime = importlib.import_module(
        package.ns + ".operators.utils._cann_ops_transformer"
    )
    from contextlib import contextmanager

    attempts = []

    @contextmanager
    def guard():
        attempts.append(1)
        yield

    with patch.object(runtime, "_without_cust_opapi_vendors", guard):
        op = runtime._GuardedOpPacket(torch.ops.aten.mm)
        with pytest.raises(RuntimeError):
            op(torch.ones(2, 3), torch.ones(2, 3))
        result = op(torch.ones(2, 3), torch.ones(3, 2))
        op(torch.ones(2, 3), torch.ones(3, 2))
    assert len(attempts) == 2
    torch.testing.assert_close(result, torch.full((2, 2), 3.0))
    assert isinstance(op, torch._ops.OpOverloadPacket)


def test_reference_metadata_stays_backend_specific(package):
    m = package.api.quant_lightning_indexer_metadata(
        1, 1, 128, 4, 2, impl="torch_native"
    )
    assert m.to("cpu") is m
    with pytest.raises(TypeError, match="metadata"):
        package.api.sparse_flash_mla(
            torch.zeros(1, 1, 2),
            ori_kv=torch.zeros(1, 2, 1, 2),
            ori_block_table=torch.tensor([[0]]),
            seqused_ori_kv=torch.tensor([2]),
            cu_seqlens_q=torch.tensor([0, 1]),
            metadata=torch.empty(0),
            impl="torch_native",
        )


@pytest.mark.parametrize("norm_type", [0, 1, 2])
@pytest.mark.parametrize("renorm", [0, 1])
def test_moe_scoring_bias_and_scaling(package, norm_type, renorm):
    import math

    logits = [-4.0, -1.0, 0.0, 2.0]
    if norm_type == 0:
        unscaled = [math.exp(v) / sum(math.exp(z) for z in logits) for v in logits]
    elif norm_type == 1:
        unscaled = [1 / (1 + math.exp(-v)) for v in logits]
    else:
        unscaled = [math.sqrt(math.log1p(math.exp(v))) for v in logits]
    # Bias changes the selected experts; it must not change their returned weights.
    weights, ids, scores = package.api.moe_gating_top_k(
        torch.tensor([logits]),
        2,
        bias=torch.tensor([100.0, 0.0, 0.0, 0.0]),
        norm_type=norm_type,
        renorm=renorm,
        out_flag=True,
        routed_scaling_factor=2.5,
        eps=1e-6,
        impl="torch_native",
    )
    assert ids.tolist() == [[0, 3]] and ids.dtype == torch.int32
    expected = [unscaled[0], unscaled[3]]
    if norm_type in (1, 2) or renorm:
        denominator = sum(expected) + 1e-6
        expected = [v / denominator for v in expected]
    torch.testing.assert_close(weights, torch.tensor([[v * 2.5 for v in expected]]))
    torch.testing.assert_close(scores, torch.tensor([unscaled]))


@pytest.mark.parametrize("group_select_mode, expected_id", [(0, 0), (1, 2)])
def test_moe_group_selection_modes(package, group_select_mode, expected_id):
    probabilities = torch.tensor([[0.9, 0.01, 0.6, 0.5]])
    logits = torch.logit(probabilities)
    weights, ids, _ = package.api.moe_gating_top_k(
        logits,
        1,
        group_count=2,
        k_group=1,
        group_select_mode=group_select_mode,
        norm_type=1,
        impl="torch_native",
    )
    assert ids.item() == expected_id
    torch.testing.assert_close(weights, torch.ones(1, 1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_moe_hash_routing_dtype_and_empty_output(package, dtype):
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=dtype)
    table = torch.tensor([[0, 2], [3, 1]], dtype=torch.int32)
    weights, ids, scores = package.api.moe_gating_top_k(
        x,
        2,
        input_ids=torch.tensor([1, 0]),
        tid2eid=table,
        norm_type=1,
        impl="torch_native",
    )
    assert ids.tolist() == [[3, 1], [0, 2]]
    assert (
        weights.dtype == dtype and scores.dtype == torch.float32 and scores.numel() == 0
    )
    selected = x.float().sigmoid().gather(1, ids.long())
    torch.testing.assert_close(
        weights, (selected / selected.sum(-1, keepdim=True)).to(dtype)
    )


def test_moe_adapter_forwards_standard_binding(package, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    backend = importlib.import_module(package.ns + ".operators.moe.moe_gating_top_k")
    x = torch.ones(2, 8)
    native = Mock(
        return_value=(
            torch.ones(2, 2),
            torch.zeros(2, 2, dtype=torch.int32),
            torch.empty(2, 8),
        )
    )
    with patch.object(
        backend.torch.ops, "npu", SimpleNamespace(npu_moe_gating_top_k=native)
    ):
        _, _, unused = package.api.moe_gating_top_k(
            x, 2, norm_type=2, routed_scaling_factor=2.0
        )
    assert native.call_args.args == (x, 2)
    assert native.call_args.kwargs["norm_type"] == 2
    assert native.call_args.kwargs["routed_scaling_factor"] == 2.0
    assert "input_ids" not in native.call_args.kwargs
    assert unused.shape == (0,)
    assert "cann_ops_transformer" not in sys.modules


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(k=0),
        dict(k=5),
        dict(k=1, group_count=3),
        dict(k=1, input_ids=torch.tensor([0])),
        dict(k=1, norm_type=3),
    ],
)
def test_moe_rejects_invalid_reference_inputs(package, kwargs):
    with pytest.raises(ValueError):
        package.api.moe_gating_top_k(torch.ones(1, 4), impl="torch_native", **kwargs)


def test_moe_empty_batch_and_extreme_logits(package):
    out = package.api.moe_gating_top_k(
        torch.empty(0, 4), 2, norm_type=2, impl="torch_native"
    )
    assert out[0].shape == out[1].shape == (0, 2)
    weights, _, _ = package.api.moe_gating_top_k(
        torch.full((1, 4), -1e10), 2, norm_type=2, impl="torch_native"
    )
    assert torch.isfinite(weights).all() and weights.count_nonzero() == 0


@pytest.fixture
def bindings(package, monkeypatch):
    """Both installations are visible; tests force one without executing NPU code."""
    runtime = importlib.import_module(
        package.ns + ".operators.utils._transformer_compat"
    )
    calls = {}
    names = (
        [
            ("cann_ops_transformer", name)
            for name in (
                "sparse_flash_mla_metadata",
                "sparse_flash_mla",
                "mixed_quant_sparse_flash_mla_metadata",
                "mixed_quant_sparse_flash_mla",
                "quant_lightning_indexer_metadata",
                "quant_lightning_indexer",
                "kv_compress_epilog",
                "indexer_quant_cache",
                "mhc_pre_sinkhorn",
                "mhc_post",
                "inplace_partial_rotary_mul",
            )
        ]
        + [
            ("custom", name)
            for name in (
                "npu_sparse_attn_sharedkv_metadata",
                "npu_sparse_attn_sharedkv",
                "npu_kv_quant_sparse_attn_sharedkv_metadata",
                "npu_kv_quant_sparse_attn_sharedkv",
                "npu_quant_lightning_indexer_metadata",
                "npu_quant_lightning_indexer",
                "indexer_compress_epilog",
                "npu_hc_pre",
                "npu_hc_post",
                "inplace_partial_rotary_mul",
                "compressor",
                "npu_moe_gating_top_k",
            )
        ]
        + [
            ("npu", name)
            for name in (
                "kv_compress_epilog",
                "compressor",
                "npu_moe_gating_top_k",
                "sparse_attn_sharedkv_metadata_host",
                "sparse_attn_sharedkv",
                "kv_quant_sparse_attn_sharedkv_metadata_host",
                "kv_quant_sparse_attn_sharedkv",
            )
        ]
    )
    for namespace, name in names:
        calls[namespace, name] = Mock(name=namespace + "." + name)
    monkeypatch.setattr(
        runtime, "optional_op", lambda namespace, name: calls.get((namespace, name))
    )
    monkeypatch.setattr(
        runtime,
        "prefer_legacy_bindings",
        lambda: runtime.backend_preference() == "legacy",
    )
    # These two public wrappers imported the preference helper by name.
    for path in ("attention.compressor", "moe.moe_gating_top_k"):
        module = importlib.import_module(package.ns + ".operators." + path)
        monkeypatch.setattr(
            module,
            "prefer_legacy_bindings",
            lambda: runtime.backend_preference() == "legacy",
        )
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(
        torch.ops,
        "npu",
        SimpleNamespace(npu_moe_gating_top_k=calls["npu", "npu_moe_gating_top_k"]),
    )
    return SimpleNamespace(runtime=runtime, calls=calls)


@pytest.mark.parametrize("backend", ["cann_ops_transformer", "legacy"])
@pytest.mark.parametrize("name", OP_NAMES)
def test_public_contract_selects_released_or_original_binding(
    package, bindings, monkeypatch, backend, name
):
    if package.arch == "ARCH22" and (
        name.startswith("mixed_quant") or name == "kv_compress_epilog"
    ):
        pytest.skip("A5-only public contract")
    monkeypatch.setenv("SGL_KERNEL_NPU_TRANSFORMER_BACKEND", backend)
    old = backend == "legacy"
    api, calls = package.api, bindings.calls
    x = torch.ones(2, 2, 4, dtype=torch.bfloat16)
    full_lengths = torch.tensor([13], dtype=torch.int32)
    kwargs = dict(
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        seqused_ori_kv=full_lengths,
        seqused_cmp_kv=full_lengths // 4,
        cmp_residual_kv=full_lengths % 4,
        cmp_ratio=4,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        layout_q="TND",
        layout_kv="PA_BBND",
    )
    if name in ("sparse_flash_mla", "mixed_quant_sparse_flash_mla"):
        quant = name.startswith("mixed")
        target = (
            (
                "custom",
                "npu_kv_quant_sparse_attn_sharedkv"
                if quant
                else "npu_sparse_attn_sharedkv",
            )
            if old
            else ("cann_ops_transformer", name)
        )
        calls[target].return_value = (x, torch.tensor(0.0))
        if quant:
            kwargs["ori_kv"] = torch.empty(
                1, 16, 1, 640 if old else 608, dtype=torch.float8_e4m3fn
            )
        out, unused = getattr(api, name)(x, **kwargs)
        assert out is x and unused.shape == (0,)
        passed = calls[target].call_args.kwargs
        if old:
            assert passed["layout_kv"] == "PA_ND"
            assert passed["seqused_kv"] is full_lengths
            assert "seqused_cmp_kv" not in passed and "cmp_residual_kv" not in passed
        else:
            assert passed["layout_kv"] == "PA_BBND"
        if quant:
            assert passed["kv_quant_mode" if old else "quant_mode"] == 1
    elif name in ("sparse_flash_mla_metadata", "mixed_quant_sparse_flash_mla_metadata"):
        quant = name.startswith("mixed")
        target = (
            (
                "custom",
                "npu_kv_quant_sparse_attn_sharedkv_metadata"
                if quant
                else "npu_sparse_attn_sharedkv_metadata",
            )
            if old
            else ("cann_ops_transformer", name)
        )
        calls[target].return_value = torch.zeros(1024, dtype=torch.int32)
        if quant:
            kwargs.update(quant_mode=1, rope_head_dim=64)
        result = getattr(api, name)(
            num_heads_q=2,
            num_heads_kv=1,
            head_dim=4,
            batch_size=1,
            has_cmp_kv=True,
            **kwargs,
        )
        assert result is calls[target].return_value
        assert (
            calls[target].call_args.kwargs["seqused_kv" if old else "seqused_ori_kv"]
            is full_lengths
        )
    elif name.startswith("quant_lightning_indexer"):
        target = ("custom", "npu_" + name) if old else ("cann_ops_transformer", name)
        args = dict(
            cu_seqlens_q=kwargs["cu_seqlens_q"],
            seqused_k=full_lengths // 4,
            cmp_residual_k=full_lengths % 4,
            cmp_ratio=4,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
        )
        if name.endswith("metadata"):
            calls[target].return_value = torch.zeros(1024, dtype=torch.int32)
            api.quant_lightning_indexer_metadata(2, 1, 4, 2, 1, **args)
        else:
            calls[target].return_value = (
                torch.zeros(2, 1, 2, dtype=torch.int32),
                torch.empty(0),
            )
            api.quant_lightning_indexer(
                x,
                x,
                x,
                x,
                x,
                2,
                1,
                block_table=torch.zeros(1, 1, dtype=torch.int32),
                **args,
            )
        passed = calls[target].call_args.kwargs
        if old:
            torch.testing.assert_close(
                passed["actual_seq_lengths_query"], torch.tensor([2], dtype=torch.int32)
            )
            torch.testing.assert_close(passed["actual_seq_lengths_key"], full_lengths)
            assert passed["layout_key"] == "PA_BSND" and passed["sparse_count"] == 2
    elif name == "kv_compress_epilog":
        target = ("npu", name) if old else ("cann_ops_transformer", name)
        cache = torch.zeros(1, 2, 1, 640 if old else 608, dtype=torch.float8_e4m3fn)
        assert (
            api.kv_compress_epilog(
                cache, torch.ones(2, 512, dtype=torch.bfloat16), torch.tensor([0, -1])
            )
            is cache
        )
        passed = calls[target].call_args
        assert passed.kwargs["quant_mode"] == (2 if old else "fp8_bf16")
        assert passed.args[0].shape == ((2, 1, 640) if old else cache.shape)
    elif name == "indexer_quant_cache":
        target = (
            ("custom", "indexer_compress_epilog")
            if old
            else ("cann_ops_transformer", name)
        )
        cache, scale = torch.empty(1, 2, 1, 4), torch.empty(1, 2)
        result = api.indexer_quant_cache(cache, scale, x, torch.tensor([0, -1]))
        assert result[0] is cache and result[1] is scale
        if old:
            assert calls[target].call_args.kwargs["indexer_compress_cache"] is cache
    elif name == "mhc_pre":
        target = (
            ("custom", "npu_hc_pre")
            if old
            else ("cann_ops_transformer", "mhc_pre_sinkhorn")
        )
        calls[target].return_value = (
            torch.ones(2, 4),
            torch.ones(2, 2),
            torch.ones(2, 2, 2),
        )
        y, _, _ = api.mhc_pre(x, x, x, x, 2, 20, 1e-6, 1e-5)
        assert y.dtype == x.dtype and y.shape == (2, 4)
        if old:
            assert calls[target].call_args.args[0] is x
            assert calls[target].call_args.kwargs["hc_sinkhorn_iters"] == 20
            assert calls[target].call_args.kwargs["norm_eps"] == 1e-5
        else:
            assert calls[target].call_args.args[0].shape == (2, 1, 2, 4)
    elif name == "mhc_post":
        target = ("custom", "npu_hc_post") if old else ("cann_ops_transformer", name)
        batched = old and package.arch == "ARCH35"
        calls[target].return_value = x.unsqueeze(0) if batched else x
        y, post, comb = x[:, 0], torch.ones(2, 2), torch.eye(2).expand(2, -1, -1)
        result = api.mhc_post(y, x, post, comb)
        assert result.shape == x.shape
        args = calls[target].call_args.args
        assert args[0].shape == ((1, 2, 4) if batched else y.shape if old else x.shape)
        if not old:
            assert args == (x, comb, y, post)
    elif name == "inplace_partial_rotary_mul":
        target = ("custom", name) if old else ("cann_ops_transformer", name)
        assert api.inplace_partial_rotary_mul(x, x, x) is x
        assert calls[target].call_args.kwargs["partial_slice"] == [0, 4]
    elif name == "compressor":
        target = ("custom" if old and package.arch == "ARCH35" else "npu", name)
        calls[target].return_value = x
        assert (
            api.compressor(x, x, x, x, x, x, rope_sin=x, rope_cos=x, cmp_ratio=4) is x
        )
        assert calls[target].call_args.args == (x, x, x, x, x, x)
        assert calls[target].call_args.kwargs["rope_sin"] is x
    else:
        target = ("custom" if old else "npu", "npu_moe_gating_top_k")
        calls[target].return_value = (
            torch.ones(2, 2),
            torch.zeros(2, 2, dtype=torch.int32),
            torch.empty(2, 4),
        )
        result = api.moe_gating_top_k(x[:, 0], 2, norm_type=2)
        assert result[0].dtype == x.dtype and result[2].shape == (0,)
        assert calls[target].call_args.kwargs["norm_type"] == 2
    calls[target].assert_called_once()


def test_auto_never_mixes_packed_reader_and_writer(package, bindings, monkeypatch):
    monkeypatch.setenv("SGL_KERNEL_NPU_TRANSFORMER_BACKEND", "auto")
    del bindings.calls["cann_ops_transformer", "kv_compress_epilog"]
    family = bindings.runtime.attention_family(True)
    assert family.legacy
    layout = importlib.import_module(package.ns + ".operators.kvcacheio.layout")
    assert layout.get_npu_kv_quant_path().packed_dim(448, 64) == 640


def test_kernel_failure_is_not_a_backend_fallback(package, bindings, monkeypatch):
    monkeypatch.setenv("SGL_KERNEL_NPU_TRANSFORMER_BACKEND", "auto")
    op = bindings.calls["cann_ops_transformer", "mhc_post"]
    op.side_effect = RuntimeError("actual kernel failure")
    with pytest.raises(RuntimeError, match="actual kernel failure"):
        package.api.mhc_post(
            torch.ones(1, 4), torch.ones(1, 2, 4), torch.ones(1, 2), torch.eye(2)[None]
        )
    bindings.calls["custom", "npu_hc_post"].assert_not_called()


def test_legacy_dspark_pairs_host_metadata_and_resolves_physical_slots(
    package, bindings, monkeypatch
):
    monkeypatch.setenv("SGL_KERNEL_NPU_TRANSFORMER_BACKEND", "legacy")
    q = torch.zeros(2, 1, 512)
    quant = package.arch == "ARCH35"
    width = 640 if quant else 512
    cache = torch.empty(5, 2, 1, width)
    table = torch.tensor([[3, 1], [4, 2]], dtype=torch.int32)
    cu = torch.tensor([0, 1, 2], dtype=torch.int32)
    kv = torch.tensor([4, 4], dtype=torch.int32)
    stem = "kv_quant_sparse_attn_sharedkv" if quant else "sparse_attn_sharedkv"
    metadata = bindings.calls["npu", stem + "_metadata_host"]
    metadata.return_value = torch.zeros(1024, dtype=torch.int32)
    attention = bindings.calls["npu", stem]
    attention.return_value = (q, torch.empty(0))
    meta_api = (
        package.api.mixed_quant_sparse_flash_mla_metadata
        if quant
        else package.api.sparse_flash_mla_metadata
    )
    compute_api = (
        package.api.mixed_quant_sparse_flash_mla
        if quant
        else package.api.sparse_flash_mla
    )
    lengths = torch.tensor([[2], [0]], dtype=torch.int32)
    kwargs = dict(
        num_heads_q=1,
        num_heads_kv=1,
        head_dim=512,
        batch_size=2,
        has_cmp_kv=False,
        cu_seqlens_q=cu,
        seqused_ori_kv=kv,
        ori_mask_mode=0,
        ori_topk=2,
        ori_topk_length=lengths,
        layout_q="TND",
        layout_kv="PA_BBND",
    )
    if quant:
        kwargs.update(quant_mode=1, rope_head_dim=64)
    with pytest.raises(ValueError, match="existing CPU int32"):
        meta_api(**kwargs)
    result = meta_api(**kwargs, cu_seqlens_q_cpu=cu, seqused_ori_kv_cpu=kv)
    assert result is metadata.return_value
    compute_api(
        q,
        ori_kv=cache,
        ori_block_table=table,
        cu_seqlens_q=cu,
        seqused_ori_kv=kv,
        ori_mask_mode=0,
        ori_sparse_indices=torch.tensor([[[0, 3]], [[1, -1]]], dtype=torch.int32),
        ori_topk_length=lengths,
        layout_q="TND",
        layout_kv="PA_BBND",
        metadata=result,
    )
    torch.testing.assert_close(
        attention.call_args.kwargs["ori_sparse_indices"],
        torch.tensor([[[6, 3]], [[-1, -1]]], dtype=torch.int32),
    )
    assert attention.call_args.kwargs["metadata"] is result


def test_legacy_packed_reference_preserves_encoding_and_skip(package):
    if package.arch != "ARCH35":
        pytest.skip("A5 packed cache")
    cache = torch.zeros(1, 2, 1, 640, dtype=torch.float8_e4m3fn)
    x = torch.ones(2, 512, dtype=torch.bfloat16)
    x[0, 448:] = 2
    package.api.kv_compress_epilog(cache, x, torch.tensor([0, -1]), impl="torch_native")
    raw = cache.view(torch.uint8)[0, 0, 0]
    torch.testing.assert_close(raw[576:583], torch.full((7,), 119, dtype=torch.uint8))
    unpack = importlib.import_module(
        package.ns + ".implementation.torch_native.kvcacheio._quantization"
    ).unpack_kv
    torch.testing.assert_close(unpack(cache)[0], x[0].float())
    assert cache.view(torch.uint8)[0, 1].count_nonzero() == 0


def test_forced_released_backend_rejects_a_partial_installation(
    package, bindings, monkeypatch
):
    monkeypatch.setenv("SGL_KERNEL_NPU_TRANSFORMER_BACKEND", "cann_ops_transformer")
    del bindings.calls["cann_ops_transformer", "sparse_flash_mla"]
    with pytest.raises(
        bindings.runtime.OperatorUnavailableError, match="complete compatible"
    ):
        bindings.runtime.attention_family(False)
    assert all(not call.called for call in bindings.calls.values())


def test_broken_released_package_is_not_treated_as_missing(package, monkeypatch):
    runtime = importlib.import_module(
        package.ns + ".operators.utils._transformer_compat"
    )
    error = ModuleNotFoundError(
        "missing dependency inside installed package", name="cann_internal_dependency"
    )
    with patch.object(runtime.importlib, "import_module", side_effect=error):
        with pytest.raises(ModuleNotFoundError, match="missing dependency"):
            runtime.has_released_ops()


def test_moe_hash_capability_selects_legacy_before_execution(
    package, bindings, monkeypatch
):
    monkeypatch.setenv("SGL_KERNEL_NPU_TRANSFORMER_BACKEND", "auto")
    namespace = "_moe_compat_" + uuid4().hex
    library = torch.library.Library(namespace, "DEF")
    library.define(
        "v1(Tensor x, int k, *, int norm_type=0) -> (Tensor, Tensor, Tensor)"
    )
    library.define(
        "hash(Tensor x, int k, *, Tensor? input_ids=None, Tensor? tid2eid=None, int norm_type=0) -> (Tensor, Tensor, Tensor)"
    )
    executed = []

    def native(x, k, *, norm_type=0):
        executed.append("native")
        raise AssertionError("v1 cannot execute a hash request")

    def legacy(x, k, *, input_ids=None, tid2eid=None, norm_type=0):
        executed.append("legacy")
        return (
            torch.ones(x.shape[0], k, dtype=x.dtype),
            tid2eid[input_ids.long()].int(),
            x.float(),
        )

    library.impl("v1", native, "CPU")
    library.impl("hash", legacy, "CPU")
    bindings.calls["npu", "npu_moe_gating_top_k"] = getattr(torch.ops, namespace).v1
    bindings.calls["custom", "npu_moe_gating_top_k"] = getattr(
        torch.ops, namespace
    ).hash
    table = torch.tensor([[3, 1]], dtype=torch.int32)
    result = package.api.moe_gating_top_k(
        torch.ones(1, 4), 2, input_ids=torch.tensor([0]), tid2eid=table, norm_type=2
    )
    assert executed == ["legacy"]
    torch.testing.assert_close(result[1], table)


def test_wrong_cache_encoding_is_rejected_before_native_execution(
    package, bindings, monkeypatch
):
    if package.arch != "ARCH35":
        pytest.skip("A5 cache contract")
    monkeypatch.setenv("SGL_KERNEL_NPU_TRANSFORMER_BACKEND", "legacy")
    with pytest.raises(ValueError, match="encoding"):
        package.api.kv_compress_epilog(
            torch.zeros(1, 1, 1, 608, dtype=torch.float8_e4m3fn),
            torch.ones(1, 512, dtype=torch.bfloat16),
            torch.tensor([0]),
        )
    bindings.calls["npu", "kv_compress_epilog"].assert_not_called()
