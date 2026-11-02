"""CPU contracts and numerical references, isolated from the NPU bootstrap.

Only device detection is simulated; all reference computations use real torch.
Run: python -m pytest tests/unit/test_dsv4_transformer.py
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
def package(request):
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
    api = importlib.import_module(ns + ".operators.dsv4")
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
        package.ns + ".implementation.torch_native.dsv4.cache"
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
    backend = importlib.import_module(
        package.ns + ".implementation.cann_ops_transformer.dsv4"
    )
    x = torch.ones(2, 2, 4, dtype=torch.bfloat16)
    fused = Mock(
        return_value=(torch.ones(2, 1, 4), torch.ones(2, 1, 2), torch.ones(2, 1, 2, 2))
    )
    with patch.object(backend, "get_op", return_value=fused):
        y, post, comb = package.api.mhc_pre(
            x, object(), object(), object(), 2, 4, 1e-6, 1e-5
        )
    assert fused.call_args.args[0].shape == (2, 1, 2, 4)
    assert fused.call_args.args[-5:] == (2, 4, 1e-6, 1e-5, False)
    assert y.shape == (2, 4) and y.dtype == x.dtype
    op = Mock(return_value=x)
    with patch.object(backend, "get_op", return_value=op):
        package.api.mhc_post(y, x, post, comb)
    assert op.call_args.args == (x, comb, y, post)


def test_vendor_guard_restores_environment_on_error(package, monkeypatch, tmp_path):
    runtime = importlib.import_module(
        package.ns + ".implementation.cann_ops_transformer._ops"
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
        package.ns + ".implementation.cann_ops_transformer._ops"
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
    backend = importlib.import_module(package.ns + ".implementation.torch_npu.dsv4")
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
