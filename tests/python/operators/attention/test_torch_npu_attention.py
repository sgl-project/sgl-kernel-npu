"""Qwen3 paged decode and DeepSeek MLA prefix-cache attention."""

import math
from itertools import accumulate

import pytest

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    require_distinct_impls,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

torch = pytest.importorskip("torch")
_FIA_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=4e-3, rtol=6e-3),
    "torch.float32": AccuracyStandard(atol=3e-3, rtol=3e-3),
}
_FIA_CASES = [
    pytest.param("fused_infer_attention_score", id="qwen3-8b-tp2-fia-paged-decode"),
    pytest.param(
        "fused_infer_attention_score_v2", id="deepseek-v2-tp8-fia-v2-tnd-prefix"
    ),
]


def generate_fia_case(name):
    generator = torch.Generator(device="cpu").manual_seed(42)

    def randn(*shape):
        return torch.randn(shape, generator=generator, device="cpu").to(
            device="npu", dtype=torch.bfloat16
        )

    if name == "fused_infer_attention_score":
        # SGLang ascend_backend.forward_decode uses BSND query and
        # flattened [pages, page_size, KV_heads * D] cache with ASCEND_USE_FIA=1.
        page_size = 128
        query = randn(2, 1, 16, 128)
        key = randn(6, page_size, 4 * 128)
        value = randn(6, page_size, 4 * 128)
        return (query, key, value), dict(
            num_heads=16,
            num_key_value_heads=4,
            input_layout="BSND",
            scale=128**-0.5,
            block_size=page_size,
            block_table=torch.tensor([[3, 1, 4], [2, 5, 0]], dtype=torch.int32).npu(),
            actual_seq_lengths_kv=[289, 159],
        )
    # DeepSeek-V2 first projects prefix latent KV back to per-head K and V.
    # Only the cached prefix reaches this FIA v2 invocation. The current
    # chunk's attention and the merge are separate calls in SGLang.
    q_lengths, kv_lengths = (32, 16), (256, 128)
    head_count, nope_dim, rope_dim = 16, 128, 64
    query = randn(sum(q_lengths), head_count, nope_dim)
    key = randn(sum(kv_lengths), head_count, nope_dim)
    value = randn(sum(kv_lengths), head_count, 128)
    return (query, key, value), dict(
        query_rope=randn(sum(q_lengths), head_count, rope_dim),
        key_rope=randn(sum(kv_lengths), 1, rope_dim)
        .expand(-1, head_count, -1)
        .contiguous(),
        num_query_heads=head_count,
        num_key_value_heads=head_count,
        input_layout="TND",
        actual_seq_qlen=list(accumulate(q_lengths)),
        actual_seq_kvlen=list(accumulate(kv_lengths)),
        sparse_mode=0,
        # DeepSeek-V2 YaRN factor=40 and mscale_all_dim=0.707.
        softmax_scale=(nope_dim + rope_dim) ** -0.5
        * (1 + 0.1 * 0.707 * math.log(40)) ** 2,
        return_softmax_lse=True,
    )


def allocate_fia_buffers(query, kwargs):
    shape = (*query.shape[:-1], 1) if kwargs.get("return_softmax_lse") else (0,)
    return [
        torch.full_like(query, float("nan")),
        torch.full(shape, float("nan"), dtype=torch.float32, device=query.device),
    ]


def has_fia_workspace(name):
    import torch_npu

    return callable(getattr(torch_npu, f"_npu_{name}_get_max_workspace", None)) and (
        hasattr(getattr(torch_npu, f"npu_{name}"), "out")
    )


@pytest.mark.parametrize("name", _FIA_CASES)
def test_fused_infer_attention_score_and_out(name):
    from sgl_kernel_npu.operators import attention

    args, kwargs = generate_fia_case(name)
    executor = check_accuracy(
        getattr(attention, name),
        args,
        kwargs=kwargs,
        standard=_FIA_STANDARD,
    )
    check_performance(executor)

    if executor.accuracy_checked and has_fia_workspace(name):
        get_workspace = getattr(attention, f"{name}_get_max_workspace")
        # The SDK side needs workspace in either comparison direction.
        workspace = get_workspace(*args, **kwargs, impl="ops-transformer")
        buffers = allocate_fia_buffers(args[0], kwargs)
        check_accuracy(
            getattr(attention, f"{name}_out"),
            args,
            kwargs={**kwargs, "out": buffers, "workspace": workspace},
            mutable_kwargs=("out",),
            standard=_FIA_STANDARD,
        )


def generate_attention_update_data():
    generator = torch.Generator(device="cpu").manual_seed(42)
    # DeepSeek-V2 TP8 has 16 local MHA heads and 128 value channels.
    # The second request has no prefix, so SGLang masks its prefix partition.
    query_lengths = (32, 16)
    heads, value_dim = 16, 128
    rows = sum(query_lengths) * heads
    lse = torch.randn(2, rows, generator=generator, device="cpu") + 4
    partial = torch.randn(2, rows, value_dim, generator=generator, device="cpu")
    lse[1, query_lengths[0] * heads :] = -torch.inf
    partial[1, query_lengths[0] * heads :] = 0
    return tuple(lse.npu().unbind()), tuple(partial.npu().unbind())


def test_attention_update_deepseek_v2_tp8_prefix_merge():
    from sgl_kernel_npu.operators.attention import attention_update

    args = generate_attention_update_data()
    executor = check_accuracy(
        attention_update,
        (*args, 0),
        result_transform=lambda result: result[0],
        standard={"torch.float32": AccuracyStandard(atol=1e-4, rtol=1e-4)},
    )
    check_performance(executor)


@pytest.mark.extra_checks
@pytest.mark.parametrize("name", _FIA_CASES)
def test_fia_out_graph_replay(name):
    from sgl_kernel_npu.operators import attention

    if get_skip_acc_check() is True:
        pytest.skip("Graph replay accuracy disabled")
    impl, impl_ref = get_test_impls()
    if impl == "torch_native":
        pytest.skip("FIA graph replay requires an NPU implementation")
    if not has_fia_workspace(name):
        pytest.skip(f"Installed TorchNPU lacks the {name} workspace/out API")
    args, kwargs = generate_fia_case(name)
    operator = getattr(attention, name)
    require_distinct_impls(operator, impl, impl_ref)
    workspace = getattr(attention, f"{name}_get_max_workspace")(
        *args, **kwargs, impl=impl
    )
    buffers = allocate_fia_buffers(args[0], kwargs)
    op_out = getattr(attention, f"{name}_out")

    def run_out():
        op_out(*args, out=buffers, workspace=workspace, **kwargs, impl=impl)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            run_out()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        run_out()
    torch.npu.synchronize()
    args[0].neg_()
    for buffer in buffers:
        buffer.fill_(float("nan"))
    torch.npu.synchronize()
    graph.update(cpu_update_input=[{}])
    graph.replay()
    torch.npu.synchronize()
    assert_accuracy(
        tuple(buffers), operator(*args, **kwargs, impl=impl_ref), _FIA_STANDARD
    )
