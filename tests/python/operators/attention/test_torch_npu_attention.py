"""Accuracy cases for the two FIA schemas and partitioned attention."""

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
    dtype: AccuracyStandard(atol=3e-3, rtol=3e-3)
    for dtype in ("torch.float16", "torch.float32")
}
_FIA_CASES = [
    pytest.param(1, 32, 1, 512, id="fia-bnsd-decode"),
    pytest.param(2, 32, 1, 512, id="fia-v2-bnsd-decode"),
]


def generate_fia_data(heads, q_len, kv_len):
    generator = torch.Generator(device="cpu").manual_seed(42)
    return tuple(
        torch.randn((1, heads, seq, 128), generator=generator, device="cpu").to(
            device="npu", dtype=torch.float16
        )
        for seq in (q_len, kv_len, kv_len)
    )


def generate_fia_case(version, heads, q_len, kv_len):
    args = generate_fia_data(heads, q_len, kv_len)
    scale = args[0].size(-1) ** -0.5
    if version == 1:
        return (
            "fused_infer_attention_score",
            args,
            dict(
                input_layout="BNSD", num_heads=heads, scale=scale, softmax_lse_flag=True
            ),
        )
    return (
        "fused_infer_attention_score_v2",
        args,
        dict(
            input_layout="BNSD",
            num_query_heads=heads,
            softmax_scale=scale,
            return_softmax_lse=True,
        ),
    )


def allocate_fia_buffers(query):
    return [
        torch.full_like(query, float("nan")),
        torch.full(
            (*query.shape[:-1], 1),
            float("nan"),
            dtype=torch.float32,
            device=query.device,
        ),
    ]


@pytest.mark.parametrize("version, heads, q_len, kv_len", _FIA_CASES)
def test_fused_infer_attention_score_and_out(version, heads, q_len, kv_len):
    from sgl_kernel_npu.operators import attention

    name, args, kwargs = generate_fia_case(version, heads, q_len, kv_len)
    executor = check_accuracy(
        getattr(attention, name), args, kwargs=kwargs, standard=_FIA_STANDARD
    )
    check_performance(executor)

    if executor.accuracy_checked:
        get_workspace = getattr(attention, f"{name}_get_max_workspace")
        # Workspace is SDK state for its out overload; the reference uses
        # the shapes and computes both outputs with ordinary PyTorch.
        workspace = get_workspace(*args, **kwargs, impl=executor.impl)
        buffers = allocate_fia_buffers(args[0])
        check_accuracy(
            getattr(attention, f"{name}_out"),
            args,
            kwargs={**kwargs, "out": buffers, "workspace": workspace},
            mutable_kwargs=("out",),
            standard=_FIA_STANDARD,
        )


def generate_attention_update_data(rows, head_dim):
    generator = torch.Generator(device="cpu").manual_seed(42)
    lse = torch.randn(2, rows, generator=generator, device="cpu").npu()
    partial = torch.randn(2, rows, head_dim, generator=generator, device="cpu").npu()
    return tuple(lse.unbind()), tuple(partial.unbind())


@pytest.mark.parametrize(
    "rows, head_dim",
    [pytest.param(64, 512, id="dsa-dcp-latent-width")],
)
def test_attention_update(rows, head_dim):
    from sgl_kernel_npu.operators.attention import attention_update

    args = generate_attention_update_data(rows, head_dim)
    executor = check_accuracy(
        attention_update,
        (*args, 1),
        standard={"torch.float32": AccuracyStandard(atol=1e-4, rtol=1e-4)},
    )
    check_performance(executor)


@pytest.mark.extra_checks
@pytest.mark.parametrize("version, heads, q_len, kv_len", _FIA_CASES)
def test_fia_out_graph_replay(version, heads, q_len, kv_len):
    from sgl_kernel_npu.operators import attention

    if get_skip_acc_check() is True:
        pytest.skip("Graph replay accuracy disabled")
    name, args, kwargs = generate_fia_case(version, heads, q_len, kv_len)
    impl, impl_ref = get_test_impls()
    operator = getattr(attention, name)
    require_distinct_impls(operator, impl, impl_ref)
    workspace = getattr(attention, f"{name}_get_max_workspace")(
        *args, **kwargs, impl=impl
    )
    buffers = allocate_fia_buffers(args[0])
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
