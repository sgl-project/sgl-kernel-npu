import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

perf_shape = [(20, 8, 5, 4, (3, 3072), 4)]
exact = {
    str(dtype): AccuracyStandard(0, 0)
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
}


def generate_data(shape, dtype, strided=False):
    layers, slots, scratch, steps, tail, requests = shape
    torch.manual_seed(42)
    storage_tail = (*tail[:-2], tail[-1], tail[-2]) if strided else tail
    dst = torch.randn(layers, slots, *storage_tail, dtype=dtype)
    src = torch.randn(layers, scratch, steps, *storage_tail, dtype=dtype)
    if strided:
        dst, src = dst.transpose(-1, -2), src.transpose(-1, -2)

    index_dtype = torch.int64 if strided else torch.int32
    dst_indices = torch.arange(requests, dtype=index_dtype)
    src_indices = (dst_indices + 1) % scratch
    step_indices = dst_indices % steps
    if strided:
        dst_indices[1] = -1
        src_indices[2] = -1
        step_indices[3] = -1
        dst_indices = dst_indices.repeat_interleave(2)[::2]
        src_indices = src_indices.repeat_interleave(2)[::2]
        step_indices = step_indices.repeat_interleave(2)[::2]
    return dst, src, dst_indices, src_indices, step_indices


@pytest.mark.parametrize(
    "shape,dtype,strided",
    [
        pytest.param((1, 2, 2, 2, (17,), 1), torch.float32, False, id="rank1-tail"),
        pytest.param(
            (2, 6, 5, 3, (2, 3, 19), 5), torch.float16, True, id="rank3-strided"
        ),
        # Kimi Linear TP4 layer count and base conv window, with a synthetic cache.
        pytest.param(
            (20, 8, 5, 4, (3, 3072), 4),
            torch.bfloat16,
            False,
            id="kimi-linear-48b-a3b-tp4-conv-shape",
        ),
    ],
)
def test_speculative_state_scatter(shape, dtype, strided):
    from sgl_kernel_npu.operators.mamba import speculative_state_scatter

    args = generate_data(shape, dtype, strided)
    executor = make_executor(speculative_state_scatter, args=args)
    if not get_skip_acc_check():
        impl, impl_ref = get_test_impls()
        require_distinct_impls(speculative_state_scatter, impl, impl_ref)
        dst = args[0]
        stride = dst.stride()
        before = dst.clone()
        reference = dst.clone()
        saved_inputs = tuple(t.clone() for t in args[1:])

        assert (
            speculative_state_scatter(reference, *args[1:], impl=impl_ref) is reference
        )
        assert executor() is dst
        assert dst.stride() == stride
        assert_accuracy(dst, reference, standard=exact)
        assert_accuracy(args[1:], saved_inputs, standard=exact)
        assert_accuracy(dst[:, shape[-1] :], before[:, shape[-1] :], standard=exact)
        if strided:
            assert_accuracy(dst[:, 1:4], before[:, 1:4], standard=exact)
        executor.accuracy_checked = True
    check_performance(executor, enable_perf=shape in perf_shape)


@pytest.mark.extra_checks
def test_empty_request_list():
    from sgl_kernel_npu.operators.mamba import speculative_state_scatter

    args = generate_data((1, 2, 2, 2, (17,), 0), torch.float32)
    dst = args[0]
    before = dst.clone()
    impl, _ = get_test_impls()
    assert speculative_state_scatter(*args, impl=impl) is dst
    assert_accuracy(dst, before, standard=exact)


@pytest.mark.extra_checks
def test_speculative_state_scatter_graph():
    from sgl_kernel_npu.operators.mamba import speculative_state_scatter

    impl, _ = get_test_impls()
    if impl not in ("auto", "triton"):
        pytest.skip("Graph replay is supported by the Triton implementation")

    args = generate_data((1, 2, 2, 2, (17,), 1), torch.float32)
    dst, src, _, _, step_indices = args
    before = dst.clone()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        speculative_state_scatter(*args, impl=impl)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        speculative_state_scatter(*args, impl=impl)
    torch.npu.synchronize()

    for step in (-1, 1):
        dst.copy_(before)
        step_indices.fill_(step)
        expected = before.clone()
        if step >= 0:
            expected[:, 0] = src[:, 1, step]
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        assert_accuracy(dst, expected, standard=exact)
