"""Paged allocation accuracy, output mutation and performance."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

# Generic paged allocation; these shapes are independent of a particular model.
perf_shape = [(7, 10000, 128), (100, 2, 128)]

ALLOC_EXTEND_CASES = [
    pytest.param([0], [7], [-1], 100, 128, id="prefill-partial"),
    pytest.param([7], [8], [134], 1000, 128, id="decode-one"),
    pytest.param([7], [9], [134], 1000, 128, id="decode-two"),
    pytest.param([0], [300], [-1], 1000, 128, id="prefill-multiple-pages"),
    pytest.param([100], [400], [227], 1000, 128, id="extend-cached-prefix"),
    pytest.param(
        [0] * 19,
        [
            734,
            720,
            705,
            778,
            725,
            710,
            772,
            735,
            726,
            729,
            732,
            737,
            720,
            728,
            720,
            761,
            723,
            735,
            698,
        ],
        [-1] * 19,
        1066,
        128,
        id="prefill-batch-19",
    ),
    pytest.param(
        [
            697,
            734,
            720,
            705,
            778,
            725,
            710,
            772,
            735,
            726,
            729,
            732,
            737,
            720,
            728,
            720,
            761,
            723,
            735,
            698,
        ],
        [
            698,
            735,
            721,
            706,
            779,
            726,
            711,
            773,
            736,
            727,
            730,
            733,
            738,
            721,
            729,
            721,
            762,
            724,
            736,
            699,
        ],
        [
            1720,
            2525,
            3279,
            4032,
            4873,
            5716,
            6469,
            7299,
            8158,
            8917,
            9688,
            10459,
            11232,
            11983,
            12759,
            13519,
            14328,
            15058,
            15838,
            16569,
        ],
        950,
        128,
        id="decode-batch",
    ),
    pytest.param(
        [
            697,
            734,
            720,
            705,
            778,
            725,
            710,
            772,
            735,
            726,
            729,
            732,
            737,
            720,
            728,
            720,
            761,
            723,
            735,
            698,
        ],
        [
            699,
            736,
            722,
            707,
            780,
            727,
            712,
            774,
            737,
            728,
            731,
            734,
            739,
            722,
            730,
            722,
            763,
            725,
            737,
            700,
        ],
        [
            1720,
            2525,
            3279,
            4032,
            4873,
            5716,
            6469,
            7299,
            8158,
            8917,
            9688,
            10459,
            11232,
            11983,
            12759,
            13519,
            14328,
            15058,
            15838,
            16569,
        ],
        950,
        128,
        id="verify-batch",
    ),
    pytest.param(
        [0] * 7,
        [10000, 8000, 3000, 5000, 4000, 6000, 7000],
        [-1] * 7,
        950,
        128,
        id="long-prefill",
    ),
    pytest.param([0] * 100, [2] * 100, [-1] * 100, 950, 128, id="batch-100"),
    pytest.param([0], [128], [-1], 100, 128, id="complete-page"),
    pytest.param(
        [0, 15, 16], [17, 18, 48], [-1, 159, 175], 100, 16, id="page-16-mixed"
    ),
]


def generate_data(prefix, sequence, last, free_stop, page_size, *, device="npu"):
    tensors = [
        torch.tensor(data, dtype=torch.int64, device=device)
        for data in (prefix, sequence, last)
    ]
    pages = torch.arange(1, free_stop, dtype=torch.int64, device=device)
    count = sum(end - start for start, end in zip(prefix, sequence))
    output = torch.full((count,), -12345, dtype=torch.int64, device=device)
    values = torch.full((1,), -98765, dtype=torch.int64, device=device)
    return (*tensors, pages, page_size, output, values)


def _clone_inputs(inputs):
    return tuple(
        value.clone() if isinstance(value, torch.Tensor) else value for value in inputs
    )


@pytest.mark.parametrize("prefix,sequence,last,free_stop,page_size", ALLOC_EXTEND_CASES)
def test_alloc_extend(prefix, sequence, last, free_stop, page_size):
    from sgl_kernel_npu.operators.mem_cache import alloc_extend

    inputs = generate_data(prefix, sequence, last, free_stop, page_size)
    impl, impl_ref = get_test_impls()
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(alloc_extend, impl)
    else:
        require_distinct_impls(alloc_extend, impl, impl_ref)
        reference = _clone_inputs(inputs)
        assert alloc_extend(*reference, impl=impl_ref) is None
    actual = _clone_inputs(inputs)
    pointers = (actual[5].data_ptr(), actual[6].data_ptr())
    assert alloc_extend(*actual, impl=impl) is None
    assert pointers == (actual[5].data_ptr(), actual[6].data_ptr())
    if not skip_accuracy:
        assert_accuracy(actual[5:], reference[5:])
    for i in range(4):
        assert_accuracy(actual[i], inputs[i])
    executor = make_executor(
        alloc_extend,
        args=actual,
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip_accuracy,
    )
    check_performance(
        executor, enable_perf=(len(prefix), max(sequence), page_size) in perf_shape
    )


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "argument",
    [0, 1, 2, 3, 5],
    ids=["pre_lens", "seq_lens", "last_loc", "free_pages", "out_indices"],
)
def test_alloc_extend_dtype_rejected(argument):
    from sgl_kernel_npu.operators.mem_cache import alloc_extend

    inputs = list(generate_data([0], [7], [-1], 100, 128))
    inputs[argument] = inputs[argument].to(torch.int32)
    impl = get_test_impls()[0]
    with pytest.raises(
        (ValueError, RuntimeError), match="Only support int64 input dtype"
    ):
        alloc_extend(*inputs, impl=impl)
    if impl in ("ascendc", "auto"):
        with pytest.raises(
            (ValueError, RuntimeError), match="Only support int64 input dtype"
        ):
            torch.ops.npu.alloc_extend(*inputs)
