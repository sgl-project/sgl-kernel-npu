"""Public-interface accuracy and performance checks for causal_conv1d.

Fixture definitions and generation are copied from this repository's
 tests/python/sgl_kernel_npu/test_conv1d_prefill.py. The nine positive cases,
 CPU generation order and seed 20260326 are retained without shrinking them.
Ordinary update reuses the same nine configurations. Two speculative update
configurations come from tests/python/sgl_kernel_npu/test_conv1d_update.py;
their shapes were confirmed for this integrated AscendC path. They do not call
the standalone causal_conv1d_update operator or the original Triton wrappers.
Scenario: SGLang AscendGDNAttnBackend calls the token-major prefill/update API
with transposed contiguous weights and caller-owned cache slots. These fixtures
exercise that protocol; no shape is claimed to be a captured production request.
"""

from dataclasses import dataclass, replace
from itertools import accumulate
from typing import Optional

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

PAD_SLOT_ID = -1
SEED = 20260326
PERF_WARMUP = 5
PERF_ACTIVE = 5

# Original prefill shapes and the two confirmed speculative update shapes.
perf_shape = [
    (2, 6, 4096),
    (3, 4, 4096),
    (10, 4096),
    (2, 5, 1024),
    (3, 7, 2048),
    (13, 2048),
    (2, 4, 3072),
    (2, 4, 1024),
    (1, 4, 1024),
    (32, 4, 12288),
]


@dataclass
class CaseConfig:
    name: str
    dtype: torch.dtype
    dim: int
    width: int
    state_len: int
    num_cache_lines: int
    activation_mode: bool
    use_bias: bool
    input_mode: str
    batch: int
    seq_len: Optional[int] = None
    lengths: Optional[list[int]] = None
    cache_indices: Optional[list[int]] = None
    has_initial_state: Optional[list[bool]] = None
    query_start_loc_dtype: torch.dtype = torch.int32


CASES = [
    CaseConfig(
        name="dense3d_all_zero_no_bias",
        dtype=torch.bfloat16,
        dim=4096,
        width=4,
        state_len=3,
        num_cache_lines=8,
        activation_mode=False,
        use_bias=False,
        input_mode="3d",
        batch=2,
        seq_len=6,
        cache_indices=[0, 1],
        has_initial_state=[False, False],
    ),
    CaseConfig(
        name="dense3d_mixed_bias_act",
        dtype=torch.bfloat16,
        dim=4096,
        width=4,
        state_len=5,
        num_cache_lines=24,
        activation_mode=True,
        use_bias=True,
        input_mode="3d",
        batch=3,
        seq_len=4,
        cache_indices=[5, 12, 20],
        has_initial_state=[True, False, True],
    ),
    CaseConfig(
        name="varlen2d_all_one_bias_act",
        dtype=torch.bfloat16,
        dim=4096,
        width=4,
        state_len=6,
        num_cache_lines=12,
        activation_mode=True,
        use_bias=True,
        input_mode="2d",
        batch=3,
        lengths=[3, 5, 2],
        cache_indices=[2, 4, 8],
        has_initial_state=[True, True, True],
    ),
    CaseConfig(
        name="varlen2d_mixed_pad_no_bias",
        dtype=torch.bfloat16,
        dim=4096,
        width=4,
        state_len=5,
        num_cache_lines=20,
        activation_mode=False,
        use_bias=False,
        input_mode="2d",
        batch=4,
        lengths=[2, 4, 1, 3],
        cache_indices=[3, PAD_SLOT_ID, 9, 15],
        has_initial_state=[True, False, False, True],
    ),
    CaseConfig(
        name="dense3d_fp16_bias_act",
        dtype=torch.float16,
        dim=1024,
        width=4,
        state_len=4,
        num_cache_lines=10,
        activation_mode=True,
        use_bias=True,
        input_mode="3d",
        batch=2,
        seq_len=5,
        cache_indices=[1, 7],
        has_initial_state=[True, False],
    ),
    CaseConfig(
        name="dense3d_width3_bias_act",
        dtype=torch.bfloat16,
        dim=2048,
        width=3,
        state_len=4,
        num_cache_lines=12,
        activation_mode=True,
        use_bias=True,
        input_mode="3d",
        batch=3,
        seq_len=7,
        cache_indices=[0, 5, 9],
        has_initial_state=[True, False, True],
    ),
    CaseConfig(
        name="varlen2d_width2_fp16_act",
        dtype=torch.float16,
        dim=2048,
        width=2,
        state_len=3,
        num_cache_lines=16,
        activation_mode=True,
        use_bias=False,
        input_mode="2d",
        batch=4,
        lengths=[3, 5, 1, 4],
        cache_indices=[1, 6, PAD_SLOT_ID, 11],
        has_initial_state=[True, True, False, True],
    ),
    CaseConfig(
        name="dense3d_dim3072",
        dtype=torch.bfloat16,
        dim=3072,
        width=4,
        state_len=5,
        num_cache_lines=8,
        activation_mode=False,
        use_bias=True,
        input_mode="3d",
        batch=2,
        seq_len=4,
        cache_indices=[0, 3],
        has_initial_state=[True, False],
    ),
    CaseConfig(
        name="dense3d_int64_query_start_loc",
        dtype=torch.bfloat16,
        dim=1024,
        width=4,
        state_len=5,
        num_cache_lines=8,
        activation_mode=False,
        use_bias=True,
        input_mode="3d",
        batch=2,
        seq_len=4,
        cache_indices=[1, 5],
        has_initial_state=[True, False],
        query_start_loc_dtype=torch.int64,
    ),
]


EXTRA_DIMS = {"prefill-cutbsd": 4224}


def generate_test_data(case_index, run_mode=0):
    """Keep the original CPU RNG stream and every declared input unchanged."""
    cases = CASES
    if isinstance(case_index, str):
        cases = [replace(CASES[0], dim=EXTRA_DIMS[case_index])]
        case_index = 0
    if case_index < len(cases):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            for case in cases[: case_index + 1]:
                lengths = (
                    [case.seq_len] * case.batch
                    if case.input_mode == "3d"
                    else case.lengths
                )
                shape = (
                    (case.batch, case.seq_len, case.dim)
                    if case.input_mode == "3d"
                    else (sum(lengths), case.dim)
                )
                x = torch.randn(shape, device="cpu", dtype=case.dtype)
                weight = torch.randn(
                    (case.width, case.dim), device="cpu", dtype=case.dtype
                )
                bias = (
                    torch.randn((case.dim,), device="cpu", dtype=case.dtype)
                    if case.use_bias
                    else None
                )
                state = torch.randn(
                    (case.num_cache_lines, case.state_len, case.dim),
                    device="cpu",
                    dtype=case.dtype,
                )
        return dict(
            x=x,
            weight=weight,
            conv_states=state,
            bias=bias,
            query_start_loc=torch.tensor(
                [0, *accumulate(lengths)],
                dtype=case.query_start_loc_dtype,
                device="cpu",
            ),
            cache_indices=torch.tensor(
                case.cache_indices, dtype=torch.int32, device="cpu"
            ),
            has_initial_state=torch.tensor(
                case.has_initial_state, dtype=torch.bool, device="cpu"
            ),
            num_accepted_tokens=None,
            activation_mode=int(case.activation_mode),
            pad_slot_id=PAD_SLOT_ID,
            run_mode=run_mode,
        )

    # The two confirmed speculative cases retain their original RNG order.
    fp16 = case_index == len(CASES)
    if run_mode != 1 or case_index not in (len(CASES), len(CASES) + 1):
        raise IndexError((run_mode, case_index))
    batch, length, dim = (1, 4, 1024) if fp16 else (32, 4, 12288)
    dtype = torch.float16 if fp16 else torch.bfloat16
    width, state_len = 4, 6
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        if fp16:
            x = torch.randn((batch, length, dim), dtype=dtype, device="cpu")
            weight = torch.randn((dim, width), dtype=dtype, device="cpu")
            weight = weight.T.contiguous()
            bias = torch.randn((dim,), dtype=dtype, device="cpu")
            history = torch.randn((batch, width - 1, dim), dtype=dtype, device="cpu")
            state = torch.zeros((batch, state_len, dim), dtype=dtype, device="cpu")
            state[:, -(width - 1) :] = history
        else:
            weight = torch.randn((width, dim), dtype=dtype, device="cpu")
            bias = None
            x = torch.randn((batch, length, dim), dtype=dtype, device="cpu")
            state = torch.randn((batch, state_len, dim), dtype=dtype, device="cpu")
    return dict(
        x=x,
        weight=weight,
        conv_states=state,
        bias=bias,
        query_start_loc=None,
        cache_indices=torch.arange(batch, dtype=torch.int32, device="cpu"),
        has_initial_state=None,
        num_accepted_tokens=torch.full((batch,), 4, dtype=torch.int32, device="cpu"),
        activation_mode=1,
        pad_slot_id=PAD_SLOT_ID,
        run_mode=1,
    )


def clone_test_data(data, device=None):
    """Fresh storage for every call, including the mutable state pool."""
    return {
        key: (
            value.to(device=device or value.device).clone()
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in data.items()
    }


UPDATE_CASE_NAMES = [f"ordinary-{case.name}" for case in CASES] + [
    "speculative-fp16-b1-l4-d1024-k4-s6-accepted4",
    "speculative-bf16-b32-l4-d12288-k4-s6-accepted4",
]


def check_result(actual_y, actual, reference_y, reference, original, label):
    """Check valid y rows, the exact full state pool and read-only inputs."""
    x = original["x"]
    for output, data in ((actual_y, actual), (reference_y, reference)):
        assert output.shape == x.shape
        assert output.dtype == x.dtype
        assert output.device == data["x"].device
        assert data["conv_states"].device == data["x"].device
    mask = torch.zeros(x.numel() // x.shape[-1], dtype=torch.bool, device="cpu")
    query = (
        [seq * x.shape[1] for seq in range(x.shape[0] + 1)]
        if x.ndim == 3
        else original["query_start_loc"].cpu().tolist()
    )
    indices = original["cache_indices"]
    slots = range(len(query) - 1) if indices is None else indices.cpu().tolist()
    for seq, slot in enumerate(slots):
        if indices is None or slot != original["pad_slot_id"]:
            mask[query[seq] : query[seq + 1]] = True
    # Padding output is undefined. State outside live slots must stay intact.
    assert_accuracy(
        actual_y.detach().cpu().reshape(-1, x.shape[-1])[mask],
        reference_y.detach().cpu().reshape(-1, x.shape[-1])[mask],
        path=f"{label}.y",
    )
    torch.testing.assert_close(
        actual["conv_states"].detach().cpu(),
        reference["conv_states"].detach().cpu(),
        atol=0,
        rtol=0,
        equal_nan=False,
        msg=lambda message: f"{label}.conv_states: {message}",
    )
    for data in (actual, reference):
        for key, value in data.items():
            if key != "conv_states" and isinstance(value, torch.Tensor):
                assert torch.equal(
                    value.cpu(), original[key].cpu()
                ), f"{key} was modified"


@pytest.mark.parametrize(
    "run_mode,case_index",
    [(0, index) for index in range(len(CASES))]
    + [(1, index) for index in range(len(UPDATE_CASE_NAMES))]
    + [
        pytest.param(0, "prefill-cutbsd", marks=pytest.mark.extra_checks),
    ],
    ids=[
        f"{mode}-{case.name}-b{case.batch}-"
        f"{'l' + str(case.seq_len) if case.seq_len is not None else 'lens' + '-'.join(map(str, case.lengths))}-"
        f"d{case.dim}-k{case.width}-state{case.state_len}-{case.dtype}"
        for mode in ("prefill", "update")
        for case in CASES
    ]
    + UPDATE_CASE_NAMES[len(CASES) :]
    + ["extra-prefill-cutbsd-b2-l6-d4224"],
)
def test_causal_conv1d(run_mode, case_index):
    from sgl_kernel_npu.operators.mamba import causal_conv1d

    original = generate_test_data(case_index, run_mode)
    impl, impl_ref = get_test_impls()
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(causal_conv1d, impl)
    else:
        require_distinct_impls(causal_conv1d, impl, impl_ref)
        actual = clone_test_data(original, "npu")
        reference = clone_test_data(original, "npu")
        state_ptr = actual["conv_states"].data_ptr()
        y_ref = causal_conv1d(**reference, impl=impl_ref)
        y = causal_conv1d(**actual, impl=impl)
        assert actual["conv_states"].data_ptr() == state_ptr
        check_result(y, actual, y_ref, reference, original, f"{impl}-vs-{impl_ref}")

    inputs = clone_test_data(original, "npu")
    executor = make_executor(
        causal_conv1d,
        kwargs=inputs,
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip_accuracy,
    )
    states = None

    def execute():
        nonlocal states
        if states is None:
            # The initial call is before profiling. Prepare all independent
            # states here so warmup/active measurements contain no copy kernels.
            states = iter(
                [
                    inputs["conv_states"].clone()
                    for _ in range(1 + PERF_WARMUP + PERF_ACTIVE)
                ]
            )
        executor.keywords["conv_states"] = next(states)
        return executor()

    execute.func = executor.func
    execute.impl = executor.impl
    execute.impl_ref = executor.impl_ref
    execute.accuracy_checked = executor.accuracy_checked
    check_performance(
        execute,
        enable_perf=tuple(original["x"].shape) in perf_shape,
        warmup=PERF_WARMUP,
        active=PERF_ACTIVE,
    )
