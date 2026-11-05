"""Public-interface accuracy and separate-process legacy replay for causal_conv1d.

Fixture definitions and generation are copied from this repository's
 tests/python/sgl_kernel_npu/test_conv1d_prefill.py. The nine positive cases,
 CPU generation order and seed 20260326 are retained without shrinking them.
Ordinary update reuses the same nine configurations. Two speculative update
configurations come from tests/python/sgl_kernel_npu/test_conv1d_update.py;
their shapes were confirmed for this integrated AscendC path. They do not call
the standalone causal_conv1d_update operator or the original Triton wrappers.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pytest
import torch

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
_REPLAY_FORMAT = "sgl-causal-conv1d-prefill-v1"
_UPDATE_REPLAY_FORMAT = "sgl-causal-conv1d-all-v2"
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


def make_query_start_loc(
    lengths: Iterable[int],
    device: torch.device,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    qsl = [0]
    for length in lengths:
        qsl.append(qsl[-1] + int(length))
    if device.type == "cpu":
        return torch.tensor(qsl, device="cpu", dtype=dtype)
    out = torch.empty((len(qsl),), device=device, dtype=dtype)
    for idx, value in enumerate(qsl):
        out[idx] = int(value)
    return out


def make_device_bool_tensor(
    values: Iterable[bool], device: torch.device
) -> torch.Tensor:
    values = list(values)
    out = torch.zeros((len(values),), device=device, dtype=torch.bool)
    for idx, value in enumerate(values):
        out[idx] = bool(value)
    return out


def make_device_int_tensor(values: Iterable[int], device: torch.device) -> torch.Tensor:
    values = list(values)
    if device.type == "cpu":
        return torch.tensor(values, device="cpu", dtype=torch.int32)
    out = torch.empty((len(values),), device=device, dtype=torch.int32)
    for idx, value in enumerate(values):
        out[idx] = int(value)
    return out


def make_host_bool_tensor(values: Iterable[bool]) -> torch.Tensor:
    return torch.tensor(list(values), device="cpu", dtype=torch.bool)


def make_case_tensors(case: CaseConfig, device: torch.device, pad_slot_id: int):
    if case.input_mode == "3d":
        assert case.seq_len is not None
        x = torch.randn(
            (case.batch, case.seq_len, case.dim), device=device, dtype=case.dtype
        )
        lengths = [case.seq_len] * case.batch
    else:
        assert case.lengths is not None
        x = torch.randn((sum(case.lengths), case.dim), device=device, dtype=case.dtype)
        lengths = case.lengths

    weight = torch.randn((case.width, case.dim), device=device, dtype=case.dtype)
    bias = (
        torch.randn((case.dim,), device=device, dtype=case.dtype)
        if case.use_bias
        else None
    )
    conv_states = torch.randn(
        (case.num_cache_lines, case.state_len, case.dim),
        device=device,
        dtype=case.dtype,
    )
    query_start_loc = make_query_start_loc(
        lengths, device, dtype=case.query_start_loc_dtype
    )
    cache_indices = make_device_int_tensor(case.cache_indices, device)
    if device.type == "cpu":
        has_initial_state = make_host_bool_tensor(case.has_initial_state)
    else:
        has_initial_state = make_device_bool_tensor(case.has_initial_state, device)

    assert cache_indices.numel() == case.batch
    assert has_initial_state.numel() == case.batch
    assert query_start_loc.numel() == case.batch + 1
    assert (cache_indices == pad_slot_id).sum().item() < case.batch

    return (
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
    )


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


def generate_test_data(case_index):
    """Preserve the old main()'s once-seeded RNG stream, including earlier cases."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        for case in CASES[: case_index + 1]:
            tensors = make_case_tensors(case, torch.device("cpu"), PAD_SLOT_ID)
    x, weight, bias, state, query, cache, initial = tensors
    return dict(
        x=x,
        weight=weight,
        conv_states=state,
        bias=bias,
        query_start_loc=query,
        cache_indices=cache,
        has_initial_state=initial,
        num_accepted_tokens=None,
        activation_mode=int(case.activation_mode),
        pad_slot_id=PAD_SLOT_ID,
        run_mode=0,
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


def generate_update_test_data(case_index):
    if case_index < len(CASES):
        data = generate_test_data(case_index)
        data["run_mode"] = 1
        return data

    # Reuse the old update test's two numerical configurations, with CPU input
    # generation and independent clones for the integrated implementation/oracle.
    fp16 = case_index == len(CASES)
    if case_index not in (len(CASES), len(CASES) + 1):
        raise IndexError(case_index)
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


def valid_rows(data):
    """Only pad y is undefined; the entire state pool is always compared."""
    x = data["x"]
    # check_result indexes CPU outputs; pytest sets the default device to NPU.
    mask = torch.zeros(x.numel() // x.shape[-1], dtype=torch.bool, device="cpu")
    query = (
        [seq * x.shape[1] for seq in range(x.shape[0] + 1)]
        if x.ndim == 3
        else data["query_start_loc"].cpu().tolist()
    )
    indices = data["cache_indices"]
    slots = range(len(query) - 1) if indices is None else indices.cpu().tolist()
    for seq, slot in enumerate(slots):
        if indices is None or slot != data["pad_slot_id"]:
            mask[int(query[seq]) : int(query[seq + 1])] = True
    return mask


def check_result(actual_y, actual_state, reference_y, reference_state, mask, label):
    # Keep BF16/FP16 dtype so assert_accuracy chooses the shared dtype standard.
    actual = actual_y.detach().cpu().reshape(-1, actual_y.shape[-1])[mask]
    reference = reference_y.detach().cpu().reshape(-1, reference_y.shape[-1])[mask]
    assert_accuracy(actual, reference, path=f"{label}.y")
    torch.testing.assert_close(
        actual_state.detach().cpu(),
        reference_state.detach().cpu(),
        atol=0,
        rtol=0,
        equal_nan=False,
        msg=lambda message: f"{label}.conv_states: {message}",
    )


def assert_readonly_inputs(data, before):
    for key, value in data.items():
        if key != "conv_states" and isinstance(value, torch.Tensor):
            assert torch.equal(value.cpu(), before[key].cpu()), f"{key} was modified"


def _make_performance_executor(operator, original, impl, impl_ref, accuracy_checked):
    inputs = clone_test_data(original, torch.device("npu"))
    executor = make_executor(
        operator,
        kwargs=inputs,
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=accuracy_checked,
    )
    states = None

    def execute():
        nonlocal states
        if states is None:
            # benchmark_npu calls once before profiling, then warmup + active.
            # Allocate/copy every independent initial state in that first call;
            # subsequent profiled calls only swap a Python reference, never copy.
            states = iter(
                [
                    inputs["conv_states"].clone()
                    for _ in range(1 + PERF_WARMUP + PERF_ACTIVE)
                ]
            )
        executor.keywords["conv_states"] = next(states)
        return executor()

    # The shared reporter reads public-operator identity and selection metadata.
    execute.func = executor.func
    execute.impl = executor.impl
    execute.impl_ref = executor.impl_ref
    execute.accuracy_checked = executor.accuracy_checked
    return execute


def _check_case(original):
    from sgl_kernel_npu.operators.mamba import causal_conv1d

    impl, impl_ref = get_test_impls()
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(causal_conv1d, impl)
    else:
        require_distinct_impls(causal_conv1d, impl, impl_ref)
        actual_inputs = clone_test_data(original, torch.device("npu"))
        reference_inputs = clone_test_data(original, torch.device("npu"))
        actual_state_ptr = actual_inputs["conv_states"].data_ptr()
        y_ref = causal_conv1d(**reference_inputs, impl=impl_ref)
        y = causal_conv1d(**actual_inputs, impl=impl)
        assert actual_inputs["conv_states"].data_ptr() == actual_state_ptr
        check_result(
            y,
            actual_inputs["conv_states"],
            y_ref,
            reference_inputs["conv_states"],
            valid_rows(original),
            f"{impl}-vs-{impl_ref}",
        )
        assert_readonly_inputs(actual_inputs, original)
        assert_readonly_inputs(reference_inputs, original)
    executor = _make_performance_executor(
        causal_conv1d, original, impl, impl_ref, not skip_accuracy
    )
    check_performance(
        executor,
        enable_perf=tuple(original["x"].shape) in perf_shape,
        warmup=PERF_WARMUP,
        active=PERF_ACTIVE,
    )


@pytest.mark.parametrize(
    "case_index", range(len(CASES)), ids=[case.name for case in CASES]
)
def test_causal_conv1d_prefill(case_index):
    _check_case(generate_test_data(case_index))


@pytest.mark.parametrize(
    "case_index", range(len(UPDATE_CASE_NAMES)), ids=UPDATE_CASE_NAMES
)
def test_causal_conv1d_update(case_index):
    _check_case(generate_update_test_data(case_index))


def _replay_cases(include_update):
    for index, case in enumerate(CASES):
        yield case.name, generate_test_data(index)
    if include_update:
        for index, name in enumerate(UPDATE_CASE_NAMES):
            yield name, generate_update_test_data(index)


def capture_legacy(path, include_update=False):
    """Run in the OLD package environment; save inputs and results for replay."""
    import sgl_kernel_npu  # noqa: F401 - register the old extension

    if not hasattr(torch.ops.npu, "causal_conv1d"):
        raise RuntimeError("Legacy torch.ops.npu.causal_conv1d is not registered")
    records = []
    for name, original in _replay_cases(include_update):
        data = clone_test_data(original, torch.device("npu"))
        y = torch.ops.npu.causal_conv1d(**data)
        assert_readonly_inputs(data, original)
        records.append(
            dict(
                name=name,
                inputs=original,
                y=y.cpu(),
                conv_states=data["conv_states"].cpu(),
            )
        )
    torch.save(
        dict(
            format=_UPDATE_REPLAY_FORMAT if include_update else _REPLAY_FORMAT,
            seed=SEED,
            torch_version=str(torch.__version__),
            cases=records,
        ),
        path,
    )
    print(f"Saved legacy replay ({len(records)} cases): {path}")


def compare_legacy(path):
    """Run in the NEW package environment; verify old/new/reference separately."""
    from sgl_kernel_npu.operators.mamba import causal_conv1d

    require_test_impl(causal_conv1d, "ascendc")
    replay = torch.load(path, map_location="cpu", weights_only=True)
    if (
        replay["format"] not in (_REPLAY_FORMAT, _UPDATE_REPLAY_FORMAT)
        or replay["seed"] != SEED
    ):
        raise ValueError("Unexpected legacy replay format or seed")
    expected_names = [case.name for case in CASES]
    if replay["format"] == _UPDATE_REPLAY_FORMAT:
        expected_names += UPDATE_CASE_NAMES
    if [record["name"] for record in replay["cases"]] != expected_names:
        raise ValueError("Legacy replay must contain the declared cases in order")
    for record in replay["cases"]:
        original = record["inputs"]
        data = clone_test_data(original, torch.device("npu"))
        reference = clone_test_data(original)
        y = causal_conv1d(**data, impl="ascendc")
        y_ref = causal_conv1d(**reference, impl="torch_native")
        mask = valid_rows(original)
        check_result(
            y,
            data["conv_states"],
            record["y"],
            record["conv_states"],
            mask,
            "new-vs-old",
        )
        check_result(
            y,
            data["conv_states"],
            y_ref,
            reference["conv_states"],
            mask,
            "new-vs-reference",
        )
        check_result(
            record["y"],
            record["conv_states"],
            y_ref,
            reference["conv_states"],
            mask,
            "old-vs-reference",
        )
        assert_readonly_inputs(data, original)
        assert_readonly_inputs(reference, original)
        print(f"PASS old/new/reference: {record['name']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture-legacy", type=Path, metavar="PATH")
    mode.add_argument("--compare-legacy", type=Path, metavar="PATH")
    parser.add_argument(
        "--include-update",
        action="store_true",
        help="Capture all 20 prefill/update cases; comparison infers cases from the replay format.",
    )
    args = parser.parse_args()
    if args.capture_legacy is not None:
        capture_legacy(args.capture_legacy.expanduser().resolve(), args.include_update)
    else:
        if args.include_update:
            parser.error("--include-update is only used with --capture-legacy")
        compare_legacy(args.compare_legacy.expanduser().resolve())


if __name__ == "__main__":
    main()
