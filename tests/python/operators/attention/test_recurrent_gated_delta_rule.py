"""Packed RGDR accuracy, independent-state profiling, and old/new replay.

Sources at upstream dc112d9ef68c:
  test_recurrent_gated_delta_rule.py::compatible_cases/TestCase.generate_input
  test_recurrent_gated_delta_rule_vs_triton.py::DEFAULT_SHAPES/make_inputs
All eight input families retain their original dimensions, BF16 data, state
capacity, gates and accepted-token settings. Compatible inputs have no fixed
seed in the source; benchmark inputs retain seed 42. Device defaults never move
CPU masks/metadata onto NPU. No Triton implementation is called by this suite.
"""

import argparse
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path

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
from tests.python.utils.performance import (
    benchmark_npu,
    check_performance,
    write_result,
)

WARMUP = 5
ACTIVE = 5
FORMAT = "sgl-rgdr-replay-v1"


@dataclass(frozen=True)
class Case:
    family: str
    shape: tuple[int, int, int, int, int, int]  # B,S,Nk,Nv,Dk,Dv

    @property
    def name(self):
        b, s, nk, nv, dk, dv = self.shape
        return f"legacy-{self.family}-b{b}-s{s}-nk{nk}-nv{nv}-dk{dk}-dv{dv}"


CASES = [
    Case("compatible", (2, 1, 8, 16, 128, 128)),
    Case("compatible", (64, 2, 8, 16, 128, 128)),
    Case("compatible", (32, 8, 8, 16, 128, 128)),
    Case("compatible", (32, 8, 16, 32, 128, 128)),
    Case("benchmark", (2, 1, 8, 16, 128, 128)),
    Case("benchmark", (4, 4, 8, 16, 128, 128)),
    Case("benchmark", (8, 8, 8, 16, 128, 128)),
    Case("benchmark", (2, 4, 8, 16, 64, 32)),
]

# Keep the original benchmark's four shapes and accepted=None input family.
perf_shape = [
    (2, 1, 8, 16, 128, 128),
    (4, 4, 8, 16, 128, 128),
    (8, 8, 8, 16, 128, 128),
    (2, 4, 8, 16, 64, 32),
]


def make_inputs(case, device):
    """Retain each source's allocation/RNG order; no smaller surrogate cases."""
    b, s, nk, nv, dk, dv = case.shape
    dtype = torch.bfloat16
    if case.family == "compatible":
        intermediate = (
            torch.rand((65, s, nv, dv, dk), dtype=dtype, device=device)
            if s > 1
            else None
        )
        recurrent = torch.rand((65, nv, dv, dk), dtype=dtype, device=device)
        v = torch.rand((b, s, nv, dv), dtype=dtype, device=device)
        q = torch.rand((b, s, nk, dk), dtype=dtype, device=device)
        k = torch.rand((b, s, nk, dk), dtype=dtype, device=device)
        g = torch.rand((b, s, nv), dtype=torch.float32, device=device) * -1.0
        beta = torch.rand((b, s, nv), dtype=dtype, device=device)
        cache = torch.randperm(65, device=device)[:b].to(torch.int32)
        offsets = torch.arange(s, dtype=torch.int32, device=device)
        indices = (cache[:, None] * s + offsets[None, :]).contiguous()
        # The old script generates accepted counts on CPU before copying to NPU.
        accepted = torch.randint(1, s + 1, (b,), device="cpu").to(
            device=device, dtype=torch.int32
        )
    else:
        torch.manual_seed(42)
        q = torch.randn((b, s, nk, dk), dtype=dtype, device=device)
        k = torch.randn((b, s, nk, dk), dtype=dtype, device=device)
        v = torch.randn((b, s, nv, dv), dtype=dtype, device=device)
        recurrent = torch.randn((b * s, nv, dv, dk), dtype=dtype, device=device) * 0.1
        cache = torch.arange(b, dtype=torch.int32, device=device)
        offsets = torch.arange(s, dtype=torch.int32, device=device)
        indices = (cache[:, None] * s + offsets[None, :]).contiguous()
        scratch = torch.zeros_like(recurrent)
        if s > 1:
            scratch[indices[:, 0].to(torch.int64)] = recurrent[cache.to(torch.int64)]
        intermediate = scratch if s > 1 else None
        g = -(0.05 + 0.95 * torch.rand((b, s, nv), dtype=torch.float32, device=device))
        beta = torch.randn((b, s, nv), dtype=dtype, device=device)
        accepted = None
    mix = torch.cat([q.flatten(2), k.flatten(2), v.flatten(2)], dim=-1).contiguous()
    return dict(
        mix_qkv=mix,
        recurrent_state=recurrent,
        beta=beta,
        scale=dk**-0.5,
        actual_seq_lengths=torch.full((b,), s, dtype=torch.int32, device=device),
        ssm_state_indices=indices,
        nk=nk,
        nv=nv,
        intermediate_state=(
            intermediate.view(-1, nv, dv, dk) if intermediate is not None else None
        ),
        cache_indices=cache if s > 1 else None,
        num_accepted_tokens=accepted,
        g=g,
        gk=None,
    )


def clone_inputs(data, device=None):
    return {
        key: (
            value.to(device=device or value.device).clone()
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in data.items()
    }


def mutable_states(data):
    return {
        key: data[key]
        for key in ("recurrent_state", "intermediate_state")
        if data[key] is not None
    }


def valid_tokens(data):
    mask = torch.zeros(data["mix_qkv"].shape[:2], dtype=torch.bool, device="cpu")
    for row, length in enumerate(data["actual_seq_lengths"].detach().cpu().tolist()):
        mask[row, :length] = True
    return mask


def check_unchanged(data, before):
    for key, value in data.items():
        if isinstance(value, torch.Tensor) and key not in (
            "recurrent_state",
            "intermediate_state",
        ):
            assert torch.equal(
                value.detach().cpu(), before[key].detach().cpu()
            ), f"modified read-only {key}"


def check_result(out, data, ref_out, ref_data, original, case, *, exact=False):
    mask = valid_tokens(original)
    got, want = out.detach().cpu()[mask], ref_out.detach().cpu()[mask]
    if exact:
        assert torch.equal(got, want), "old/new outputs differ bitwise"
    else:
        assert_accuracy(got, want)
        # Preserve existing stricter diagnostics as assertions; a wider public
        # BF16 default must not hide regressions in the original eight families.
        atol, rtol = (2**-8, 2**-8) if case.family == "compatible" else (0.01, 0)
        torch.testing.assert_close(got, want, atol=atol, rtol=rtol, equal_nan=False)
    del got, want
    intermediate = original["intermediate_state"] is not None
    work_name = "intermediate_state" if intermediate else "recurrent_state"
    work = data[work_name]
    untouched = torch.ones(work.shape[0], dtype=torch.bool, device="cpu")
    indices = original["ssm_state_indices"].detach().cpu()
    for seq, length in enumerate(
        original["actual_seq_lengths"].detach().cpu().tolist()
    ):
        untouched[indices[seq, :length].to(torch.int64)] = False
    for key, value in mutable_states(data).items():
        got, want = value.detach().cpu(), ref_data[key].detach().cpu()
        before = original[key].detach().cpu()
        # Cover every slot without allocating FP32 comparison temporaries for
        # the entire 520 MiB compatible pool at once. This changes no test shape.
        slot_bytes = got[0].numel() * got.element_size()
        chunk = max(1, (8 << 20) // slot_bytes)
        for start in range(0, got.shape[0], chunk):
            end = start + chunk
            if exact:
                assert torch.equal(
                    got[start:end], want[start:end]
                ), f"old/new {key} differs bitwise"
            else:
                assert_accuracy(got[start:end], want[start:end], path=key)
                torch.testing.assert_close(
                    got[start:end],
                    want[start:end],
                    atol=atol,
                    rtol=rtol,
                    equal_nan=False,
                )
            if key == work_name:
                mask = untouched[start:end]
                assert torch.equal(
                    got[start:end][mask], before[start:end][mask]
                ), f"modified unused {key} slots"
            else:
                assert torch.equal(
                    got[start:end], before[start:end]
                ), "MTP changed the recurrent seed pool"
        del before
        del got, want
    check_unchanged(data, original)
    check_unchanged(ref_data, original)


def independent_state_executor(operator, inputs, *, impl, impl_ref, accuracy_checked):
    """Prepare writable pools in the unprofiled call; never reset inside timing."""
    executor = make_executor(
        operator,
        kwargs=inputs,
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=accuracy_checked,
    )
    key = (
        "intermediate_state"
        if inputs["intermediate_state"] is not None
        else "recurrent_state"
    )
    states = None

    def execute():
        nonlocal states
        if states is None:
            states = iter([inputs[key].clone() for _ in range(1 + WARMUP + ACTIVE)])
        executor.keywords[key] = next(states)
        return executor()

    execute.func = executor.func
    execute.impl = executor.impl
    execute.impl_ref = executor.impl_ref
    execute.accuracy_checked = executor.accuracy_checked
    return execute


@pytest.mark.parametrize("case", [pytest.param(case, id=case.name) for case in CASES])
def test_recurrent_gated_delta_rule(case):
    from sgl_kernel_npu.operators.attention import recurrent_gated_delta_rule

    impl, impl_ref = get_test_impls()
    skip = bool(get_skip_acc_check())
    if skip:
        require_test_impl(recurrent_gated_delta_rule, impl)
    else:
        require_distinct_impls(recurrent_gated_delta_rule, impl, impl_ref)
    inputs = make_inputs(case, "npu")
    if not skip:
        actual, reference = clone_inputs(inputs), clone_inputs(inputs)
        identities = {
            key: (id(value), value.data_ptr())
            for key, value in mutable_states(actual).items()
        }
        ref_out = recurrent_gated_delta_rule(**reference, impl=impl_ref)
        out = recurrent_gated_delta_rule(**actual, impl=impl)
        for key, value in mutable_states(actual).items():
            assert identities[key] == (id(value), value.data_ptr()), key
        check_result(out, actual, ref_out, reference, inputs, case)
        del actual, reference, out, ref_out
    # The four large compatible pools remain accuracy cases by default. The
    # standard CLI may explicitly enable profiling for them without changing shape.
    executor = independent_state_executor(
        recurrent_gated_delta_rule,
        inputs,
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip,
    )
    check_performance(
        executor,
        enable_perf=case.family == "benchmark" and case.shape in perf_shape,
        warmup=WARMUP,
        active=ACTIVE,
    )


def _legacy(*, impl, **kwargs):
    import sgl_kernel_npu  # noqa: F401 - load the old wheel's extension

    return torch.ops.npu.recurrent_gated_delta_rule(**kwargs)


def _benchmark_replay(operator, inputs, directory, case, impl, *, accuracy_checked):
    import torch_npu

    executor = independent_state_executor(
        operator,
        inputs,
        impl=impl,
        impl_ref="torch_native",
        accuracy_checked=accuracy_checked,
    )
    value = benchmark_npu(
        executor, directory / f"{case.name}-{impl}", warmup=WARMUP, active=ACTIVE
    )
    write_result(
        directory / "performance.csv",
        case=case.name,
        shape=str(case.shape),
        impl=impl,
        accuracy_checked=accuracy_checked,
        metric="mean_device_task_us",
        time_us=f"{value:.3f}",
        warmup=WARMUP,
        active=ACTIVE,
        device=torch.npu.get_device_name(),
        torch_version=str(torch.__version__),
        torch_npu_version=str(torch_npu.__version__),
        sdk_root=os.environ.get("ASCEND_HOME_PATH", ""),
    )


def capture(directory, benchmark=False):
    import sgl_kernel_npu  # noqa: F401

    directory.mkdir(parents=True, exist_ok=False)
    for case in CASES:
        original = make_inputs(case, "npu")
        actual = clone_inputs(original)
        out = _legacy(impl="legacy", **actual)
        check_unchanged(actual, original)
        torch.save(
            dict(
                format=FORMAT,
                name=case.name,
                inputs=clone_inputs(original, "cpu"),
                out=out.detach().cpu(),
                states={k: v.detach().cpu() for k, v in mutable_states(actual).items()},
            ),
            directory / f"{case.name}.pt",
        )
        del actual, out
        if benchmark and case.family == "benchmark":
            # Capture records the old result; reference validation happens only
            # during compare, so do not claim an accuracy pass in this CSV.
            _benchmark_replay(
                _legacy, original, directory, case, "legacy", accuracy_checked=False
            )
        del original
        gc.collect()
    (directory / "manifest.json").write_text(
        json.dumps(dict(format=FORMAT, cases=[c.name for c in CASES]), indent=2)
    )


def compare(directory, benchmark_directory=None):
    from sgl_kernel_npu.operators.attention import recurrent_gated_delta_rule

    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest != dict(format=FORMAT, cases=[c.name for c in CASES]):
        raise ValueError(
            "Replay manifest does not match the eight original input families"
        )
    if benchmark_directory is not None:
        benchmark_directory.mkdir(parents=True, exist_ok=False)
    for case in CASES:
        record = torch.load(
            directory / f"{case.name}.pt", map_location="cpu", weights_only=True
        )
        if record["format"] != FORMAT or record["name"] != case.name:
            raise ValueError("Replay format or case identity mismatch")
        original = record["inputs"]
        actual, reference = clone_inputs(original, "npu"), clone_inputs(original, "npu")
        out = recurrent_gated_delta_rule(**actual, impl="ascendc")
        ref_out = recurrent_gated_delta_rule(**reference, impl="torch_native")
        old = {**original, **record["states"]}
        check_result(out, actual, record["out"], old, original, case, exact=True)
        check_result(out, actual, ref_out, reference, original, case)
        check_result(record["out"], old, ref_out, reference, original, case)
        print(f"PASS old/new/reference: {case.name}")
        del actual, reference, out, ref_out, old
        if benchmark_directory is not None and case.family == "benchmark":
            _benchmark_replay(
                recurrent_gated_delta_rule,
                clone_inputs(original, "npu"),
                benchmark_directory,
                case,
                "ascendc",
                accuracy_checked=True,
            )
        del original, record
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture-legacy", type=Path)
    mode.add_argument("--compare-legacy", type=Path)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Also profile the four original benchmark configurations",
    )
    parser.add_argument(
        "--prof-dir",
        type=Path,
        help="Required with --compare-legacy --benchmark; must be a new directory",
    )
    args = parser.parse_args()
    import torch_npu  # noqa: F401

    if args.capture_legacy is not None:
        if args.prof_dir is not None:
            parser.error(
                "Legacy profiles are written under --capture-legacy; omit --prof-dir"
            )
        capture(args.capture_legacy.expanduser().resolve(), args.benchmark)
    else:
        if args.benchmark != (args.prof_dir is not None):
            parser.error(
                "--compare-legacy requires --benchmark and --prof-dir together"
            )
        compare(
            args.compare_legacy.expanduser().resolve(),
            args.prof_dir.expanduser().resolve() if args.prof_dir else None,
        )


if __name__ == "__main__":
    main()
