"""Packed RGDR accuracy and profiling with independent mutable state.

Sources at upstream dc112d9ef68c:
  test_recurrent_gated_delta_rule.py::compatible_cases/TestCase.generate_input
  test_recurrent_gated_delta_rule_vs_triton.py::DEFAULT_SHAPES/make_inputs
All eight input families retain their original dimensions, BF16 data, state
capacity, gates and accepted-token settings. Compatible inputs have no fixed
seed in the source; benchmark inputs retain seed 42. Device defaults never move
CPU masks/metadata onto NPU. No Triton implementation is called by this suite.
Scenario: SGLang AscendGDNAttnBackend.fused_recurrent_gated_delta_rule_update
passes packed QKV, BF16 beta, per-sequence lengths and per-token state slots.
These source fixtures check that protocol, including S=1/S=8 and Dk != Dv;
their dimensions are not presented as captured production requests.
"""

from copy import deepcopy

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

WARMUP = 5
ACTIVE = 5


# Original B,S,Nk,Nv,Dk,Dv configurations; accepted=None for benchmark inputs.
perf_shape = [
    (2, 1, 8, 16, 128, 128),
    (4, 4, 8, 16, 128, 128),
    (8, 8, 8, 16, 128, 128),
    (2, 4, 8, 16, 64, 32),
]
CASES = [
    ("compatible", (2, 1, 8, 16, 128, 128)),
    ("compatible", (64, 2, 8, 16, 128, 128)),
    ("compatible", (32, 8, 8, 16, 128, 128)),
    ("compatible", (32, 8, 16, 32, 128, 128)),
    *[("benchmark", shape) for shape in perf_shape],
]
CASE_IDS = [
    f"legacy-{family}-b{b}-s{s}-nk{nk}-nv{nv}-dk{dk}-dv{dv}"
    for family, (b, s, nk, nv, dk, dv) in CASES
]
STATE_KEYS = ("recurrent_state", "intermediate_state")


def generate_test_data(family, shape, device):
    """Retain each source's allocation/RNG order; no smaller surrogate cases."""
    b, s, nk, nv, dk, dv = shape
    dtype = torch.bfloat16
    if family == "compatible":
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


def check_result(out, data, ref_out, ref_data, original, family):
    # Preserve both the public standards and the original stricter thresholds.
    atol, rtol = (2**-8, 2**-8) if family == "compatible" else (0.01, 0)
    output_shape = (
        *original["mix_qkv"].shape[:2],
        original["nv"],
        original["recurrent_state"].shape[-2],
    )
    for result, inputs in ((out, data), (ref_out, ref_data)):
        assert result.shape == output_shape, "output shape mismatch"
        assert result.dtype == inputs["mix_qkv"].dtype, "output dtype mismatch"
        assert result.device == inputs["mix_qkv"].device, "output device mismatch"
    lengths = original["actual_seq_lengths"].detach().cpu()
    valid = torch.arange(out.shape[1], device="cpu")[None, :] < lengths[:, None]
    got, want = out.detach().cpu()[valid], ref_out.detach().cpu()[valid]
    assert_accuracy(got, want)
    torch.testing.assert_close(got, want, atol=atol, rtol=rtol, equal_nan=False)
    work_name = (
        "intermediate_state"
        if original["intermediate_state"] is not None
        else "recurrent_state"
    )
    untouched = torch.ones(original[work_name].shape[0], dtype=torch.bool, device="cpu")
    indices = original["ssm_state_indices"].detach().cpu()
    untouched[indices[valid].long()] = False
    for key, before in original.items():
        if not isinstance(before, torch.Tensor):
            continue
        for value in (data[key], ref_data[key]):
            assert value.shape == before.shape, f"{key}: shape mismatch"
            assert value.dtype == before.dtype, f"{key}: dtype mismatch"
            assert value.device == before.device, f"{key}: device mismatch"
        got, want, before = (
            data[key].detach().cpu(),
            ref_data[key].detach().cpu(),
            before.detach().cpu(),
        )
        if key not in STATE_KEYS:
            assert torch.equal(got, before), f"modified read-only {key}"
            assert torch.equal(want, before), f"reference modified read-only {key}"
            continue
        # Compare every slot in bounded chunks, including the 520 MiB pool.
        chunk = max(1, (8 << 20) // (got[0].numel() * got.element_size()))
        for start in range(0, got.shape[0], chunk):
            end = start + chunk
            assert_accuracy(got[start:end], want[start:end], path=key)
            torch.testing.assert_close(
                got[start:end], want[start:end], atol=atol, rtol=rtol, equal_nan=False
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


@pytest.mark.parametrize("family,shape", CASES, ids=CASE_IDS)
def test_recurrent_gated_delta_rule(family, shape):
    from sgl_kernel_npu.operators.attention.gdn import recurrent_gated_delta_rule

    impl, impl_ref = get_test_impls()
    skip = bool(get_skip_acc_check())
    if skip:
        require_test_impl(recurrent_gated_delta_rule, impl)
    else:
        require_distinct_impls(recurrent_gated_delta_rule, impl, impl_ref)
    inputs = generate_test_data(family, shape, "npu")
    if not skip:
        actual, reference = deepcopy(inputs), deepcopy(inputs)
        identities = {
            key: (id(value), value.data_ptr())
            for key, value in actual.items()
            if key in STATE_KEYS and value is not None
        }
        ref_out = recurrent_gated_delta_rule(**reference, impl=impl_ref)
        out = recurrent_gated_delta_rule(**actual, impl=impl)
        for key, identity in identities.items():
            assert identity == (id(actual[key]), actual[key].data_ptr()), key
        check_result(out, actual, ref_out, reference, inputs, family)
        del actual, reference, out, ref_out
    # The four large compatible pools remain accuracy cases by default. The
    # standard CLI may explicitly enable profiling for them without changing shape.
    state_name = (
        "intermediate_state"
        if inputs["intermediate_state"] is not None
        else "recurrent_state"
    )
    states = None

    executor = make_executor(
        recurrent_gated_delta_rule,
        kwargs=inputs,
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip,
    )

    def execute():
        nonlocal states
        # Prepare every working pool before profiler collection, not in timed calls.
        if states is None:
            states = iter(
                [inputs[state_name].clone() for _ in range(1 + WARMUP + ACTIVE)]
            )
        executor.keywords[state_name] = next(states)
        return executor()

    execute.func = executor.func
    execute.impl = executor.impl
    execute.impl_ref = executor.impl_ref
    execute.accuracy_checked = executor.accuracy_checked
    check_performance(
        execute,
        enable_perf=family == "benchmark" and shape in perf_shape,
        warmup=WARMUP,
        active=ACTIVE,
    )
