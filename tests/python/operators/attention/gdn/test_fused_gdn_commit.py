"""Tests for attention.gdn.fused_gdn_commit public API."""

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

# Production Qwen3.8-27B (48 GDN layers, GQA=3) and Qwen3.5-9B (24 GDN layers, GQA=2)
perf_shape = [
    (48, 1, 4, 16, 48),
    (48, 32, 4, 16, 48),
    (48, 32, 8, 16, 48),
    (24, 32, 4, 16, 32),
]


def generate_data(num_layers, bs, k_ver, nk, nv, with_track):
    torch.npu.manual_seed(99)
    d = 128
    num_slots = max(16, 2 * bs + 6)
    ssm_states_all = (
        torch.randn((num_layers, num_slots, nv, d, d), dtype=torch.bfloat16) * 0.1
    )
    rec_k = torch.randn((num_layers, bs, k_ver, nk, d), dtype=torch.bfloat16) * 0.15
    rec_v = torch.randn((num_layers, bs, k_ver, nv, d), dtype=torch.bfloat16) * 0.15
    # Negative cumulative decay log-probs (monotonically decreasing along K)
    g_steps = -torch.rand((num_layers, bs, k_ver, nv), dtype=torch.float32) * 0.25
    rec_g = torch.cumsum(g_steps, dim=2)

    dst_slots = torch.arange(1, bs + 1, dtype=torch.int64)
    last_steps = torch.arange(bs, dtype=torch.int64) % k_ver
    if bs >= 4:
        # Exercise negative skip slot on last row
        dst_slots[-1] = -1

    if with_track:
        track_slots = torch.arange(bs + 2, 2 * bs + 2, dtype=torch.int64)
        track_steps = torch.clamp(last_steps - 1, min=0)
        track_steps[1::2] = -1
        if bs >= 4:
            track_slots[-1] = -1
    else:
        track_slots = None
        track_steps = None

    return (
        ssm_states_all,
        rec_k,
        rec_v,
        rec_g,
        dst_slots,
        last_steps,
        track_slots,
        track_steps,
    )


@pytest.mark.parametrize(
    "num_layers,bs,k_ver,nk,nv",
    [
        pytest.param(48, 1, 4, 16, 48, id="qwen38-27b-48L-mtp-bs1"),
        pytest.param(48, 32, 4, 16, 48, id="qwen38-27b-48L-mtp-bs32"),
        pytest.param(48, 32, 8, 16, 48, id="qwen38-27b-48L-dspark-bs32"),
        pytest.param(24, 32, 4, 16, 32, id="qwen35-9b-24L-gqa2-bs32"),
        pytest.param(4, 6, 12, 8, 32, id="edge-gqa4-k12"),
        pytest.param(4, 5, 3, 16, 16, id="edge-gqa1-k3"),
    ],
)
@pytest.mark.parametrize("with_track", [False, True])
def test_fused_gdn_commit(num_layers, bs, k_ver, nk, nv, with_track):
    from sgl_kernel_npu.operators.attention.gdn import fused_gdn_commit

    (
        ssm_states_all,
        rec_k,
        rec_v,
        rec_g,
        dst_slots,
        last_steps,
        track_slots,
        track_steps,
    ) = generate_data(num_layers, bs, k_ver, nk, nv, with_track)
    impl, impl_ref = get_test_impls()
    kw = {"track_slots": track_slots, "track_steps": track_steps}
    args = (ssm_states_all, rec_k, rec_v, rec_g, dst_slots, last_steps)

    accuracy_checked = False
    if not get_skip_acc_check():
        require_distinct_impls(fused_gdn_commit, impl, impl_ref)
        ssm_ref = ssm_states_all.clone()
        ssm_act = ssm_states_all.clone()
        ret_ref = fused_gdn_commit(
            ssm_ref, rec_k, rec_v, rec_g, dst_slots, last_steps, **kw, impl=impl_ref
        )
        ret_act = fused_gdn_commit(
            ssm_act, rec_k, rec_v, rec_g, dst_slots, last_steps, **kw, impl=impl
        )
        assert ret_act.data_ptr() == ssm_act.data_ptr()
        assert ret_ref.data_ptr() == ssm_ref.data_ptr()
        assert_accuracy(
            ssm_act,
            ssm_ref,
            standard={"torch.bfloat16": AccuracyStandard(atol=2e-2, rtol=2e-2)},
        )
        accuracy_checked = True

    executor = make_executor(
        fused_gdn_commit,
        args=args,
        kwargs=kw,
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=accuracy_checked,
    )
    check_performance(
        executor,
        enable_perf=((num_layers, bs, k_ver, nk, nv) in perf_shape and not with_track),
    )
