"""Tests for attention.gdn.fused_gdn_prefill public API."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Production Qwen3.8-27B (16/48, GQA=3) and Qwen3.5-9B / Qwen3.6-35B-A3B (16/32, GQA=2)
perf_shape = [
    (16, 48, (733,)),
    (16, 48, (512, 512)),
    (16, 32, (512,)),
]


def generate_data(nk, nv, seq_lens, pad_extra, with_ckpt):
    torch.npu.manual_seed(123)
    d = 128
    qkv_dim = (2 * nk + nv) * d
    total_valid = sum(seq_lens)
    t_padded = total_valid + pad_extra
    b_size = len(seq_lens)

    mixed_qkv = torch.randn((t_padded, qkv_dim), dtype=torch.bfloat16) * 0.5
    a = torch.randn((t_padded, nv), dtype=torch.bfloat16) * 0.3
    b_in = torch.randn((t_padded, nv), dtype=torch.bfloat16) * 0.5
    A_log = torch.randn((nv,), dtype=torch.float32) * 0.2 - 1.0
    dt_bias = torch.randn((nv,), dtype=torch.bfloat16) * 0.1

    num_slots = max(16, b_size * 2 + 4)
    ssm_states = torch.randn((num_slots, nv, d, d), dtype=torch.bfloat16) * 0.1
    cache_indices = torch.arange(1, b_size + 1, dtype=torch.int32)

    cu_list = [0]
    for s_len in seq_lens:
        cu_list.append(cu_list[-1] + s_len)
    cu_seqlens = torch.tensor(cu_list, dtype=torch.int32)

    from sgl_kernel_npu.operators.attention.gdn import build_gdn_prefill_chunk_meta

    chunk_indices, chunk_offsets, _, tv = build_gdn_prefill_chunk_meta(
        seq_lens, device=mixed_qkv.device, chunk_size=64
    )

    if with_ckpt:
        ckpt_chunk_idx = torch.tensor(
            [max(0, (s_len // 64) - 1) for s_len in seq_lens], dtype=torch.int32
        )
        ckpt_dst_slots = torch.arange(b_size + 2, 2 * b_size + 2, dtype=torch.int32)
        if b_size >= 3:
            ckpt_dst_slots[2] = -1
            ckpt_chunk_idx[2] = -1
    else:
        ckpt_chunk_idx = None
        ckpt_dst_slots = None

    return (
        mixed_qkv,
        a,
        b_in,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        nk,
        nv,
        tv,
        ckpt_chunk_idx,
        ckpt_dst_slots,
    )


@pytest.mark.parametrize(
    "nk,nv,seq_lens",
    [
        pytest.param(16, 48, (733,), id="qwen38-27b-gqa3-single"),
        pytest.param(16, 48, (129, 64, 15), id="qwen38-27b-gqa3-varlen"),
        pytest.param(16, 32, (130, 45), id="qwen35-9b-gqa2x2-varlen"),
        pytest.param(16, 16, (70, 17), id="generic-gqa1-varlen"),
        pytest.param(8, 32, (65, 13), id="generic-gqa4-varlen"),
    ],
)
@pytest.mark.parametrize("pad_extra", [0, 29])
@pytest.mark.parametrize("with_ckpt", [False, True])
def test_fused_gdn_prefill(nk, nv, seq_lens, pad_extra, with_ckpt):
    from sgl_kernel_npu.operators.attention.gdn import fused_gdn_prefill

    (
        mixed_qkv,
        a,
        b_in,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        num_k_heads,
        num_v_heads,
        total_valid,
        ckpt_chunk_idx,
        ckpt_dst_slots,
    ) = generate_data(nk, nv, seq_lens, pad_extra, with_ckpt)

    impl, impl_ref = get_test_impls()
    accuracy_checked = get_skip_acc_check() is not True
    if accuracy_checked:
        require_distinct_impls(fused_gdn_prefill, impl, impl_ref)
    else:
        require_test_impl(fused_gdn_prefill, impl)

    if accuracy_checked:
        ref_ssm = ssm_states.clone()
        act_ssm = ssm_states.clone()
        o_ref = fused_gdn_prefill(
            mixed_qkv,
            a,
            b_in,
            A_log,
            dt_bias,
            ref_ssm,
            cache_indices,
            cu_seqlens,
            chunk_indices,
            chunk_offsets,
            num_k_heads,
            num_v_heads,
            total_valid_tokens=total_valid,
            ckpt_chunk_idx=ckpt_chunk_idx,
            ckpt_dst_slots=ckpt_dst_slots,
            impl=impl_ref,
        )
        o_act = fused_gdn_prefill(
            mixed_qkv,
            a,
            b_in,
            A_log,
            dt_bias,
            act_ssm,
            cache_indices,
            cu_seqlens,
            chunk_indices,
            chunk_offsets,
            num_k_heads,
            num_v_heads,
            total_valid_tokens=total_valid,
            ckpt_chunk_idx=ckpt_chunk_idx,
            ckpt_dst_slots=ckpt_dst_slots,
            impl=impl,
        )
        if pad_extra > 0:
            assert o_act[total_valid:].abs().max().item() == 0.0
        cos_o = F.cosine_similarity(
            o_act[:total_valid].float().flatten(),
            o_ref[:total_valid].float().flatten(),
            dim=0,
        ).item()
        assert cos_o > 0.9998, f"Output cosine similarity too low: {cos_o}"
        for slot in cache_indices.tolist():
            cos_h = F.cosine_similarity(
                act_ssm[slot].float().flatten(),
                ref_ssm[slot].float().flatten(),
                dim=0,
            ).item()
            assert cos_h > 0.9998, f"Final state cosine similarity too low: {cos_h}"
        if with_ckpt:
            for dst in ckpt_dst_slots.tolist():
                if dst >= 0:
                    cos_c = F.cosine_similarity(
                        act_ssm[dst].float().flatten(),
                        ref_ssm[dst].float().flatten(),
                        dim=0,
                    ).item()
                    assert (
                        cos_c > 0.9998
                    ), f"Ckpt state cosine similarity too low: {cos_c}"

    warmup, active = 3, 5
    states = None

    def executor():
        nonlocal states
        if states is None:
            states = iter([ssm_states.clone() for _ in range(1 + warmup + active)])
        return fused_gdn_prefill(
            mixed_qkv,
            a,
            b_in,
            A_log,
            dt_bias,
            next(states),
            cache_indices,
            cu_seqlens,
            chunk_indices,
            chunk_offsets,
            num_k_heads,
            num_v_heads,
            total_valid_tokens=total_valid,
            ckpt_chunk_idx=ckpt_chunk_idx,
            ckpt_dst_slots=ckpt_dst_slots,
            impl=impl,
        )

    executor.func = fused_gdn_prefill
    executor.impl = impl
    executor.impl_ref = impl_ref
    executor.accuracy_checked = accuracy_checked
    check_performance(
        executor,
        enable_perf=(
            (nk, nv, seq_lens) in perf_shape and pad_extra == 0 and not with_ckpt
        ),
        warmup=warmup,
        active=active,
    )


def test_fused_gdn_prefill_strongly_correlated_keys():
    """Verify FP16 2-level Euler solve_tril accuracy under strongly correlated keys."""
    from sgl_kernel_npu.operators.attention.gdn import (
        build_gdn_prefill_chunk_meta,
        fused_gdn_prefill,
    )

    torch.npu.manual_seed(0)
    nk, nv, d, seq_len = 16, 48, 128, 256
    A_log = torch.full((nv,), -3.0, dtype=torch.float32)
    dt_bias = torch.full((nv,), -2.0, dtype=torch.bfloat16)
    u1 = torch.randn((1, nk * d), dtype=torch.bfloat16)
    q_raw = u1 + 0.2 * torch.randn((seq_len, nk * d), dtype=torch.bfloat16)
    k_raw = u1 + 0.2 * torch.randn((seq_len, nk * d), dtype=torch.bfloat16)
    v_raw = torch.randn((seq_len, nv * d), dtype=torch.bfloat16) * 3.0
    mixed_qkv = torch.cat([q_raw, k_raw, v_raw], dim=-1).contiguous()
    a = torch.full((seq_len, nv), -4.0, dtype=torch.bfloat16)
    b_in = torch.full((seq_len, nv), 2.5, dtype=torch.bfloat16)
    ssm_ref = torch.zeros((1, nv, d, d), dtype=torch.bfloat16)
    ssm_act = ssm_ref.clone()
    cu = torch.tensor([0, seq_len], dtype=torch.int32)
    cache_idx = torch.tensor([0], dtype=torch.int32)
    cidx, coff, _, tv = build_gdn_prefill_chunk_meta([seq_len], mixed_qkv.device)

    impl, impl_ref = get_test_impls()
    if get_skip_acc_check() is True:
        require_test_impl(fused_gdn_prefill, impl)
        return
    require_distinct_impls(fused_gdn_prefill, impl, impl_ref)
    o_ref = fused_gdn_prefill(
        mixed_qkv,
        a,
        b_in,
        A_log,
        dt_bias,
        ssm_ref,
        cache_idx,
        cu,
        cidx,
        coff,
        nk,
        nv,
        tv,
        impl=impl_ref,
    )
    o_act = fused_gdn_prefill(
        mixed_qkv,
        a,
        b_in,
        A_log,
        dt_bias,
        ssm_act,
        cache_idx,
        cu,
        cidx,
        coff,
        nk,
        nv,
        tv,
        impl=impl,
    )
    o_rel = (o_act.float() - o_ref.float()).norm().item() / o_ref.float().norm().item()
    h_rel = (ssm_act[0].float() - ssm_ref[0].float()).norm().item() / ssm_ref[
        0
    ].float().norm().item()
    assert o_rel <= 0.025, f"Correlated-key output rel L2 too high: {o_rel}"
    assert h_rel <= 0.015, f"Correlated-key state rel L2 too high: {h_rel}"
