import os
import time
from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel_npu.fla.utils import (
    fused_qkvzba_split_reshape_cat,
    fused_qkvzba_split_reshape_cat_torch,
)

LAUNCH_MIN = 2
LAUNCH_CNT = max(2, LAUNCH_MIN)
device = "npu"


def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    else:
        assert error_rate < ratio, msg


def print_diff(name, ref, tri, atol=0.005):
    abs_diff = torch.abs(ref - tri)
    max_abs_diff = abs_diff.max().item()
    print(f"[{name}] Max absolute difference: {max_abs_diff:.6f}")
    if max_abs_diff > atol:
        print(f"Exceeds tolerance ({atol})!")


@pytest.mark.parametrize(
    ("B", "num_heads_qk", "num_heads_v", "head_qk", "head_v", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-num_heads_qk{}-num_heads_v{}-head_qk{}-head_v{}-{}".format(*test),
        )
        for test in [
            # (B, num_heads_qk, num_heads_v, head_qk, head_v, dtype)
            (1, 4, 8, 128, 128, torch.float16),
            (2, 4, 8, 128, 128, torch.float16),
            (4, 4, 8, 128, 128, torch.float16),
            (8, 4, 8, 128, 128, torch.float16),
            (1, 4, 8, 128, 128, torch.bfloat16),
            (2, 4, 8, 128, 128, torch.bfloat16),
            (4, 4, 8, 128, 128, torch.bfloat16),
            (8, 4, 8, 128, 128, torch.bfloat16),
            (1, 4, 8, 128, 128, torch.float32),
            (2, 4, 8, 128, 128, torch.float32),
            (4, 4, 8, 128, 128, torch.float32),
            (8, 4, 8, 128, 128, torch.float32),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_FUSED_QKVZBA") == "1",
    reason="Skipping test_fused_qkvzba because SKIP_TEST_FUSED_QKVZBA is set",
)
def test_fused_qkvzba(
    B: int,
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
    dtype: torch.dtype,
):
    if head_v not in [64, 128, 256]:
        pytest.skip(reason="fused_qkvzba only supports head_v in [64,128,256]")

    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    mixed_qkvz = torch.randn((B, 3072), dtype=dtype)
    mixed_ba = torch.randn((B, 16), dtype=dtype)

    mixed_qkvz, mixed_ba = map(
        lambda x: x.to(device).requires_grad_(), (mixed_qkvz, mixed_ba)
    )

    tri_mixed_qkv, tri_z, tri_b, tri_a = None, None, None, None
    begin_time = 0
    for i in range(LAUNCH_CNT):
        if i == 1 or LAUNCH_CNT == 1:
            torch.npu.synchronize()
            begin_time = time.time()

        tri_mixed_qkv, tri_z, tri_b, tri_a = fused_qkvzba_split_reshape_cat(
            mixed_qkvz=mixed_qkvz.clone(),
            mixed_ba=mixed_ba.clone(),
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_qk=head_qk,
            head_v=head_v,
        )

    torch.npu.synchronize()
    use_time = time.time() - begin_time
    avg_time = use_time * 1000 / (LAUNCH_CNT - 1) if LAUNCH_CNT > 1 else use_time * 1000
    print(f"[DEBUG] fused_qkvzba (target) using time: {avg_time:.2f} ms")

    ref_mixed_qkv, ref_z, ref_b, ref_a = None, None, None, None
    begin_time = 0
    for i in range(LAUNCH_CNT):
        if i == 1 or LAUNCH_CNT == 1:
            torch.npu.synchronize()
            begin_time = time.time()

        ref_mixed_qkv, ref_z, ref_b, ref_a = fused_qkvzba_split_reshape_cat_torch(
            mixed_qkvz=mixed_qkvz.clone(),
            mixed_ba=mixed_ba.clone(),
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_qk=head_qk,
            head_v=head_v,
        )

    torch.npu.synchronize()
    use_time = time.time() - begin_time
    avg_time = use_time * 1000 / (LAUNCH_CNT - 1) if LAUNCH_CNT > 1 else use_time * 1000
    print(f"[DEBUG] fused_qkvzba (torch) using time: {avg_time:.2f} ms")

    print_diff("mixed_qkv", ref_mixed_qkv, tri_mixed_qkv, 0.005)
    print_diff("z", ref_z, tri_z, 0.005)
    print_diff("b", ref_b, tri_b, 0.005)
    print_diff("a", ref_a, tri_a, 0.005)

    assert_close("mixed_qkv", ref_mixed_qkv, tri_mixed_qkv, 0.005, err_atol=1e-6)
    assert_close("z", ref_z, tri_z, 0.005, err_atol=1e-6)
    assert_close("b", ref_b, tri_b, 0.005, err_atol=1e-6)
    assert_close("a", ref_a, tri_a, 0.005, err_atol=1e-6)
