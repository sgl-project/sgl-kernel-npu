"""L1 accuracy test for fused_shared_expert_mlp (run on NPU).

Compares against the production eager chain (F.linear MatMulV2 + npu_swiglu +
fused_sigmoid_mul_broadcast) stage by stage: inter (K1/SwiGLU), gate (K1 gate
dot), out (K2 + sigmoid mul).

Measured on NPU (see report §7): the K1 gate dot matches MatMulV2 BIT-FOR-BIT
(0.00 mismatch at all shapes) — tl.dot and MatMulV2 use identical cube
reduction. The inter/out differences (~1 bf16 ulp on a fraction of elements)
are inherited from the vendor elementwise ops' internal arithmetic
(npu_swiglu / fused_sigmoid_mul_broadcast are not bit-equal to the textbook
fp32 formula — see test_npu_swiglu_provenance). max_rel spikes in the report
come from near-zero elements, not real error. The strict gate for the fused
path is the e2e greedy-token check, not per-tensor bit equality.

Model dims covered:
  - Qwen3.5-35B-A3B : hidden 2048, shared_expert_intermediate 512
  - Qwen3.5-122B-A10B: hidden 3072, shared_expert_intermediate 1024

Usage: pytest tests/python/sgl_kernel_npu/test_fused_shared_expert.py -s
"""

import pytest
import torch

try:
    import torch_npu  # noqa: F401

    HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
except Exception:
    HAS_NPU = False

if HAS_NPU:
    from sgl_kernel_npu.activation.fused_sigmoid_mul import (
        fused_sigmoid_mul_broadcast,
    )
    from sgl_kernel_npu.fused.fused_shared_expert import (
        fused_shared_expert_mlp,
        fused_shared_expert_mlp_debug,
    )

pytestmark = pytest.mark.skipif(not HAS_NPU, reason="requires NPU")

# (IN, HALF, N, tag)
SHAPES = [
    (2048, 512, 2048, "35B-A3B"),
    (3072, 1024, 3072, "122B-A10B"),
]


def eager_chain(hidden, wgu, wd, wg, residual=None):
    gu = hidden @ wgu.t()
    inter = torch_npu.npu_swiglu(gu)
    down = inter @ wd.t()
    gate = hidden @ wg.t()
    out = fused_sigmoid_mul_broadcast(down, gate)
    if residual is not None:
        out = residual.add(out)
    return out, inter, gate


def bit_mismatch_rate(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.view(torch.int16) != b.view(torch.int16)).float().mean().item()


def stage_report(name, fused, ref):
    """Print bit-mismatch / max-abs / max-rel for one stage."""
    af, rf = fused.float(), ref.float()
    abs_diff = (af - rf).abs()
    rel_diff = abs_diff / rf.abs().clamp_min(1e-6)
    print(
        f"    {name}: bit-mismatch={bit_mismatch_rate(fused, ref):.2e} "
        f"max_abs={abs_diff.max().item():.2e} max_rel={rel_diff.max().item():.2e}"
    )


@pytest.mark.parametrize("IN,HALF,N,tag", SHAPES)
@pytest.mark.parametrize("M", [1, 8, 32, 128, 256])
def test_fused_shared_expert_mlp(M, IN, HALF, N, tag):
    torch.manual_seed(0)
    hidden = torch.randn(M, IN).to(torch.bfloat16).npu()
    wgu = (torch.randn(2 * HALF, IN) / IN**0.5).to(torch.bfloat16).npu()
    wd = (torch.randn(N, HALF) / HALF**0.5).to(torch.bfloat16).npu()
    wg = (torch.randn(1, IN) / IN**0.5).to(torch.bfloat16).npu()

    ref, ref_inter, ref_gate = eager_chain(hidden, wgu, wd, wg)
    out, inter, gate = fused_shared_expert_mlp_debug(hidden, wgu, wd, wg)

    print(f"\n{tag} M={M}:")
    stage_report("inter", inter, ref_inter)
    stage_report("gate ", gate.view(M, 1), ref_gate)
    stage_report("out  ", out, ref)
    assert not torch.isnan(out.float()).any(), "fused output contains NaN"
    torch.testing.assert_close(out.float(), ref.float(), rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize("IN,HALF,N,tag", SHAPES)
def test_npu_swiglu_provenance(IN, HALF, N, tag):
    """Isolate the vendor op: npu_swiglu vs the textbook fp32 SiLU formula on
    the SAME bf16 input. Any mismatch here is vendor-internal arithmetic and
    cannot be bit-matched by any reimplementation — it bounds how close the
    fused kernel can get to the eager chain at the inter stage."""
    torch.manual_seed(0)
    gu = torch.randn(32, 2 * HALF).to(torch.bfloat16).npu()
    ref = torch_npu.npu_swiglu(gu)
    x1 = gu[:, :HALF].float()
    x2 = gu[:, HALF:].float()
    manual = (x1 * torch.sigmoid(x1) * x2).to(torch.bfloat16)
    print(f"\nnpu_swiglu provenance {tag}:")
    stage_report("swiglu", ref, manual)


@pytest.mark.parametrize("IN,HALF,N,tag", SHAPES)
def test_fused_shared_expert_mlp_residual(IN, HALF, N, tag):
    M = 32
    torch.manual_seed(1)
    hidden = torch.randn(M, IN).to(torch.bfloat16).npu()
    wgu = (torch.randn(2 * HALF, IN) / IN**0.5).to(torch.bfloat16).npu()
    wd = (torch.randn(N, HALF) / HALF**0.5).to(torch.bfloat16).npu()
    wg = (torch.randn(1, IN) / IN**0.5).to(torch.bfloat16).npu()
    residual = torch.randn(M, N).to(torch.bfloat16).npu()

    ref, _, _ = eager_chain(hidden, wgu, wd, wg, residual=residual)
    out = fused_shared_expert_mlp(hidden, wgu, wd, wg, residual=residual)
    torch.testing.assert_close(out.float(), ref.float(), rtol=1.6e-2, atol=1e-2)
    print(f"\nresidual {tag}: out bit-mismatch={bit_mismatch_rate(out, ref):.2e}")
