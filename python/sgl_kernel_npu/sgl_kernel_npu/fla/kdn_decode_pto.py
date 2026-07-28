"""PTO-ISA backend for the KDA recurrent decode step (KDN_DECODE_PTO_BACKEND=1).

Drop-in replacement for the triton ``fused_sigmoid_gating_delta_rule_update_npu``
decode path, backed by ``torch.ops.npu.kdn_decode``.

Two differences from the triton kernel are worth knowing:

* **The gating is not fused yet.** The triton kernel computes
  ``g = -exp(A_log) * softplus(a + dt_bias)`` and ``beta = sigmoid(b)`` inside
  the kernel; here they are torch ops in front of the launch. That costs a
  handful of extra elementwise kernels per step, and it rounds ``g`` to the
  kernel's fp16 wire format, where the triton path keeps it fp32. Folding the
  gating into the kernel is the planned follow-up.
* **The state layout is taken as V-major** ``[slots, H, V, K]``, matching
  sglang's ``temporal_state`` pool (``mem_cache/memory_pool.py``), the prefill
  ``chunk_delta_h`` block pointer ``(V, K)/(K, 1)``, and the CUDA reference
  decode kernel. Note the triton NPU kernel indexes the same pool K-major.
"""

import os
from typing import Optional

import torch
import torch.nn.functional as F

HEAD_DIM = 128
_PTO_ENV = "KDN_DECODE_PTO_BACKEND"


def pto_backend_enabled() -> bool:
    return os.environ.get(_PTO_ENV, "0") not in ("", "0", "false", "False")


def kdn_decode_pto(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = True,
) -> torch.Tensor:
    """Run one recurrent decode step; returns ``o`` shaped like ``v``."""
    if not is_kda:
        raise NotImplementedError(
            "the PTO decode backend implements the per-channel KDA gate only (is_kda=True)"
        )
    if cu_seqlens is None:
        raise NotImplementedError("the PTO decode backend requires cu_seqlens")

    B, T, H, K = k.shape
    V = v.shape[-1]
    HV = v.shape[2]
    if HV != H:
        raise NotImplementedError(
            f"the PTO decode backend does not implement GQA grouping (H={H}, HV={HV})"
        )
    if K != HEAD_DIM or V != HEAD_DIM:
        raise NotImplementedError(f"the PTO decode backend supports K=V=128, got K={K}, V={V}")

    if scale is None:
        scale = K**-0.5

    # ---- gating, still in torch (see module docstring) ----------------------
    # g = -exp(A_log) * softplus(a + dt_bias), per (token, head, k) for KDA.
    a_f32 = a.reshape(B, T, HV, K).float()
    dt_bias_f32 = dt_bias.reshape(HV, K).float()
    # F.softplus(x, beta, threshold) is exactly the triton branch:
    #   beta*x <= threshold ? log1p(exp(beta*x))/beta : x
    softplus_x = F.softplus(a_f32 + dt_bias_f32, beta=softplus_beta, threshold=softplus_threshold)
    g = (-A_log.reshape(1, 1, HV, 1).float().exp() * softplus_x).to(torch.float16)
    beta = torch.sigmoid(b.reshape(B, T, HV).float()).to(torch.float16)

    # ---- wire format: fp16, contiguous -------------------------------------
    q16 = q.reshape(B, T, H, K).to(torch.float16).contiguous()
    k16 = k.reshape(B, T, H, K).to(torch.float16).contiguous()
    v16 = v.reshape(B, T, HV, V).to(torch.float16).contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    out = torch.zeros_like(v16)

    cu32 = cu_seqlens.to(torch.int32).contiguous()
    idx32 = initial_state_indices.to(torch.int32).contiguous()

    # block_dim is chosen host-side from GetCoreNumAiv(); the kernel is
    # vector-only, so one AIV block is one worker.
    torch.ops.npu.kdn_decode(
        q16,
        k16,
        v16,
        g,
        beta,
        initial_state_source,
        out,
        idx32,
        cu32,
        float(scale),
        bool(use_qk_l2norm_in_kernel),
    )
    return out.to(v.dtype).reshape(v.shape)
