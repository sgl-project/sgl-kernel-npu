"""PTO-ISA backend for the KDA recurrent decode step (KDA_DECODE_PTO_BACKEND=1).

Drop-in replacement for the triton ``fused_sigmoid_gating_delta_rule_update_npu``
decode path, backed by ``torch.ops.npu.kda_decode``.

* **The gating is fused into the kernel.** ``g = -exp(A_log) * softplus(a +
  dt_bias)`` and ``beta = sigmoid(b)`` are computed on the vector core in fp32,
  so ``g`` never round-trips through a narrow wire format the way it did when
  torch precomputed it.
* **q/k/v/out stay bfloat16**, the model's own dtype -- the kernel converts on
  the way in and out, so there is no ``.to(float16)`` pass either.
* **The state layout is taken as V-major** ``[slots, H, V, K]``, matching
  sglang's ``temporal_state`` pool (``mem_cache/memory_pool.py``), the prefill
  ``chunk_delta_h`` block pointer ``(V, K)/(K, 1)``, and the CUDA reference
  decode kernel. Note the triton NPU kernel indexes the same pool K-major.
"""

import os
from typing import Optional

import torch

HEAD_DIM = 128
_PTO_ENV = "KDA_DECODE_PTO_BACKEND"


def pto_backend_enabled() -> bool:
    return os.environ.get(_PTO_ENV, "0") not in ("", "0", "false", "False")


def kda_decode_pto(
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
        raise NotImplementedError(
            f"the PTO decode backend supports K=V=128, got K={K}, V={V}"
        )

    if scale is None:
        scale = K**-0.5

    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.dtype != torch.bfloat16:
            raise NotImplementedError(
                f"{name} must be bfloat16 for the PTO decode backend, got {tensor.dtype}"
            )
    for name, tensor in (("A_log", A_log), ("a", a), ("dt_bias", dt_bias), ("b", b)):
        if tensor.dtype != torch.float32:
            raise NotImplementedError(
                f"{name} must be float32 for the fused gating, got {tensor.dtype}"
            )

    q_in = q.reshape(B, T, H, K).contiguous()
    k_in = k.reshape(B, T, H, K).contiguous()
    v_in = v.reshape(B, T, HV, V).contiguous()
    out = torch.empty_like(v_in)

    cu32 = cu_seqlens.to(torch.int32).contiguous()
    idx32 = initial_state_indices.to(torch.int32).contiguous()

    # block_dim is chosen host-side from GetCoreNumAiv(); the kernel is
    # vector-only, so one AIV block is one worker.
    torch.ops.npu.kda_decode(
        q_in,
        k_in,
        v_in,
        A_log.reshape(-1).contiguous(),
        a.reshape(-1).contiguous(),
        dt_bias.reshape(-1).contiguous(),
        b.reshape(-1).contiguous(),
        initial_state_source,
        out,
        idx32,
        cu32,
        float(scale),
        bool(use_qk_l2norm_in_kernel),
        float(softplus_beta),
        float(softplus_threshold),
    )
    return out.reshape(v.shape)
