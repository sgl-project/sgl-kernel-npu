"""ACLNN Gemma RMSNorm provider, staged as ``norm/gemma_rmsnorm.py`` on 950.

Ascend 950 does not register ``npu_gemma_rms_norm``, so the Gemma weight offset
is applied explicitly and the result handed to plain RMSNorm. Exactly one
provider module is staged into the wheel (see ``setup.py``), so this file is
never importable under its own name from an installed package.
"""

from __future__ import annotations

import torch
import torch_npu


def npu_gemma_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply standard RMSNorm with Gemma weight semantics on Ascend 950.

    ``1.0 + weight`` is evaluated in the weight's own dtype, not fp32, because
    ``npu_rms_norm`` requires gamma to match the input dtype. For bf16 that
    rounds the effective scale to ~2**-8 near 1.0, so this is slightly less
    accurate than the 910 native operator. It matches what vllm-ascend
    (``ops/layernorm.py``) and SGLang's own CUDA ``gemma_weight`` buffer do --
    an accepted trade-off, not an oversight.
    """
    return torch_npu.npu_rms_norm(input, 1.0 + weight, eps)
