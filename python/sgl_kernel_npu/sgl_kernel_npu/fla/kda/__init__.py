# Ascend NPU KDA (Kimi Delta Attention) chunked-prefill kernels.
# Vendored from flash-linear-attention (fla/ops/kda/backends/triton_ascend),
# with fla.* imports rewritten to sgl_kernel_npu.fla.* and the intra kernel
# wired to the NPU token-parallel helper.
from sgl_kernel_npu.fla.kda.chunk_intra import chunk_kda_fwd_intra_npu

__all__ = ["chunk_kda_fwd_intra_npu"]
