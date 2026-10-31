"""Public operators backed by the external batch_invariant_ops library."""

# The external package is not bundled with this repo. Download the CANN op
# package for your device and the torch extension from:
# 910B x86_64:
# https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260901/cann-ops-batch_invariant-910b-1.0.0-linux.x86_64.run
# A3 aarch64:
# https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260901/cann-ops-batch_invariant-A3-1.0.0-linux.aarch64.run
# A3 x86_64:
# https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260901/cann-ops-batch_invariant-A3-1.0.0-linux.x86_64.run
# 950 x86_64:
# https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260901/cann-ops-batch_invariant-950-1.0.0-linux.x86_64.run
# 950 aarch64:
# https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260901/cann-ops-batch_invariant-950-1.0.0-linux.aarch64.run
# torch extension:
# https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260901/batch_invariant-torch_ops_extension-1.0.0.zip

from .add_rms_norm import add_rms_norm_batch_invariant
from .fused_infer_attention_score import (
    fused_infer_attention_score_batch_invariant,
    fused_infer_attention_score_batch_invariant_get_max_workspace,
    fused_infer_attention_score_batch_invariant_infer_output,
    fused_infer_attention_score_batch_invariant_out,
)
from .fused_infer_attention_score_v2 import (
    fused_infer_attention_score_batch_invariant_v2,
    fused_infer_attention_score_batch_invariant_v2_get_max_workspace,
    fused_infer_attention_score_batch_invariant_v2_infer_output,
)
from .fused_linear_online_max_sum import fused_linear_online_max_sum_batch_invariant
from .log_softmax import log_softmax_batch_invariant
from .matmul import matmul_batch_invariant
from .mm import mm_batch_invariant
from .reduce_mean import reduce_mean_batch_invariant
from .reduce_sum import reduce_sum_batch_invariant
from .softmax import softmax_batch_invariant

__all__ = [
    "add_rms_norm_batch_invariant",
    "fused_infer_attention_score_batch_invariant",
    "fused_infer_attention_score_batch_invariant_get_max_workspace",
    "fused_infer_attention_score_batch_invariant_infer_output",
    "fused_infer_attention_score_batch_invariant_out",
    "fused_infer_attention_score_batch_invariant_v2",
    "fused_infer_attention_score_batch_invariant_v2_get_max_workspace",
    "fused_infer_attention_score_batch_invariant_v2_infer_output",
    "fused_linear_online_max_sum_batch_invariant",
    "log_softmax_batch_invariant",
    "matmul_batch_invariant",
    "mm_batch_invariant",
    "reduce_mean_batch_invariant",
    "reduce_sum_batch_invariant",
    "softmax_batch_invariant",
]
