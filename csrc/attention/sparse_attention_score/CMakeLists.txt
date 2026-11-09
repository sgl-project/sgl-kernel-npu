# CANN copy of csrc/sparse_attention_score; one entry retains the existing
# architecture branches. The legacy build remains independent.
sgl_register_operator(
    ID sparse_attention_score
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglSparseAttentionScore
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/sparse_attention_score_def.cpp
    INFER_SOURCES op_host/sparse_attention_score_infer.cpp
    TILING_SOURCES op_host/tiling/sparse_attention_score_tiling.cpp
    TORCH_ADAPTER torch_adapter.cpp
    KERNEL_SOURCES op_kernel/sparse_attention_score.cpp
)
