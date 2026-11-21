# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the BSD 3-Clause License (the "License"). Please refer to the License for details.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
# -----------------------------------------------------------------------------------------------------------
# Adapted from the A3 QSA Cube operator in the vLLM NPU plugin 74b02d912.
sgl_register_operator(
    ID qsa_cube_attention
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglQsaCubeAttention
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/qsa_cube_attention_def.cpp
    TILING_SOURCES op_host/qsa_cube_attention_tiling.cpp
    INFER_SOURCES op_host/qsa_cube_attention_infer.cpp
    KERNEL_SOURCES op_kernel/qsa_cube_attention.cpp
    KERNEL_COMPILE_OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -mllvm -align-all-functions=8
)
