# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Hand-written aclnn op (mirrors block_sparse_attention): the def is excluded
# from aclnn auto-gen, and the aclnn + L0 op api live under op_api/ (compiled
# into cust_opapi.so). Host log/err macros are shimmed by sas_compat.h.

add_ops_compile_options(
    OP_NAME SparseAttentionScore
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -fpermissive
)

set(sparse_attention_score_depends sparse_attention_score PARENT_SCOPE)

# opdev def -> exclude target (proto only; aclnn is hand-written in op_api/).
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        sparse_attention_score_def.cpp
    )
endif()

target_sources(optiling PRIVATE
    sparse_attention_score_tiling.cpp
)
target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
    sparse_attention_score_infershape.cpp
)

file(GLOB SAS_OPAPI_SRCS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/op_api/*.cpp)
if (TARGET opapi AND SAS_OPAPI_SRCS)
    target_sources(opapi PRIVATE ${SAS_OPAPI_SRCS})
    target_include_directories(opapi PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/op_api)
endif()
