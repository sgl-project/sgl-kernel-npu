# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)  # custom
    target_sources(op_host_aclnnInner PRIVATE
        zerc_moe_def.cpp
    )
    add_modules_sources(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OP_MC2_ENABLE ON
        OPTYPE zerc_moe ACLNNTYPE aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
    set(MC2_OPT ON PARENT_SCOPE)
    set(zerc_moe_depends mc2/common mc2/3rd PARENT_SCOPE)
    # MegaMoe only reuses the canonical MXFP8 kernel helpers. Declaring dispatch_v2 as an op dependency also
    # builds its full host/API/graph stack and requires unrelated host logging headers.
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../../moe_distribute_dispatch_v2/op_kernel/common.h
        DESTINATION ${IMPL_INSTALL_DIR}/ascendc/moe_distribute_dispatch_v2
    )

    set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
    # Timer V2 is compiled out by default. Profile builds opt in through
    # build.sh --ops-compile-options '-DENABLE_TIMER=1'.
    add_ops_compile_options(
        OP_NAME ZercMoe
        OPTIONS --cce-auto-sync=off
        OPTIONS -DCATLASS_ARCH=3510
        OPTIONS -I${OPTENSOR_INCLUDE_ROOT}
        OPTIONS -I${TENSOR_API_INCLUDE_ROOT}
        OPTIONS -I${TENSOR_API_IMPL_INCLUDE_ROOT}
        OPTIONS -I${CMAKE_CURRENT_SOURCE_DIR}/../../3rd/catlass/include
        OPTIONS -I${CMAKE_CURRENT_SOURCE_DIR}/../../3rd
        OPTIONS -I${CMAKE_CURRENT_SOURCE_DIR}/../../
    )

else() # host
    add_mc2_modules_sources(OPTYPE zerc_moe ACLNNTYPE aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
endif()
