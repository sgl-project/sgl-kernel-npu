# set custom compile options
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    npu_op_kernel_options(ALL OPTIONS -g -O0 )
endif()

message(STATUS "ASCEND include path: ${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/moe_distribute_dispatch")

# copy moe_distribute_base head file
set(MOE_BASE_H_SRC "${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/inc/kernel/moe_distribute_base.h")
set(MOE_COMM_CTX_H_SRC "${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/inc/kernel/moe_distribute_comm_ctx.h")

set(MOE_BASE_H_DST "${CMAKE_CURRENT_BINARY_DIR}/../../op_kernel/moe_distribute_base.h")
set(MOE_COMM_CTX_H_DST "${CMAKE_CURRENT_BINARY_DIR}/../../op_kernel/moe_distribute_comm_ctx.h")

execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy "${MOE_BASE_H_SRC}" "${MOE_BASE_H_DST}"
        RESULT_VARIABLE copy_base_result
)
if(copy_base_result EQUAL 0)
    message(STATUS "Successfully copied moe_distribute_base.h")
else()
    message(WARNING "Failed to copy moe_distribute_base.h from ${MOE_BASE_H_SRC}")
endif()

# copy moe_distribute_comm_ctx.h
execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy "${MOE_COMM_CTX_H_SRC}" "${MOE_COMM_CTX_H_DST}"
        RESULT_VARIABLE copy_comm_result
)
if(copy_comm_result EQUAL 0)
    message(STATUS "Successfully copied moe_distribute_comm_ctx.h")
else()
    message(WARNING "Failed to copy moe_distribute_comm_ctx.h from ${MOE_COMM_CTX_H_SRC}")
endif()

npu_op_kernel_sources(ascendc_kernels
    KERNEL_DIR ./
)

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${package_name}
    LIBRARY ascendc_kernels
)

npu_op_kernel_options("" ALL OPTIONS -DASCENDC_DUMP=0 --cce-auto-sync=off)
