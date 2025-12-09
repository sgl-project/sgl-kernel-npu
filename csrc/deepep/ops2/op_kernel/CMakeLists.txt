# set custom compile options
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0 )
endif()

message(STATUS "ASCEND include path: ${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/moe_distribute_dispatch")

# copy moe_distribute_base head file
execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy
    "${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/moe_distribute_dispatch/moe_distribute_base.h"
    "${CMAKE_CURRENT_BINARY_DIR}/../../op_kernel/moe_distribute_base.h"
    RESULT_VARIABLE copy_result
)

# check moe_distribute_base head file
if(copy_result EQUAL 0)
    message(STATUS "Successfully copied moe_distribute_base.h")
else()
    message(WARNING "Failed to copy moe_distribute_base.h")
endif()

add_ops_compile_options(ALL OPTIONS -DASCENDC_DUMP=1 --cce-auto-sync=off)

add_kernels_compile()
