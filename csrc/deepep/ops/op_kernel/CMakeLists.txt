# set custom compile options
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    npu_op_kernel_options(ALL OPTIONS -g -O0 )
endif()

message(STATUS "ASCEND include path: ${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/moe_distribute_dispatch")

set(_DISPATCH_FFN_INC_OPTS)
if (EXISTS ${CMAKE_SOURCE_DIR}/third_party/catlass/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${CMAKE_SOURCE_DIR}/third_party/catlass/include)
else()
    message(FATAL_ERROR "dependency catlass is missing")
endif()
message(WARNING "_DISPATCH_FFN_INC_OPTS: ${_DISPATCH_FFN_INC_OPTS}")

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

npu_op_kernel_options("" ALL OPTIONS -DASCENDC_DUMP=0 --cce-auto-sync=off ${_DISPATCH_FFN_INC_OPTS})
