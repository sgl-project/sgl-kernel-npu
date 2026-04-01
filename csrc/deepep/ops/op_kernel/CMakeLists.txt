# set custom compile options
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0 )
endif()

message(STATUS "ASCEND include path: ${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/moe_distribute_dispatch")

set(_DISPATCH_FFN_INC_OPTS)
if (EXISTS ${CMAKE_SOURCE_DIR}/third_party/catlass/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${CMAKE_SOURCE_DIR}/third_party/catlass/include)
else()
    message(FATAL_ERROR "dependency catlass is missing")
endif()
message(WARNING "_DISPATCH_FFN_INC_OPTS: ${_DISPATCH_FFN_INC_OPTS}")

add_ops_compile_options(ALL OPTIONS -DASCENDC_DUMP=0 --cce-auto-sync=off ${_DISPATCH_FFN_INC_OPTS})

add_kernels_compile()
