if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    npu_op_kernel_options(ascendc_kernels ALL OPTIONS -g -O0)
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

npu_op_kernel_options(ascendc_kernels ALL OPTIONS
    --cce-auto-sync=off
    -Wno-deprecated-declarations
    -Werror
    -mllvm -cce-aicore-hoist-movemask=false
    --op_relocatable_kernel_binary=true
)
