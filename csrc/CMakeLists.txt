# set the library output dir to the python dir for wheel package build
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/sgl_kernel_npu/sgl_kernel_npu/lib)

# host side files
FILE(GLOB OP_SRCS
    ${PROJECT_OP_SRC_BASE}/pytorch_extensions.cpp
    ${PROJECT_OP_SRC_BASE}/helloworld/op_host/helloworld.cpp
    ${PROJECT_OP_SRC_BASE}/cache_location_assign/op_host/cache_loc_assign.cpp
    ${PROJECT_OP_SRC_BASE}/assign_cache_op/op_host/assign_cache.cpp
    ${PROJECT_OP_SRC_BASE}/mla_preprocess/op_host/mla_preprocess.cpp
    )

# set the so name
set(OP_PLUGIN_NAME sgl_kernel_npu)

# kernel side files without workspace
ascendc_library(no_workspace_kernel STATIC
    ${PROJECT_OP_SRC_BASE}/helloworld/op_kernel/kernel_helloworld.cpp
    ${PROJECT_OP_SRC_BASE}/assign_cache_op/op_kernel/assign_cache_op.cpp
)

ascendc_library(workspace_kernel STATIC
    ${PROJECT_OP_SRC_BASE}/cache_location_assign/op_kernel/cache_loc_assign_kernel_int32.cpp
    ${PROJECT_OP_SRC_BASE}/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp
)

ascendc_compile_definitions(workspace_kernel PRIVATE
       -DHAVE_WORKSPACE
       -DHAVE_TILING
)

# create shared library libsgl_kernel_npu.so
add_library(${OP_PLUGIN_NAME} SHARED ${OP_SRCS})

target_link_libraries(${OP_PLUGIN_NAME} PRIVATE
        workspace_kernel
        no_workspace_kernel
        torch_npu
        ascendcl
        tiling_api
        register
        platform
        ascendalog
        dl
)

target_link_directories(${OP_PLUGIN_NAME} PRIVATE
        ${TORCH_DIR}/lib
        ${TORCH_NPU_DIR}/lib
)

target_include_directories(${OP_PLUGIN_NAME} PRIVATE
        ${PROJECT_OP_SRC_BASE}/utils
        ${PROJECT_SOURCE_DIR}/include
        ${TORCH_DIR}/include
        ${TORCH_DIR}/include/torch/csrc/api/include
        ${TORCH_NPU_DIR}/include
)