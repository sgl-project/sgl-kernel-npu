# Entered only for selected HOST operators at the KERNEL checkpoint.
include("${CMAKE_CURRENT_LIST_DIR}/../torch.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../host_ops.cmake")
sgl_collect_host_sources(host_sources host_adapters host_definitions
    ARCH "${SGL_ARCH_VERSION}" OPERATORS ${host_operators})

# Keep Torch registrations in the final extension, so static linking cannot
# discard their initializers. Only the implementations belong in this archive.
add_library(sgl_host_operators STATIC ${host_sources})
target_compile_features(sgl_host_operators PRIVATE cxx_std_17)
target_compile_options(sgl_host_operators PRIVATE ${torch_flags})
target_compile_definitions(sgl_host_operators PRIVATE ${host_definitions})
# Match the legacy Release defaults without changing the parent CANN configuration.
# Explicit build configurations keep their toolchain-provided flags.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    target_compile_options(sgl_host_operators PRIVATE -O3)
    target_compile_definitions(sgl_host_operators PRIVATE NDEBUG)
endif()
target_include_directories(sgl_host_operators PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../../aclnn_torch_adapter"
    "${torch_npu_root}/include" "${ASCEND_CANN_PACKAGE_PATH}/include"
    ${Python3_INCLUDE_DIRS})
if("shm_allocator" IN_LIST host_operators)
    find_path(SGL_ASCEND_DRIVER_INCLUDE driver/ascend_hal_define.h
        HINTS "${ASCEND_DRIVER_PATH}/include" "$ENV{ASCEND_DRIVER_PATH}/include"
            "${ASCEND_DRIVER_PATH}/kernel/inc" "$ENV{ASCEND_DRIVER_PATH}/kernel/inc"
            "/usr/local/Ascend/driver/include" "/usr/local/Ascend/driver/kernel/inc"
            "${ASCEND_CANN_PACKAGE_PATH}/include" REQUIRED)
    target_include_directories(sgl_host_operators PRIVATE "${SGL_ASCEND_DRIVER_INCLUDE}")
endif()
target_link_libraries(sgl_host_operators PRIVATE ${TORCH_LIBRARIES})
set_target_properties(sgl_host_operators PROPERTIES
    POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# RUN installs this archive; build_torch() later links it with the Torch adapters.
# Executing the installer does not execute or register the HOST operators.
npu_op_package_add(sgl_kernel_npu
    FILES "${CMAKE_CURRENT_BINARY_DIR}/libsgl_host_operators.a"
    PACKAGE_PATH op_impl/host/lib)
