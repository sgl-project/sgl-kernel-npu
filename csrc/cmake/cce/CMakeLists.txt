# CCE dispatch is plain C++; Torch registrations remain in the final shared library.
include("${CMAKE_CURRENT_LIST_DIR}/../host_ops.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../cce_kernels.cmake")
sgl_collect_host_sources(sources adapters definitions
    ARCH "${SGL_ARCH_VERSION}" OPERATORS ${cce_operators})
add_library(sgl_cce_operators STATIC ${sources})
target_compile_features(sgl_cce_operators PRIVATE cxx_std_17)
target_include_directories(sgl_cce_operators PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include")
target_compile_definitions(sgl_cce_operators PRIVATE ${definitions})
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    target_compile_options(sgl_cce_operators PRIVATE -O3)
endif()
set_target_properties(sgl_cce_operators PROPERTIES POSITION_INDEPENDENT_CODE ON
    CXX_EXTENSIONS OFF CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
sgl_add_cce_kernels(sgl_cce_operators ${cce_operators})
npu_op_package_add(sgl_kernel_npu
    FILES "${CMAKE_CURRENT_BINARY_DIR}/libsgl_cce_operators.a"
    PACKAGE_PATH op_impl/cce/lib)
