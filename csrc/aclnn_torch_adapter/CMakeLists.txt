cmake_minimum_required(VERSION 3.20)
project(sgl_cann_torch LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The CANN Torch extension requires Linux")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/torch.cmake")
find_library(TORCH_NPU_LIBRARY torch_npu PATHS "${torch_npu_root}/lib" NO_DEFAULT_PATH REQUIRED)
set(sdk "$ENV{ASCEND_HOME_PATH}")
find_library(ACL_LIBRARY ascendcl PATHS "${sdk}/lib64" NO_DEFAULT_PATH REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/operators.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/domains.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/host_ops.cmake")
sgl_add_operator_domains()
set(cann_operators)
set(cann_includes)
set(cann_libraries)
if(SGL_CANN_OPERATORS)
    if(NOT IS_DIRECTORY "${SGL_CANN_VENDOR}/op_api/include")
        message(FATAL_ERROR "SGL_CANN_VENDOR must name this build's installed CANN vendor")
    endif()
    find_library(NNOPBASE_LIBRARY nnopbase PATHS "${sdk}/lib64" NO_DEFAULT_PATH REQUIRED)
    list(APPEND cann_includes "${SGL_CANN_VENDOR}/op_api/include")
    list(APPEND cann_libraries "${NNOPBASE_LIBRARY}")
    sgl_select_cann_operators(cann_operators ARCH "${SGL_ARCH_VERSION}" OPERATORS ${SGL_CANN_OPERATORS})
endif()
sgl_collect_host_sources(host_sources host_adapters host_definitions
    ARCH "${SGL_ARCH_VERSION}" OPERATORS ${SGL_HOST_OPERATORS})
set(torch_adapters)
foreach(id IN LISTS cann_operators)
    sgl_get_operator_field(adapter "${id}" TORCH_ADAPTER)
    if(NOT adapter)
        message(FATAL_ERROR "No Torch adapter declared for selected operator: ${id}")
    endif()
    list(APPEND torch_adapters "${adapter}")
endforeach()
list(APPEND torch_adapters ${host_adapters})
list(REMOVE_DUPLICATES torch_adapters)
if(NOT torch_adapters)
    message(FATAL_ERROR "No CANN or HOST operators selected for the Torch library")
endif()
message(STATUS "Torch CANN operators: ${cann_operators}; HOST operators: ${SGL_HOST_OPERATORS}")
add_library(sgl_kernel_npu SHARED ${torch_adapters})
if(SGL_HOST_OPERATORS)
    # HOST implementations were compiled and installed by the native RUN build.
    find_library(SGL_HOST_LIBRARY sgl_host_operators
        PATHS "${SGL_CANN_VENDOR}/op_impl/host/lib" NO_DEFAULT_PATH REQUIRED)
    target_link_libraries(sgl_kernel_npu PRIVATE "${SGL_HOST_LIBRARY}")
endif()
target_compile_features(sgl_kernel_npu PRIVATE cxx_std_17)
target_compile_options(sgl_kernel_npu PRIVATE ${torch_flags})
target_compile_definitions(sgl_kernel_npu PRIVATE ${host_definitions})
target_include_directories(sgl_kernel_npu PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}" "${torch_npu_root}/include" "${sdk}/include"
    ${Python3_INCLUDE_DIRS})
target_link_libraries(sgl_kernel_npu PRIVATE ${TORCH_LIBRARIES})
set_target_properties(sgl_kernel_npu PROPERTIES CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
# Only CANN adapters need the repository's installed vendor and nnopbase.
target_include_directories(sgl_kernel_npu PRIVATE ${cann_includes})
target_link_libraries(sgl_kernel_npu PRIVATE "${TORCH_NPU_LIBRARY}"
    "${ACL_LIBRARY}" ${cann_libraries} ${CMAKE_DL_LIBS})
# Dependencies are loaded by torch/torch_npu and the initialized CANN runtime.
# Do not bake the build machine's absolute directories into the wheel.
set_target_properties(sgl_kernel_npu PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${SGL_TORCH_LIBRARY_DIR}"
    SKIP_BUILD_RPATH ON INSTALL_RPATH "")
