# this is the cmakelist file for deepep build
# deepep will be built as separated wheel package

if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    set(DEEPEP_ARCH "x86_64")
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    set(DEEPEP_ARCH "aarch64")
else()
    message(FATAL_ERROR "Unsupported host processor: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(PROJECT_BUILD_PATH ${PROJECT_BINARY_DIR})
set(TARGET_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
set(ASCEND_HOME_PATH ${ASCEND_HOME_PATH})

find_program(PYTHON_EXECUTABLE NAMES python3 python)
message(STATUS "PYTHON_EXECUTABLE:${PYTHON_EXECUTABLE}")

list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_DIR})
find_package(pybind11 REQUIRED)
include(CheckCXXSourceCompiles)

message(STATUS "CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH}")

file(GLOB SOURCES_DEEPEP *.cpp)
list(APPEND SOURCES_DEEPEP
    profiling/core/profile_exporter.cpp
    profiling/core/profile_runtime.cpp
    profiling/core/profile_session.cpp
    profiling/adapters/fused_deep_moe_a5/fused_deep_moe_a5_profile_traits.cpp
    profiling/adapters/fused_deep_moe_a5/fused_deep_moe_a5_profile_adapter.cpp
)

pybind11_add_module(deep_ep_cpp pybind_extension.cpp ${SOURCES_DEEPEP})
set_target_properties(deep_ep_cpp PROPERTIES CXX_STANDARD 17)

# Define __DAV_C310__ for C310/A5 chip builds so that #ifdef __DAV_C310__ guards
# work in host/pybind compilation context (they only exist natively in AscendC device compilation)
# DEEPEP_IS_A5_BUILD is passed from build.sh when SOC_VERSION=Ascend950
if("${DEEPEP_IS_A5_BUILD}" STREQUAL "ON")
    target_compile_definitions(deep_ep_cpp PRIVATE __DAV_C310__)
endif()
target_include_directories( deep_ep_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}

    ${TORCH_DIR}/include
    ${TORCH_DIR}/include/torch/csrc/api/include
    ${TORCH_NPU_DIR}/include
    ${TORCH_NPU_DIR}/include/third_party/acl/inc/acl
    ${TORCH_NPU_DIR}/include/third_party/acl/inc
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/${DEEPEP_ARCH}-linux/include/experiment/platform
)

set(CMAKE_REQUIRED_INCLUDES
    ${TORCH_NPU_DIR}/include
    ${TORCH_NPU_DIR}/include/third_party/acl/inc/acl
    ${TORCH_NPU_DIR}/include/third_party/acl/inc
    ${ASCEND_HOME_PATH}/include
)
set(_DEEPEP_OLD_TRY_COMPILE_TARGET_TYPE "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
check_cxx_source_compiles(
    "#include <acl/acl_rt.h>\nint main() { auto fn = &aclrtEventGetTimestamp; (void)fn; return 0; }\n"
    DEEPEP_HAS_ACLRT_EVENT_GET_TIMESTAMP)
if(DEFINED _DEEPEP_OLD_TRY_COMPILE_TARGET_TYPE AND NOT "${_DEEPEP_OLD_TRY_COMPILE_TARGET_TYPE}" STREQUAL "")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_DEEPEP_OLD_TRY_COMPILE_TARGET_TYPE}")
else()
    unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
endif()
unset(_DEEPEP_OLD_TRY_COMPILE_TARGET_TYPE)
unset(CMAKE_REQUIRED_INCLUDES)

if(DEEPEP_HAS_ACLRT_EVENT_GET_TIMESTAMP)
    target_compile_definitions(deep_ep_cpp PRIVATE DEEPEP_HAS_ACLRT_EVENT_GET_TIMESTAMP=1)
    message(STATUS "deep_ep profiling timestamp alignment: enabled")
else()
    target_compile_definitions(deep_ep_cpp PRIVATE DEEPEP_HAS_ACLRT_EVENT_GET_TIMESTAMP=0)
    message(STATUS "deep_ep profiling timestamp alignment: disabled (profiling collection still enabled)")
endif()
target_link_directories(deep_ep_cpp PRIVATE
        ${TORCH_DIR}/lib
        ${TORCH_NPU_DIR}/lib
        ${ASCEND_HOME_PATH}/lib64
)
target_link_libraries(deep_ep_cpp PRIVATE
    torch_python
    ascendcl
    hccl
    torch_npu
    opapi
)

message(STATUS "TARGET_INSTALL_DIR = ${TARGET_INSTALL_DIR}")
install(TARGETS deep_ep_cpp
    LIBRARY DESTINATION ${TARGET_INSTALL_DIR}/lib
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
)
