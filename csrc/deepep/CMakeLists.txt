# this is the cmakelist file for deepep build
# deepep will be built as separated wheel package

set(PROJECT_BUILD_PATH ${PROJECT_BINARY_DIR})
set(TARGET_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
set(ASCEND_HOME_PATH ${ASCEND_HOME_PATH})

find_program(PYTHON_EXECUTABLE NAMES python3 python)
message(STATUS "PYTHON_EXECUTABLE:${PYTHON_EXECUTABLE}")

list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_DIR})
find_package(pybind11 REQUIRED)

message(STATUS "CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH}")

file(GLOB_RECURSE SOURCES_DEEPEP *.cpp)

pybind11_add_module(deep_ep_cpp pybind_extension.cpp ${SOURCES_DEEPEP})
set_target_properties(deep_ep_cpp PROPERTIES CXX_STANDARD 17)
target_include_directories( deep_ep_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/deepep

    ${TORCH_DIR}/include
    ${TORCH_DIR}/include/torch/csrc/api/include
    ${TORCH_NPU_DIR}/include
    ${TORCH_NPU_DIR}/include/third_party/acl/inc/acl
    ${TORCH_NPU_DIR}/include/third_party/acl/inc
    ${ASCEND_HOME_PATH}/include
)
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
)

message(STATUS "TARGET_INSTALL_DIR = ${TARGET_INSTALL_DIR}")
install(TARGETS deep_ep_cpp
    LIBRARY DESTINATION ${TARGET_INSTALL_DIR}/lib
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
)
