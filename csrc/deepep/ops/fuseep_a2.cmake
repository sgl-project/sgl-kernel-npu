# Operator-only package: do not replace the normal A2 ops2 package/wheel.
set(package_name fuseep_a2)
set(DEEPEP_CATLASS_INCLUDE_DIR "" CACHE PATH "catlass-v1-stable include directory")
if(NOT EXISTS "${DEEPEP_CATLASS_INCLUDE_DIR}/catlass/catlass.hpp")
    message(FATAL_ERROR "Set DEEPEP_CATLASS_INCLUDE_DIR to catlass-v1-stable/include")
endif()
npu_op_package(${package_name} TYPE RUN CONFIG INSTALL_PATH ${CMAKE_BINARY_DIR}/)
set(ops_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/dispatch_ffn_combine_def.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/dispatch_ffn_combine_infer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/dispatch_ffn_combine_tiling.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/op_host)
npu_op_code_gen(SRC ${ops_srcs} PACKAGE ${package_name}
    COMPILE_OPTIONS -g OUT_DIR ${ASCEND_AUTOGEN_PATH})
file(COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_api/aclnn_dispatch_ffn_combine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_api/aclnn_dispatch_ffn_combine.h
    DESTINATION ${ASCEND_AUTOGEN_PATH})
file(GLOB aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn*.cpp)
npu_op_library(cust_opapi ACLNN ${aclnn_src})
target_compile_options(cust_opapi PRIVATE -fvisibility=hidden)
set(proto_src ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
set_source_files_properties(${proto_src} PROPERTIES GENERATED TRUE)
npu_op_library(cust_op_proto GRAPH ${ops_srcs} ${proto_src})
npu_op_library(cust_optiling TILING ${ops_srcs})
npu_op_package_add(${package_name} LIBRARY cust_optiling cust_opapi cust_op_proto)
add_subdirectory(op_kernel)
