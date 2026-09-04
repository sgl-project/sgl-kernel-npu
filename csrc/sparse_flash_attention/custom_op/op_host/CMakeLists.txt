aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} sparse_flash_attention_host_srcs)

npu_op_code_gen(
    SRC ${sparse_flash_attention_host_srcs}
    PACKAGE ${package_name}
    COMPILE_OPTIONS -g
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
)

file(GLOB_RECURSE sparse_flash_attention_pregen_api
    "${CMAKE_CURRENT_SOURCE_DIR}/op_api/*"
)
file(COPY ${sparse_flash_attention_pregen_api} DESTINATION ${ASCEND_AUTOGEN_PATH})

file(GLOB sparse_flash_attention_aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn*.cpp)
file(GLOB sparse_flash_attention_inner_src ${ASCEND_AUTOGEN_PATH}/aclnnInner_*.cpp)
set_source_files_properties(${sparse_flash_attention_inner_src} PROPERTIES GENERATED TRUE)

npu_op_library(cust_opapi ACLNN ${sparse_flash_attention_aclnn_src})
target_compile_options(cust_opapi PRIVATE -fvisibility=hidden)
target_include_directories(cust_opapi PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
    ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn_kernels
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc
)

file(GLOB sparse_flash_attention_proto_src ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
set_source_files_properties(${sparse_flash_attention_proto_src} PROPERTIES GENERATED TRUE)
npu_op_library(cust_op_proto GRAPH
    ${sparse_flash_attention_host_srcs}
    ${sparse_flash_attention_proto_src}
)
target_compile_options(cust_op_proto PRIVATE -fvisibility=hidden)
target_include_directories(cust_op_proto PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB sparse_flash_attention_fallback_src ${ASCEND_AUTOGEN_PATH}/fallback_*.cpp)
set_source_files_properties(${sparse_flash_attention_fallback_src} PROPERTIES GENERATED TRUE)
npu_op_library(cust_optiling TILING
    ${sparse_flash_attention_host_srcs}
    ${sparse_flash_attention_fallback_src}
)
target_compile_options(cust_optiling PRIVATE -fvisibility=hidden)
target_include_directories(cust_optiling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

npu_op_package_add(${package_name}
    LIBRARY
        cust_optiling
        cust_opapi
        cust_op_proto
)
