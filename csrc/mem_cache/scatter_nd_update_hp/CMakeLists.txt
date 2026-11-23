sgl_register_operator(
    ID scatter_nd_update_hp
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglScatterNdUpdateSk
    SUPPORTED_ARCHS 2201 3510
    OPDEF_SOURCES op_host/scatter_nd_update_sk_def.cpp
    INFER_SOURCES op_host/scatter_nd_update_sk_infershape.cpp
    ACLNN_SOURCES
        op_host/op_api/aclnn_scatter_nd_update_sk.cpp
        op_host/op_api/scatter_nd_update_sk.cpp
    ACLNN_HEADERS op_host/op_api/aclnn_scatter_nd_update_sk.h
    TILING_SOURCES op_host/scatter_nd_update_sk_tiling.cpp op_host/scatter_nd_update_sk_tiling_arch35.cpp
    TILING_INCLUDE_DIRECTORIES op_host/compat
    KERNEL_COMPILE_OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations
    TORCH_ADAPTER torch_adapter.cpp
)
sgl_register_operator_variant(
    ID scatter_nd_update_hp ARCH 2201
    KERNEL_SOURCES op_kernel/scatter_nd_update_sk.cpp
)
sgl_register_operator_variant(
    ID scatter_nd_update_hp ARCH 3510
    KERNEL_SOURCES op_kernel/scatter_nd_update_sk_apt.cpp
)

function(sgl_configure_scatter_nd_update_hp_dependencies)
    if("scatter_nd_update_hp" IN_LIST cann_operators AND TARGET sgl_cann_opapi)
        target_link_libraries(sgl_kernel_npu_ascendc_cust_opapi PRIVATE opapi_math)
    endif()
endfunction()

# The SDK creates the final ACLNN target after operator discovery.
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL sgl_configure_scatter_nd_update_hp_dependencies)
