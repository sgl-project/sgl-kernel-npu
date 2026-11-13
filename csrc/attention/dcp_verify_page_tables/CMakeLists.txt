sgl_register_operator(
    ID dcp_verify_page_tables
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglDcpVerifyPageTables
    SUPPORTED_ARCHS 2201
    TORCH_ADAPTER torch_adapter.cpp
    OPDEF_SOURCES op_host/dcp_verify_page_tables.cpp
    INFER_SOURCES op_host/dcp_verify_page_tables.cpp
    TILING_SOURCES op_host/dcp_verify_page_tables.cpp
    KERNEL_SOURCES op_kernel/dcp_verify_page_tables.cpp
    KERNEL_COMPILE_DEFINITIONS SGL_DCP_CANN_PACKAGE
)
