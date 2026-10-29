# The shared kernel is compiled for both supported architectures.
sgl_register_operator(
    ID cache_loc_update
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglCacheLocUpdate
    TORCH_ADAPTER ../cache_loc_common/torch_adapter/cache_loc_update.cpp
    SUPPORTED_ARCHS 2201 3510
    HOST_SOURCES ../cache_loc_common/op_host/cache_loc_update.cpp
    KERNEL_SOURCES ../cache_loc_common/op_kernel/cache_loc_update.cpp
)
