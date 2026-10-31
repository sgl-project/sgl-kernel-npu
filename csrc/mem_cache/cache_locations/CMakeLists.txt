# These interfaces share one Host, Kernel and Torch adapter and are packaged together.
if(("cache_loc_assign" IN_LIST SGL_CANN_OPERATORS AND NOT "cache_loc_update" IN_LIST SGL_CANN_OPERATORS)
   OR ("cache_loc_update" IN_LIST SGL_CANN_OPERATORS AND NOT "cache_loc_assign" IN_LIST SGL_CANN_OPERATORS))
    message(FATAL_ERROR "cache_loc_assign and cache_loc_update must be selected together; use both IDs or ALL")
endif()

foreach(mode IN ITEMS assign update)
    if(mode STREQUAL "assign")
        set(op_type SglCacheLocAssign)
    else()
        set(op_type SglCacheLocUpdate)
    endif()
    sgl_register_operator(
        ID cache_loc_${mode}
        EXECUTION_UNIT AICORE
        CANN_OP_TYPE ${op_type}
        TORCH_ADAPTER torch_adapter.cpp
        SUPPORTED_ARCHS 2201 3510
        HOST_SOURCES op_host/cache_loc_host.cpp
        KERNEL_SOURCES op_kernel/cache_loc_kernel.cpp
    )
endforeach()
