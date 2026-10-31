# The shared Host declares both OpTypes, so both need Kernel source mappings.
# Update the cache because domain discovery runs through nested function scopes.
if(NOT "ALL" IN_LIST SGL_CANN_OPERATORS)
    if("cache_loc_assign" IN_LIST SGL_CANN_OPERATORS OR "cache_loc_update" IN_LIST SGL_CANN_OPERATORS)
        set(cache_loc_operators ${SGL_CANN_OPERATORS})
        foreach(mode IN ITEMS assign update)
            if(NOT "cache_loc_${mode}" IN_LIST cache_loc_operators)
                list(APPEND cache_loc_operators "cache_loc_${mode}")
            endif()
        endforeach()
        # Preserve the requested IDs in CMakeCache for Python's incremental-build check.
        # Selection and target setup use the expanded list before this deferred restore.
        cmake_language(EVAL CODE "
            cmake_language(DEFER DIRECTORY [[${CMAKE_SOURCE_DIR}]] CALL
                set SGL_CANN_OPERATORS [[${SGL_CANN_OPERATORS}]] CACHE STRING
                [[ALL or a semicolon-separated list of operator IDs]] FORCE)
        ")
        set(SGL_CANN_OPERATORS "${cache_loc_operators}" CACHE STRING
            "ALL or a semicolon-separated list of operator IDs" FORCE)
        unset(cache_loc_operators)
    endif()
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
