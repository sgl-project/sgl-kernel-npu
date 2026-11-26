function(sgl_configure_shm_allocator_driver operator_id source)
    if(TARGET sgl_host_operators AND "${operator_id}" IN_LIST host_operators)
        find_path(SGL_ASCEND_DRIVER_INCLUDE driver/ascend_hal_define.h
            HINTS "${ASCEND_DRIVER_PATH}/include" "$ENV{ASCEND_DRIVER_PATH}/include"
                "${ASCEND_DRIVER_PATH}/kernel/inc" "$ENV{ASCEND_DRIVER_PATH}/kernel/inc"
                "/usr/local/Ascend/driver/include" "/usr/local/Ascend/driver/kernel/inc"
                "${ASCEND_CANN_PACKAGE_PATH}/include" REQUIRED)
        set_property(SOURCE "${source}"
            TARGET_DIRECTORY sgl_host_operators APPEND PROPERTY INCLUDE_DIRECTORIES
            "${SGL_ASCEND_DRIVER_INCLUDE}")
    endif()
    if(TARGET sgl_kernel_npu AND "${operator_id}" IN_LIST SGL_HOST_OPERATORS)
        find_library(SGL_ASCEND_HAL_LIBRARY NAMES ascend_hal
            HINTS "${ASCEND_DRIVER_PATH}/lib64/driver" "$ENV{ASCEND_DRIVER_PATH}/lib64/driver"
                "${ASCEND_DRIVER_PATH}/lib64/common" "$ENV{ASCEND_DRIVER_PATH}/lib64/common"
                "/usr/local/Ascend/driver/lib64/driver" "/usr/local/Ascend/driver/lib64/common"
                "${sdk}/lib64" "${sdk}/lib64/stub"
                "${sdk}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/linux/${CMAKE_SYSTEM_PROCESSOR}"
                "${sdk}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib" REQUIRED)
        target_link_libraries(sgl_kernel_npu PRIVATE "${SGL_ASCEND_HAL_LIBRARY}")
    endif()
endfunction()
