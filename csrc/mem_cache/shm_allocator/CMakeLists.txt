sgl_register_operator(
    ID shm_allocator
    EXECUTION_UNIT HOST
    SUPPORTED_ARCHS 2201
    CPP_SOURCES shm_allocator.cpp shm_allocator_op.cpp
    TORCH_ADAPTER torch_adapter.cpp
)

function(sgl_configure_shm_allocator_driver)
    if(TARGET sgl_host_operators AND "shm_allocator" IN_LIST host_operators)
        find_path(SGL_ASCEND_DRIVER_INCLUDE driver/ascend_hal_define.h
            HINTS "${ASCEND_DRIVER_PATH}/include" "$ENV{ASCEND_DRIVER_PATH}/include"
                "${ASCEND_DRIVER_PATH}/kernel/inc" "$ENV{ASCEND_DRIVER_PATH}/kernel/inc"
                "/usr/local/Ascend/driver/include" "/usr/local/Ascend/driver/kernel/inc"
                "${ASCEND_CANN_PACKAGE_PATH}/include" REQUIRED)
        set_property(SOURCE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/shm_allocator.cpp"
            TARGET_DIRECTORY sgl_host_operators APPEND PROPERTY INCLUDE_DIRECTORIES
            "${SGL_ASCEND_DRIVER_INCLUDE}")
    endif()
    if(TARGET sgl_kernel_npu AND "shm_allocator" IN_LIST SGL_HOST_OPERATORS)
        find_library(SGL_ASCEND_HAL_LIBRARY NAMES ascend_hal
            HINTS "${ASCEND_DRIVER_PATH}/lib64/driver" "$ENV{ASCEND_DRIVER_PATH}/lib64/driver"
                "${ASCEND_DRIVER_PATH}/lib64/common" "$ENV{ASCEND_DRIVER_PATH}/lib64/common"
                "/usr/local/Ascend/driver/lib64/driver" "/usr/local/Ascend/driver/lib64/common"
                "${sdk}/lib64" "${sdk}/lib64/stub" REQUIRED)
        target_link_libraries(sgl_kernel_npu PRIVATE "${SGL_ASCEND_HAL_LIBRARY}")
    endif()
endfunction()

# The HOST archive and Torch extension are created after operator discovery.
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL sgl_configure_shm_allocator_driver)
