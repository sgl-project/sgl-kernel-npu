include_guard(GLOBAL)

function(sgl_initialize_cann_package)
    npu_op_package(sgl_kernel_npu TYPE RUN
        CONFIG INSTALL_PATH "${CMAKE_CURRENT_BINARY_DIR}/package-stage")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/package-stage" PARENT_SCOPE)
endfunction()
