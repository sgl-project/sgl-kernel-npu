# Existing repository kernels join the same Torch library as ACLNN adapters.
include_guard(GLOBAL)
function(sgl_add_native_operators target)
    if(NOT SGL_NATIVE_OPERATORS)
        return()
    endif()
    set(ASCEND_HOME_PATH "${sdk}")
    set(ASCEND_CANN_PACKAGE_PATH "${sdk}")
    set(SOC_VERSION "${SGL_NATIVE_SOC}")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_CXX_STANDARD 17)
    foreach(candidate IN ITEMS tools/tikcpp/ascendc_kernel_cmake compiler/tikcpp/ascendc_kernel_cmake ascendc_devkit/tikcpp/samples/cmake)
        if(EXISTS "${sdk}/${candidate}/ascendc.cmake")
            set(ASCENDC_CMAKE_DIR "${sdk}/${candidate}")
            break()
        endif()
    endforeach()
    if(NOT ASCENDC_CMAKE_DIR)
        message(FATAL_ERROR "Cannot find ascendc.cmake in the initialized SDK")
    endif()
    include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")
    set(registration "${CMAKE_CURRENT_BINARY_DIR}/native_registration.cpp")
    execute_process(COMMAND "${Python3_EXECUTABLE}"
        "${SGL_SOURCE_ROOT}/python/sgl_kernel_npu/build_tools/native.py"
        --source-root "${SGL_SOURCE_ROOT}" --output "${registration}"
        --arch-version "${SGL_ARCH_VERSION}" --ops ${SGL_NATIVE_OPERATORS}
        RESULT_VARIABLE registration_result)
    if(NOT registration_result EQUAL 0)
        message(FATAL_ERROR "Cannot generate repository native registrations")
    endif()
    if(SGL_ARCH_VERSION STREQUAL "2201")
        set(kernel_arch arch22)
        set(arch_definition SGL_KERNEL_ARCH_22 SGL_KERNEL_ENABLE_A3_ONLY_OPS)
    elseif(SGL_ARCH_VERSION STREQUAL "3510")
        set(kernel_arch arch35)
        set(arch_definition SGL_KERNEL_ARCH_35 SGL_KERNEL_ENABLE_A5_ONLY_OPS)
    else()
        message(FATAL_ERROR "Unsupported native architecture: ${SGL_ARCH_VERSION}")
    endif()
    set(root "${SGL_SOURCE_ROOT}/csrc")
    set(host_sources "${registration}")
    set(kernel_targets)
    set(host_includes)
    foreach(op IN LISTS SGL_NATIVE_OPERATORS)
        if(op STREQUAL "compressor")
            ascendc_library(native_compressor_kernel STATIC "${root}/compressor/op_kernel/compressor.cpp")
            ascendc_include_directories(native_compressor_kernel PRIVATE
                "${root}/compressor/op_kernel" "${root}/compressor/op_kernel/${kernel_arch}"
                "${root}/utils/kernel")
            list(APPEND host_sources "${root}/compressor/op_host/compressor.cpp"
                "${root}/compressor/op_host/${kernel_arch}/compressor_tiling.cpp")
            list(APPEND kernel_targets native_compressor_kernel)
        elseif(op STREQUAL "sparse_attn_sharedkv" AND SGL_ARCH_VERSION STREQUAL "2201")
            ascendc_library(native_sparse_kernel STATIC "${root}/sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv.cpp")
            ascendc_include_directories(native_sparse_kernel PRIVATE
                "${root}/sparse_attn_sharedkv/op_kernel" "${root}/sparse_attn_sharedkv/op_host" "${root}/utils/kernel")
            ascendc_compile_options(native_sparse_kernel PRIVATE --cce-auto-sync=off -Wno-deprecated-declarations
                "SHELL:-mllvm -cce-vf-remove-membar=false")
            list(APPEND host_sources "${root}/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv.cpp"
                "${root}/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_tiling.cpp")
            list(APPEND host_includes "${root}/sparse_attn_sharedkv/op_host")
            list(APPEND kernel_targets native_sparse_kernel)
        elseif(op STREQUAL "sparse_attn_sharedkv_metadata_host" AND SGL_ARCH_VERSION STREQUAL "2201")
            list(APPEND host_sources "${root}/sparse_attn_sharedkv_metadata/op_host/sparse_attn_sharedkv_metadata.cpp")
        elseif(op STREQUAL "kv_quant_sparse_attn_sharedkv" AND SGL_ARCH_VERSION STREQUAL "3510")
            ascendc_library(native_quant_sparse_kernel STATIC "${root}/kv_quant_sparse_attn_sharedkv/op_kernel/kv_quant_sparse_attn_sharedkv.cpp")
            ascendc_include_directories(native_quant_sparse_kernel PRIVATE
                "${root}/kv_quant_sparse_attn_sharedkv/op_kernel" "${root}/kv_quant_sparse_attn_sharedkv/op_host"
                "${root}/attentions/csrc/ops/utils/inc/kernel" "${root}/utils/kernel")
            ascendc_compile_options(native_quant_sparse_kernel PRIVATE --cce-auto-sync=off -Wno-deprecated-declarations
                "SHELL:-mllvm -cce-vf-remove-membar=false" "SHELL:-mllvm -cce-aicore-hoist-movemask=false")
            foreach(unit IN ITEMS kv_quant_sparse_attn_sharedkv kv_quant_sparse_attn_sharedkv_check
                    kv_quant_sparse_attn_sharedkv_check_consistancy kv_quant_sparse_attn_sharedkv_check_existance
                    kv_quant_sparse_attn_sharedkv_check_feature kv_quant_sparse_attn_sharedkv_check_single_para
                    kv_quant_sparse_attn_sharedkv_tiling)
                list(APPEND host_sources "${root}/kv_quant_sparse_attn_sharedkv/op_host/${unit}.cpp")
            endforeach()
            list(APPEND host_includes "${root}/kv_quant_sparse_attn_sharedkv/op_host")
            list(APPEND kernel_targets native_quant_sparse_kernel)
        elseif(op STREQUAL "kv_quant_sparse_attn_sharedkv_metadata_host" AND SGL_ARCH_VERSION STREQUAL "3510")
            list(APPEND host_sources "${root}/kv_quant_sparse_attn_sharedkv_metadata/op_host/kv_quant_sparse_attn_sharedkv_metadata.cpp")
        elseif(op STREQUAL "kv_compress_epilog" AND SGL_ARCH_VERSION STREQUAL "3510")
            ascendc_library(native_kv_writer_kernel STATIC "${root}/kv_compress_epilog/op_kernel/kv_compress_epilog.cpp")
            ascendc_include_directories(native_kv_writer_kernel PRIVATE "${root}/kv_compress_epilog/op_kernel" "${root}/utils/kernel")
            list(APPEND host_sources "${root}/kv_compress_epilog/op_host/kv_compress_epilog.cpp")
            list(APPEND kernel_targets native_kv_writer_kernel)
        else()
            message(FATAL_ERROR "Unsupported native operator/architecture: ${op}/${SGL_ARCH_VERSION}")
        endif()
    endforeach()
    if(NOT SGL_NATIVE_OPERATORS)
        message(FATAL_ERROR "A native operator selection is required")
    endif()
    foreach(target IN LISTS kernel_targets)
        ascendc_compile_definitions(${target} PRIVATE -DHAVE_WORKSPACE -DHAVE_TILING)
    endforeach()


    target_sources(${target} PRIVATE ${host_sources})
    target_compile_definitions(${target} PRIVATE ${arch_definition})
    target_include_directories(${target} BEFORE PRIVATE
        "${root}/utils" "${SGL_SOURCE_ROOT}/include" ${host_includes}
        "${sdk}/include" "${sdk}/include/external"
        "${sdk}/include/experiment/platform" "${sdk}/include/experiment/runtime")
    foreach(name IN ITEMS tiling_api register platform ascendalog)
        find_library(${name}_library NAMES ${name} PATHS "${sdk}/lib64" NO_DEFAULT_PATH REQUIRED)
        target_link_libraries(${target} PRIVATE "${${name}_library}")
    endforeach()
    if("sparse_attn_sharedkv_metadata_host" IN_LIST SGL_NATIVE_OPERATORS OR
            "kv_quant_sparse_attn_sharedkv_metadata_host" IN_LIST SGL_NATIVE_OPERATORS)
        find_package(OpenMP REQUIRED COMPONENTS CXX)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_link_libraries(${target} PRIVATE ${kernel_targets})
    target_link_options(${target} PRIVATE "-Wl,-Bsymbolic-functions")
endfunction()
