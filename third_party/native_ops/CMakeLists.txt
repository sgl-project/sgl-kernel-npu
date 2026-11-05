cmake_minimum_required(VERSION 3.20)
project(sgl_native_ops LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Bundled native operators require Linux and CANN")
endif()
set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}")
set(SOC_VERSION "${SGL_NATIVE_SOC}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
foreach(candidate IN ITEMS tools/tikcpp/ascendc_kernel_cmake compiler/tikcpp/ascendc_kernel_cmake ascendc_devkit/tikcpp/samples/cmake)
    if(EXISTS "${ASCEND_HOME_PATH}/${candidate}/ascendc.cmake")
        set(ASCENDC_CMAKE_DIR "${ASCEND_HOME_PATH}/${candidate}")
        break()
    endif()
endforeach()
if(NOT ASCENDC_CMAKE_DIR)
    message(FATAL_ERROR "Cannot find ascendc.cmake in the initialized SDK")
endif()
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
    "import pathlib, torch, torch_npu; print(torch.utils.cmake_prefix_path); print(pathlib.Path(torch_npu.__file__).parent)"
    RESULT_VARIABLE python_result OUTPUT_VARIABLE python_paths OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT python_result EQUAL 0)
    message(FATAL_ERROR "The build Python must provide torch and torch_npu")
endif()
string(REPLACE "\n" ";" python_paths "${python_paths}")
list(GET python_paths 0 torch_prefix)
list(GET python_paths 1 torch_npu_root)
list(PREPEND CMAKE_PREFIX_PATH "${torch_prefix}")
find_package(Torch REQUIRED)
if(SGL_ARCH_VERSION STREQUAL "2201")
    set(kernel_arch arch22)
    set(arch_definition SGL_KERNEL_ARCH_22 SGL_KERNEL_ENABLE_A3_ONLY_OPS)
elseif(SGL_ARCH_VERSION STREQUAL "3510")
    set(kernel_arch arch35)
    set(arch_definition SGL_KERNEL_ARCH_35 SGL_KERNEL_ENABLE_A5_ONLY_OPS)
else()
    message(FATAL_ERROR "Unsupported native architecture: ${SGL_ARCH_VERSION}")
endif()
if(NOT SGL_NATIVE_PACKAGE MATCHES "^sgl_native_[a-z0-9_]+$")
    message(FATAL_ERROR "Invalid private native package")
endif()
set(root "${SGL_NATIVE_SOURCE}/csrc")
set(host_sources "${SGL_REGISTRATION_SOURCE}")
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

add_library(${SGL_NATIVE_PACKAGE} SHARED ${host_sources})
separate_arguments(torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(${SGL_NATIVE_PACKAGE} PRIVATE ${torch_flags})
target_compile_definitions(${SGL_NATIVE_PACKAGE} PRIVATE ${arch_definition})
get_target_property(torch_cxx_standard torch CXX_STANDARD)
if(torch_cxx_standard)
    set_property(TARGET ${SGL_NATIVE_PACKAGE} PROPERTY CXX_STANDARD ${torch_cxx_standard})
endif()
# Keep SDK GE headers before torch_npu's copies, preserving the legacy build order.
target_include_directories(${SGL_NATIVE_PACKAGE} PRIVATE
    "${root}/utils" "${SGL_NATIVE_SOURCE}/include" ${host_includes}
    "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/include/external"
    "${ASCEND_HOME_PATH}/include/experiment/platform" "${ASCEND_HOME_PATH}/include/experiment/runtime"
    "${torch_npu_root}/include" ${Python3_INCLUDE_DIRS})
foreach(name IN ITEMS torch_npu ascendcl tiling_api register platform ascendalog)
    find_library(${name}_library NAMES ${name}
        PATHS "${torch_npu_root}/lib" "${ASCEND_HOME_PATH}/lib64" NO_DEFAULT_PATH REQUIRED)
    target_link_libraries(${SGL_NATIVE_PACKAGE} PRIVATE "${${name}_library}")
endforeach()
if("sparse_attn_sharedkv_metadata_host" IN_LIST SGL_NATIVE_OPERATORS OR
        "kv_quant_sparse_attn_sharedkv_metadata_host" IN_LIST SGL_NATIVE_OPERATORS)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(${SGL_NATIVE_PACKAGE} PRIVATE OpenMP::OpenMP_CXX)
endif()
target_link_libraries(${SGL_NATIVE_PACKAGE} PRIVATE ${kernel_targets} ${TORCH_LIBRARIES} ${CMAKE_DL_LIBS})
# Another plugin exporting legacy C++ symbols must not replace this pinned copy.
target_link_options(${SGL_NATIVE_PACKAGE} PRIVATE "-Wl,-Bsymbolic-functions")
set_target_properties(${SGL_NATIVE_PACKAGE} PROPERTIES
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY "${SGL_NATIVE_OUTPUT}"
    SKIP_BUILD_RPATH ON INSTALL_RPATH "")
