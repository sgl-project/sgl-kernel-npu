cmake_minimum_required(VERSION 3.20)
project(sgl_native_compressor LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The bundled compressor requires Linux and CANN")
endif()
set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}")
set(SOC_VERSION "${SGL_COMPRESSOR_SOC}")
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
    set(compressor_arch arch22)
    set(compressor_definition SGL_KERNEL_ARCH_22)
elseif(SGL_ARCH_VERSION STREQUAL "3510")
    set(compressor_arch arch35)
    set(compressor_definition SGL_KERNEL_ARCH_35)
else()
    message(FATAL_ERROR "Unsupported compressor architecture: ${SGL_ARCH_VERSION}")
endif()
set(compressor "${SGL_NATIVE_SOURCE}/csrc/compressor")
ascendc_library(compressor_kernel STATIC "${compressor}/op_kernel/compressor.cpp")
ascendc_include_directories(compressor_kernel PRIVATE
    "${compressor}/op_kernel" "${compressor}/op_kernel/${compressor_arch}"
    "${SGL_NATIVE_SOURCE}/csrc/utils/kernel")
ascendc_compile_definitions(compressor_kernel PRIVATE -DHAVE_WORKSPACE -DHAVE_TILING)

add_library(sgl_native_compressor SHARED
    "${SGL_REGISTRATION_SOURCE}"
    "${compressor}/op_host/compressor.cpp"
    "${compressor}/op_host/${compressor_arch}/compressor_tiling.cpp")
separate_arguments(torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(sgl_native_compressor PRIVATE ${torch_flags})
target_compile_definitions(sgl_native_compressor PRIVATE ${compressor_definition})
# Match the installed Torch headers while retaining C++17 for the device kernel.
get_target_property(torch_cxx_standard torch CXX_STANDARD)
if(torch_cxx_standard)
    set_property(TARGET sgl_native_compressor PROPERTY CXX_STANDARD ${torch_cxx_standard})
endif()
# SDK GE headers must precede torch_npu's bundled copies, as in the legacy build.
target_include_directories(sgl_native_compressor PRIVATE
    "${SGL_NATIVE_SOURCE}/csrc/utils" "${SGL_NATIVE_SOURCE}/include"
    "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/include/external"
    "${ASCEND_HOME_PATH}/include/experiment/platform"
    "${ASCEND_HOME_PATH}/include/experiment/runtime"
    "${torch_npu_root}/include" ${Python3_INCLUDE_DIRS})
foreach(name IN ITEMS torch_npu ascendcl tiling_api register platform ascendalog)
    find_library(${name}_library NAMES ${name}
        PATHS "${torch_npu_root}/lib" "${ASCEND_HOME_PATH}/lib64" NO_DEFAULT_PATH REQUIRED)
    target_link_libraries(sgl_native_compressor PRIVATE "${${name}_library}")
endforeach()
target_link_libraries(sgl_native_compressor PRIVATE compressor_kernel ${TORCH_LIBRARIES} ${CMAKE_DL_LIBS})
# Bind this copy's host/launch functions locally even if another plugin exported
# the same legacy C++ symbols before the bundled library was loaded.
target_link_options(sgl_native_compressor PRIVATE "-Wl,-Bsymbolic-functions")
set_target_properties(sgl_native_compressor PROPERTIES
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY "${SGL_NATIVE_OUTPUT}"
    SKIP_BUILD_RPATH ON INSTALL_RPATH "")
