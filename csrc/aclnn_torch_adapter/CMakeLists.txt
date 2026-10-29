cmake_minimum_required(VERSION 3.20)
project(sgl_cann_torch LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The CANN Torch extension requires Linux")
endif()
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
find_library(TORCH_NPU_LIBRARY torch_npu PATHS "${torch_npu_root}/lib" NO_DEFAULT_PATH REQUIRED)
set(sdk "$ENV{ASCEND_HOME_PATH}")
find_library(ACL_LIBRARY ascendcl PATHS "${sdk}/lib64" NO_DEFAULT_PATH REQUIRED)
find_library(NNOPBASE_LIBRARY nnopbase PATHS "${sdk}/lib64" NO_DEFAULT_PATH REQUIRED)
if(NOT IS_DIRECTORY "${SGL_CANN_VENDOR}/op_api/include")
    message(FATAL_ERROR "SGL_CANN_VENDOR must name this build's installed CANN vendor")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/operators.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/domains.cmake")
sgl_add_operator_domains()
sgl_select_operators(selected ARCH "${SGL_ARCH_VERSION}" OPERATORS ${SGL_CANN_OPERATORS})
set(adapters)
foreach(id IN LISTS selected)
    sgl_get_operator(adapter "${id}" TORCH_ADAPTER)
    if(NOT adapter)
        message(FATAL_ERROR "No Torch adapter declared for selected operator: ${id}")
    endif()
    list(APPEND adapters "${adapter}")
endforeach()
separate_arguments(torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
foreach(target IN ITEMS sgl_kernel_npu)
    add_library(${target} SHARED ${adapters})
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_options(${target} PRIVATE ${torch_flags})
    target_include_directories(${target} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${torch_npu_root}/include" "${sdk}/include" "${SGL_CANN_VENDOR}/op_api/include"
        ${Python3_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE ${TORCH_LIBRARIES} "${TORCH_NPU_LIBRARY}"
        "${ACL_LIBRARY}" "${NNOPBASE_LIBRARY}" ${CMAKE_DL_LIBS})
    # Dependencies are loaded by torch/torch_npu and the initialized CANN runtime.
    # Do not bake the build machine's absolute directories into the wheel.
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
        LIBRARY_OUTPUT_DIRECTORY "${SGL_TORCH_LIBRARY_DIR}"
        SKIP_BUILD_RPATH ON INSTALL_RPATH "")
endforeach()
