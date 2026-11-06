# Torch headers and ABI flags shared by HOST implementations and Torch adapters.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
    "import torch; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE python_result OUTPUT_VARIABLE torch_prefix OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT python_result EQUAL 0)
    message(FATAL_ERROR "The build Python must provide torch")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${torch_prefix}")
find_package(Torch REQUIRED)
separate_arguments(torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

# Only callers declaring TORCH_NPU need its headers or Python import.
function(sgl_find_torch_npu output)
    execute_process(COMMAND "${Python3_EXECUTABLE}" -c
        "import pathlib, torch_npu; print(pathlib.Path(torch_npu.__file__).parent)"
        RESULT_VARIABLE python_result OUTPUT_VARIABLE torch_npu_root OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT python_result EQUAL 0)
        message(FATAL_ERROR "The build Python must provide torch_npu")
    endif()
    set(${output} "${torch_npu_root}" PARENT_SCOPE)
endfunction()
