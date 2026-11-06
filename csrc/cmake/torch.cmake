# Torch headers and ABI flags shared by HOST implementations and Torch adapters.
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
separate_arguments(torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
