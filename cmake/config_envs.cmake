# find python binary
find_program(PYTHON_EXECUTABLE NAMES python3)

if (NOT EXISTS ${PYTHON_EXECUTABLE})
    message(FATAL_ERROR "python3 is not found, install python firstly")
endif ()

# get torch path, torch npu path, pybind11 path via python script
execute_process(
        COMMAND ${PYTHON_EXECUTABLE} "-c"
        "import torch; import torch_npu; import os; import pybind11;
torch_dir = os.path.realpath(os.path.dirname(torch.__file__));
torch_npu_dir = os.path.realpath(os.path.dirname(torch_npu.__file__));
pybind11_dir = os.path.realpath(os.path.dirname(pybind11.__file__));
abi_enabled=torch.compiled_with_cxx11_abi();
print(torch_dir, torch_npu_dir, pybind11_dir, abi_enabled, end='');
quit(0)
        "
        RESULT_VARIABLE EXEC_RESULT
        OUTPUT_VARIABLE OUTPUT_ENV_DEFINES)

# if failed to run the python script
if (NOT ${EXEC_RESULT} EQUAL 0)
    message(FATAL_ERROR "failed to get run python script to get ENVS like TORCH_DIR etc")
else ()
    message(STATUS "run python script successfully, output string is [${OUTPUT_ENV_DEFINES}]")
endif ()

# extract TORCH_DIR and set it
execute_process(
        COMMAND sh -c "echo \"${OUTPUT_ENV_DEFINES}\" | awk '{print $1}'"
        OUTPUT_VARIABLE TORCH_DIR
        RESULT_VARIABLE EXEC_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# extract TORCH_NPU_DIR and set it
execute_process(
        COMMAND sh -c "echo \"${OUTPUT_ENV_DEFINES}\" | awk '{print $2}'"
        OUTPUT_VARIABLE TORCH_NPU_DIR
        RESULT_VARIABLE EXEC_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# extract PYBIND11_DIR and set it
execute_process(
        COMMAND sh -c "echo \"${OUTPUT_ENV_DEFINES}\" | awk '{print $3}'"
        OUTPUT_VARIABLE PYBIND11_DIR
        RESULT_VARIABLE EXEC_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# extract PYTROCH_ABI and set it
execute_process(
        COMMAND sh -c "echo \"${OUTPUT_ENV_DEFINES}\" | awk '{print $4}'"
        OUTPUT_VARIABLE TORCH_API_ENABLED
        RESULT_VARIABLE EXEC_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "SOC_VERSION=${SOC_VERSION}")
message(STATUS "TORCH_DIR=${TORCH_DIR}")
message(STATUS "TORCH_NPU_DIR=${TORCH_NPU_DIR}")
message(STATUS "PYBIND11_DIR=${PYBIND11_DIR}")

# set _GLIBCXX_USE_CXX11_ABI
if (${TORCH_API_ENABLED} STREQUAL "True")
    add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=1)
    message(STATUS "_GLIBCXX_USE_CXX11_ABI=1")
else ()
    add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)
    message(STATUS "_GLIBCXX_USE_CXX11_ABI=0")
endif ()