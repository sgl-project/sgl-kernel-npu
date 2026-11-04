# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE-CANN in the root of the software repository for the full text of the License.
# Adapted from vllm-ascend csrc/cmake/{func,symbol,modules/Findaicpu}.cmake
# at af7c3fc267f8648ce6cac044111bc523478e76fb: HCC OBJECT targets and device linking.
# JSON packaging follows ops-transformer cmake/symbol.cmake at
# 39b52ad5e252d98e13df76ed45b6c68b21b98859, with Python invoked directly.

# Entered only for selected AICPU operators at the KERNEL checkpoint.
find_program(SGL_AICPU_CXX aarch64-target-linux-gnu-g++
    PATHS "${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin"
    NO_DEFAULT_PATH REQUIRED)
find_file(SGL_AICPU_CONTEXT libaicpu_context.a
    PATHS "${ASCEND_CANN_PACKAGE_PATH}/ops_base/lib64" "${ASCEND_CANN_PACKAGE_PATH}/lib64"
    NO_DEFAULT_PATH REQUIRED)
find_file(SGL_AICPU_PROTOBUF libbase_ascend_protobuf.a
    PATHS "${ASCEND_CANN_PACKAGE_PATH}/ops_base/lib64" "${ASCEND_CANN_PACKAGE_PATH}/lib64"
    NO_DEFAULT_PATH REQUIRED)
set(CMAKE_CXX_COMPILER "${SGL_AICPU_CXX}")

# Installed CANN exposes these headers at the root or the host platform prefix.
# Upstream dlog_headers supplies pkg_inc/base or include/toolchain for log_weak.h.
set(aicpu_includes)
foreach(prefix IN ITEMS "${ASCEND_CANN_PACKAGE_PATH}"
        "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux")
    foreach(suffix IN ITEMS include/aicpu include/experiment include/experiment/cce
            include/experiment/msprof pkg_inc pkg_inc/aicpu
            pkg_inc/base include/toolchain
            pkg_inc/aicpu_common/context pkg_inc/aicpu_common/context/common
            pkg_inc/aicpu_common/context/cpu_proto pkg_inc/aicpu_common/context/utils
            include/aicpu_common/context include/aicpu_common/context/common
            include/aicpu_common/context/cpu_proto include/aicpu_common/context/utils
            include/experiment/datagw/aicpu/common)
        if(IS_DIRECTORY "${prefix}/${suffix}")
            list(APPEND aicpu_includes "${prefix}/${suffix}")
        endif()
    endforeach()
endforeach()
find_path(SGL_AICPU_KERNEL_INCLUDE cpu_kernel.h
    PATHS ${aicpu_includes} NO_DEFAULT_PATH REQUIRED)

# One OBJECT target per operator keeps its private settings separate.
set(aicpu_targets)
set(aicpu_objects)
set(aicpu_json_sources)
set(aicpu_json_arguments)
foreach(id IN LISTS aicpu_operators)
    sgl_resolve_operator(op "${id}" "${SGL_ARCH_VERSION}")
    set(target "sgl_aicpu_${id}_objects")
    add_library(${target} OBJECT ${op_KERNEL_SOURCES})
    target_include_directories(${target} PRIVATE
        ${aicpu_includes} ${op_KERNEL_INCLUDE_DIRECTORIES})
    target_compile_definitions(${target} PRIVATE
        _FORTIFY_SOURCE=2 _GLIBCXX_USE_CXX11_ABI=1 DISABLE_COMPILE_V1
        google=ascend_private ${op_KERNEL_COMPILE_DEFINITIONS})
    target_compile_options(${target} PRIVATE
        -O2 -fPIC -fstack-protector-all -fvisibility=hidden
        -fvisibility-inlines-hidden -fno-common ${op_KERNEL_COMPILE_OPTIONS})
    list(APPEND aicpu_targets ${target})
    list(APPEND aicpu_objects "$<TARGET_OBJECTS:${target}>")
    list(APPEND aicpu_json_sources ${op_AICPU_JSON})
    list(APPEND aicpu_json_arguments --operator "${op_CANN_OP_TYPE}=${op_AICPU_JSON}")
endforeach()

# Keep the CpuKernel registry and protobuf code inside the device shared library.
set(aicpu_library "${CMAKE_CURRENT_BINARY_DIR}/libsgl_kernel_npu_aicpu.so")
add_custom_command(
    OUTPUT "${aicpu_library}"
    COMMAND "${SGL_AICPU_CXX}" -shared ${aicpu_objects}
        -Wl,--whole-archive "${SGL_AICPU_CONTEXT}" "${SGL_AICPU_PROTOBUF}"
        -Wl,--no-whole-archive -Wl,-Bsymbolic
        -Wl,--exclude-libs=libbase_ascend_protobuf.a -Wl,-z,now
        -o "${aicpu_library}"
    DEPENDS ${aicpu_targets} ${aicpu_objects}
        "${SGL_AICPU_CONTEXT}" "${SGL_AICPU_PROTOBUF}"
    VERBATIM COMMAND_EXPAND_LISTS)
add_custom_target(sgl_cann_aicpu DEPENDS "${aicpu_library}")

# Registration is checked separately so device-only builds need no JSON.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(aicpu_json "${CMAKE_CURRENT_BINARY_DIR}/cust_aicpu_kernel.json")
set(merge_json_script "${CMAKE_CURRENT_SOURCE_DIR}/merge_json.py")
get_filename_component(aicpu_library_name "${aicpu_library}" NAME)
add_custom_command(
    OUTPUT "${aicpu_json}"
    COMMAND "${Python3_EXECUTABLE}" "${merge_json_script}"
        --output "${aicpu_json}" --library "${aicpu_library_name}"
        ${aicpu_json_arguments}
    DEPENDS ${aicpu_json_sources} "${merge_json_script}"
    VERBATIM COMMAND_EXPAND_LISTS)
add_custom_target(sgl_cann_aicpu_metadata DEPENDS "${aicpu_json}")

# Use the existing RUN package and the upstream custom AICPU vendor layout.
npu_op_package_add(sgl_kernel_npu
    FILES "${aicpu_json}" PACKAGE_PATH op_impl/cpu/config)
npu_op_package_add(sgl_kernel_npu
    FILES "${aicpu_library}" PACKAGE_PATH op_impl/cpu/aicpu_kernel/impl)
