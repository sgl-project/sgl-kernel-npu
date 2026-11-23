# Entered only for selected HOST operators at the KERNEL checkpoint.
include("${CMAKE_CURRENT_LIST_DIR}/../torch.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../host_ops.cmake")
sgl_collect_host_sources(host_sources host_adapters host_definitions
    ARCH "${SGL_ARCH_VERSION}" OPERATORS ${host_operators})

# Keep Torch registrations in the final extension, so static linking cannot
# discard their initializers. Only the implementations belong in this archive.
add_library(sgl_host_operators STATIC ${host_sources})
target_compile_features(sgl_host_operators PRIVATE cxx_std_17)
target_compile_options(sgl_host_operators PRIVATE ${torch_flags})
target_compile_definitions(sgl_host_operators PRIVATE ${host_definitions})
# Match the legacy Release defaults without changing the parent CANN configuration.
# Explicit build configurations keep their toolchain-provided flags.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    target_compile_options(sgl_host_operators PRIVATE -O3)
    target_compile_definitions(sgl_host_operators PRIVATE NDEBUG)
endif()
target_include_directories(sgl_host_operators PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../../aclnn_torch_adapter"
    "${torch_npu_root}/include" "${ASCEND_CANN_PACKAGE_PATH}/include"
    ${Python3_INCLUDE_DIRS})
target_link_libraries(sgl_host_operators PRIVATE ${TORCH_LIBRARIES})
set_target_properties(sgl_host_operators PROPERTIES
    POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# CCE kernels declared by the selected HOST operators: the bisheng compiler
# emits one object per declared variant and the objects are archived into the
# same HOST library, so build_torch() links the device kernels directly into
# libsgl_kernel_npu.so without any separate runtime packaging.
sgl_collect_cce_kernels(cce_kernel_specs ARCH "${SGL_ARCH_VERSION}"
    OPERATORS ${host_operators})
set(cce_kernel_objects)
if(cce_kernel_specs)
    # Prefer the bisheng shipped with the CANN the build already uses, so the
    # compiler and its headers stay consistent; fall back to $BISHENG, then PATH.
    if(ASCEND_CANN_PACKAGE_PATH AND EXISTS "${ASCEND_CANN_PACKAGE_PATH}/bin/bisheng")
        set(sgl_cce_compiler "${ASCEND_CANN_PACKAGE_PATH}/bin/bisheng")
    elseif(DEFINED ENV{BISHENG} AND NOT "$ENV{BISHENG}" STREQUAL "")
        set(sgl_cce_compiler "$ENV{BISHENG}")
    else()
        find_program(sgl_cce_compiler NAMES bisheng REQUIRED)
    endif()
    if(SGL_ARCH_VERSION STREQUAL "3510")
        set(sgl_cce_npu_arch dav-3510)
    else()
        message(FATAL_ERROR "No CCE --npu-arch mapping for architecture ${SGL_ARCH_VERSION}")
    endif()
    # bisheng's internal objcopy step needs the output directory to exist.
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/cce")
    foreach(spec IN LISTS cce_kernel_specs)
        string(REPLACE "|" ";" spec_fields "${spec}")
        list(GET spec_fields 0 cce_id)
        list(GET spec_fields 1 cce_variant)
        list(GET spec_fields 2 cce_source)
        list(GET spec_fields 3 cce_definition_text)
        string(REPLACE "," ";" cce_definitions "${cce_definition_text}")
        set(cce_object "${CMAKE_CURRENT_BINARY_DIR}/cce/${cce_id}-${cce_variant}.o")
        set(cce_flags -std=c++17 --npu-arch=${sgl_cce_npu_arch} -O2 -Werror -fPIC -c
            "-I${ASCEND_CANN_PACKAGE_PATH}/include")
        foreach(definition IN LISTS cce_definitions)
            list(APPEND cce_flags "-D${definition}")
        endforeach()
        add_custom_command(
            OUTPUT "${cce_object}"
            COMMAND "${sgl_cce_compiler}" ${cce_flags} "${cce_source}" -o "${cce_object}"
            DEPENDS "${cce_source}"
            COMMENT "CCE kernel ${cce_id}/${cce_variant} (${cce_source})"
            VERBATIM)
        list(APPEND cce_kernel_objects "${cce_object}")
    endforeach()
    target_sources(sgl_host_operators PRIVATE ${cce_kernel_objects})
    set_source_files_properties(${cce_kernel_objects} PROPERTIES
        EXTERNAL_OBJECT TRUE GENERATED TRUE)
endif()

# RUN installs this archive; build_torch() later links it with the Torch adapters.
# Executing the installer does not execute or register the HOST operators.
npu_op_package_add(sgl_kernel_npu
    FILES "${CMAKE_CURRENT_BINARY_DIR}/libsgl_host_operators.a"
    PACKAGE_PATH op_impl/host/lib)
