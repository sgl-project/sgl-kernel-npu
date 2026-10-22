include_guard(GLOBAL)

function(sgl_generate_cann_interfaces output_dir)
    if(NOT COMMAND npu_op_code_gen)
        message(FATAL_ERROR "The selected CANN SDK does not provide npu_op_code_gen")
    endif()
    set(host_sources)
    foreach(id IN LISTS ARGN)
        sgl_get_operator(sources "${id}" HOST_SOURCES)
        list(APPEND host_sources ${sources})
        foreach(source IN LISTS sources)
            get_filename_component(directory "${source}" DIRECTORY)
            file(GLOB_RECURSE headers CONFIGURE_DEPENDS "${directory}/*.h" "${directory}/*.hpp")
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${headers})
        endforeach()
    endforeach()

    set(config_dir "${CMAKE_CURRENT_BINARY_DIR}/operator-config")
    file(MAKE_DIRECTORY "${config_dir}")
    file(CONFIGURE OUTPUT "${config_dir}/sgl_operator_config.h" CONTENT
        "#pragma once\n#define SGL_CANN_SOC_CONFIG \"${SGL_CANN_SOC}\"\n")
    # Opbuild executes during configuration. Track handwritten inputs for rebuilds.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${host_sources})
    file(REMOVE_RECURSE "${output_dir}")
    file(MAKE_DIRECTORY "${output_dir}")
    npu_op_code_gen(
        SRC ${host_sources}
        PACKAGE sgl_kernel_npu
        COMPILE_OPTIONS "-I${config_dir}"
        OUT_DIR "${output_dir}"
    )
endfunction()

function(sgl_add_cann_tiling)
    set(host_sources)
    foreach(id IN LISTS ARGN)
        sgl_get_operator(sources "${id}" HOST_SOURCES)
        list(APPEND host_sources ${sources})
    endforeach()
    npu_op_library(sgl_cann_tiling TILING ${host_sources})
    target_include_directories(sgl_cann_tiling PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/operator-config")
    set(SGL_CANN_HOST_SOURCES "${host_sources}" PARENT_SCOPE)
endfunction()

function(sgl_add_cann_kernels output_dir)
    file(GLOB api_sources "${output_dir}/aclnn*.cpp")
    if(NOT api_sources OR NOT EXISTS "${output_dir}/op_proto.cc")
        message(FATAL_ERROR "SDK did not generate the required ACLNN / op_proto sources in ${output_dir}")
    endif()
    npu_op_library(sgl_cann_opapi ACLNN ${api_sources})
    npu_op_library(sgl_cann_proto GRAPH ${SGL_CANN_HOST_SOURCES} "${output_dir}/op_proto.cc")
    target_include_directories(sgl_cann_proto PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/operator-config")
    foreach(id IN LISTS ARGN)
        sgl_get_operator(op_type "${id}" CANN_OP_TYPE)
        sgl_get_operator(sources "${id}" KERNEL_SOURCES)
        list(LENGTH sources count)
        if(NOT count EQUAL 1)
            message(FATAL_ERROR "This checkpoint requires one Kernel entry source per operator: ${id}")
        endif()
        get_filename_component(directory "${sources}" DIRECTORY)
        get_filename_component(filename "${sources}" NAME)
        file(RELATIVE_PATH relative_dir "${CMAKE_CURRENT_SOURCE_DIR}" "${directory}")
        # Device selection is handled before this shared source mapping.
        # COMPUTE_UNIT here requires a chip model, not a compute-unit family.
        npu_op_kernel_sources(sgl_cann_kernels OP_TYPE "${op_type}"
            KERNEL_DIR "${relative_dir}" KERNEL_FILE "${filename}")
    endforeach()
    npu_op_kernel_library(sgl_cann_kernels SRC_BASE "${CMAKE_CURRENT_SOURCE_DIR}"
        TILING_LIBRARY sgl_cann_tiling)
    npu_op_package_add(sgl_kernel_npu LIBRARY sgl_cann_opapi sgl_cann_proto sgl_cann_tiling sgl_cann_kernels)
    if(NOT TARGET binary)
        message(FATAL_ERROR "The SDK did not create the required binary build target")
    endif()
    add_custom_target(sgl_cann_kernel_check DEPENDS binary sgl_cann_opapi sgl_cann_proto)
endfunction()
