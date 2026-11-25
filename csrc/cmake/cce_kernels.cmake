# CCE objects are linked into the CCE archive.
include_guard(GLOBAL)

# CCE kernels declared through sgl_register_cce_kernel() for the selected CCE
# operators. Each spec is one bisheng compilation encoded as
# "<operator-id>|<variant>|<source>|<definition>,<definition>,..." with the
# definition list comma-joined so a spec stays a single CMake list element.
function(sgl_collect_cce_kernels out_specs)
    cmake_parse_arguments(PARSE_ARGV 1 selection "" "ARCH" "OPERATORS")
    if(NOT "${selection_UNPARSED_ARGUMENTS}" STREQUAL "" OR "ARCH" IN_LIST selection_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR "Invalid CCE compilation selection")
    endif()
    if(NOT DEFINED selection_OPERATORS OR "${selection_OPERATORS}" STREQUAL "")
        message(FATAL_ERROR "CCE compilation selection requires OPERATORS")
    endif()
    set(specs)
    foreach(id IN LISTS selection_OPERATORS)
        sgl_get_operator_field(unit "${id}" EXECUTION_UNIT)
        if(NOT unit STREQUAL "CCE")
            message(FATAL_ERROR "CCE compilation requires EXECUTION_UNIT CCE: ${id}")
        endif()
        set(prefix "SGL_OPERATOR_${id}_CCE_ARCH_${selection_ARCH}")
        get_property(variants GLOBAL PROPERTY "${prefix}_VARIANTS")
        foreach(variant IN LISTS variants)
            get_property(source GLOBAL PROPERTY "${prefix}_${variant}_SOURCE")
            get_property(definitions GLOBAL PROPERTY "${prefix}_${variant}_COMPILE_DEFINITIONS")
            string(REPLACE ";" "," definition_text "${definitions}")
            list(APPEND specs "${id}|${variant}|${source}|${definition_text}")
        endforeach()
    endforeach()
    set(${out_specs} "${specs}" PARENT_SCOPE)
endfunction()


function(sgl_add_cce_kernels target)
    # Compile variants independently; their objects share the CCE archive.
    sgl_collect_cce_kernels(cce_kernel_specs ARCH "${SGL_ARCH_VERSION}" OPERATORS ${ARGN})
    if(cce_kernel_specs)
        if(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/bin/bisheng")
            set(cce_compiler "${ASCEND_CANN_PACKAGE_PATH}/bin/bisheng")
        elseif(NOT "$ENV{BISHENG}" STREQUAL "")
            set(cce_compiler "$ENV{BISHENG}")
        else()
            find_program(cce_compiler NAMES bisheng REQUIRED)
        endif()
        if(NOT SGL_ARCH_VERSION STREQUAL "3510")
            message(FATAL_ERROR "No CCE architecture mapping for ${SGL_ARCH_VERSION}")
        endif()
        set(cce_dir "${CMAKE_CURRENT_BINARY_DIR}/cce")
        file(MAKE_DIRECTORY "${cce_dir}")
        foreach(spec IN LISTS cce_kernel_specs)
            string(REPLACE "|" ";" fields "${spec}")
            list(GET fields 0 id)
            list(GET fields 1 variant)
            list(GET fields 2 source)
            list(GET fields 3 definitions)
            string(REPLACE "," ";" definitions "${definitions}")
            list(TRANSFORM definitions PREPEND "-D")
            set(flags -std=c++17 --npu-arch=dav-3510 -O2 -Werror -fPIC -c
                "-I${ASCEND_CANN_PACKAGE_PATH}/include" ${definitions})
            set(object "${cce_dir}/${id}-${variant}.o")
            # Track project headers conservatively without relying on mixed-mode depfile support.
            get_filename_component(op_root "${source}" DIRECTORY)
            get_filename_component(op_root "${op_root}" DIRECTORY)
            file(GLOB_RECURSE headers CONFIGURE_DEPENDS
                "${op_root}/*.h" "${op_root}/*.hpp" "${op_root}/*.inc"
                "${PROJECT_SOURCE_DIR}/../include/*.h" "${PROJECT_SOURCE_DIR}/../include/*.hpp")
            # file(CONFIGURE) preserves the timestamp when the command is unchanged.
            file(CONFIGURE OUTPUT "${object}.command" CONTENT "${cce_compiler};${source};${flags};${headers}" @ONLY)
            add_custom_command(OUTPUT "${object}"
                COMMAND "${CMAKE_COMMAND}" -E make_directory "${cce_dir}"
                COMMAND "${cce_compiler}" ${flags} "${source}" -o "${object}"
                DEPENDS "${source}" ${headers} "${object}.command" "${cce_compiler}"
                COMMENT "CCE ${id}/${variant}" VERBATIM)
            set_source_files_properties("${object}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
            target_sources(${target} PRIVATE "${object}")
        endforeach()
    endif()

endfunction()
