# Operator declarations only: this module does not load Torch or the CANN SDK.
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/operator_variants.cmake")

function(_sgl_operator_source_paths output base_dir)
    # Resolve declaration-relative paths and return them to the caller.
    set(paths)
    foreach(source IN LISTS ARGN)
        get_filename_component(path "${source}" ABSOLUTE BASE_DIR "${base_dir}")
        list(APPEND paths "${path}")
    endforeach()
    set(${output} "${paths}" PARENT_SCOPE)
endfunction()

# Paths are relative to the declaring CMake file, including when it is included
# from another directory. Only handwritten inputs belong in these source lists;
# generated SDK outputs will be described by the backend's build rules.
# Derive SUPPORTED_ARCHS and any per-architecture sources from the legacy build.
# The current build target or validation machine must not redefine that scope.
function(sgl_register_operator)
    # Parse the declaration and reject unknown, missing, or invalid fields.
    _sgl_operator_build_fields(build_fields)
    _sgl_check_duplicate_keywords("Operator"
        "ID;EXECUTION_UNIT;CANN_OP_TYPE;TORCH_ADAPTER;SUPPORTED_ARCHS;${build_fields}" ${ARGV})
    cmake_parse_arguments(PARSE_ARGV 0 op ""
        "ID;EXECUTION_UNIT;CANN_OP_TYPE;TORCH_ADAPTER"
        "SUPPORTED_ARCHS;${build_fields}")
    if(NOT "${op_UNPARSED_ARGUMENTS}" STREQUAL "")
        message(FATAL_ERROR "Unknown operator arguments: ${op_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT "${op_KEYWORDS_MISSING_VALUES}" STREQUAL "")
        message(FATAL_ERROR "Operator fields need values: ${op_KEYWORDS_MISSING_VALUES}")
    endif()
    foreach(field IN ITEMS ID EXECUTION_UNIT)
        if(NOT DEFINED op_${field} OR "${op_${field}}" STREQUAL "")
            message(FATAL_ERROR "Operator declaration is missing required field: ${field}")
        endif()
    endforeach()
    if(NOT "${op_ID}" MATCHES "^[a-z][a-z0-9_]*$")
        message(FATAL_ERROR "Operator ID must use lowercase letters, digits and underscores: ${op_ID}")
    endif()

    if(NOT "${op_EXECUTION_UNIT}" STREQUAL "AICORE"
       AND NOT "${op_EXECUTION_UNIT}" STREQUAL "AICPU")
        message(FATAL_ERROR "Unknown EXECUTION_UNIT: ${op_EXECUTION_UNIT}")
    endif()
    foreach(field IN ITEMS CANN_OP_TYPE SUPPORTED_ARCHS)
        if(NOT DEFINED op_${field} OR "${op_${field}}" STREQUAL "")
            message(FATAL_ERROR "Operator declaration is missing required field: ${field}")
        endif()
    endforeach()

    # Check architecture coverage and keep operator IDs and CANN types unique.
    foreach(arch IN LISTS op_SUPPORTED_ARCHS)
        if(NOT arch MATCHES "^(2201|3510)$")
            message(FATAL_ERROR "Unsupported ARCH in ${op_ID}: ${arch}")
        endif()
    endforeach()
    get_property(ids GLOBAL PROPERTY SGL_OPERATOR_IDS)
    if(op_ID IN_LIST ids)
        message(FATAL_ERROR "Duplicate operator ID: ${op_ID}")
    endif()
    foreach(id IN LISTS ids)
        get_property(op_type GLOBAL PROPERTY SGL_OPERATOR_${id}_CANN_OP_TYPE)
        if("${op_type}" STREQUAL "${op_CANN_OP_TYPE}")
            message(FATAL_ERROR "Duplicate CANN_OP_TYPE: ${op_CANN_OP_TYPE} (${id}, ${op_ID})")
        endif()
    endforeach()

    # Normalize handwritten input paths relative to the declaring CMake file.
    foreach(field IN LISTS build_fields)
        if(field MATCHES "SOURCES$|DIRECTORIES$")
            _sgl_operator_source_paths(value "${CMAKE_CURRENT_LIST_DIR}" ${op_${field}})
        else()
            set(value "${op_${field}}")
        endif()
        set_property(GLOBAL PROPERTY SGL_OPERATOR_${op_ID}_${field} "${value}")
    endforeach()
    set(torch_adapter)
    if(DEFINED op_TORCH_ADAPTER)
        _sgl_operator_source_paths(torch_adapter "${CMAKE_CURRENT_LIST_DIR}" "${op_TORCH_ADAPTER}")
    endif()

    # Store metadata shared across domains; this does not set compiler flags.
    foreach(field IN ITEMS ID EXECUTION_UNIT CANN_OP_TYPE SUPPORTED_ARCHS)
        set_property(GLOBAL PROPERTY SGL_OPERATOR_${op_ID}_${field} "${op_${field}}")
    endforeach()
    set_property(GLOBAL PROPERTY SGL_OPERATOR_${op_ID}_TORCH_ADAPTER "${torch_adapter}")
    set_property(GLOBAL APPEND PROPERTY SGL_OPERATOR_IDS "${op_ID}")
endfunction()

function(sgl_get_registered_operators output)
    # Return the registered IDs through the caller's named output variable.
    get_property(ids GLOBAL PROPERTY SGL_OPERATOR_IDS)
    set(${output} "${ids}" PARENT_SCOPE)
endfunction()

function(sgl_get_operator output id field)
    # Validate the lookup before returning one field from the shared registry.
    get_property(ids GLOBAL PROPERTY SGL_OPERATOR_IDS)
    if(NOT id IN_LIST ids)
        message(FATAL_ERROR "Unknown operator ID: ${id}")
    endif()
    _sgl_operator_build_fields(fields)
    list(APPEND fields ID EXECUTION_UNIT CANN_OP_TYPE SUPPORTED_ARCHS TORCH_ADAPTER)
    if(NOT field IN_LIST fields)
        message(FATAL_ERROR "Unknown operator field: ${field}")
    endif()
    get_property(value GLOBAL PROPERTY SGL_OPERATOR_${id}_${field})
    set(${output} "${value}" PARENT_SCOPE)
endfunction()

# ALL selects every registered operator supporting the explicit target.
# Explicit IDs remain strict: an unsupported request must not be dropped.
function(sgl_select_operators output)
    # Validate the requested architecture and operator list.
    cmake_parse_arguments(PARSE_ARGV 1 selection "" "ARCH" "OPERATORS")
    if(NOT "${selection_UNPARSED_ARGUMENTS}" STREQUAL "")
        message(FATAL_ERROR "Unknown selection arguments: ${selection_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT "${selection_KEYWORDS_MISSING_VALUES}" STREQUAL "")
        message(FATAL_ERROR "Selection fields need values: ${selection_KEYWORDS_MISSING_VALUES}")
    endif()
    foreach(field IN ITEMS ARCH OPERATORS)
        if(NOT DEFINED selection_${field} OR "${selection_${field}}" STREQUAL "")
            message(FATAL_ERROR "Selection is missing required field: ${field}")
        endif()
    endforeach()
    _sgl_validate_operator_arch("${selection_ARCH}")
    # Expand ALL to the registry; explicit selections remain strict.
    set(requested ${selection_OPERATORS})
    if("ALL" IN_LIST requested)
        list(LENGTH requested count)
        if(NOT count EQUAL 1)
            message(FATAL_ERROR "ALL cannot be combined with explicit operator IDs")
        endif()
        sgl_get_registered_operators(requested)
    endif()
    # Filter by support scope and backend; resolved inputs are checked later.
    set(selected)
    foreach(id IN LISTS requested)
        if(id IN_LIST selected)
            message(FATAL_ERROR "Duplicate requested operator: ${id}")
        endif()
        sgl_get_operator(supported_archs "${id}" SUPPORTED_ARCHS)
        if(NOT selection_ARCH IN_LIST supported_archs)
            if(selection_OPERATORS STREQUAL "ALL")
                message(STATUS "Excluded CANN operator: ${id} (supports ${supported_archs}, target ${selection_ARCH})")
                continue()
            endif()
            message(FATAL_ERROR "Operator ${id} does not support requested ARCH: ${selection_ARCH}")
        endif()
        sgl_get_operator(unit "${id}" EXECUTION_UNIT)
        if(NOT unit STREQUAL "AICORE")
            message(FATAL_ERROR "Selected EXECUTION_UNIT backend is not implemented: ${id}/${unit}")
        endif()
        list(APPEND selected "${id}")
    endforeach()
    # Reject an empty result and return the selected IDs to the caller.
    if(NOT selected)
        message(FATAL_ERROR "No CANN operators support requested ARCH: ${selection_ARCH}")
    endif()
    set(${output} "${selected}" PARENT_SCOPE)
endfunction()
