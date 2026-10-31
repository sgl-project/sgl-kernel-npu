include_guard(GLOBAL)
# Match public operator domains with their CANN declaration directories.
get_filename_component(SGL_OPERATOR_DOMAIN_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
get_filename_component(SGL_PUBLIC_OPERATORS_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../../python/sgl_kernel_npu/sgl_kernel_npu/operators" ABSOLUTE)

function(sgl_add_operator_domains)
    if(NOT IS_DIRECTORY "${SGL_PUBLIC_OPERATORS_ROOT}")
        message(FATAL_ERROR "Public operator domains directory not found: ${SGL_PUBLIC_OPERATORS_ROOT}")
    endif()

    # Track both sides so newly added domain directories or CMake entries reconfigure.
    file(GLOB public_entries LIST_DIRECTORIES true CONFIGURE_DEPENDS "${SGL_PUBLIC_OPERATORS_ROOT}/*")
    file(GLOB cann_entries CONFIGURE_DEPENDS "${SGL_OPERATOR_DOMAIN_ROOT}/*/CMakeLists.txt")
    list(SORT public_entries)
    foreach(entry IN LISTS public_entries)
        if(NOT IS_DIRECTORY "${entry}")
            continue()
        endif()
        get_filename_component(domain "${entry}" NAME)
        # Ignore utils, hidden directories, and Python's double-underscore entries.
        if(domain STREQUAL "utils" OR domain MATCHES "^__" OR domain MATCHES "^\\.")
            continue()
        endif()
        set(declaration "${SGL_OPERATOR_DOMAIN_ROOT}/${domain}/CMakeLists.txt")
        if(declaration IN_LIST cann_entries)
            message(STATUS "Discovered CANN operator domain: ${domain}")
            add_subdirectory("${SGL_OPERATOR_DOMAIN_ROOT}/${domain}" "declarations/${domain}")
        endif()
    endforeach()
endfunction()

function(sgl_discover_operators)
    # Discover direct child declarations, then load them in a stable order.
    file(GLOB declarations CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*/CMakeLists.txt")
    list(SORT declarations)
    foreach(declaration IN LISTS declarations)
        get_filename_component(directory "${declaration}" DIRECTORY)
        add_subdirectory("${directory}")
    endforeach()
endfunction()
