include_guard(GLOBAL)
# Register domains once; each domain discovers only its direct operator children.
set(SGL_OPERATOR_DOMAIN_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
function(sgl_add_operator_domains)
    # Enter each registered domain using a separate declaration build directory.
    foreach(domain IN ITEMS sampler attention)
        add_subdirectory("${SGL_OPERATOR_DOMAIN_ROOT}/${domain}" "declarations/${domain}")
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
