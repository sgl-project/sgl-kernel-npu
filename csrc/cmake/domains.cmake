include_guard(GLOBAL)
# Register domains once; each domain discovers only its direct operator children.
set(SGL_OPERATOR_DOMAIN_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
# HOST declarations are discovered before the legacy CANN declarations so the
# unified request can be partitioned without exposing HOST IDs to CANN helpers.
function(sgl_add_host_operator_domains)
    foreach(domain IN ITEMS data_move)
        add_subdirectory("${SGL_OPERATOR_DOMAIN_ROOT}/${domain}" "declarations/${domain}")
    endforeach()
endfunction()

function(sgl_add_cann_operator_domains)
    foreach(domain IN ITEMS sampler attention mem_cache lora examples activation matmul embedding indexer speculative mamba)
        add_subdirectory("${SGL_OPERATOR_DOMAIN_ROOT}/${domain}" "declarations/${domain}")
    endforeach()
endfunction()

# Callers with already-separated IDs (e.g. the Torch build) discover both phases.
function(sgl_add_operator_domains)
    sgl_add_host_operator_domains()
    sgl_add_cann_operator_domains()
endfunction()

function(sgl_discover_operators)
    file(GLOB declarations CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*/CMakeLists.txt")
    list(SORT declarations)
    foreach(declaration IN LISTS declarations)
        get_filename_component(directory "${declaration}" DIRECTORY)
        add_subdirectory("${directory}")
    endforeach()
endfunction()
