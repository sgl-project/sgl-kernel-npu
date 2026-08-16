add_ops_compile_options(
        OP_NAME Compressor
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)

set(compressor_depends compressor PARENT_SCOPE)
target_sources(op_host_aclnn PRIVATE
        compressor_def.cpp
)

target_sources(optiling PRIVATE
        compressor_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        compressor_tiling.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        compressor_proto.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
