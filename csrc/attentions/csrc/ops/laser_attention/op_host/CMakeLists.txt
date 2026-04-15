add_ops_compile_options(
        OP_NAME LaserAttention
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)

set(laser_attention_depends laser_attention PARENT_SCOPE)
target_sources(op_host_aclnn PRIVATE
        laser_attention_def.cpp
)

target_sources(optiling PRIVATE
        laser_attention_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        laser_attention_tiling.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        laser_attention_proto.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)