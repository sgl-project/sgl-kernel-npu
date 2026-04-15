add_ops_compile_options(
        OP_NAME BlockSparseAttention
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)

set(block_sparse_attention_depends block_sparse_attention PARENT_SCOPE)
target_sources(op_host_aclnn PRIVATE
        block_sparse_attention_def.cpp
)

target_sources(optiling PRIVATE
        block_sparse_attention_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        block_sparse_attention_tiling.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        block_sparse_attention_proto.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)