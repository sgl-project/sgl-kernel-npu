add_ops_compile_options(
        OP_NAME AdaBlockSparseAttention
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)

set(ada_block_sparse_attention_depends ada_block_sparse_attention PARENT_SCOPE)
target_sources(op_host_aclnn PRIVATE
        ada_block_sparse_attention_def.cpp
)

target_sources(optiling PRIVATE
        ada_block_sparse_attention_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        ada_block_sparse_attention_tiling.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        ada_block_sparse_attention_proto.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
