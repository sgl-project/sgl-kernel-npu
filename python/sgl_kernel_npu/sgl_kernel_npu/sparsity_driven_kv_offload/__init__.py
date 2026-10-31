from sgl_kernel_npu.sparsity_driven_kv_offload.ops import (
    create_shm_tensor,
    free_shm,
    fused_lightning_indexer_manage,
    fused_quant_scatter_copy_sparse_flash_attention,
    slot_map_lookup,
    unidex_copy_inplace,
)

__all__ = [
    "create_shm_tensor",
    "free_shm",
    "fused_lightning_indexer_manage",
    "fused_quant_scatter_copy_sparse_flash_attention",
    "plan_sparse_kv_cache",
    "slot_map_lookup",
    "unidex_copy_inplace",
]


def __getattr__(name):
    # Keep the planner module optional for older offload-only deployments.
    if name == "plan_sparse_kv_cache":
        from sgl_kernel_npu.sparsity_driven_kv_offload.cache_planner import (
            plan_sparse_kv_cache,
        )

        return plan_sparse_kv_cache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
