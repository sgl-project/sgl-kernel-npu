# Native DSV4 tree attention

AscendC implementation of the DSV4 DSpark tree sparse-attention operator,
ported from vLLM-Ascend and exposed through the `sgl-kernel-npu` torch
extension as:

```python
torch.ops.npu.sparse_attn_sharedkv_tree
```

## Integration

- Host entry: `op_host/sparse_attn_sharedkv_tree.cpp`
- Host tiling: `op_host/sparse_attn_sharedkv_tree_tiling.cpp`
- Kernel entries: `sparse_attn_sharedkv_tree_fp16` and
  `sparse_attn_sharedkv_tree_bf16`
- CMake target: `sparse_attn_sharedkv_tree_kernel` (A3 only)
- The host wrapper uses `ge_helper::TilingContext`, `EXEC_KERNEL_CMD` and the
  repository tiling-tensor cache; it does not use ACLNN or a custom-op vendor
  package.

The optional tree inputs are `cmp_tree_kv`, `tree_ancestors` and
`tree_prefix`. `metadata` is produced by the existing host metadata scheduler:

```python
metadata = torch.ops.npu.sparse_attn_sharedkv_metadata_host(...)
out, lse = torch.ops.npu.sparse_attn_sharedkv_tree(
    q, metadata=metadata, return_softmax_lse=True, **tree_inputs)
```

## Contract

- One TND query record per tree node, including the root.
- `seqused_kv` is the logical prefix length plus ancestor count.
- `ori_sparse_indices` contains physical original-KV slots in chronological
  ancestor order, padded with `-1`; its width is a multiple of 128.
- The valid original-slot count must match the logical sliding-window length.
- C4 supplies logical `cmp_sparse_indices`; C128 omits it and traverses the
  full compressed history, including the tree tail.
- `cmp_tree_kv` is contiguous `[nodes, 512]` FP16/BF16, `tree_ancestors` is
  int64 `[queries, max_depth]`, and `tree_prefix` is int32 `[queries]`.
- Original KV rows are shared; no branch KV expansion is performed.

## Build and test

```bash
./build.sh -a kernels
python -m pytest tests/python/sgl_kernel_npu/test_tree_sparse_attn_sharedkv_tree.py::test_native_attention_parity -q
```

The parity test exercises the torch registration, host tiling adapter,
workspace sizing and kernel launch against the independent CPU oracle.
