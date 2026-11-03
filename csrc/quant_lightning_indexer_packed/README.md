# Legacy Packed Quant Lightning Indexer

Ascend950 direct-launch implementation, following the legacy
`apply_token_bitmask` layout:

- `op_host/quant_lightning_indexer_packed.cpp`: Tensor validation, workspace,
  stream lifetime and direct kernel launch.
- `op_kernel/quant_lightning_indexer.cpp`: direct-launch entry receiving each
  host-computed tiling field by value and rebuilding the local tiling record.
  No tiling Tensor or host launcher wrapper is needed.
- `op_kernel/arch35` and `op_kernel/common`: self-contained computation headers.

Build with `bash build.sh -a kernels Ascend950`. This wheel contains the existing
legacy operators, including NEXTN tree building, and packed QLI. `build.sh`
itself is unchanged. Install the exact wheel emitted by that build.

The migrated CANN/ACLNN implementation is independently maintained in
`csrc/indexer/quant_lightning_indexer_packed`. Neither native source tree includes
the other; the two build paths produce alternative wheels, not additive ones.
Both expose `torch.ops.npu.sgl_quant_lightning_indexer_packed` and use the same
public Python API. To retire this implementation, remove this directory and its
legacy CMake target, header declaration and A5 registration entries. The migrated
CANN/ACLNN implementation does not need to change.

Computation derives from ops-transformer commit
`aacba10c247cc454768c07704c7d0d026adc3d6d`; original notices remain in source files.
