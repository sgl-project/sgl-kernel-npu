# Over-embedding operators

The LongCat2 over-embedding path first calls
[update_oe_token_table](update_oe_token_table.md) to write the current tokens into
request history, then calls [compute_n_gram_ids](compute_n_gram_ids.md) to calculate
embedding-table indices from that updated history.

Both public APIs provide `auto`, `ascendc`, and the independent `torch_native`
reference on arch22. `auto` selects `ascendc` without fallback. Input descriptions
record the existing caller contract; they do not imply additional validation by
the public dispatch layer.

Build from the repository root with the REFACTOR entry on an arch22 Linux/CANN
environment, then install the wheel produced by that build:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 setup.py --arch-version 2201 \
  --ops compute_n_gram_ids,update_oe_token_table --artifact wheel
```

`2201` corresponds to arch22 and builds for `ascend910b` and `ascend910_93`.
Both operators retain the official standalone CANN host, tiling, and kernel,
invoked through this repository's ACLNN Torch adapters. CANN manages tiling and
dispatch; the handwritten tiling cache from the old direct-launch path is no
longer used. The existing `torch.ops.npu.compute_n_gram_ids` and
`torch.ops.npu.update_oe_token_table` interfaces remain; update still modifies and
returns the same table.

`--ops all` on arch22 includes both operators. Add `mlp_lightning_indexer` to the
list above to build all three LongCat2 operators; use `--artifact run` for a CANN
installer alone. For a full A2/A3 wheel with the original native operators and
all three LongCat2 ACLNN operators, use `bash build.sh -a kernels`.
Subset wheels replace an installation rather than adding to an old wheel.
See the [build guide](../../../../../python/sgl_kernel_npu/README.md) for installation.

Architecture declarations describe implementation scope. NPU compilation,
accuracy, performance, and the existing graph capture/replay cases must still be
validated for the ACLNN build. Earlier direct-launch results do not validate this
new execution path.
