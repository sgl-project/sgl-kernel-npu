# Over-embedding operators

The LongCat2 over-embedding path first calls
[update_oe_token_table](update_oe_token_table.md) to write the current tokens into
request history, then calls [compute_n_gram_ids](compute_n_gram_ids.md) to calculate
embedding-table indices from that updated history.

Both public APIs provide `auto`, `ascendc`, and the independent `torch_native`
reference on arch22. `auto` selects `ascendc` without fallback. Input descriptions
record the existing caller contract; they do not imply additional validation by
the public dispatch layer.

Build these native operators using the legacy entry on an arch22 Linux/CANN
environment, then install the wheel produced by that build:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh -a kernels 910
```

The root `setup.py --ops all` entry does not include these unmigrated native
operators. See the [build guide](../../../../../python/sgl_kernel_npu/README.md)
for installation. Architecture declarations describe implementation scope;
NPU accuracy, performance, and graph results must be reported for the tested
commit and environment.
