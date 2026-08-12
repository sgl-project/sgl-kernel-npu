# Sparse Attention SharedKV

This directory ports the main `sparse_attn_sharedkv` operator from
`vllm-ascend` commit `de35b40196dbd6c21d075a31183982355149c9b0` to the
direct `sgl_kernel_npu` AscendC launch path.

The metadata-producing operator is intentionally not included. The input
`metadata` must be produced separately and must follow
`op_kernel/sparse_attn_sharedkv_metadata.h` (`int32[1024]`).

The initial supported contract is:

- A2/A3, FP16/BF16
- query layout `BSND` or `TND`
- KV layout `PA_ND`
- 64 query heads, head dimension 512, and one KV head
- SWA, CFA, and SCFA dispatch modes
- `return_softmax_lse=False`

The PyTorch entry point is `torch.ops.npu.sparse_attn_sharedkv`. Its argument
order and defaults follow the upstream main operator. `metadata` remains a
required external `int32[1024]` input; this port does not register or build a
metadata-producing operator.

Build and test on an Ascend development machine:

```bash
./build.sh -a kernels Ascend910_9382  # A3
# ./build.sh -a kernels Ascend910B1   # A2
python tests/python/sgl_kernel_npu/test_sparse_attn_sharedkv.py -v
```

The unit test uses a single-active-core metadata fixture only to isolate the
main operator. Integration with the separately ported metadata operator should
be tested in addition to this fixture. The fixture covers the SWA/BSND/PA_ND
main path, a non-identity page table, and more than one S2 tile against an FP32
PyTorch reference.
