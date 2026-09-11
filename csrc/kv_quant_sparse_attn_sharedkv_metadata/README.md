# `kv_quant_sparse_attn_sharedkv_metadata`

This directory provides the metadata producer for the A5 quantized-KV sparse
attention kernel. It returns a device `int32[1024]` schedule containing FA and
FD core assignments.

## Torch interface

```python
metadata = torch.ops.npu.kv_quant_sparse_attn_sharedkv_metadata(
    num_heads_q=64,
    num_heads_kv=1,
    head_dim=512,
    kv_quant_mode=1,
    cu_seqlens_q=cu_seqlens_q,
    seqused_kv=seqused_kv,
    layout_q="TND",
    layout_kv="PA_ND",
    has_ori_kv=True,
    has_cmp_kv=True,
    device="npu",
)
```

The metadata scheduler uses the existing host scheduling implementation and
converts its eight-field FA records to the nine-field A5 layout, including the
scheduled `FA_S2_MAX_NUM` value. For `num_heads_q == 128`, A5 uses split-G
execution: the scheduler runs on half the core count and each FA record is
duplicated into the two logical records consumed by the kernel. Sequence
tensors are staged as CPU `int32` tensors by this launcher, then the completed
metadata is copied to the input device.

## Build

Build and install the kernels wheel for A5:

```bash
bash build.sh -a kernels Ascend950PR_9599
pip install output/sgl_kernel_npu*.whl
```

Use the same metadata tensor for the matching attention invocation. Device
validation on a real A5/CANN environment is still required.
