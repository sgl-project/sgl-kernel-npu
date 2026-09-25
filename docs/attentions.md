# Attention kernels

`laser_attn`, `ada_block_sparse_attention` and `sparse_block_estimate` are built
into `libsgl_kernel_npu.so` and packaged in the `sgl_kernel_npu` wheel:

```bash
./build.sh -a kernels Ascend910B1       # A2
./build.sh -a kernels Ascend910_9382    # A3
```

Use `sgl_kernel_npu.attentions` (or `torch.ops.npu`):

```python
from sgl_kernel_npu.attentions import ada_block_sparse_attention, sparse_block_estimate

# q/k/v: FP16 or BF16 NPU tensors in [B, N, S, D] layout.
scale = q.shape[-1] ** -0.5
mask, counts = sparse_block_estimate(
    q, k, num_heads=q.shape[1], num_key_value_heads=k.shape[1],
    scale_value=scale, causal=True,
)
out = ada_block_sparse_attention(
    q, k, v, mask, counts,
    num_heads=q.shape[1], num_key_value_heads=k.shape[1],
    scale_value=scale, causal=True,
)
```

The separate `attentions` wheel, `torch.ops.attentions` namespace and
`./build.sh -a attentions` target have been removed. The old FastLayerNorm and
RainFusionAttention wrappers are no longer provided. This change does not add
a replacement RainFusionAttention implementation.

## Device and input support

The three kernels are enabled only for `SOC_VERSION` matching `Ascend910B*`
or `Ascend910_93*`. They are excluded from compilation, linking and operator
registration for other targets, including A5. Python entry points remain
importable and report an unavailable operator explicitly. This exclusion does
not establish A5 support for the other kernels in the project. Use the actual
AscendC SoC name for an A5 kernel build; `Ascend950` remains the DeepEP alias
rejected by the existing `kernels` build target.

Ada and the estimator accept FP16/BF16 inputs in BNSD, BSND or BSH layout.
BSND is normalized to the existing BSH specialization. TND is not exposed.
Inputs must have positive dimensions and matching devices/dtypes; optional
sequence lengths contain either one value or one value per batch, each in
`(0, padded_sequence_length]`. Sparse sizes are multiples of 128 in `[128, 512]`;
the estimator stride must divide the sparse size. Ada rejects tiling keys that
have no specialization in the original kernel. In particular, the original
FP16 high-precision specialization is absent (`inner_precise=0` or `2`).

The estimator returns an int8 mask of shape
`[B, Nq, ceil(Sq / sparse_size), align32(ceil(Skv / sparse_size))]` and int32
counts of shape `[B, Nq, ceil(Sq / sparse_size)]`. User-supplied masks must have
0/1 entries, zero padding and matching row counts; these data-dependent
conditions are the caller's responsibility. For causal attention, select only
causally allowed blocks and retain the diagonal block. The output dtype and
layout match the query. Laser Attention retains its FP16, BNSD, head-dimension
128 and sequence-alignment restrictions.

## Implementation and validation

All three use the existing `ascendc_library` and `EXEC_KERNEL_CMD` path. There
is no custom OPP install, generated ACLNN wrapper or additional shared library.
Ada keeps the existing host tiling calculation behind a tensor adapter. Device
tilings are explicit records matching the host serialization, and the former
`TILING_KEY_VAR` branches are dispatched using the host's tiling key. The
estimator's tile factors are local to each invocation so that a previous shape
cannot change later launches. Tiling and sequence-length device copies are
cached per device and stream, with a bounded cache, following the Laser
Attention approach.

Platform queries use the CANN
[PlatformAscendCManager kernel-launch API](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_1039.html),
which uses the already-linked `tiling_api` and `platform` libraries. The release
CI matrix on this branch uses CANN 8.5.0/9.0.0 and PyTorch 2.8.0/2.10.0.

On A2/A3, build and run:

```bash
pytest -q tests/python/sgl_kernel_npu/test_laser_attention.py \
          tests/python/sgl_kernel_npu/test_sparse_attention.py
```

The sparse tests compare Ada against a masked PyTorch reference for FP16/BF16,
MHA/GQA, causal/noncausal and all three layouts. Estimator checks cover mask
counts, causal structure and independence from previous input shapes. Build,
numerical execution and NPU graph replay must still be validated with CANN and
NPU hardware; CPU-only source/configuration checks cannot establish them.
