# sparse_block_estimate

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Supported | Unsupported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.attention import sparse_block_estimate

def sparse_block_estimate(
    query, key, actual_seq_lengths=None, actual_seq_lengths_kv=None,
    input_layout="BNSD", stride=8, sparse_size=128, num_heads=1,
    num_key_value_heads=1, scale_value=1, threshold=1, causal=True,
    keep_sink=True, keep_recent=True, row_sparse=1, *, impl="auto",
):
    ...
```

`auto` and `ascendc` use the same entry, without fallback. Importing sgl_kernel_npu also registers the native `torch.ops.npu.sparse_block_estimate`. Its business parameters and defaults match this signature; it has no impl argument.

## Description

Estimate sparse attention blocks and return their mask and per-row counts. For contiguous BNSD input, stride $s$ and head dimension $D$, Q is reversed within each stride group and multiplied by contiguous K groups:

$$
A_{ij}=\sum_{r=0}^{s-1}\sum_{d=0}^{D-1}Q_{is+s-1-r,d}K_{js+r,d},\qquad Z_{ij}=\mathrm{scale\_value}\,A_{ij}.
$$

With $R=\mathrm{sparse\_size}/s$, online softmax exponents are cast to FP16 and reduced over $R$ column samples. FP16 rescaling and row reductions produce block scores. Original sorting, cumulative thresholds, row_sparse and forced columns determine $M_{uv}\in\{0,1\}$ and $C_u=\sum_v M_{uv}$. Rounding and tile order follow the implementation, rather than standard Top-k over dense attention.

## Parameters and return value

The wrapper adds no validation. The native adapter retains the legacy nonzero checks for num_heads and sparse_size and the Tensor conversion helper's rejection of unmapped dtypes. Tensor operations, ACLNN and legacy Tiling can reject other inputs.

| Parameter | Type / shape / dtype | Default | Meaning |
| --- | --- | --- | --- |
| `query` | NPU Tensor; BNSD `[B,Nq,Sq,D]`, BSND `[B,Sq,Nq,D]`, BSH `[B,Sq,Nq*D]`; FP16/BF16 | Required | Legacy storage handling, without a new contiguity check |
| `key` | NPU Tensor; corresponding layout with Nkv,Sk; FP16/BF16 | Required | Ordinary safe calls use the same length and dtype as query |
| `actual_seq_lengths` | list[int] or None | None | Actual Q lengths; empty/None uses full lengths, one element broadcasts, multiple elements index batches |
| `actual_seq_lengths_kv` | list[int] or None | None | Actual KV lengths with the same rules; not a Tensor |
| `input_layout` | str | BNSD | BNSD, BSND or BSH; legacy Torch does not define TND outputs |
| `stride` | int | 8 | Ordinary cases use a positive divisor of sparse_size; no added check |
| `sparse_size` | int | 128 | Sparse block size, retaining the nonzero check |
| `num_heads` | int | 1 | Q heads used by Host scheduling; supply the actual head count |
| `num_key_value_heads` | int | 1 | KV heads for original GQA mapping |
| `scale_value` | float | 1 | Scale after Matmul; explicit zero stays zero |
| `threshold` | float | 1 | Original cumulative block-score threshold |
| `causal` | bool | True | Original causal template branch, without substituting a tokenwise triangular mask |
| `keep_sink` | bool | True | Original first-column handling; final forced first column is retained |
| `keep_recent` | bool | True | Original last-column handling; noncausal execution also forces it |
| `row_sparse` | float | 1 | Original retention-ratio branch for at least 10 active columns; no clamping |
| `impl` | Keyword-only str | auto | auto, ascendc or torch_native |

Let $L=\lceil Sq/\mathrm{sparse\_size}\rceil$, $L_{32}=32\lceil L/32\rceil$:

| Output | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `sparse_mask` | `[B,Nq,L,L32]` | int8 | New output on query's device, retaining NPU format allocation |
| `sparse_count_table` | `[B,Nq,L]` | int32 | Per-row block counts on query's device |

Rows beyond actual Q lengths are not guaranteed to be written. Mask allocation uses the **Q length**, whereas the kernel's write stride uses the **K length**. Unequal lengths retain legacy limitations and are not general cross-attention support. Inputs should not be modified.

## ascendc backend

Calls bundled SglSparseBlockEstimate ACLNN/Host/Kernel without requiring the old attentions package. A2/A3 share one implementation. Torch causal=True and CANN OpDef false remain distinct; the adapter explicitly forwards the Torch value.

The computation class is unchanged, including raw BSND/BSH copy strides, Matmul addressing, forced columns, mutable global Tiling bases and unwritten output regions. Call order can affect tiling.

## torch_native backend

Independent PyTorch implementation of the old addressing, reversed Q groups, online max/sum, FP16 reductions, sorting and threshold boundaries. It preserves the final forced first column even with keep_sink=False, reads the NPU vector-core count and maintains its own Tiling history.

Shared check_accuracy requires exact equality for both integer outputs. Portable CPU checks do not establish device accuracy. Device-specific Matmul/Softmax numerical alignment and legacy/new execution comparisons remain pending. Comparisons need the same call sequence for both backends.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import sparse_block_estimate

q = torch.randn(1, 1, 2048, 64, device="npu", dtype=torch.float16) * 0.1
k = torch.randn_like(q)
mask, count = sparse_block_estimate(q, k, threshold=0.8, impl="ascendc")
mask_ref, count_ref = sparse_block_estimate(q, k, threshold=0.8, impl="torch_native")
torch.testing.assert_close(mask, mask_ref, atol=0, rtol=0)
torch.testing.assert_close(count, count_ref, atol=0, rtol=0)
```
