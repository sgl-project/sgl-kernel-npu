# Attention kernels

`laser_attn`, `ada_block_sparse_attention` and `sparse_block_estimate` are built
for Ascend A2/A3. These are forward-only operators.

Import `sgl_kernel_npu` to register the operators, then call `torch.ops.npu`:

```python
import torch
import sgl_kernel_npu

# q/k/v: FP16 or BF16 NPU tensors in [B, N, S, D] layout.
scale = q.shape[-1] ** -0.5
stride = 8
mask, counts = torch.ops.npu.sparse_block_estimate(
    q, k, num_heads=q.shape[1], num_key_value_heads=k.shape[1],
    stride=stride, sparse_size=128, scale_value=scale / stride, causal=True,
)
out = torch.ops.npu.ada_block_sparse_attention(
    q, k, v, mask, counts,
    num_heads=q.shape[1], num_key_value_heads=k.shape[1],
    scale_value=scale, causal=True,
)
```

## Device support

All three operators are built for `Ascend910`.
They are excluded from compilation and operator registration for A5.

## Input notation

`B` is the batch size, `Nq` the query head count, `Nkv` the key/value head
count, `Sq` the query sequence length, `Skv` the key/value sequence length,
and `D` the per-head dimension. Q/K/V must have the same dtype and reside
on the same NPU. K and V must have identical shapes. MHA, MQA and GQA use
`Nq % Nkv == 0`; heads are not broadcast automatically.

All dimensions must be positive; empty inputs are unsupported.

## Laser Attention (`torch.ops.npu.laser_attn`)

| Input / parameter | Constraint |
| --- | --- |
| Q | FP16, `[B, Nq, Sq, D]` |
| K, V | FP16, `[B, Nkv, Skv, D]` |
| `input_layout` | `"BNSD"` only |
| Sequence lengths | Both `Sq` and `Skv` must be multiples of 256; K/V lengths must match |
| `head_num` | Must equal `Nq`; `Nq` must be divisible by `Nkv` |
| `head_dim` | Must be divisible by 128 |
| `scale_value` | Finite scalar, normally `1 / sqrt(128)`; the default is `1.0`, not an inferred scale |
| Result | `(softmax_log_max_sum, attention_out)` |

## Ada Block Sparse Attention (`torch.ops.npu.ada_block_sparse_attention`)

Let `R = ceil(Sq / sparse_size)`, `C = ceil(Skv / sparse_size)` and
`Cp = 32 * ceil(C / 32)`.

| Input / parameter | Constraint |
| --- | --- |
| Q | FP16 or BF16; BNSD: `[B, Nq, Sq, D]`, BSND: `[B, Sq, Nq, D]`, BSH: `[B, Sq, Nq * D]`; on the current NPU device |
| K, V | Same dtype and device as Q; BNSD: `[B, Nkv, Skv, D]`, BSND: `[B, Skv, Nkv, D]`, BSH: `[B, Skv, Nkv * D]` |
| `input_layout` | `"BNSD"`, `"BSND"` or `"BSH"`; the same for Q/K/V; TND and quantized inputs are unsupported |
| `num_heads`, `num_key_value_heads` | Set to `Nq` and `Nkv`, respectively; `Nq % Nkv == 0`; BSH hidden sizes must be divisible by their head counts |
| Head dimension `D` | Equal for Q/K/V; `D <= 512`; the shape must admit a compiled tiling specialization |
| `sparse_size` | Only 128 is supported |
| `sparse_mask` | int8 `[B, Nq, R, Cp]` on Q's device; `1` selects a key block, `0` skips it; columns `[C, Cp)` must be zero |
| `sparse_count_table` | int32 `[B, Nq, R]` on Q's device; must equal `sparse_mask.sum(-1)` converted to int32 |
| `scale_value` | Finite scalar, normally `1 / sqrt(D)`; defaults to `1.0` |
| `causal` | Boolean, defaults to `True`; the block mask must select only causally allowed blocks and retain the diagonal block when enabled |
| `inner_precise` | Use `1` (default) |
| `pre_tokens`, `next_tokens` | Nonnegative int32 values; retain the defaults unless deliberately configuring an attention window |
| `actual_seq_lengths`, `actual_seq_lengths_kv` | Unsupported; leave as `None` |
| Result | `(attention_out, softmax_log_max_sum)` |

### Block mask and counts

The mask is per query head, even for GQA/MQA. It is a block-selection mask,
not an additive token-level attention mask. Counts must equal
`sparse_mask.sum(-1)` converted to int32. Supply at least one valid selected
block for each active query row. For causal attention, select only causally
allowed blocks and retain the diagonal block; the kernel handles token-level
causal masking within the selected blocks.

The host validates shapes, dtypes and devices, but does not inspect the mask
values or verify counts on the device. The caller must maintain these
invariants. `sparse_block_estimate` produces the mask/count pair in this format.

See usage example:
https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/layers/attention/backends/block_sparse_attn.py

## Sparse Block Estimate (`torch.ops.npu.sparse_block_estimate`)

Use the same layout, head counts, `sparse_size` and `causal` setting as the
subsequent Ada Block Sparse Attention call. `R`, `C` and `Cp` have the same
definitions as above.

| Input / parameter | Constraint |
| --- | --- |
| Q | FP16 or BF16; BNSD: `[B, Nq, Sq, D]`, BSND: `[B, Sq, Nq, D]`, BSH: `[B, Sq, Nq * D]`; on the current NPU device |
| K | Same dtype and device as Q; BNSD: `[B, Nkv, Skv, D]`, BSND: `[B, Skv, Nkv, D]`, BSH: `[B, Skv, Nkv * D]` |
| `input_layout` | `"BNSD"`, `"BSND"` or `"BSH"`; the same for Q/K |
| `num_heads`, `num_key_value_heads` | Set to `Nq` and `Nkv`, respectively; `Nq % Nkv == 0`; BSH hidden sizes must be divisible by their head counts |
| Head dimension `D` | Equal for Q/K; use a dimension supported by the subsequent Ada call |
| `sparse_size` | Use 128, matching the verified Ada block size |
| `stride` | Positive integer dividing `sparse_size`; defaults to 8 |
| `scale_value` | Finite scalar; use the attention scale divided by `stride` to match MindIE-SD; defaults to `1.0` |
| `threshold` | CDF threshold in `(0, 1]`; defaults to `1.0`, disabling CDF pruning |
| `row_sparse` | Retained fraction in `(0, 1]`, equivalent to `1 - sparsity`; defaults to `1.0`, disabling the retained-fraction limit; see the minimum below |
| `causal` | Boolean, defaults to `True`; must match the Ada call |
| `keep_sink` | Boolean, defaults to `True`; requests retention of the first key block |
| `keep_recent` | Boolean, defaults to `True`; retains the diagonal block in causal mode; noncausal mode retains the last valid key block regardless of this flag |
| `actual_seq_lengths`, `actual_seq_lengths_kv` | Leave as `None` when producing masks for the supported Ada path |
| Result | `(sparse_mask, sparse_count_table)`: int8 `[B, Nq, R, Cp]` and int32 `[B, Nq, R]`, on Q's device |

When the retained-fraction limit applies (rows with at least ten valid key
blocks), `floor(valid_key_blocks * row_sparse)` must be at least one. Otherwise,
the kernel can read index `-1` when selecting its cutoff. The host checks
finiteness but does not enforce these semantic ranges. Sink/diagonal retention,
tied scores and the absence of this limit for rows with fewer than ten valid
blocks mean that `row_sparse` does not guarantee exact final sparsity.

## Tests

On A2/A3, build and run:

```bash
pytest -q tests/python/sgl_kernel_npu/test_laser_attention.py \
          tests/python/sgl_kernel_npu/test_sparse_attention.py
```
