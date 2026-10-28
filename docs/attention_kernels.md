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

All three operators use `SGL_KERNEL_ENABLE_A3_ONLY_OPS` for both compilation
and registration. The build enables this flag for A2/A3 and disables it for
A5 by default. The kernels require Ascend A2/A3.

## Input notation

`B` is the batch size, `Nq` the query head count, `Nkv` the key/value head
count, `Sq` the query sequence length, `Skv` the key/value sequence length,
and `D` the per-head dimension. Q/K/V must have the same dtype and reside
on the same NPU. K and V must have identical shapes. MHA, MQA and GQA use
`Nq % Nkv == 0`; heads are not broadcast automatically.

All dimensions must be positive; empty inputs are unsupported.

BSH and BSND inputs are converted to contiguous BNSD internally. Ada returns
its output in the original layout. This path adds layout-conversion copies.

## Laser Attention (`torch.ops.npu.laser_attn`)

| Input / parameter | Constraint |
| --- | --- |
| Q | FP16, `[B, Nq, Sq, D]` |
| K, V | FP16, `[B, Nkv, Skv, D]` |
| `input_layout` | `"BNSD"` only |
| Sequence lengths | Both `Sq` and `Skv` must be multiples of 256; K/V lengths must match |
| `head_num` | Must equal `Nq`; `Nq` must be divisible by `Nkv` |
| `head_dim` | Must equal 128 |
| `scale_value` | Finite scalar, normally `1 / sqrt(128)`; the default is `1.0`, not an inferred scale |
| `atten_mask` | Optional FP16 tensor on Q's device; leave as `None` for the supported dense path |
| `alibi_mask` | Optional FP16 tensor on Q's device; leave as `None`; the device entry point does not consume it |
| `drop_mask` | Optional uint8 tensor on Q's device; leave as `None`; the device entry point does not consume it |
| `keep_prob` | Finite scalar in `(0, 1]`, default `1.0`; dropout is not implemented, so use `1.0` |
| `pre_tokens` | KV padding count in `[1, 255]` when the original KV length is padded to a multiple of 256; use `2147483647` (default) when no padding was added. The host accepts any nonnegative int32, but that does not establish support for other values |
| `is_highPrecision` | Boolean, default `True`; the device path always enables high precision |
| Result | `(softmax_log_max_sum, attention_out)`: FP32 tensors of shapes `[B, Nq, Sq]` and `[B, Nq, Sq, 128]`; the first is an uninitialized compatibility placeholder and must not be read |

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
| `inner_precise` | Integer in `[0, 3]`; use `1` (default). FP16 modes 0 and 2 require a specialization not present in this build and fail the tiling-key check |
| `pre_tokens` | Previous-token window bound in the general Ada implementation; nonnegative int32, default `214748647` |
| `next_tokens` | Following-token window bound in the general Ada implementation; nonnegative int32, default `0`. |
| `actual_seq_lengths`, `actual_seq_lengths_kv` | Optional integer lists; `None` or empty uses the padded length, one entry applies to every batch item, otherwise exactly `B` entries; each entry in `(0, Sq]` or `(0, Skv]`. Use `None` for the validated full-length path |
| Result | `attention_out` tensor with the same shape, dtype and layout as Q; no softmax statistics are returned |

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
| `actual_seq_lengths`, `actual_seq_lengths_kv` | Same list convention and bounds as Ada; causal calls require actual Q length <= actual KV length in every batch. Use `None` for the validated full-length Ada path |
| Result | `(sparse_mask, sparse_count_table)`: int8 `[B, Nq, R, Cp]` and int32 `[B, Nq, R]`, on Q's device |

When the retained-fraction limit applies (rows with at least ten valid key
blocks), the cutoff rank is `max(1, floor(valid_key_blocks * row_sparse))`.
The host enforces `threshold` and `row_sparse` in `(0, 1]` and rejects causal
calls whose actual Q length exceeds the actual KV length in any batch. Sink/diagonal retention,
tied scores and the absence of this limit for rows with fewer than ten valid
blocks mean that `row_sparse` does not guarantee exact final sparsity.

## Tests

On A2/A3, build and run:

```bash
pytest -q tests/python/sgl_kernel_npu/test_laser_attention.py \
          tests/python/sgl_kernel_npu/test_ada_block_sparse_attention.py
```
