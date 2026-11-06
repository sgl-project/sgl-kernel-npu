# compressor_v2

Experimental CompressorV2 for **CANN 9.2.0 / A3 (arch22)**. `auto` and
`cann_ops_transformers_cann920` select the same bundled backend, without fallback.

Upstream documentation: [CompressorV2 Torch API](https://gitcode.com/cann/ops-transformer/blob/e7da18105f08b06e01ef0fff78f439098d483746/experimental/attention/compressor_v2/docs/torchapi_compressor.md).

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` / `auto` | Supported | Unsupported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.attention import compressor_v2

def compressor_v2(
    x, wkv, wgate, state_cache, cmp_ratio=4, *,
    state_block_table=None, cu_seqlens=None, seqused=None, start_pos=None,
    impl="auto",
): ...
```

## Description

Let `K = x.float() @ wkv.float().T`, `G = x.float() @ wgate.float().T`.
For each completed absolute-position group of cmp_ratio tokens, compute
`sum(softmax(G, token_axis) * K)` feature-wise, then cast to x.dtype.
An unaligned first group reads earlier tokens from the cache. Incomplete groups
produce no output. Only the final `min(seqused[b],C)` new tokens update
`state_cache[block_id, absolute_position % C, :]`: K in the first D channels,
G in the last D. Other entries persist. coff=1 and cache_mode=2 are fixed;
there is no APE input.

**The cache is mutated; only cmp_kv is returned.** BSH output has shape
`[B,ceil(S/cmp_ratio),D]`, with
`floor(((start_pos[b] % cmp_ratio)+seqused[b])/cmp_ratio)` valid leading rows per
batch. TH output has shape `[min(T,T//cmp_ratio+B),D]`: valid groups are compacted
in batch order, followed by padding. Padding is unspecified and must be excluded
from accuracy comparisons. Continuation calls must use the updated cache and
correct start_pos.

## Parameters and return value

| Parameter | dtype | Shape / constraints |
| --- | --- | --- |
| x | fp16/bf16 | `[B,S,H]` or packed `[T,H]`; H=1024..10240, multiple of 512 |
| wkv, wgate | same as x | `[D,H]`, D=128/512 |
| state_cache | fp32 | `[P,C,2*D]`, P>=B, P>0, C=1..1024; updated in place |
| state_block_table | int32 | `[B]`, required on A3; unique physical block IDs in `[0,P)`; 0 is valid |
| cu_seqlens | int32 | TH: `[B+1]`, cumulative nondecreasing boundaries from 0 to T; BSH: None |
| seqused | int32 | Optional `[B]`, 0..sequence length; None uses the whole sequence |
| start_pos | int32 | Optional `[B]`, nonnegative absolute positions; None means zero |
| cmp_ratio | int | 2/4/8/16/32/64/128, default 4 |

All tensors share one device. x, weights and index tensors are contiguous.
Only axis 0 of state_cache may be noncontiguous. Empty B/S/T are supported;
weights and cache remain nonempty. A3 tiling requires state_block_table despite
the upstream Python default of None. Launch validation checks metadata only;
the caller guarantees valid tensor contents and available history.

## cann_ops_transformers_cann920 backend

Build with the `ops_transformer` profile. Its `additional_builds` entry schedules
CompressorV2 with `--experimental` on both upstream commands. The experimental
vendor and Python package differ from the regular build, so both can ship in one
wheel. The package includes the complete vendor,
LICENSE and manifest; this experimental dependency has no AICPU. First use may
JIT-compile C++ bindings and requires a compiler and matching CANN/Torch/torch_npu.

Mapping: build/OpType `compressor_v2` / `CompressorV2`; Python export `compressor`
from `_third_party.sgl_ops_transformer_experimental.ops.ds41`; Torch name
`cann_ops_transformer::ds41.compressor`; ACLNN `aclnnCompressorV2`.

## torch_native backend

`torch_native` independently computes in CPU FP32, supports BSH/TH and circular
history, and requires C>=cmp_ratio. It reads host metadata and is not a
performance backend. The pinned upstream custom_op declares `mutates_args=()`
despite modifying the cache. This integration targets eager inference;
functionalization, torch.compile, TorchAir capture and autograd are not validated.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import compressor_v2

x = torch.randn(1, 16, 1024, dtype=torch.bfloat16, device="npu")
wkv = torch.randn(128, 1024, dtype=x.dtype, device=x.device)
wgate = torch.randn_like(wkv)
cache = torch.zeros(1, 16, 256, dtype=torch.float32, device=x.device)
blocks = torch.tensor([0], dtype=torch.int32, device=x.device)
y = compressor_v2(x, wkv, wgate, cache, 4, state_block_table=blocks)
assert y.shape == (1, 4, 128)
```

Upstream copyright: Huawei Technologies Co., Ltd.; CANN Open Software License Agreement Version 2.0.
