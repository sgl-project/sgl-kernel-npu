# Fused KV compressor

## Hardware support

| Implementation | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformer_cann920` | supported | supported |
| `ops_transformer_cann910` | unsupported | supported |
| `ascendc` | supported | unsupported |
| `torch_native` | supported | supported |

## Signature

```python
from sgl_kernel_npu.operators.attention import compressor

def compressor(x, wkv, wgate, state_cache, ape, norm_weight, *, impl='auto', **kwargs):
    ...
```

`auto` selects the initialized SDK production path: original bindings on CANN 9.1, released bindings on CANN 9.2. Execution failures never trigger another backend.

## Description

Preserves the fused RMSNorm/RoPE contract. Do not replace with the released core-only compressor. State is mutated in place.

Project KV and gates; combine each compression window using normalized gate weights and positional bias; apply RMSNorm and partial RoPE. Return completed compressed rows and retain unfinished/overlap state.

## Parameters

| Name | Type | Default | Contract |
| --- | --- | --- | --- |
| x | Tensor | required | Input activation; shape described below; production device NPU |
| wkv | Tensor | required | [coff*D,C] KV projection weight |
| wgate | Tensor | required | [coff*D,C] gate projection weight |
| state_cache | Tensor | required | Mutable FP32 [pages,slots,2*coff*D] history |
| ape | Tensor | required | [cmp_ratio,coff*D] positional gate bias |
| norm_weight | Tensor | required | [D] RMSNorm weight |
| impl | str | 'auto' | Registered implementation; auto selects the SDK path; never falls back to torch_native |

### Keyword arguments

| Name | Type | Contract |
| --- | --- | --- |
| rope_sin / rope_cos | FP32 Tensor | Rotary factors for compressed output rows. |
| rope_head_dim / cmp_ratio | int | Rotary width / compression ratio, required. |
| state_block_table | int32 Tensor | Required: paged table in mode 1; A3 explicit slots or A5 request-bank IDs in mode 2. |
| cu_seqlens / seqused / start_pos | int32 Tensor / None | Cumulative TH offsets (None for BSH), per-request counts, absolute starts. |
| coff / norm_eps | int / float | Defaults 1 / 1e-6; reference coff is 1 or 2. |
| rotary_mode / cache_mode | int | Defaults 2 / 2; reference supports 1 or 2. |

## Return value

| Name | Type | Contract |
| --- | --- | --- |
| output | Tensor | TH: [capacity,D]; BSH: [B,capacity,D]. Only completed compressed rows are valid; padding is unspecified. state_cache is also mutated. |

## cann_ops_transformer_cann920

Uses the CANN 9.2 binding and rejects a different initialized SDK. `auto` dispatches by the cached SDK version. A5 KV reader/writer formats are derived from the same version before allocation; each operator requires only its own binding.

`torch.ops.npu.compressor` supplies the fused contract; CANN 9.1 A5 uses `torch.ops.custom.compressor`.

## ops_transformer_cann910

Source: [A5 ops-transformer experimental](https://gitcode.com/cann/ops-transformer/tree/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/experimental/attention/compressor). The kernel source is experimental ops-transformer. Its custom_ops Python/C++ bridge may be distributed from cann-recipes-infer; the implementation name records kernel provenance, not the bridge package name. The CANN 9.1 ABI conversion stays in this operator module; an initialized CANN 9.2 SDK is rejected.

## ascendc

Source: [A3 sgl-kernel-npu device kernel](https://github.com/sgl-project/sgl-kernel-npu/blob/26ec66efe01f8acef43db55da679107702876b23/csrc/compressor/op_kernel/compressor.cpp). The C++ wrapper directly launches the checked-in Ascend C kernel; this is not inferred from the Torch namespace. The CANN 9.1 ABI conversion stays in this operator module; an initialized CANN 9.2 SDK is rejected.

## torch_native

Independent eager PyTorch reference for correctness; never an automatic production or graph-capture fallback. Metadata descriptors are not native tiling bytes. Modes outside the documented reference domain are rejected.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import compressor

x = torch.randn(4, 1024, device="npu", dtype=torch.bfloat16)
wkv = torch.randn(128, 1024, device="npu", dtype=x.dtype) * 0.02
wgate = torch.randn_like(wkv) * 0.02
state = torch.zeros(2, 16, 256, device="npu", dtype=torch.float32)
ape = torch.zeros(4, 128, device="npu", dtype=torch.float32)
norm = torch.ones(128, device="npu", dtype=torch.float32)
i32 = lambda values: torch.tensor(values, device="npu", dtype=torch.int32)
y = compressor(x, wkv, wgate, state, ape, norm,
    rope_sin=torch.zeros(2, 64, device="npu"),
    rope_cos=torch.ones(2, 64, device="npu"), rope_head_dim=64, cmp_ratio=4,
    state_block_table=i32([[1]]), cu_seqlens=i32([0, 4]),
    seqused=i32([4]), start_pos=i32([0]), cache_mode=1, impl="torch_native")
```

Build/install the current wheel and initialize the SDK first. The example demonstrates the API; it does not establish NPU accuracy, performance or model inference.

```bash
python3 -m pytest tests/python/operators/attention/test_compressor.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[Upstream operators](https://github.com/Ascend/ops-transformer)
