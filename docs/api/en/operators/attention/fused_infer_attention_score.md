# fused_infer_attention_score

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ops-transformer` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `ops-transformer` wherever it is registered.

```python
from sgl_kernel_npu.operators.attention import (
    fused_infer_attention_score,
    fused_infer_attention_score_out,
    fused_infer_attention_score_get_max_workspace,
)


def fused_infer_attention_score(query, key, value, *, impl="auto", **kwargs): ...


def fused_infer_attention_score_out(
    query, key, value, *, out, impl="auto", **kwargs
): ...


def fused_infer_attention_score_get_max_workspace(
    query, key, value, *, impl="auto", **kwargs
): ...
```

## Description

Runs fused decode or prefill attention. For dense unquantized inputs, $Y=\mathrm{softmax}(sQK^\top+M)V$, with an optional additive mask or bias $M$. The second output carries row-level log-sum-exp when requested. The out overload writes to buffers provided by the caller.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `query` | Tensor | [B, Hq, Sq, D] in BNSD, FP16/BF16 in example | Required | Query tensor. Other layouts and quantized modes follow the SDK. |
| `key`, `value` | Tensor | [B, Hkv, Skv, D] in BNSD, query dtype in example | Required | Key/value tensors with matching KV length. The SDK also supports paged layouts. |
| `num_heads`, `num_key_value_heads` | int | - | `1` / `0` | Optional query and KV head counts. Defaults are 1 and 0, where 0 infers the KV count. |
| `scale` | float | - | `1.0` | Optional logit scale, default 1.0. |
| `input_layout` | str | - | `"BSH"` | Optional SDK layout tag, default BSH. The example uses BNSD. |
| `actual_seq_lengths`, `actual_seq_lengths_kv` | sequence[int] or None | [B] for ordinary batched layouts | `None` | Optional valid query and KV lengths, interpreted according to the input layout. |
| `block_table`, `atten_mask` | Tensor or None | INT32 page table and SDK-compatible mask | `None` | Optional logical page mapping and mask. |
| `softmax_lse_flag` | bool | - | `False` | Optional. Requests the LSE result when true. Defaults to false. |
| `out` | sequence[Tensor] | two buffers matching the output shape and dtype | Required | Required by the out overload. Contains attention output followed by LSE. |
| `workspace` | Tensor or None | SDK workspace | SDK default | Optional workspace for the out overload. Obtain it using the matching helper and arguments. |
| `impl` | str | - | `"auto"` | `auto`, `ops-transformer`, `torch_native`. |
| `**kwargs` | dict | - | - | Additional supported TorchNPU keyword arguments. The linked vendor contract specifies their defaults and constraints. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `attention_output`, `softmax_lse` | tuple[Tensor, Tensor] | [B, Hq, Sq, D] in the example; FP32 LSE | Main call returns both tensors. For BNSD with LSE enabled, its shape is [B, Hq, Sq, 1]. The out overload uses the given buffers. |
| `workspace` | Tensor | SDK workspace shape and dtype | The workspace helper returns this buffer for the matching configuration. |

## ops-transformer backend

Calls `torch_npu.npu_fused_infer_attention_score`. Shape, dtype and layout combinations follow the installed SDK. Pass vendor options supported by the installed SDK. V1 uses scale and softmax_lse_flag. Match the main call, the workspace helper and the out overload's layout, optional inputs and flags. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/en/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score.md) [CANN ops-transformer source](https://gitcode.com/cann/ops-transformer/tree/7326dcc4f1e2dde5d2a802458877f927570267d1/attention/fused_infer_attention_score)

## torch_native backend

The main and `out` APIs share an FP16/BF16 reference implementation for these cases.

- Dense BNSD, BSND and BSH attention with optional explicit masks, bias and LSE.
- Single-token BSND decode with flattened KV pages `[num_blocks, block_size, kv_heads * head_dim]`, an int32 block table and one KV length per request.
- Packed TND attention with cumulative query and KV lengths and optional separate query and key RoPE tensors.

Paged BSND and packed TND require `sparse_mode=0`, the default token windows, and no attention mask or bias. LSE uses FP32 with shape `[B, N, S, 1]` for batched layouts and `[T, N, 1]` for TND. With LSE disabled, the second tensor has shape `[0]`. The workspace helper is provided by `ops-transformer`.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import (
    fused_infer_attention_score,
    fused_infer_attention_score_out,
    fused_infer_attention_score_get_max_workspace,
)

q = torch.randn((1, 32, 1, 128), dtype=torch.float16, device="npu")
k = torch.randn((1, 32, 512, 128), dtype=q.dtype, device=q.device)
v = torch.randn_like(k)
kwargs = dict(
    num_heads=32,
    scale=128**-0.5,
    input_layout="BNSD",
    softmax_lse_flag=True,
    impl="ops-transformer",
)
y, lse = fused_infer_attention_score(q, k, v, **kwargs)
workspace = fused_infer_attention_score_get_max_workspace(q, k, v, **kwargs)
out = [torch.empty_like(y), torch.empty_like(lse)]
fused_infer_attention_score_out(q, k, v, out=out, workspace=workspace, **kwargs)
```
