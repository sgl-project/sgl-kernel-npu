# mega_moe

## Chip support

| impl | arch22 (A3 test target) | arch35 (A5) |
| --- | --- | --- |
| `auto` | Registered; hardware tests pending | Registered; hardware tests pending |
| `cann` | Registered; hardware tests pending | Registered; hardware tests pending |

Requires CANN 9.2.0, matching PyTorch/torch_npu and a `cann_ops_transformer` provider exporting MegaMoE, including its `ninja` JIT dependency. Registration does not certify every provider version or input mode.

## Signature

```python
from sgl_kernel_npu.operators.moe import mega_moe
mega_moe(*args, impl="auto", **kwargs)
```

`auto` and `cann` alias the same implementation, with no fallback. The first call imports the provider and caches its functions. All arguments except `impl`, object identities and results pass through unchanged. Import/execution failures propagate; an unknown impl fails before provider import.

## Behavior

Dispatch tokens to expert ranks, compute FFNs, then combine weighted results. Unquantized SwiGLU semantics follow below (g/u split the first projection). Quantization, rounding and saturation follow the installed provider; this wrapper performs no numerical conversion.

`y[t] = sum_k routing[t,k] * (silu(g[t,e]) * u[t,e]) @ W2[e]`, `e = ids[t,k]`, `[g,u] = x[t] @ W1[e]`.

## Arguments and returns

| Argument | Type, shape and constraints |
| --- | --- |
| `x` | NPU Tensor `[T,H]`; tested input is BF16. A3 uses buffer capacity T and caller padding; A5 uses actual token length. |
| `topk_ids` | NPU int32 `[T,K]`, global expert IDs; E must be divisible by EP world size. |
| `topk_weights` | NPU `[T,K]`; tested FP32 on A3, BF16/FP32 on A5. |
| `l1_weights`, `l2_weights` | Local expert TensorLists; A3 BF16 `[H,2I]` / `[I,H]` NZ; see backend notes for quantized/A5 layouts. |
| `sym_buffer` | Live provider buffer for the same EP group and configuration. |
| `l1_weights_sf`, `l2_weights_sf` | Optional quantization scale TensorLists; omit/use None for BF16; must match quantized weights. |
| `l1_bias`, `l2_bias` | Optional compensation biases; used by A3 W4A8, must be None on A5. |
| `x_active_mask` | Optional A3 int8 `[T]` indicating active tokens; omit/use None on A5. |
| `activation` | Tests cover `swiglu` and `situglu` (SiTU on A5). |
| `activation_params` | Optional dict, e.g. `beta`, `linear_beta`; defaults belong to the provider. |
| `weight1_type`, `weight2_type` | Provider enum for A5 FP4 weights; omit/use None otherwise. |
| `impl` | Keyword-only, default `"auto"`; also accepts `"cann"`. |

| Return | Description |
| --- | --- |
| `output` | NPU `[T,H]`; BF16 tests preserve x dtype; caller slices active A3 tokens. |
| `expert_token_nums` | Counts for local experts; count/cumulative format follows provider environment settings. Tests set `MOE_EXPERT_TOKEN_NUMS_TYPE=1`. |

## cann

A3 coverage includes BF16, W8A8 and W4A8; A5 includes MXFP8 E4M3/E5M2, MXFP4 E2M1 and MXFP8-W4. A5 uses platform-specific weight layouts and ND E8M0 scales. FP4 storage must follow provider packing rules; A8W4-FP uses FRACTAL_NZ_C0_32. A3 weights cannot be reused directly on A5.

The interface intentionally retains `*args/**kwargs`, without redefining external provider defaults or constraints. Consult the matching provider version for exact restrictions.

## Example

```python
# Save as example.py; run: torchrun --standalone --nproc-per-node=4 example.py
import os
import torch
import torch.distributed as dist
import torch_npu
from sgl_kernel_npu.operators.moe import mega_moe, get_symm_buffer_for_mega_moe

torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("hccl")
T, H, I, E, K = 32, 2048, 1024, 8, 4  # A3 BF16 example
local_experts = E // dist.get_world_size()
x = torch.randn(T, H, device="npu", dtype=torch.bfloat16)
ids = torch.arange(K, device="npu", dtype=torch.int32).expand(T, K).contiguous()
routing = torch.full((T, K), 1 / K, device="npu", dtype=torch.float32)
w1 = [torch_npu.npu_format_cast(torch.randn(H, 2*I, device="npu",
      dtype=torch.bfloat16) / H**0.5, 29) for _ in range(local_experts)]
w2 = [torch_npu.npu_format_cast(torch.randn(I, H, device="npu",
      dtype=torch.bfloat16) / I**0.5, 29) for _ in range(local_experts)]
buffer = None
try:
    buffer = get_symm_buffer_for_mega_moe(
        dist.group.WORLD, num_experts=E, num_max_tokens_per_rank=T,
        num_topk=K, hidden=H, intermediate_hidden=I,
        max_recv_token_num=0, dispatch_quant_mode=0,
        dispatch_quant_out_dtype=None, impl="cann",
    )
    output, counts = mega_moe(
        x=x, topk_ids=ids, topk_weights=routing, l1_weights=w1, l2_weights=w2,
        sym_buffer=buffer, activation="swiglu", impl="cann",
    )
    torch.npu.synchronize()
finally:
    if buffer is not None:
        torch.npu.synchronize()
        buffer.destroy()
    dist.destroy_process_group()
```

Distributed tests, the independent-reference exception and performance commands are in [MoE test README](../../../../../tests/python/operators/moe/README.md).
