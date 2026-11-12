# get_symm_buffer_for_mega_moe

## Hardware support

| impl | arch22 (A3 test target) | arch35 (A5) |
| --- | --- | --- |
| `auto` | Registered; hardware tests pending | Registered; hardware tests pending |
| `cann_ops_transformer` | Registered; hardware tests pending | Registered; hardware tests pending |

Only ARCH22 and ARCH35 are declared supported; this is not an all-generation registration. Hardware validation remains pending.

Requires CANN 9.2.0, matching PyTorch/torch_npu and a `cann_ops_transformer` provider exporting MegaMoE, including its `ninja` JIT dependency. Registration does not certify every provider version or input mode.

## Signature

```python
from sgl_kernel_npu.operators.moe import get_symm_buffer_for_mega_moe
get_symm_buffer_for_mega_moe(*args, impl="auto", **kwargs)
```

`auto` and `cann_ops_transformer` alias the same implementation, with no fallback. The first call imports the provider and caches its functions. All arguments except `impl`, object identities and results pass through unchanged. Import/execution failures propagate; an unknown impl fails before provider import.

## Description

The caller owns lifetime: reuse only with the same configuration, create a new buffer when configuration changes, synchronize completed work before `destroy()`, then destroy the process group. The wrapper does not cache, pad or destroy buffers. Every rank must execute collectives in the same order. This resource-creation interface does not require a second numerical implementation.

## Parameters and return value

Defaults in the parameter table refer to the pinned provider linked below. The public signature remains `(*args, impl="auto", **kwargs)` and only defines the `impl` default; other provider options pass through unchanged. See the original document for additional optional parameters.

| Parameter | Default | Type, meaning and constraints |
| --- | --- | --- |
| First positional argument group | Required | Initialized HCCL EP `torch.distributed.ProcessGroup`. |
| `num_experts` | Required | int, positive global expert count E divisible by group size. |
| `num_max_tokens_per_rank` | Required | int, per-rank capacity; compatible settings on every rank. |
| `num_topk` | Required | int, number of selected experts K per token. |
| `hidden`, `intermediate_hidden` | Required | int, model H and expert I; obey provider mode alignment. |
| `max_recv_token_num` | `0` | int, receive capacity setting; tests explicitly use 0, interpreted by provider. |
| `dispatch_quant_mode` | `0` | int; tests use 0 (BF16), 2 (INT8), or 4 (MXFP). |
| `dispatch_quant_out_dtype` | `None` | dtype or provider FP4 enum matching quantization mode; BF16 tests explicitly pass None. |
| `impl` | `"auto"` | Keyword-only, default `"auto"`; also accepts `"cann_ops_transformer"`. |

| Return value | Type | Description |
| --- | --- | --- |
| `buffer` | Provider `SymmBuffer` | Live EP communication resource; synchronize before `destroy()`. |

## cann_ops_transformer backend

Original operator documentation: [ops-transformer MegaMoE](https://gitcode.com/cann/ops-transformer/blob/be9c0b2766e827b523c4a2b8728e738d470b5a36/mc2/mega_moe/docs/torchapi_mega_moe.md). This pinned source describes provider signatures, defaults, quantization formulas and layout restrictions; use the documentation matching the installed provider version.

A3 coverage includes BF16, W8A8 and W4A8; A5 includes MXFP8 E4M3/E5M2, MXFP4 E2M1 and MXFP8-W4. A5 uses platform-specific weight layouts and ND E8M0 scales. FP4 storage must follow provider packing rules; A8W4-FP uses FRACTAL_NZ_C0_32. A3 weights cannot be reused directly on A5.

The interface intentionally retains `*args/**kwargs`, without redefining external provider defaults or constraints. Consult the matching provider version for exact restrictions.

## Example

For complete four-rank A3 input construction, execution and cleanup, see [mega_moe](mega_moe.md).

```python
# Save as buffer_example.py; run: torchrun --standalone --nproc-per-node=4 buffer_example.py
import os
import torch
import torch.distributed as dist
import torch_npu
from sgl_kernel_npu.operators.moe import get_symm_buffer_for_mega_moe

device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
torch.npu.set_device(device)
dist.init_process_group("hccl")
buffer = None
try:
    dist.all_reduce(torch.ones(1, device=device))
    buffer = get_symm_buffer_for_mega_moe(
        dist.group.WORLD, num_experts=8, num_max_tokens_per_rank=32,
        num_topk=4, hidden=2048, intermediate_hidden=1024,
        impl="cann_ops_transformer",
    )
finally:
    if buffer is not None:
        torch.npu.synchronize()
        buffer.destroy()
    dist.destroy_process_group()
```
