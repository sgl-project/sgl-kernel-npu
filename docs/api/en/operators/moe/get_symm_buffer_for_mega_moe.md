# get_symm_buffer_for_mega_moe

## Chip support

| impl | arch22 (A3 test target) | arch35 (A5) |
| --- | --- | --- |
| `auto` | Registered; hardware tests pending | Registered; hardware tests pending |
| `cann` | Registered; hardware tests pending | Registered; hardware tests pending |

Requires CANN 9.2.0, matching PyTorch/torch_npu and a `cann_ops_transformer` provider exporting MegaMoE, including its `ninja` JIT dependency. Registration does not certify every provider version or input mode.

## Signature

```python
from sgl_kernel_npu.operators.moe import get_symm_buffer_for_mega_moe
get_symm_buffer_for_mega_moe(*args, impl="auto", **kwargs)
```

`auto` and `cann` alias the same implementation, with no fallback. The first call imports the provider and caches its functions. All arguments except `impl`, object identities and results pass through unchanged. Import/execution failures propagate; an unknown impl fails before provider import.

## Behavior

The caller owns lifetime: reuse only with the same configuration, create a new buffer when configuration changes, synchronize completed work before `destroy()`, then destroy the process group. The wrapper does not cache, pad or destroy buffers. Every rank must execute collectives in the same order. This resource-creation interface does not require a second numerical implementation.

## Arguments and returns

| Argument | Type, meaning and constraints |
| --- | --- |
| First positional argument group | Initialized HCCL EP `torch.distributed.ProcessGroup`. |
| `num_experts` | int, positive global expert count E divisible by group size. |
| `num_max_tokens_per_rank` | int, per-rank capacity; compatible settings on every rank. |
| `num_topk` | int, number of selected experts K per token. |
| `hidden`, `intermediate_hidden` | int, model H and expert I; obey provider mode alignment. |
| `max_recv_token_num` | int, receive capacity setting; tests explicitly use 0, interpreted by provider. |
| `dispatch_quant_mode` | int; tests use 0 (BF16), 2 (INT8), or 4 (MXFP). |
| `dispatch_quant_out_dtype` | dtype or provider FP4 enum matching quantization mode; BF16 tests explicitly pass None. |
| `impl` | Keyword-only, default `"auto"`; also accepts `"cann"`. |

Returns the provider buffer object. Provider defaults belong to the installed version; only impl has a wrapper-defined default.

## cann

A3 coverage includes BF16, W8A8 and W4A8; A5 includes MXFP8 E4M3/E5M2, MXFP4 E2M1 and MXFP8-W4. A5 uses platform-specific weight layouts and ND E8M0 scales. FP4 storage must follow provider packing rules; A8W4-FP uses FRACTAL_NZ_C0_32. A3 weights cannot be reused directly on A5.

The interface intentionally retains `*args/**kwargs`, without redefining external provider defaults or constraints. Consult the matching provider version for exact restrictions.

## Example

For complete four-rank A3 input construction, execution and cleanup, see [mega_moe](mega_moe.md).
