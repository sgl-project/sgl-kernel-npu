# get_symm_buffer_for_mega_moe

Create the persistent EP communication buffer used by [mega_moe](mega_moe.md), through the external CANN provider.

```python
from sgl_kernel_npu.operators.moe import get_symm_buffer_for_mega_moe

get_symm_buffer_for_mega_moe(*args, impl="auto", **kwargs)
```

The registered name is `moe.get_symm_buffer_for_mega_moe`. `auto` and `cann` resolve to the same implementation, explicitly registered for `Generation.ARCH22` and `Generation.ARCH35`. The normal Ascend runtime and a compatible `cann_ops_transformer` installation are required. The provider is imported only on the first backend call, shared with the MegaMoE wrapper.

All arguments except `impl` are forwarded unchanged to `cann_ops_transformer.ops.get_symm_buffer_for_mega_moe`; its return object is returned unchanged. An unknown implementation raises `ImplementationNotFoundError` before loading the backend. Provider errors propagate without fallback.

The existing integration supplies a process group and capacity/layout arguments:

```python
buffer = get_symm_buffer_for_mega_moe(
    group,
    num_experts=num_experts,
    num_max_tokens_per_rank=max_tokens,
    num_topk=topk,
    hidden=hidden,
    intermediate_hidden=intermediate_hidden,
    max_recv_token_num=0,
    dispatch_quant_mode=dispatch_quant_mode,
    dispatch_quant_out_dtype=dispatch_quant_out_dtype,
    impl="cann",
)
```

Argument constraints and defaults follow the installed provider. Keep the returned buffer alive while MegaMoE uses it, and release it with the provider's `destroy()` method after outstanding work completes. This wrapper does not cache buffer objects or destroy them automatically; callers must coordinate collective operations and buffer lifecycle across ranks.
