# mega_moe

Run the external CANN MegaMoE provider, which combines token dispatch, expert FFN computation and result combination. This package registers and forwards the interface; it does not implement the kernel or manage communication buffers.

```python
from sgl_kernel_npu.operators.moe import mega_moe

mega_moe(*args, impl="auto", **kwargs)
```

`auto` and `cann` select the same provider on `Generation.ARCH22` and `Generation.ARCH35`. Other generations are not declared supported. The package requires its normal Ascend NPU runtime; `cann_ops_transformer.ops` is loaded lazily on the first backend call. Install a compatible `cann_ops_transformer` provider and its JIT build dependencies (including `ninja`). Provider import and execution errors propagate to the caller without fallback.

All positional and keyword arguments except `impl` are forwarded unchanged to `cann_ops_transformer.ops.mega_moe`. Shapes, dtypes, quantization modes, activation parameters and return values follow the installed provider's contract. Generation registration does not guarantee that every provider version supports every input configuration.

The existing integration passes `x`, `topk_ids`, `topk_weights`, per-expert `l1_weights` and `l2_weights`, and a persistent `sym_buffer`. Optional arguments include weight scales, biases, `x_active_mask`, `activation` and `activation_params`. The provider returns `(output, expert_token_nums)`.

Create the communication buffer using [get_symm_buffer_for_mega_moe](get_symm_buffer_for_mega_moe.md). The caller owns its lifetime and any required token padding or output slicing; the registration wrapper performs neither.

```python
from sgl_kernel_npu.operators import get_impl, list_impls
from sgl_kernel_npu.operators.moe import mega_moe

assert list_impls(mega_moe) == ["auto", "cann"]
assert get_impl(mega_moe, "auto") is get_impl(mega_moe, "cann")
# output, counts = mega_moe(..., sym_buffer=buffer, impl="cann")
```
