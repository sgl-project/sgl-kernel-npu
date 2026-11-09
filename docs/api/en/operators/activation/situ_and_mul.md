# situ_and_mul

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

arch22 was verified on A3. A2 was not tested separately. arch35 was verified on A5.
The available implementations are Triton and PyTorch.

## Signature

```python
from sgl_kernel_npu.operators.activation import situ_and_mul

def situ_and_mul(
    x: torch.Tensor,
    group_list: Optional[torch.Tensor] = None,
    group_list_type: Optional[int] = None,
    beta: float = 4.0,
    linear_beta: Optional[float] = 25.0,
    *,
    impl="auto",
) -> torch.Tensor:
    ...
```

`auto` and `triton` are registered to the same entry point. The default uses Triton, with backend modules imported lazily. Unavailable implementations raise an error without automatic fallback. Callers must satisfy the input constraints below; unsupported inputs are not comprehensively validated.

## Description

Applies the SiTU activation used by Kimi-K3 to gate/up halves of an input, then multiplies them. This API computes the floating-point activation without quantization. The [Kimi-K3 configuration](https://huggingface.co/moonshotai/Kimi-K3/blob/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721/config.json) uses gate/up bounds 4/25. [SGLang KimiK3MLP](https://github.com/sgl-project/sglang/blob/53a24c448ecd6c26bf6461b80ffb8d55397d57cf/python/sglang/srt/models/kimi_k3.py#L290-L344) applies this activation between the packed gate/up projection and the down projection.

For input shape `[..., 2*d]`, split the last dimension into gate and up halves:

$$
g = x[..., :d], \qquad u = x[..., d:].
$$

$$
a = \beta\tanh(g / \beta)\,\sigma(g), \qquad
v = \begin{cases}
L\tanh(u/L), & L = \mathrm{linear\_beta} \ne \mathrm{None},\\
u, & \mathrm{linear\_beta} = \mathrm{None},
\end{cases}
\qquad y = a v.
$$

All intermediates use float32; the result is cast to `x.dtype`.

Leading dimensions are flattened into `N` token rows. Without `group_list`, all rows are processed. With `group_list_type=1`, the valid row count is `sum(group_list)`; with `group_list_type=0`, it is `group_list[-1]`. Cumulative counts have one entry per expert, without a required leading zero. Only the valid prefix is written. Remaining output rows are padding with unspecified values and must not be consumed or compared.

Group counts must be nonnegative; cumulative counts must be nonnegative and nondecreasing. The total must satisfy `0 <= total <= N`. These device-value preconditions are the caller's responsibility and are not checked on the Triton path, avoiding host synchronization. Participating floating-point inputs are expected to be finite.

## Parameters and return values

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[..., 2*d]` | `torch.float32`, `torch.float16`, `torch.bfloat16` | Required | Strided tensor with at least one dimension and `d > 0`. Leading dimensions may be empty. Gate is the first half; up is the second half. |
| `group_list` | `torch.Tensor` or `None` | `[E]` | `torch.int32`, `torch.int64` | `None` | Optional contiguous, nonempty per-expert vector on x's device. With None, all token rows are valid. |
| `group_list_type` | `int` or `None` | — | — | `None` | `0` for cumulative counts, `1` for counts; required with group_list, otherwise ignored. |
| `beta` | `int` or `float` | — | — | `4.0` | Finite positive Python scalar, excluding bool; gate saturation bound. |
| `linear_beta` | `int`, `float` or `None` | — | — | `25.0` | Finite positive Python scalar, excluding bool; up saturation bound. None disables up saturation. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"triton"`, or `"torch_native"`. |

The public operator performs the three legacy checks once before dispatch: an even last input dimension, group_list_type in {0, 1} when groups are supplied, and int32/int64 group dtype. It also rejects an empty group_list. Both implementations share these checks, including for empty inputs. Input dtype/device/layout, contiguous one-dimensional groups and finite positive scalar bounds remain caller preconditions and are not explicitly checked. Backend implementations do not repeat these shared checks; Triton additionally checks the row-capacity limit described below.

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[..., d]` | x.dtype | Newly allocated contiguous tensor on x's device; inputs are not modified. With groups, only the valid flattened prefix is defined. |

For Triton, the flattened input capacity must satisfy `N <= 2**31 - C`, where `C` is the selected NPU vector-core count. The wrapper rejects a larger capacity before output allocation or kernel launch. This leaves room for ceil-div row partitioning in the preserved int32 scheduler. Group totals must still satisfy `0 <= total <= N`. The PyTorch reference does not use this scheduler.

Empty token inputs return a correctly shaped empty output without a kernel launch. A grouped total of zero writes no output values.

## triton backend

- Located in `implementation/triton/activation/situ_and_mul.py`, with kernel `_situ_and_mul_kernel`. Nonempty inputs require an NPU.
- Preserves the existing row distribution over vector cores, H-tiling, masked feature tails, FP32 arithmetic and autotuning over `BLOCK_H` in `{1024, 2048, 4096, 8192}` with multibuffering. Int64 row offsets support large flattened input offsets.
- Materializes non-contiguous inputs as packed rows; contiguous inputs need no copy. Group metadata must already be contiguous. Copy costs are included when profiling such inputs.
- Reads the final cumulative-count entry at `E - 1`. The count-reduction extent is rounded up to a power of two with masked expert tails, including non-power-of-two expert counts. Group totals stay on device.

## torch_native backend

- Uses independent PyTorch operations without calling Triton; supports strided inputs and can be checked separately on CPU.
- Computes tanh, sigmoid and multiplication in float32, then converts to x.dtype. Grouped calls synchronize the total to the host to select the valid prefix; this reference behavior is unsuitable for graph capture.
- Backend math approximations may differ. Use dtype-appropriate tolerances; bitwise equality is not guaranteed.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import situ_and_mul

x = torch.randn(17, 6144, device="npu", dtype=torch.bfloat16)
output = situ_and_mul(x)
reference = situ_and_mul(x, impl="torch_native")
torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)

# Routed MoE: only the first 12 flattened token rows are valid.
counts = torch.tensor([0, 3, 5, 4], device="npu", dtype=torch.int64)
routed = situ_and_mul(x, counts, 1, impl="triton")
routed_ref = situ_and_mul(x, counts, 1, impl="torch_native")
torch.testing.assert_close(routed[:12], routed_ref[:12], rtol=2e-2, atol=2e-2)
```
