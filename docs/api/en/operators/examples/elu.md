# elu

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Supported | Unsupported |
| `torch_native` | Supported | Supported |

The Ascend C Kernel is declared for arch22 only, so `ascendc` is not available on
arch35. `auto` is an alias of `ascendc`, not a separate implementation, and does
not fall back when the Ascend C operator is missing. Install a wheel built for the matching
architecture with `--ops elu`; see [build instructions](../../../../../python/sgl_kernel_npu/README.md).

## Signature

```python
from sgl_kernel_npu.operators.examples import elu

def elu(x, alpha=1.0, *, impl="auto"):
    ...
```

Omitting `impl` or passing `"auto"` selects `ascendc`. On arch35 the Ascend C
implementation is not declared; request `impl="torch_native"` explicitly there.

## Description

Computes the Exponential Linear Unit element-wise, where $\alpha$ is the
coefficient of the negative branch:

$$
\mathrm{elu}(x)=
\begin{cases}
x, & x > 0 \\
\alpha\left(e^{x}-1\right), & x \le 0
\end{cases}
$$

The Ascend C implementation does not branch per element. It rewrites the
piecewise definition with $\max(x,0)=x-\min(x,0)$:

$$
\mathrm{elu}(x)=\alpha\left(e^{\min(x,0)}-1\right)+\left(x-\min(x,0)\right)
$$

so the vector unit executes one straight sequence of instructions per tile.
`Exp` only ever receives $\min(x,0)\le 0$, whose result stays in $(0,1]$, so
fp16 cannot overflow. The `torch_native` reference evaluates the definition
literally in float32 with `torch.expm1`, which keeps the relative accuracy of
$e^{x}-1$ near zero, and casts back to the input dtype.

Numerical note: the fp16 kernel evaluates $e^{\min(x,0)}-1$ directly in fp16, so
the *relative* error grows as $x\to0^-$ (about 2-3% for $|x|\le0.01$). The
*absolute* error in that band is only about 2.3e-5, which the `atol` term of the
usual tolerance absorbs, so the standard `1e-3` still holds. Use fp32 when
relative accuracy in that narrow band matters.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | Any shape with at least one dimension and one element | `torch.float16`, `torch.float32` | Required | Contiguous NPU tensor in ND storage format. The operator treats the tensor as a flat element sequence, so no alignment or rank requirement applies. |
| `alpha` | `float` | — | — | `1.0` | Coefficient of the negative branch. Must be finite. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only. One of `"auto"`, `"ascendc"`, `"torch_native"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `y` | `torch.Tensor` | Same as `x` | Same as `x` | Newly allocated contiguous ND tensor on the same NPU; the input is not modified. |

The public function checks dtype, device, contiguity, storage format, non-empty
input and `alpha` before dispatch. Unsupported inputs raise instead of being
converted silently.

## ascendc backend

- Declared as an AICORE operator in `csrc/examples/elu/` with
  `SUPPORTED_ARCHS 2201`, and called through `torch.ops.npu.elu`.
- Host tiling derives `tileLength` from the UB capacity and dtype (512-element
  vector aligned, capped at 4096), participates with at most one AIV core per
  tile, and splits the element count evenly. The Kernel handles a short last
  tile with `DataCopyPad`, so arbitrary element counts need no host padding.
- The Kernel runs a double-buffered CopyIn / Compute / CopyOut pipeline over two
  single-buffered scratch buffers and computes ELU with one branch-free vector
  sequence per tile.

## torch_native backend

Uses PyTorch operations only: the piecewise definition is evaluated with a
boolean mask in float32 and returned in the input dtype. It accepts any shape,
including non-contiguous inputs, and serves as the independent reference.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.examples import elu

x = torch.randn(16, 7168, device="npu", dtype=torch.float16)

print(list_impls(elu))  # ['ascendc', 'auto', 'torch_native']
y_default = elu(x)  # auto -> ascendc
y_ascendc = elu(x, alpha=1.0, impl="ascendc")
y_native = elu(x, alpha=1.0, impl="torch_native")

torch.testing.assert_close(y_default, y_ascendc, rtol=0, atol=0)
torch.testing.assert_close(y_ascendc, y_native, rtol=1e-3, atol=1e-3)
```
