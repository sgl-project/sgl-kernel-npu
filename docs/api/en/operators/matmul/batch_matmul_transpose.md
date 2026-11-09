# batch_matmul_transpose

## Hardware support

| Implementation | arch22 (A2/A3, 2201) | arch35 (A5, 3510) |
| --- | --- | --- |
| `ascendc` | Supported | Unsupported |
| `torch_native` | Supported | Unverified |

The AscendC kernel supports Ascend A2/A3 (arch22, 2201).

## Signature

```python
from sgl_kernel_npu.operators.matmul import batch_matmul_transpose

def batch_matmul_transpose(
    tensor_a, tensor_b, tensor_c, format_mode=None, quant_mode=None, *, impl="auto"
):
    ...
```

`auto` selects `ascendc`, with no fallback. The function writes into the
caller-allocated `tensor_c` and returns `None`.

## Description

Compute a batched matrix product with interleaved token and batch axes:

$$
C_{m,b,n} = \sum_{k=0}^{K-1} A_{m,b,k} W_{b,k,n}.
$$

Here $M$ is the token count, $B$ the independent matrix/head count, $K$ the
reduction dimension, and $N$ the output feature dimension. Both implementations
accumulate in FP32 and store the result in the input dtype.

For explicit NZ weights, the supplied four-dimensional ND tensor contains
$W^{NZ}_{b,j,k,r}=W_{b,k,16j+r}$. This is a physical packing convention, not a
request to convert an opaque `torch_npu` storage format. For example, pack ND
weights with `w.reshape(B, K, N // 16, 16).permute(0, 2, 1, 3).contiguous()`.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `tensor_a` | Tensor | `[M, B, K]` | FP16 / BF16 | Required | Contiguous; `1 <= M <= 1024`. |
| `tensor_b` | Tensor | ND: `[B, K, N]`; NZ: `[B, N/16, K, 16]` | Same as A | Required | Contiguous; NZ requires K and N divisible by 16. |
| `tensor_c` | Tensor | `[M, B, N]` | Same as A | Required | Caller-allocated, contiguous output, overwritten in place; must not overlap A or B. |
| `format_mode` | str / None | — | — | `None` | `None` means `"ND"`; accepts `"ND"` and `"NZ"`. |
| `quant_mode` | str / None | — | — | `None` | Accepts `per_channel_symm`, `per_channel_asymm`, and `per_token_symm`. `None` selects `per_channel_symm`. All supported values use floating-point computation without quantization. |
| `impl` | str | — | — | `"auto"` | Keyword-only; `auto`, `ascendc`, or `torch_native`. |

| Return value | Type | Shape | Dtype | Side effects |
| --- | --- | --- | --- | --- |
| None | NoneType | — | — | Writes C; A and B remain unchanged. |

All dimensions must be positive. Each tensor has at most `UINT32_MAX` elements.
All tensors must have the same dtype and device. `ascendc` requires an NPU;
directly loading the torch-native implementation also permits CPU algebra checks.
The public package itself requires its normal NPU/native runtime prerequisites.
Passing unsupported layouts, dtypes, shapes or mode names raises an error.

## ascendc backend

The Torch adapter launches the bundled ACLNN operator on the current NPU stream.
The CANN tiling callback chooses tile sizes and block scheduling from the input
dimensions and device properties. CANN manages the tiling data that the kernel
uses to compute the batched matrix product.

## torch_native backend

This implementation unpacks NZ weights and computes the product with PyTorch
`bmm` using FP32 operands. It casts the result to the output dtype and copies it
into `tensor_c`. Different accumulation orders can cause numerical differences.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.matmul import batch_matmul_transpose

a = torch.randn(3, 2, 32, device="npu", dtype=torch.float16)
b = torch.randn(2, 32, 48, device="npu", dtype=a.dtype)
c = torch.empty(3, 2, 48, device="npu", dtype=a.dtype)
assert batch_matmul_transpose(a, b, c, impl="ascendc") is None
reference = torch.empty_like(c)
batch_matmul_transpose(a, b, reference, impl="torch_native")
torch.testing.assert_close(c, reference, atol=1e-3, rtol=1e-3)
```

## Build and test

From the repository root, build and install the single-operator wheel, then run the operator tests on Ascend A2/A3.

```bash
python3 setup.py --make-clean --arch-version 2201 --ops batch_matmul_transpose \
  --artifact wheel --without-third-party -j 8
python3 -m pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl

python3 -m pytest tests/python/operators/matmul/test_batch_matmul_transpose.py \
  --impl=ascendc --impl-ref=torch_native --enable-perf=false -v -rs
```
