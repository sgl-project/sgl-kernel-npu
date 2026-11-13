# batch_matmul_transpose

## Hardware support

| Implementation | arch22 (A2/A3, 2201) | arch35 (A5, 3510) |
| --- | --- | --- |
| `ascendc` | Supported | Unsupported |
| `torch_npu` | Limited support | Supported |
| `torch_native` | Supported | Unverified |

The AscendC kernel supports Ascend A2/A3 (arch22, 2201).

## Signature

```python
from sgl_kernel_npu.operators.matmul import batch_matmul_transpose

def batch_matmul_transpose(tensor_a, tensor_b, *, impl="auto"):
    ...
```

The function takes ND tensors and returns a new tensor of shape `[M, B, N]`.

MLA uses `auto`, which selects `ascendc` on arch22 when any of `M`, `B * K`, or
`N` is at least 65536. It selects `torch_npu` for other shapes on arch22 and for
all shapes on arch35. DSA explicitly selects `impl="torch_npu"` for its query
and value projections in every forward mode.

## Description

Compute a batched matrix product with interleaved token and batch axes:

$$
C_{m,b,n} = \sum_{k=0}^{K-1} A_{m,b,k} W_{b,k,n}.
$$

Here $M$ is the token count, $B$ the independent matrix/head count, $K$ the
reduction dimension, and $N$ the output feature dimension. `ascendc` and
`torch_native` accumulate in FP32 and store the result in the input dtype.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `tensor_a` | Tensor | `[M, B, K]` | FP16 / BF16 | Required | `ascendc` and `torch_native` require `1 <= M <= 1024`. |
| `tensor_b` | Tensor | `[B, K, N]` | Same as A | Required | `ascendc` and `torch_native` require contiguous weights. |
| `impl` | str | — | — | `"auto"` | Keyword-only. Supports `auto`, `ascendc`, `torch_npu`, and `torch_native`. |

| Return value | Type | Shape | Dtype | Side effects |
| --- | --- | --- | --- | --- |
| Result | Tensor | `[M, B, N]` | Same as A | Newly allocated output. |

For `ascendc` and `torch_native`, all dimensions must be positive and each
tensor has at most `UINT32_MAX` elements. `ascendc` makes A contiguous as
needed. Both inputs must share dtype and device. `ascendc` requires an NPU.
The public package itself requires its normal NPU/native runtime prerequisites.
Passing unsupported layouts, dtypes or shapes raises an error.

## ascendc backend

The Torch adapter launches the bundled ACLNN operator on the current NPU stream.
The CANN tiling callback chooses tile sizes and block scheduling from the input
dimensions and device properties. CANN manages the tiling data that the kernel
uses to compute the batched matrix product.

## torch_npu backend

This implementation calls `torch_npu.npu_transpose_batchmatmul` with ND inputs
`[M, B, K]` and `[B, K, N]` and returns its output. By default it uses
`perm_x1=(1, 0, 2)`, `perm_x2=(0, 1, 2)`, and `perm_y=(1, 0, 2)`.

On arch22 (A2/A3), CANN 9.1 specifies `1 <= B <= 65535` and `1 <= N <= 65535`.
The fused input transpose from `[M, B, K]` also requires `B * K <= 65535`.
When the first input is already in `[B, M, K]` layout, CANN instead checks
`K <= 65535` for that input stride.
[CANN 9.1 TransposeBatchMatMul reference](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnTransposeBatchMatMul.md)

On arch22, when `B * K >= 65536`, this backend transposes the first input to
contiguous `[B, M, K]` and calls TorchNPU with `perm_x1=(0, 1, 2)`. The output
has shape `[M, B, N]` because `perm_y=(1, 0, 2)`. The limits on `B`, `K`,
and `N` for the `[B, M, K]` layout still apply.

## torch_native backend

This implementation computes the product with PyTorch `bmm` using FP32
operands and casts the result to the input dtype. Different accumulation
orders can cause numerical differences.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.matmul import batch_matmul_transpose

a = torch.randn(3, 2, 32, device="npu", dtype=torch.float16)
b = torch.randn(2, 32, 48, device="npu", dtype=a.dtype)
output = batch_matmul_transpose(a, b, impl="ascendc")
reference = batch_matmul_transpose(a, b, impl="torch_native")
torch.testing.assert_close(output, reference, atol=1e-3, rtol=1e-3)
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
