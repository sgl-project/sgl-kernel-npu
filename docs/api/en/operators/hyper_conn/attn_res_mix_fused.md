# attn_res_mix_fused

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

## Signature

```python
from sgl_kernel_npu.operators.hyper_conn import attn_res_mix_fused

def attn_res_mix_fused(
    prefix_sum,
    bank,
    num_valid_blocks,
    combined_weight,
    variance_epsilon,
    *,
    impl="auto",
):
    ...
```

`auto` and `triton` are registered to the same entry point. The default uses Triton, with backend modules imported lazily. Unsupported inputs or unavailable implementations raise an error without automatic fallback.

## Description

Computes normalized scores, a row-wise softmax, and a weighted mixture for Kimi-K3 attention residual. Returns the mixture before the final RMSNorm. The operator reads existing snapshots without writing snapshots or adding residual inputs.

Let $N$ be the token count, $H$ the hidden dimension, and $B$ the number of valid snapshots. For each token $t$, form $B+1$ rows by appending `prefix_sum[t, :]` to `bank[t, :B, :]`:

$$
v_{t,r,h}=\begin{cases}
\mathrm{bank}_{t,r,h}, & 0\le r<B,\\
\mathrm{prefix\_sum}_{t,h}, & r=B.
\end{cases}
$$

Using $w=\mathrm{combined\_weight}$ and $\varepsilon=\mathrm{variance\_epsilon}$, compute:

$$
q_{t,r}=\left(\frac{1}{H}\sum_{h=0}^{H-1}v_{t,r,h}^{2}+\varepsilon\right)^{-1/2},
\qquad s_{t,r}=\sum_{h=0}^{H-1}(v_{t,r,h}q_{t,r})w_h.
$$

$$
p_{t,r}=\frac{\exp(s_{t,r}-\max_j s_{t,j})}{\sum_{j=0}^{B}\exp(s_{t,j}-\max_k s_{t,k})},
\qquad y_{t,h}=\sum_{r=0}^{B}p_{t,r}v_{t,r,h}.
$$

The weighted sum uses the original row values, rather than the normalized row values. No output RMSNorm is applied. All intermediate computations use float32; the result is cast to `prefix_sum.dtype`.

In SGLang, `combined_weight` is the elementwise product of the score RMSNorm weight and score projection weight, usually a contiguous float32 vector. The caller must compute this weight beforehand. The bank capacity may exceed the number of valid snapshots. `bank[:, B:, :]` is excluded from computation and may contain uninitialized values or NaNs. Participating inputs are expected to contain finite values.

## Parameters and return values

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `prefix_sum` | `torch.Tensor` | `[N, H]` | `torch.float32`, `torch.float16`, `torch.bfloat16` | Required | Current residual prefix; `N >= 0` and `H > 0`. |
| `bank` | `torch.Tensor` | `[N, C, H]` | `torch.float32`, `torch.float16`, `torch.bfloat16` | Required | Snapshot bank; `C >= 0`. Must share prefix_sum's device, but may have a different dtype. Only the first `num_valid_blocks` rows are read. |
| `num_valid_blocks` | `int` | — | — | Required | Python integer excluding bool; must satisfy `0 <= num_valid_blocks <= C`. Triton raises `ValueError` for an out-of-range value. |
| `combined_weight` | `torch.Tensor` | `[H]` | `torch.float32`, `torch.float16`, `torch.bfloat16` | Required | Combined score weight. Must share prefix_sum's device, but may have a different dtype. |
| `variance_epsilon` | `int` or `float` | — | — | Required | Python real scalar excluding bool; must be finite and positive, such as `1e-6`. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"triton"`, or `"torch_native"`. |

All three tensors require `torch.strided` layout. Triton validates metadata before computation, including for empty inputs; the public operator handles dispatch. The `torch_native` implementation assumes valid inputs and serves as a mathematical reference without explicit metadata validation.

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[N, H]` | prefix_sum.dtype | On prefix_sum's device. Nonempty calls allocate a new output without modifying any input. |

When `N == 0`, returns `prefix_sum` itself without launching a kernel. When `N > 0` and `num_valid_blocks == 0`, only the prefix row participates. For finite inputs, the output equals prefix_sum in value but is still a new tensor.

## triton backend

- Located in `implementation/triton/hyper_conn/attn_residual_mix.py`, with kernel `_attn_res_mix_fused_kernel`.
- Nonempty inputs must be on an NPU. The last-dimension strides of prefix_sum and bank must be 1, as must combined_weight's stride. Token strides and bank row strides may differ; non-contiguous views satisfying these constraints are supported.
- Expands the hidden dimension to `BLOCK_H = next_power_of_2(H)`. Loads and stores are masked, padded channels contribute zero, and mean square is divided by the actual `H`. The hidden dimension no longer has to be a power of two; masked tails handle dimensions such as `7168`.
- Fuses scoring, softmax, and mixing into one persistent vector-core kernel. `N` and `B` remain dynamic; `NB` is `B + 1` rounded up to a power of two. No score matrix is allocated.

## torch_native backend

- Uses independent PyTorch operations without calling Triton. Supports general valid `H` and non-contiguous strides. Its mathematical behavior can also be checked separately on CPU.
- Concatenates only the valid bank rows and prefix row. Computes mean square, scores, softmax, and weighted mixing in float32, then casts to prefix_sum.dtype.
- Reduction order and fused multiply-add operations may cause numerical differences. Bitwise equality is not guaranteed; use dtype-appropriate tolerances.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.hyper_conn import attn_res_mix_fused

prefix_sum = torch.randn(17, 1024, device="npu", dtype=torch.float32)
bank = torch.randn(17, 8, 1024, device="npu", dtype=torch.float32)
combined_weight = torch.randn(1024, device="npu", dtype=torch.float32) * 0.01

output = attn_res_mix_fused(prefix_sum, bank, 3, combined_weight, 1e-6)
reference = attn_res_mix_fused(
    prefix_sum, bank, 3, combined_weight, 1e-6, impl="torch_native"
)
torch.testing.assert_close(output, reference, rtol=1e-4, atol=1e-4)
```
