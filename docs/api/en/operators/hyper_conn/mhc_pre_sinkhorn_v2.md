# mhc_pre_sinkhorn_v2

On Ascend A3, the `ops_transformer` profile bundles a dedicated CANN operator
for the DeepSeek-V4.1 mHC forward. It computes the current sublayer's
coefficients and uses the previous sublayer's `pre_mix` to mix the residual.
The V2 bundle uses the private CANN operator type `SglMhcPreSinkhorn` so it
can load alongside the existing `MhcPreSinkhorn` operator.

```python
from sgl_kernel_npu.operators.hyper_conn import mhc_pre_sinkhorn_v2

y, post, comb, pre = mhc_pre_sinkhorn_v2(
    x, phi, alpha, bias, pre_mix=previous_pre,
)
```

`x` is FP16/BF16 [T,N,D]. `phi` is FP32 [(2+N)*N,N*D], `alpha` is
FP32 [3], `bias` is FP32 [(2+N)*N], and `pre_mix` is FP32 [T,N].
The defaults are `hc_mult=4`, `num_iters=20`, `hc_eps=1e-6`, and
`norm_eps=1e-6`. The outputs have shapes [T,D], [T,N], [T,N*N], and [T,N].
`auto` selects the bundled HF32 CANN implementation on A3. `impl="torch_native"`
selects the independent FP32 reference. The CANN forward returns the four
outputs without allocating backward intermediates.

[Pinned upstream implementation](https://gitcode.com/cann/ops-transformer/blob/53f5c56996be158b52579d9cdd484b54a38dfb64/mhc/mhc_pre_sinkhorn/docs/aclnnMhcPreSinkhornV2_en.md)
