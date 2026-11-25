# mhc_pre_sinkhorn_v2

A3 的 `ops_transformer` profile 为 DeepSeek-V4.1 打包这个 mHC 前向算子。
它使用上一子层的 `pre_mix` 混合残差，同时计算当前子层的系数。
V2 的 CANN 算子类型是 `SglMhcPreSinkhorn`，可以和现有
`MhcPreSinkhorn` 在同一进程加载。

```python
from sgl_kernel_npu.operators.hyper_conn import mhc_pre_sinkhorn_v2

y, post, comb, pre = mhc_pre_sinkhorn_v2(
    x, phi, alpha, bias, pre_mix=previous_pre,
)
```

`x` 为 FP16/BF16 [T,N,D]，`phi` 为 FP32 [(2+N)*N,N*D]，`alpha`
为 FP32 [3]，`bias` 为 FP32 [(2+N)*N]，`pre_mix` 为 FP32 [T,N]。
默认参数为 `hc_mult=4`、`num_iters=20`、`hc_eps=1e-6`、
`norm_eps=1e-6`。四项输出的形状依次为 [T,D]、[T,N]、[T,N*N]、[T,N]。
A3 的 `auto` 选择 wheel 内置的 HF32 CANN 实现，`impl="torch_native"`
选择独立 FP32 参考。CANN 前向直接返回四项输出，省去反向辅助张量分配。

[固定版本的上游实现](https://gitcode.com/cann/ops-transformer/blob/53f5c56996be158b52579d9cdd484b54a38dfb64/mhc/mhc_pre_sinkhorn/docs/aclnnMhcPreSinkhornV2.md)
