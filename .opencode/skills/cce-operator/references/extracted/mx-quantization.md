# MX 量化 Cube 参考

遇到带分组 scale 的低精度 Matmul，或需要计算其有效 FLOPs、理论数据量时查阅。本页只提供契约与指标方法，不提供完整可运行的 MXFP8 Kernel。Intrinsic 支持与 scale 物理布局需查目标 SDK 和[公开参考实现索引](../resource-index.md)。

## 先明确契约

分别记录 A/B 的 FP8 格式、scale 的编码和分组轴、每组元素数、scale 的 padding、B 的转置或物理布局、累加与输出精度。scale 的物理 shape 不能只由逻辑矩阵 shape 推断。正确性参考先按相同分组规则恢复数值，再按约定精度 Matmul。

## 指标片段

对于 A[M,K]、转置存放的 B[N,K]，若两者均为每元素 1 字节、每 32 个 K 元素有一个 1 字节 scale，且 scale 的 K 维数量补齐到偶数、输出 FP32，可用：

```python
def mx_minimum_bytes(m, n, k):
    scale_k = ((k + 31) // 32 + 1) // 2 * 2
    return m*k + n*k + m*scale_k + n*scale_k + 4*m*n
```

这是该明确布局下的理论最少字节数，包含 scale 和输出，不包含缓存重复读取、转换、workspace 及 padding 的额外流量。有效 Matmul FLOPs 为 `2*M*N*K`。FP8 与 FP16 的 MFU 分母各用目标设备对应精度的峰值，不能共用一个未注明精度的峰值。性能计时见[通用模板](benchmark-template.md)。
