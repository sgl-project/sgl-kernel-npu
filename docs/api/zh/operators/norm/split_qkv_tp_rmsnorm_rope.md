# split_qkv_tp_rmsnorm_rope

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |
| `cpu` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.norm import split_qkv_tp_rmsnorm_rope

q, k, v = split_qkv_tp_rmsnorm_rope(
    input, cos, sin, q_hidden_size, kv_hidden_size, head_dim, eps,
    q_weight, k_weight, rotary_dim, tp_world, tp_group, impl="auto",
)
```

## 参数与返回值

`input` 是按 Q、K、V 顺序拼接的本 rank 稠密输入，形状为 `[B, q_hidden_size + 2 * kv_hidden_size]`。两个宽度都是本地分片宽度。每行 Q、K 各自跨本地完整宽度计算均方值。各 rank 需使用相同的批量大小及相同的分片宽度。`tp_world` 与 `tp_group` 的大小一致。单卡传 `1, None`。

`head_dim` 是 2 的幂，本地宽度按完整 head 划分，Q 宽度是 KV 宽度的整数倍。`q_weight`、`k_weight` 分别对应本地 Q、K 通道。`eps` 加入全局均方值后再取平方根。`cos`、`sin` 每行有 `rotary_dim` 个元素，取前半行对每个 head 前 `rotary_dim` 个通道做 NeoX 旋转。

返回独立的 Q、K、V，形状分别为 `[B, q_hidden_size]`、`[B, kv_hidden_size]`、`[B, kv_hidden_size]`，与输入同设备、同 dtype。V 仅拆分。Q/K 使用 FP32 统计量，加权归一化后先舍入到输入精度，再做 RoPE。

## 实现

`auto` 选择 Triton。`tp_world > 1` 时在两个计算阶段之间对每行 Q/K 的均方值执行 SUM all-reduce，再除以 TP 大小。`torch_native` 在输入设备上计算。`cpu` 在 CPU 计算参考值，多卡时将统计量放回输入设备完成进程组归约，并将结果返回输入设备。
