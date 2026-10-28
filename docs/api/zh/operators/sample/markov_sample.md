# markov_sample

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 待验证 |
| `torch_native` | 支持 | 待验证 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

默认 `impl="auto"` 与 `triton` 注册到同一入口。注册成功不代表所有芯片均已完成验证。

## 函数原型

```python
from sgl_kernel_npu.operators.sample import markov_sample

def markov_sample(
    *, base_logits, bias, temperatures, key, offset, step, step_dtype,
    vocab_start=0, gather=None, gather_logits=None, logits_out=None,
    has_sampled_rows=None, exp_noise=None, impl="auto"
):
    ...
```

## 功能说明

将 `bias` 加到 `base_logits` 的有效列，将结果舍入到 `step_dtype`，然后为每行返回一个全局词表索引。温度为 0 时选择最大 logit；非零温度时选择 `logit / temperature - log(noise)` 的最大值，其中 noise 为指数分布样本。`bias` 宽度之外的填充列不参与选择。

使用 Philox 噪声时，`key`、`offset`、`step`、行号和全局词表索引共同确定噪声流。也可以通过 `exp_noise` 提供预生成噪声。

## 参数与返回值

张量须位于同一 NPU。公开接口将参数传给所选后端；下表描述当前实现的要求。

| 参数 | 类型 | Shape / dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- |
| `base_logits` | `torch.Tensor` | `[B, P]`，浮点 | 必填 | 补齐后的本地 logits，`B > 0`、`P > 0`，列步长为 1。 |
| `bias` | `torch.Tensor` | `[B, V]`，浮点 | 必填 | 前 `V` 列的偏置，`0 <= V <= P`，列步长为 1。 |
| `temperatures` | `torch.Tensor` | `[B]`，`torch.float32` | 必填 | 每行温度，连续存储；0 表示贪心选择。建议使用非负有限值。 |
| `key` | `int` | — | 必填 | Philox key；提供 `exp_noise` 时忽略。 |
| `offset` | `torch.Tensor` | 单元素，`torch.int64` | 必填 | 输入设备上的 Philox 偏移；提供 `exp_noise` 时仍须传入。 |
| `step` | `int` | — | 必填 | Philox 步数；提供 `exp_noise` 时忽略。 |
| `step_dtype` | `torch.dtype` | — | 必填 | 舍入类型，可选 `torch.float32`、`torch.float16` 或 `torch.bfloat16`。 |
| `vocab_start` | `int` | — | `0` | 本地分片的全局起始索引，须能被 4 整除。 |
| `gather` | callable 或 `None` | — | `None` | 分片选择时交换各 rank 的候选张量，沿 batch 维度堆叠返回；候选布局取决于后端。 |
| `gather_logits` | callable 或 `None` | — | `None` | 沿词表维度汇聚本地 logits；同时提供 `gather` 和 `logits_out` 时必填。 |
| `logits_out` | `torch.Tensor` 或 `None` | 本地为 `[B, P]`，汇聚时为 `[B, V_global]`；`step_dtype` | `None` | 可选输出缓冲区，原地写入舍入后的 logits；列步长为 1。 |
| `has_sampled_rows` | `bool` 或 `None` | — | `None` | 仅当所有行均为贪心模式时可设为 `False`，跳过噪声生成。 |
| `exp_noise` | `torch.Tensor` 或 `None` | 与本地 logits 或汇聚后的 `logits_out` 同形状；`torch.float32` | `None` | 输入设备上的正指数分布样本，列步长为 1；随机数序列由调用方管理。 |
| `impl` | `str` | — | `"auto"` | 可选 `"auto"`、`"triton"` 或 `"torch_native"`。 |

| 返回值 | 类型 | Shape / dtype | 描述 |
| --- | --- | --- | --- |
| `tokens` | `torch.Tensor` | `[B]`，`torch.int64` | 输入设备上新建的全局 token ID；本地有效分片为空时贡献哨兵值 `2147483647`。 |

不保留 `logits_out` 的分布式选择由 `gather` 交换后端的候选缓冲区。保留汇聚 logits 时，`gather_logits` 返回拼接结果；若提供 `exp_noise`，其形状应与 `logits_out` 相同。调用方须确保回调和分片布局与所选后端兼容。

## triton 后端

使用 Triton kernel 进行分块选择和最终归约。`exp_noise=None` 时在 kernel 内生成 Philox 噪声；提供预生成噪声时跳过该过程，调用方负责在图重放等场景下刷新噪声。

## torch_native 后端

使用 Torch 运算和相同的 Philox 计数器协议或外部噪声，可作为 Triton 后端的参考实现。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.sample import markov_sample

base = torch.randn(2, 1024, device="npu", dtype=torch.float32)
bias = torch.randn_like(base)
temperatures = torch.tensor([0.0, 1.0], device="npu", dtype=torch.float32)
offset = torch.zeros(1, device="npu", dtype=torch.int64)
tokens = markov_sample(base_logits=base, bias=bias, temperatures=temperatures,
                       key=1234, offset=offset, step=0, step_dtype=torch.float32)
```
