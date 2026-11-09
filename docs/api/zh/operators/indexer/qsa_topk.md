# qsa_topk

## 硬件支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Yes | No |
| `torch_native` | Yes | No |

## 签名

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.indexer import qsa_topk

def qsa_topk(score, lengths, topk, row_starts=None, *, impl="auto"):
    ...
```

## 说明

选择序列内相对块索引。输入 FP32 score[R,M]，连续 int32 lengths[R]，可选 row_starts[R]。topk 只支持 512/2048。输出连续 int32[R,topk]。短行按 0..length-1 排列并以 -1 补齐；长行选择最大的 topk 个值，并列值允许不同合法索引。有效区间内分数必须有限。

所有张量位于同一 NPU，输入不修改；零行返回空输出。`auto` 与 `triton` 指向同一入口，保留实现中的原生 PyTorch 子步骤，不捕获异常切换实现。`torch_native` 为独立 reference：短卷积使用原生 NPU 卷积，其余使用 CPU 计算（浮点 oracle 使用 FP64）后返回输入设备。reference 会同步，仅用于图外精度验证。

## 参数与返回值

| 参数 | 类型 | 形状 / 类型 | 默认值 | 约束 |
| --- | --- | --- | --- | --- |
| `score` | Tensor | [R,M], FP32 | required | NPU；列连续、允许行间距；M <= 2**24；有效分数有限。 |
| `lengths` | Tensor | [R], int32 | required | score 同设备连续张量；有效区间不得越界。 |
| `topk` | int | — | required | 512 或 2048。 |
| `row_starts` | Tensor or None | [R], int32 | None | score 同设备连续张量；None 表示全零。 |
| `impl` | str | — | "auto" | 仅关键字参数：auto、triton、torch_native。 |

| 返回值 | 类型 | 形状 | 类型 | 设备 / 布局 |
| --- | --- | --- | --- | --- |
| `output` | Tensor | [R,topk] | int32 | 输入设备；新分配的连续张量。 |

## triton

默认推理实现。启动前检查不支持的元数据；输入内容由调用方保证符合上述约束。支持 graph capture 和原地更新输入、权重后的 replay。实现中已有的原生子步骤保留在此分支。

## torch_native

针对上述输入契约的独立精度 reference。允许同步主机传输，应在图外运行；不会在默认实现失败后自动切换到此分支。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import qsa_topk

score = torch.randn(1, 1024, device="npu", dtype=torch.float32)
lengths = torch.full((1,), 1024, device="npu", dtype=torch.int32)
y = qsa_topk(score, lengths, 512, impl="auto")
```
