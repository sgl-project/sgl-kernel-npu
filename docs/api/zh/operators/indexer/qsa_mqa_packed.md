# qsa_mqa_packed

## 硬件支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Yes | No |
| `torch_native` | Yes | No |

## 签名

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.indexer import qsa_mqa_packed

def qsa_mqa_packed(q, k, starts, ends, *, impl="auto"):
    ...
```

## 说明

压缩键的 QSA 打分：BF16 Q[R,4,128]、K[M,1,128]，连续 int32 starts/ends[R]。输出 FP32[R,M]，有效区间为 [starts,ends)，区间外为负无穷。分数是各头 ReLU(QK) 的和除以 sqrt(128)。

所有张量位于同一 NPU，输入不修改；零行返回空输出。`auto` 与 `triton` 指向同一入口，保留实现中的原生 PyTorch 子步骤，不捕获异常切换实现。`torch_native` 为独立 reference：短卷积使用原生 NPU 卷积，其余使用 CPU 计算（浮点 oracle 使用 FP64）后返回输入设备。reference 会同步，仅用于图外精度验证。

`score[r,j] = sum_h max(dot(q[r,h], k[j,0]), 0) / sqrt(128)`；累加和输出均为 FP32。

## 参数与返回值

| 参数 | 类型 | 形状 / 类型 | 默认值 | 约束 |
| --- | --- | --- | --- | --- |
| `q` | Tensor | [R,4,128], BF16 | required | 连续 NPU query。 |
| `k` | Tensor | [M,1,128], BF16 | required | 连续 NPU 压缩键。 |
| `starts / ends` | Tensor | [R], int32 | required | q 同设备连续张量；调用方保证 0 <= starts <= ends <= M。 |
| `impl` | str | — | "auto" | 仅关键字参数：auto、triton、torch_native。 |

| 返回值 | 类型 | 形状 | 类型 | 设备 / 布局 |
| --- | --- | --- | --- | --- |
| `output` | Tensor | [R,M] | FP32 | 输入设备；新分配的连续张量。 |

## triton

默认推理实现。启动前检查不支持的元数据；输入内容由调用方保证符合上述约束。支持 graph capture 和原地更新输入、权重后的 replay。实现中已有的原生子步骤保留在此分支。

## torch_native

针对上述输入契约的独立精度 reference。允许同步主机传输，应在图外运行；不会在默认实现失败后自动切换到此分支。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import qsa_mqa_packed

q = torch.randn(1, 4, 128, device="npu", dtype=torch.bfloat16)
k = torch.randn(16, 1, 128, device="npu", dtype=q.dtype)
start = torch.zeros(1, device="npu", dtype=torch.int32)
y = qsa_mqa_packed(q, k, start, start + 16, impl="auto")
```
