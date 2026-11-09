# expand_qsa_blocks

## 硬件支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Yes | No |
| `torch_native` | Yes | No |

## 签名

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.indexer import expand_qsa_blocks

def expand_qsa_blocks(blocks, positions, lengths, ratio, topk, *, impl="auto"):
    ...
```

## 说明

将 int32/int64 blocks[R,K] 展开为 token 索引，K=512/2048，topk=K*ratio。positions/lengths 为每行的查询位置和序列长度。输出连续 int32[R,topk+ratio-1]，依次为有效块、局部尾部、-1 补齐。块列表必须为有效前缀接 -1 后缀。

所有张量位于同一 NPU，输入不修改；零行返回空输出。`auto` 与 `triton` 指向同一入口，保留实现中的原生 PyTorch 子步骤，不捕获异常切换实现。`torch_native` 为独立 reference：短卷积使用原生 NPU 卷积，其余使用 CPU 计算（浮点 oracle 使用 FP64）后返回输入设备。reference 会同步，仅用于图外精度验证。

## 参数与返回值

| 参数 | 类型 | 形状 / 类型 | 默认值 | 约束 |
| --- | --- | --- | --- | --- |
| `blocks` | Tensor | [R,K], int32/int64 | required | NPU，接受带步长视图；K=512/2048；有效前缀之后为 -1。 |
| `positions / lengths` | Tensor | [R], int32/int64 | required | 同设备；查询位置和序列长度；接受带步长视图。 |
| `ratio` | int | — | required | 至少为 2。 |
| `topk` | int | — | required | K*ratio。 |
| `impl` | str | — | "auto" | 仅关键字参数：auto、triton、torch_native。 |

| 返回值 | 类型 | 形状 | 类型 | 设备 / 布局 |
| --- | --- | --- | --- | --- |
| `output` | Tensor | [R,topk+ratio-1] | int32 | 输入设备；新分配的连续张量。 |

## triton

默认推理实现。启动前检查不支持的元数据；输入内容由调用方保证符合上述约束。支持 graph capture 和原地更新输入、权重后的 replay。实现中已有的原生子步骤保留在此分支。

## torch_native

针对上述输入契约的独立精度 reference。允许同步主机传输，应在图外运行；不会在默认实现失败后自动切换到此分支。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import expand_qsa_blocks

blocks = torch.full((1, 512), -1, device="npu", dtype=torch.int32)
position = torch.zeros(1, device="npu", dtype=torch.int32)
y = expand_qsa_blocks(blocks, position, position + 1, 4, 2048, impl="auto")
```
