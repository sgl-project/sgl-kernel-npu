# verify_ngram_tree_greedy

## 芯片支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 未验证 |
| `torch_native` | 支持 | 未验证 |

## 函数原型

`auto` 选择 `triton`；两个后端均未限制架构注册。所有 Tensor 使用同一 NPU。

```python
from sgl_kernel_npu.operators.speculative import verify_ngram_tree_greedy

def verify_ngram_tree_greedy(predicts, accept_index, accept_token_num, candidates, retrive_index, retrive_next_token, retrive_next_sibling, target_predict, *, impl="auto"): ...
```

## 功能说明

沿匹配的子节点和兄弟节点遍历，直到不匹配、叶节点或接受宽度限制；写入根节点和已接受的全局位置、草稿接受数及最后的目标预测，未写入的输出保持原值。

## 参数与返回值

| 参数 | 类型、形状及约束 |
| --- | --- |
| `predicts` | 原地 int32 `[B*D]`；按全局槽排列的 token ID。 |
| `accept_index` | 原地 int32 `[B,S]`，`S >= 1`；按路径排列的根和接受槽。 |
| `accept_token_num` | 原地 int32 `[B]`；接受草稿数，不含最后一个目标预测。 |
| `candidates` | int32/int64 `[B,D]`；候选 token ID。 |
| `retrive_index` | int32/int64 `[B,D]`；把局部节点映射到合法全局槽。 |
| `retrive_next_token` | int32/int64 `[B,D]`；局部首子节点索引或 -1。 |
| `retrive_next_sibling` | int32/int64 `[B,D]`；局部下一个兄弟节点索引或 -1。 |
| `target_predict` | int32/int64 `[B,D]`；按平铺全局槽寻址的目标下一个 token ID。 |
| `impl` | str：`auto`、`triton` 或 `torch_native`，默认 `auto`。 |

所有张量连续，写入缓冲区不能交叠。子节点和兄弟节点链接必须构成合法树。

| 返回值 | 类型和含义 |
| --- | --- |
| `None` | 原地更新输出缓冲区。 |

## triton 后端

Triton 内核实现。

## torch_native 后端

独立 PyTorch 参考实现；已在 Ascend A3（arch22）验证。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.speculative import verify_ngram_tree_greedy

device = "npu"
x = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
links = torch.tensor([[1, -1]], dtype=torch.int32, device=device)
siblings = torch.full_like(x, -1)
predicts = torch.zeros(2, dtype=torch.int32, device=device)
accepted = torch.full_like(x, -1)
count = torch.zeros(1, dtype=torch.int32, device=device)
verify_ngram_tree_greedy(predicts, accepted, count, x, x, links, siblings, x, impl="triton")
```
