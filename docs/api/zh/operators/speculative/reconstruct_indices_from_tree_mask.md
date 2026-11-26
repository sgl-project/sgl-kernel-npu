# reconstruct_indices_from_tree_mask

## 芯片支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 未验证 |
| `torch_native` | 支持 | 未验证 |

## 函数原型

`auto` 选择 `triton`；两个后端均未限制架构注册。所有 Tensor 使用同一 NPU。

```python
from sgl_kernel_npu.operators.speculative import reconstruct_indices_from_tree_mask

def reconstruct_indices_from_tree_mask(tree_mask, verified_seq_len, positions, retrive_index, retrive_next_token, retrive_next_sibling, batch_size, draft_token_num, *, impl="auto"): ...
```

## 功能说明

根据父节点在前的祖先掩码重建深度位置、全局索引及局部首子节点和下一兄弟节点链接；缺失链接为 -1。

## 参数与返回值

| 参数 | 类型、形状及约束 |
| --- | --- |
| `tree_mask` | 连续 bool `[B,D,D]` 或一维；父节点先于子节点的祖先矩阵。 |
| `verified_seq_len` | 连续 int32/int64 `[B]`；已接受前缀长度。 |
| `positions` | 原地 int64 `[B*D]` 或 `[B,D]`；前缀长度加严格祖先数。 |
| `retrive_index` | 原地 int64 `[B,D]`；全局槽 `b*D+j`。 |
| `retrive_next_token` | 原地 int64 `[B,D]`；局部首子节点槽或 -1。 |
| `retrive_next_sibling` | 原地 int64 `[B,D]`；局部下一个兄弟节点槽或 -1。 |
| `batch_size` | int `B`；请求数。 |
| `draft_token_num` | int `D`；含根节点的候选宽度。 |
| `impl` | str：`auto`、`triton` 或 `torch_native`，默认 `auto`。 |

所有输出缓冲区连续且不能交叠。

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
from sgl_kernel_npu.operators.speculative import reconstruct_indices_from_tree_mask

device = "npu"
mask = torch.tensor([[[1, 0], [1, 1]]], dtype=torch.bool, device=device)
lens = torch.tensor([3], dtype=torch.int32, device=device)
out = [torch.empty((1, 2), dtype=torch.int64, device=device) for _ in range(4)]
reconstruct_indices_from_tree_mask(mask, lens, *out, 1, 2, impl="triton")
```
