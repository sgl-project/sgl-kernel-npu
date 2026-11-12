# build_tree

## 芯片支持

| 实现 | arch 2201 | arch 3510 |
| --- | --- | --- |
| `ascendc` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

默认 `auto` 与 `ascendc` 使用同一入口。

```python
from sgl_kernel_npu.operators.speculative import build_tree

def build_tree(
    parent_list, selected_index, verified_seq_len, tree_mask, positions,
    retrive_index, retrive_next_token, retrive_next_sibling,
    topk, depth, draft_token_num, tree_mask_mode, *, impl="auto",
) -> None:
    ...
```

## 功能说明

根据父候选关系和已选节点构造推测解码验证树，原地写入注意力掩码、绝对位置、
全局节点索引及子节点、兄弟节点链接。节点 0 为根，掩码允许每个节点关注自身及其祖先。

记批量为 `B`，`D = draft_token_num`，`k = topk`，请求 `b` 的已验证前缀长度为 `L_b`。
节点 `i` 的绝对位置为 `L_b + 节点深度`，全局索引为 `b*D+i`；子节点按索引升序链接。

## 参数与返回值

| 参数 | Shape / 类型 | 说明 |
| --- | --- | --- |
| `parent_list` | int64 `[B,k*(depth-1)+1]` | 父候选节点表。 |
| `selected_index` | int64 `[B,D-1]` | 不含根的已选候选索引；索引唯一，且包含所选节点的全部祖先。 |
| `verified_seq_len` | int64 `[B]` | 每条请求的已验证前缀长度。 |
| `tree_mask` | bool 一维 Tensor | 原地写入掩码，容量由 `tree_mask_mode` 决定。 |
| `positions` | int64 `[B*D]` 或连续 `[B,D]` | 原地写入绝对位置。 |
| `retrive_index` | int64 `[B,D]` | 原地写入全局节点索引。 |
| `retrive_next_token` | int64 `[B,D]` | 原地写入首个子节点索引，无子节点为 `-1`。 |
| `retrive_next_sibling` | int64 `[B,D]` | 原地写入下一个兄弟节点索引，无兄弟节点为 `-1`。 |
| `topk` | `int`，必填 | 每步候选数量。 |
| `depth` | `int`，必填 | 候选树深度。 |
| `draft_token_num` | `int`，必填 | 包含根节点的 draft 节点数。 |
| `tree_mask_mode` | `int`，必填 | `0` 为 FULL_MASK；其他模式见下文。 |
| `impl` | `str`，默认 `"auto"` | 仅限关键字，可选 `auto`、`ascendc`、`torch_native`。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| `None` | `NoneType` | 原地写入五个输出缓冲，保留其存储；只读输入不修改。 |


所有 Tensor 应位于同一设备，使用容量充足的连续缓冲区，五个输出之间不能重叠。

## 掩码模式

- `0`（FULL_MASK）：顺次拼接每条请求的 `[D,L_b+D]` 掩码，总长度为
  `D*sum(L_b)+B*D*D`。只写每行最后 `D` 列，已验证前缀列由调用方初始化并保持不变。
- `1`（QLEN_ONLY）：掩码为展平的 `[B,D,D]`。当前 `ascendc` 后端保留旧行地址问题，
  可能覆盖错误行或越界；请使用 FULL_MASK。`torch_native` 按正确逐节点行地址计算。
- `2`（QLEN_ONLY_BITPACKING）：未实现，调用时报错。

## ascendc 后端

- 要求所有 Tensor 位于同一 NPU，按连续物理存储读取，不支持一般 stride 语义。
- 原地写入调用方提供的五个输出，不创建连续副本或额外回写。

## torch_native 后端

- 使用独立 PyTorch 运算构造树及掩码。
- 保持相同的输出形状、dtype 和原地写入接口。

## 调用示例

沿用现有 FULL_MASK 配置，两后端使用独立的输出缓冲区。

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.speculative import build_tree

inputs = dict(
    parent_list=torch.tensor([
        [-1, 0, 1, 2, 3, 4, 8, 9, 10, 20, 24, 21, 28],
        [-1, 0, 1, 2, 3, 4, 5, 6, 7, 24, 28, 20, 21],
    ], device="npu", dtype=torch.int64),
    selected_index=torch.tensor([
        [0, 1, 4, 8, 20, 24, 36], [0, 4, 5, 6, 7, 24, 36],
    ], device="npu", dtype=torch.int64),
    verified_seq_len=torch.tensor([5, 10], device="npu", dtype=torch.int64),
    topk=4, depth=4, draft_token_num=8, tree_mask_mode=0,
)
outputs = dict(
    tree_mask=torch.ones((5 + 10) * 8 + 2 * 8 * 8, device="npu", dtype=torch.bool),
    positions=torch.full((16,), -1, device="npu", dtype=torch.int64),
    retrive_index=torch.full((2, 8), -1, device="npu", dtype=torch.int64),
    retrive_next_token=torch.full((2, 8), -1, device="npu", dtype=torch.int64),
    retrive_next_sibling=torch.full((2, 8), -1, device="npu", dtype=torch.int64),
)
reference = {name: tensor.clone() for name, tensor in outputs.items()}
result = build_tree(**inputs, **outputs)  # auto: ascendc
build_tree(**inputs, **reference, impl="torch_native")
assert result is None
print(outputs["positions"].reshape(2, 8))
```
