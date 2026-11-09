# verify_tree_greedy

## 芯片支持

| 实现 | arch 2201 | arch 3510 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

默认选项 `auto` 与 `triton` 使用同一个入口。

```python
from sgl_kernel_npu.operators.speculative import verify_tree_greedy

def verify_tree_greedy(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    target_predict,
    *,
    impl="auto",
):
    ...
```

## 功能说明

在贪心推测解码中，将草稿前缀与目标模型的下一 token 预测逐项比较。
从 `j=1` 起，满足 `candidates[b,j] == target_predict[b,j-1]` 即接受，
第一次不匹配时停止。

接受 `a` 个草稿后，将根槽和接受槽写入 `accept_index[b,0:a+1]`，并设置
`accept_token_num[b] = a`。标准槽位映射下，写入
`predicts[b*D+j] = target_predict[b,j]`，其中 `0 <= j <= a`，最后一项为
最终目标 token。调用方在调用前通过 argmax 计算 `target_predict`。

## 参数与返回值

`B` 为请求数，`D` 为含根节点的候选宽度。两个后端的共同输入范围为线性链，
要求张量在同一 NPU 上连续存储，`B >= 1`、`D >= 1`、`accept_index[B,D]`，
并采用 `retrive_index[b,j] = b*D+j` 的标准槽位。native 需要有效的链链接。
token 和槽位须可用 int32 输出表示。

| 参数 | shape / dtype | 含义 |
| --- | --- | --- |
| `predicts` | 一维，长度至少 `B*D`，int32 | 调用方分配的原地 token 输出。 |
| `accept_index` | `[B,D]`，int32 | 原地根槽及接受槽索引；未写位置保留初值。 |
| `accept_token_num` | `[B]`，int32 | 接受的草稿数，不含最终补出 token。 |
| `candidates` | `[B,D]`，int32 或 int64 | 含根的候选 token，根列的 token 值不参与比较。 |
| `retrive_index` | `[B,D]`，整数 | 标准映射 `b*D+j`。 |
| `retrive_next_token` | `[B,D]`，整数 | 线性链的首子节点局部编号；Triton 忽略此参数，native 使用。 |
| `retrive_next_sibling` | `[B,D]`，整数 | 线性链中为 `-1`；Triton 忽略此参数，native 使用。 |
| `target_predict` | `[B,D]`，int32 或 int64 | 目标模型为每个节点预测的下一个 token ID。 |
| `impl` | str，默认 `"auto"` | `"auto"`、`"triton"`、`"torch_native"`。 |

返回 `None`。调用方分配并初始化三只输出缓冲，未写位置保留初值。
各输出的写入范围须互不重叠，并与输入分离。

## triton 后端

- 每个请求一个程序，按补齐到 2 的幂的宽度掩码读取，随后顺序比较前缀，
  忽略子节点和兄弟链接。
- 要求线性链，接受索引的行跨度须为 `D`。不支持多分支树和宽度小于 `D` 的
  `accept_index`。wrapper 直接转发参数，不执行校验。

## torch_native 后端

- 复用已有 `speculative.py::verify_tree_greedy_native` 的原地遍历路径，
  将返回值适配为 `None`。
- 使用子节点和兄弟链接，也支持多分支树。后端精度对比采用共同的线性链范围。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.speculative import verify_tree_greedy

# B=2、D=2，复用原 int32 线性链用例。
candidates = torch.tensor([[10375, 28], [223, 15098]], dtype=torch.int32, device="npu")
target = torch.tensor([[28, 334], [15098, 18]], dtype=torch.int32, device="npu")
slots = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="npu")
children = torch.tensor([[1, -1], [1, -1]], dtype=torch.int32, device="npu")
siblings = torch.full_like(children, -1)

for impl in ("triton", "torch_native"):
    predicts = torch.zeros(4, dtype=torch.int32, device="npu")
    accept_index = torch.full((2, 2), -1, dtype=torch.int32, device="npu")
    accept_token_num = torch.zeros(2, dtype=torch.int32, device="npu")
    result = verify_tree_greedy(
        predicts, accept_index, accept_token_num,
        candidates, slots, children, siblings, target, impl=impl,
    )
    assert result is None
    print(impl, predicts, accept_index, accept_token_num)
```

## 测试

[算子测试](../../../../../tests/python/operators/speculative/test_verify_tree_greedy.py)
覆盖全接受、提前不匹配、模型形状、仅根节点、补齐宽度及 int32/int64 输入，
使用独立输出并精确比较整数结果。另有多分支树回归单独验证 `torch_native` 的固定输出。
精度与性能执行参数见[测试框架说明](../../../../../tests/python/README.md)。
