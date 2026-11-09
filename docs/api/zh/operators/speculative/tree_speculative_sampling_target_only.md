# tree_speculative_sampling_target_only

## 芯片支持

| 实现 | arch 2201 | arch 3510 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

默认选项 `auto` 与 `triton` 使用同一个入口。

```python
from sgl_kernel_npu.operators.speculative import tree_speculative_sampling_target_only

def tree_speculative_sampling_target_only(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single=1.0,
    threshold_acc=1.0,
    deterministic=True,
    *,
    impl="auto",
):
    ...
```

## 功能说明

根据目标概率在候选树中选择一条接受路径，然后补采一个最终 token。从根节点
0 出发，按兄弟链接顺序尝试子节点，将目标概率累加到 `prob_acc`。候选概率为
`p` 时，满足以下任一条件即接受。

- `coin <= prob_acc / max(float(threshold_acc), 1e-9)`。
- `p >= threshold_single`。

接受后记录节点槽号，使用该节点的概率行和随机数继续遍历。拒绝后将概率写入
`draft_probs`，尝试下一个兄弟节点。接受路径包含根和最多 `S-1` 个草稿节点。

在最终概率行 `r`，使用 `max(target_probs[b,r] - draft_probs[b,r], 0)` 补采。
选择累计权重严格大于 `uniform_samples_for_final_sampling[b] * sum(weights)`
的第一个 token。

## 参数与返回值

`B` 为请求数，`D` 为含根节点的节点数，`S` 为接受路径容量，`V` 为词表大小。
张量须连续并位于同一 NPU，要求 `D >= 1`、`1 <= S <= D`、词表非空。
调用方提供无环树、有效的 token ID 和输出槽、概率和随机数。

| 参数 | shape / dtype | 含义 |
| --- | --- | --- |
| `predicts` | 一维，通常 `[B*D]`，int32 | 原地输出接受的草稿及最终 token，写入 `retrive_index` 指定的全局槽。 |
| `accept_index` | `[B,S]`，int32 | 原地输出根槽和随后接受的节点槽，有效前缀长度为 `accept_token_num[b]+1`。 |
| `accept_token_num` | `[B]`，int32 | 原地输出接受的草稿数，范围为 `[0,S-1]`，最终补采 token 另计。 |
| `candidates` | `[B,D]`，int64 | 根和候选节点的词号，遍历时读取子节点的词号。 |
| `retrive_index` | `[B,D]`，int64 | 局部节点对应的全局输出槽，须落在 `predicts` 范围内且可用 int32 表示。 |
| `retrive_next_token` | `[B,D]`，int64 | 第一个子节点的局部编号，`-1` 表示没有子节点。 |
| `retrive_next_sibling` | `[B,D]`，int64 | 下一个兄弟节点的局部编号，`-1` 表示兄弟链结束。 |
| `uniform_samples` | `[B,D]`，float32 | 调用方提供的 `[0,1)` 随机数，按当前接受节点的局部编号读取。 |
| `uniform_samples_for_final_sampling` | `[B]`，float32 | 调用方提供的 `[0,1)` 最终采样随机数。 |
| `target_probs` | `[B,D,V]`，float32 | 只读的目标概率，每个节点提供下一步验证或补采所需的概率行。 |
| `draft_probs` | `[B,D,V]`，float32 | 第四个可变输出，即拒绝概率 scratch。每次调用先清零，再记录被拒绝兄弟的目标概率。 |
| `threshold_single` | float，默认 `1.0` | 单词概率达到该阈值时直接接受，取值范围 `[0,1]`。 |
| `threshold_acc` | float，默认 `1.0` | 累计概率接受阈值，取值范围 `[0,1]`，计算时使用 `max(float(threshold_acc),1e-9)`。 |
| `deterministic` | bool，默认 `True` | 保留参数，当前实现忽略其值，所有随机数均由调用方提供。 |
| `impl` | str，默认 `"auto"` | 可选 `"auto"`、`"triton"`、`"torch_native"`。 |

返回 `None`。调用方分配并初始化三只整数输出缓冲，未写位置保留初值。
非空调用每次清零 `draft_probs`，可用 `empty_like(target_probs)` 分配。
四只可变缓冲的写入范围须互不重叠，并与输入分离。Triton 会拒绝
`draft_probs` 与目标张量的 `data_ptr()` 相等的情况。
`B == 0` 时，Triton 在维数检查后直接返回，native 清零空 scratch 后返回。

## triton 后端

- 清零 scratch 后启动三个 kernel，依次执行树遍历接受、按 2048 个 token
  分块求残差权重和、最终采样。wrapper 检查形状、dtype、设备和连续性。
- 到达累计分布末尾时，取所选词块内最后一个正权重位置，该块无正权重时取 `V-1`。

## torch_native 后端

- 复用仓库已有测试的 `target_only_tree_reference`，适配公开签名，以
  `draft_probs` 作为拒绝概率 scratch。
- 用 Python 标量遍历树。最终补采使用 FP64 累计残差权重，并取累计结果的最后一项
  作为总权重，减少采样边界附近的舍入差异。末尾回退在整行查找。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.speculative import tree_speculative_sampling_target_only

# B=2、D=S=4、V=32，线性候选树。
candidates = torch.tensor([[0, 3, 4, 5], [0, 7, 8, 9]], device="npu")
slots = torch.arange(8, device="npu").reshape(2, 4)
children = torch.tensor([[1, 2, 3, -1], [1, 2, 3, -1]], device="npu")
siblings = torch.full_like(children, -1)
target = torch.full((2, 4, 32), 1.0 / 32, device="npu")
for b in range(2):
    for j in range(1, 4):
        target[b, j - 1] *= 0.2
        target[b, j - 1, candidates[b, j]] += 0.8
        target[b, j - 1] /= target[b, j - 1].sum()
coins = torch.tensor([[0.1, 0.9, 0.2, 0.0], [0.7, 0.2, 0.95, 0.0]], device="npu")
final_coin = torch.tensor([0.25, 0.75], device="npu")

for impl in ("triton", "torch_native"):
    predicts = torch.full((8,), -1, dtype=torch.int32, device="npu")
    accept_index = torch.full((2, 4), -1, dtype=torch.int32, device="npu")
    accept_token_num = torch.zeros(2, dtype=torch.int32, device="npu")
    scratch = torch.empty_like(target)
    result = tree_speculative_sampling_target_only(
        predicts, accept_index, accept_token_num,
        candidates, slots, children, siblings, coins, final_coin, target, scratch,
        impl=impl,
    )
    assert result is None
    print(impl, predicts, accept_index, accept_token_num)
```

## 测试

[算子测试](../../../../../tests/python/operators/speculative/test_tree_speculative_sampling_target_only.py)
覆盖多分支树、线性链、阈值及模型形状。两个后端各用独立输出与 scratch，全部零容差比较。
精度与性能执行参数见[测试框架说明](../../../../../tests/python/README.md)。
