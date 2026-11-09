# chain_speculative_sampling

## 芯片支持

| 实现 | arch 2201 | arch 3510 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

默认选项 `auto` 与 `triton` 使用同一个入口。

```python
from sgl_kernel_npu.operators.speculative import chain_speculative_sampling

def chain_speculative_sampling(
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

在非贪心推测解码中验证一条线性草稿链。从 `j=1` 起，对候选
`x = candidates[b,j]` 检查接受条件
`uniform_samples[b,j-1] * draft_probs[b,j-1,x] < target_probs[b,j-1,x]`，
第一次拒绝即停止。

接受 `a` 个草稿后，使用第 `a` 行补采一个最终 token。全部接受时采样权重为
`target_probs[b,a]`，发生拒绝时为
`max(target_probs[b,a] - draft_probs[b,a], 0)`。选择累计权重严格大于
`uniform_samples_for_final_sampling[b] * sum(weights)` 的第一个 token。

## 参数与返回值

`B` 为请求数，`D` 为包含根节点的候选宽度，`R` 为草稿概率行数，`V` 为词表
大小。张量须连续并位于同一 NPU，要求 `D >= 1`、`R >= max(D-1,1)`、词表非空。
调用方提供有效的 token ID、全局输出槽、概率和随机数。

| 参数 | shape / dtype | 含义 |
| --- | --- | --- |
| `predicts` | 一维，通常 `[B*D]`，int32 | 输出：在 `retrive_index` 指定的槽位写入接受的草稿和最终 token。 |
| `accept_index` | `[B,D]`，int32 | 输出：根槽位及接受槽位；有效前缀长度为 `accept_token_num[b] + 1`。 |
| `accept_token_num` | `[B]`，int32 | 输出：接受的草稿数，不含最终补采 token；SGLang 加 1 得到本轮推进长度。 |
| `candidates` | `[B,D]`，int64 | 根及草稿 token ID；根 token 本身不参与验证或采样。 |
| `retrive_index` | `[B,D]`，int64 | 有效且互不冲突的全局输出槽位，须落在 `predicts` 范围内且能用 int32 输出索引表示。 |
| `retrive_next_token`、`retrive_next_sibling` | 忽略 | 兼容参数，不启用树遍历。 |
| `uniform_samples` | `[B,D]`，float32 | 调用方提供的 `[0,1)` 接受随机数；最后一列不使用。 |
| `uniform_samples_for_final_sampling` | `[B]`，float32 | 调用方提供的 `[0,1)` 最终采样随机数。 |
| `target_probs` | `[B,D,V]`，float32 | 调用方完成采样变换后的目标概率。 |
| `draft_probs` | `[B,R,V]`，float32 | 同一词表上的草稿概率，允许额外行。 |
| `threshold_single`、`threshold_acc` | float，默认 `1.0` | 忽略的兼容参数，不改变接受条件。 |
| `deterministic` | bool，默认 `True` | 忽略的兼容参数；算子内部不生成随机数。 |
| `impl` | str，默认 `"auto"` | 可选 `"auto"`、`"triton"`、`"torch_native"`。 |

返回 `None`。调用方分配并初始化三只输出缓冲，未写位置保留初值。各输出的
写入范围须互不重叠，并与输入分离。`B == 0` 时，Triton 检查候选和目标张量
的维数后直接返回。

## triton 后端

- 使用三个 kernel，依次执行前缀接受、按 2048 个 token 分块求残差权重和、
  最终采样。wrapper 检查形状、dtype、设备和连续性。
- 到达累计分布末尾时，取所选词块内最后一个正权重位置，该块无正权重时取 `V-1`。

## torch_native 后端

- 复用仓库已有测试的 `chain_rejection_reference`，适配公开签名和原地输出缓冲。
- 用 Python 标量判断接受条件，用 PyTorch 整行 `sum`/`cumsum` 采样，末尾回退
  在整行查找。接近接受或采样边界时，浮点舍入可能影响 token 选择。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.speculative import chain_speculative_sampling

# B=1、D=2、R=1、V=4，接受判断恰好落在拒绝边界。
candidates = torch.tensor([[0, 1]], dtype=torch.int64, device="npu")
slots = torch.arange(2, dtype=torch.int64, device="npu").reshape(1, 2)
links = torch.full_like(candidates, -1)
target = torch.full((1, 2, 4), 0.25, dtype=torch.float32, device="npu")
draft = torch.tensor([[[0.25, 0.5, 0.25, 0.0]]], device="npu")
coins = torch.tensor([[0.5, 0.125]], device="npu")
final_coin = torch.tensor([0.37], device="npu")

for impl in ("triton", "torch_native"):
    predicts = torch.full((2,), -1, dtype=torch.int32, device="npu")
    accept_index = torch.full((1, 2), -1, dtype=torch.int32, device="npu")
    accept_token_num = torch.zeros(1, dtype=torch.int32, device="npu")
    result = chain_speculative_sampling(
        predicts, accept_index, accept_token_num,
        candidates, slots, links, links, coins, final_coin, target, draft,
        impl=impl,
    )
    assert result is None
    print(impl, predicts, accept_index, accept_token_num)
```

## 测试

[算子测试](../../../../../tests/python/operators/speculative/test_chain_speculative_sampling.py)
覆盖全接受、提前拒绝、模型形状、旧回归及边界，使用独立输出缓冲并精确比较整数结果。
精度与性能执行参数见[测试框架说明](../../../../../tests/python/README.md)。
