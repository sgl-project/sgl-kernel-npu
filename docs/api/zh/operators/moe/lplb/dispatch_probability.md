# dispatch_probability

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `triton_cann920` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

`triton_cann920` 要求 CANN 9.2.x；在其他版本上显式选择会在启动前抛出 `RuntimeError`。

## 函数签名

`auto` 在 CANN 9.2.x 上选择 `triton_cann920`，其他版本选择 `triton`。

```python
from sgl_kernel_npu.operators.moe import dispatch_probability


def dispatch_probability(
    topk_ids, log2phy_prob, log2phy_map, random_vals=None, *, impl="auto"
): ...
```

## 功能描述

按照 `lp_post` 得到的副本概率，为每个 token 的每个路由槽位采样一个物理专家：

```text
p = log2phy_prob[topk_ids]                     # (..., max_copies)
p = where(p.sum(-1) > 0, p, (log2phy_map[topk_ids] >= 0))
physical = log2phy_map[topk_ids, multinomial(p)]
```

若某一行概率全为零，则在有效副本（`log2phy_map != -1`）上均匀采样。采样用每个槽位单个均匀随机数的逆 CDF 完成，因此结果是 `random_vals` 与两张表的确定性函数。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `topk_ids` | Tensor | `Required` | int32/int64 逻辑专家 id，形状任意；非连续输入会先物化。 |
| `log2phy_prob` | Tensor | `Required` | 连续 float32 [num_logical,max_copies] 副本概率。 |
| `log2phy_map` | Tensor | `Required` | 连续 int32/int64 [num_logical,max_copies] 物理 id；`-1` 表示未使用的副本。形状与设备须与 `log2phy_prob` 一致。 |
| `random_vals` | Tensor \| None | `None` | 可选的连续 float32 [numel(topk_ids)] `[0, 1)` 均匀随机数；省略时自动生成。 |
| `impl` | str | `'auto'` | auto 在 CANN 9.2.x 上选择 triton_cann920，其他版本选择 triton；也可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| physical | Tensor | 采样得到的物理 id，形状与类型同 `topk_ids`。 |

## triton 后端

- 每个 program 先把整张 `[num_logical, max_copies]` 表载入 UB（转置后以副本轴为归约轴），随后按 64 个 token 一块跨步处理。
- 每个槽位用 `tl.cumsum` 求出 CDF 行，把均匀随机数换算为阈值 `r * total`，统计不大于阈值的 CDF 元素个数，再 gather 出物理 id；用计数代替搜索，避免对副本数的串行循环。
- 逻辑数与副本数都用带掩码的 2 的幂 tile 填充，因此对调用方不施加 2 的幂限制；选中的副本下标会截断到 `max_copies - 1`。
- 填充后的表或 `[max_copies, 64]` gather tile 超过 32768 个元素时，会在启动前以 `ValueError` 拒绝。

## triton_cann920 后端

- 与 `triton` 契约相同，但把 `num_logical` 与 `max_copies` 作为编译期常量，并用 `tl.trans` 转置常驻表，因此表不需要按 2 的幂填充。
- `auto` 在 CANN 9.2.x 上选择此后端；在其他版本上显式选择会在启动前被拒绝。
- 表与 gather tile 在启动前同样按 32768 元素的 UB 预算校验。

## torch_native 后端

- 用表 gather 构建 `topk_probs`，应用零行回退掩码，使用相同的 `random_vals` 做逆 CDF 采样，最后 gather `log2phy_map`。
- 与内核实现相互独立，且在给定 `random_vals` 时确定，可作为精度基准。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.moe import dispatch_probability

log2phy_prob = torch.tensor([[0.4, 0.6], [0.0, 0.0], [1.0, 0.0]], device="npu")
log2phy_map = torch.tensor([[0, 1], [2, -1], [3, -1]], dtype=torch.int32, device="npu")
topk_ids = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="npu")
random_vals = torch.tensor([0.1, 0.5, 0.9], device="npu")

physical = dispatch_probability(
    topk_ids, log2phy_prob, log2phy_map, random_vals, impl="triton"
)
```
