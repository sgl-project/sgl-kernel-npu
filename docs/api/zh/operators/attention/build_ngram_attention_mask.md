# build_ngram_attention_mask

## 芯片支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 未验证 |
| `torch_native` | 支持 | 未验证 |

## 函数原型

`auto` 选择 `triton`；两个后端均未限制架构注册。所有 Tensor 使用同一 NPU。

```python
from sgl_kernel_npu.operators.attention import build_ngram_attention_mask

def build_ngram_attention_mask(tree_mask, prefix_lens, draft_token_num, padded_kv_len, *, impl="auto"): ...
```

## 功能说明

构造分页 FIA 掩码：True 表示屏蔽。前缀和树祖先可见，兄弟分支及未使用的尾部位置被屏蔽。

## 参数与返回值

| 参数 | 类型、形状及约束 |
| --- | --- |
| `tree_mask` | 连续 bool `[B,D,D]` 或一维；`True` 表示祖先或节点本身。 |
| `prefix_lens` | 连续 int32/int64 `[B]`；可见前缀长度。 |
| `draft_token_num` | int `D`；含根节点的树宽。 |
| `padded_kv_len` | int；补齐后的键宽，非空批次至少为最大前缀长度加 `D`。 |
| `impl` | str：`auto`、`triton` 或 `torch_native`，默认 `auto`。 |

| 返回值 | 类型和含义 |
| --- | --- |
| mask | bool `[B,1,D,padded_kv_len]`，与 `tree_mask` 在同一设备。 |

## triton 后端

Triton 内核实现。

## torch_native 后端

独立 PyTorch 参考实现；已在 Ascend A3（arch22）验证。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import build_ngram_attention_mask

device = "npu"
mask = torch.tensor([[[1, 0], [1, 1]]], dtype=torch.bool, device=device)
lens = torch.tensor([3], dtype=torch.int32, device=device)
out = build_ngram_attention_mask(mask, lens, 2, 5, impl="triton")
```
