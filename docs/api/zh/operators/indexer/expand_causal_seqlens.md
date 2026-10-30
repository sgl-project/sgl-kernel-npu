# expand_causal_seqlens

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 未验证 |
| `torch_native` | 支持 | 未验证 |

支持范围为已验收的 A3 用例；架构注册不代表所有芯片、shape 或 dtype 均已验证。

## 函数原型

```python
import torch
from sgl_kernel_npu.operators.indexer import expand_causal_seqlens

def expand_causal_seqlens(
    cache_seqlens: torch.Tensor,
    query_lens: torch.Tensor,
    device: torch.device,
    total_q: int,
    max_query_len: int,
    *,
    seqlens_out: torch.Tensor | None = None,
    token_to_batch_out: torch.Tensor | None = None,
    impl="auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

`auto` 与 `triton` 指向同一实现；`torch_native` 是独立 Torch 实现。后端延迟导入，不执行自动回退。导入公共 registry 仍需要可用的 NPU 架构检测环境。

## 功能说明

将每请求的 KV 总长度与 query 长度展开成 packed query 的逐 token 因果长度，同时生成 token→请求编号。令请求 `b` 的总长度为 `S_b`，query 长度为 `Q_b`，输出起点为 `P_b = sum(Q[:b])`：

```text
seqlens[P_b + j] = max(S_b - Q_b + 1, 0) + j
token_to_batch[P_b + j] = b                 (0 <= j < Q_b)
```

`cache_seqlens` 已包含本次 query；起点不足 0 时先截到 0，再按 token 递增。`Q_b=0` 不输出该请求的 token。两个输入只读。

## 参数与返回值

| 参数 | 类型 / Shape / Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- |
| `cache_seqlens` | Tensor `[B]`，int32 | 必填 | 每请求包含 query 的 KV 总长度；一维连续。 |
| `query_lens` | Tensor `[B]`，int32 | 必填 | 每请求非负 query 数；一维连续，与 cache 同 batch。 |
| `device` | `torch.device` | 必填 | 必须与输入设备完全一致；推荐传 `cache_seqlens.device`。 |
| `total_q` | `int` | 必填 | 非负，必须等于 `sum(query_lens)`；由调用方提供 host 值。 |
| `max_query_len` | `int` | 必填 | 非负，必须覆盖所有 `query_lens`；`total_q>0` 时须大于 0，可大于实际最大值。 |
| `seqlens_out` | Tensor `[total_q]`，int32 | `None` | 可选连续输出缓冲；传入时原地覆盖并返回同一对象。 |
| `token_to_batch_out` | Tensor `[total_q]`，int64 | `None` | 可选连续输出缓冲；传入时原地覆盖并返回同一对象。 |
| `impl` | `str` | `"auto"` | 仅限关键字；`auto`、`triton`、`torch_native`。 |

| 返回值 | Shape / Dtype | 含义 |
| --- | --- | --- |
| `seqlens` | `[total_q]`，int32 | 每个 packed query token 可见的因果长度。 |
| `token_to_batch` | `[total_q]`，int64 | 每个 packed token 所属的请求编号。 |

所有 tensor 必须位于同一 NPU；返回值是两个 `torch.Tensor` 组成的 tuple。两个输出可分别省略；省略的缓冲由后端分配。输出不得与输入或彼此重叠。调用方保证长度非负、和及索引计算不溢出 int32；wrapper 不读取设备长度来核验 `total_q` 或最大值上界。

## triton 后端

Triton 位于 `implementation/triton/indexer/expand_causal_seqlens.py`，每个请求按 query tile 写输出；通过对前序请求长度求和计算 packed 起点。query tile 是 `max(32, min(256, next_power_of_2(max_query_len)))`，batch 归约宽度为 `next_power_of_2(B)`。`B=0` 或 `total_q=0` 时分配/返回输出，不启动 Triton kernel。

## torch_native 后端

Torch reference 独立位于 `implementation/torch_native/indexer/expand_causal_seqlens.py`，使用 `repeat_interleave`、`cumsum` 和 `index_select`，不调用 Triton；`max_query_len` 仅用于 Triton launch，native 不使用它参与计算。

SGLang 的 metadata 初始化分配输出；重放传回原 `seqlens_out` / `token_to_batch_out`，保持消费者引用的地址。`total_q` 必须符合已有输出 shape；设备和 dtype 不变。算子不负责管理 graph，也不增加 host/device 长度镜像。

## 调用示例

需已安装提供此公共 API 的 sgl-kernel-npu、torch_npu 和 Triton-Ascend。下面先分配输出，再复用相同缓冲处理另一组 query 长度。

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import expand_causal_seqlens

cache = torch.tensor([10, 8, 20], dtype=torch.int32, device="npu:0")
queries = torch.tensor([2, 0, 3], dtype=torch.int32, device=cache.device)
seqlens, requests = expand_causal_seqlens(cache, queries, cache.device, 5, 3, impl="triton")
# seqlens: [9, 10, 18, 19, 20]; requests: [0, 0, 2, 2, 2]

queries = torch.tensor([0, 3, 2], dtype=torch.int32, device=cache.device)
replayed = expand_causal_seqlens(
    cache, queries, cache.device, 5, 3,
    seqlens_out=seqlens, token_to_batch_out=requests, impl="triton",
)
assert replayed[0] is seqlens and replayed[1] is requests
# seqlens: [6, 7, 8, 19, 20]; requests: [1, 1, 1, 2, 2]
```
