# kpool_write_tail_and_maybe_compress

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 未验证 |
| `torch_native` | 支持 | 未验证 |

## 函数原型

```python
from typing import Optional
import torch
from sgl_kernel_npu.operators.mem_cache import kpool_write_tail_and_maybe_compress

def kpool_write_tail_and_maybe_compress(
    index_k_cache: torch.Tensor,
    key: torch.Tensor,
    score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    ape: torch.Tensor,
    req_pool_indices: torch.Tensor,
    write_start: torch.Tensor,
    tail_logical_start: torch.Tensor,
    write_loc: torch.Tensor,
    out_cache_loc: torch.Tensor,
    num_draft_tokens: int,
    effective_n_per_batch: Optional[torch.Tensor] = None,
    *,
    num_programs: Optional[int] = None,
    impl="auto",
) -> None:
    ...
```

`auto` 固定选择 `triton`，省略 impl 效果相同；不执行运行时后端回退。

## 功能说明

令 `K=ape.shape[0]`、`N=num_draft_tokens`，tail 长度严格为 `T=K+N`。有效 batch 将所有 N 个草稿写到 `(write_start[b]+j)%T`；`out_cache_loc[b*N]==0` 跳过整个 batch。

设 `s=write_start[b]`、`e=N` 或 `clamp(effective_n_per_batch[b],0,N)`：

$$c_b=\lfloor(s+e)/K\rfloor-\lfloor s/K\rfloor.$$

第 p 个完成 pool 从 `tail_logical_start[b]+p*K` 起读取 K 项，写到 `write_loc[b,p]`。**effective 只控制 pool 完成判定，不减少 N 个 tail 写入**。返回 None，修改 cache、tail key 和 tail score。

逐通道 softmax pooling 使用 FP32，令 `a[j,d]=score[j,d]+ape[j,d]`、`m[d]=max_j a[j,d]`：

$$z_d=\operatorname{BF16}\left(\frac{\sum_j e^{a_{j,d}-m_d}k_{j,d}}{\sum_j e^{a_{j,d}-m_d}}\right).$$

不执行 Hadamard、FP8 量化或逐 slot scale 输出。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `index_k_cache` | `torch.Tensor` | 连续存储，可展平为 `[cache_rows, 128]` | `torch.bfloat16` | 必填 | pool 填满时原地更新的 cache。 |
| `key` | `torch.Tensor` | `[B * N, 128]` | `torch.bfloat16` | 必填 | draft key。 |
| `score` | `torch.Tensor` | 与 `key` 相同 | `torch.float16`、`torch.bfloat16` 或 `torch.float32` | 必填 | 与 `key` 配对的 score。 |
| `tail_k` | `torch.Tensor` | `[R, pool_size + N, 128]` | `torch.bfloat16` | 必填 | 原地更新的 tail key 环形缓冲。 |
| `tail_score` | `torch.Tensor` | 与 `tail_k` 相同 | `torch.float16`、`torch.bfloat16` 或 `torch.float32` | 必填 | 原地更新的 tail score 环形缓冲。 |
| `ape` | `torch.Tensor` | `[pool_size, 128]` | `torch.float32` | 必填 | pool 位置的加性 score；`pool_size > 0`。 |
| `req_pool_indices` | `torch.Tensor` | `[>= B]` | `torch.int64` | 必填 | 每个 batch 项对应的请求/环形缓冲行。 |
| `write_start` | `torch.Tensor` | `[>= B]` | `torch.int32` 或 `torch.int64` | 必填 | 每个 batch 开始写入的逻辑 tail 位置。 |
| `tail_logical_start` | `torch.Tensor` | `[>= B]` | `torch.int32` 或 `torch.int64` | 必填 | 候选待填满 pool 的逻辑起点。 |
| `write_loc` | `torch.Tensor` | `[B, ceil(N / pool_size)]` | `torch.int64` | 必填 | 候选 cache 行；只读，必须预先为连续张量。 |
| `out_cache_loc` | `torch.Tensor` | `[>= B * N]` | `torch.int32` 或 `torch.int64` | 必填 | token cache 位置；某 batch 的首项为 0 时跳过整个 batch。只读。 |
| `num_draft_tokens` | `int` | — | — | 必填 | 静态 `N`；必须为正，且整除 `key.shape[0]`。 |
| `effective_n_per_batch` | `torch.Tensor` 或 `None` | `[>= B]` 或 — | `torch.int32` 或 `torch.int64` | `None` | 可选有效 draft 数，只用于判断 pool 是否填满。 |
| `num_programs` | `int` 或 `None` | — | — | `None` | 仅限关键字参数；Triton grid 大小，显式提供时必须为正且会截断到 `B`。默认使用可用 Vector Core 数并截断到 `B`。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；`auto`, `triton`, `torch_native` |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `None` | `NoneType` | — | — | 原地更新范围见功能说明。 |

所有 tensor 位于同一 NPU。上游保证索引值域、容量与读入历史有效，避免原地状态的并发写冲突；wrapper 不完整检查所有值域。

## triton 后端

固定通道 128，grid-stride 循环。cache、tail、write_loc 必须连续，其余只读输入可内部转连续。显式 num_programs 为正并截断到 B，默认按 Vector Core 数选择；查询不可用时使用实现的上限。

## torch_native 后端

独立按 batch 写 tail 并做 FP32 softmax；effective 为 0 仍写所有草稿的 tail。num_programs 不参与此后端运算。

公共 registry 导入时检测 NPU 架构，`torch_native` 入口也需要该环境。支持范围为已验收的 A3 用例；架构注册不代表所有芯片、shape 或 dtype 均已验证。

## 调用示例

需已安装 sgl-kernel-npu、torch_npu 和 NPU 运行环境。本例显式选择 `torch_native`。 也可选择 `triton` 或 `auto`；比较后端时需隔离各自的可变状态。

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import kpool_write_tail_and_maybe_compress
b, n, pool_size, h = (1, 2, 2, 128)
cache = torch.zeros(1, h, device='npu', dtype=torch.bfloat16)
key = torch.randn(b * n, h, device='npu', dtype=torch.bfloat16)
score = torch.zeros(b * n, h, device='npu', dtype=torch.float32)
tail_k = torch.zeros(1, pool_size + n, h, device='npu', dtype=torch.bfloat16)
tail_score = torch.zeros(1, pool_size + n, h, device='npu', dtype=torch.float32)
kpool_write_tail_and_maybe_compress(
    cache,
    key,
    score,
    tail_k,
    tail_score,
    ape=torch.zeros(pool_size, h, device='npu', dtype=torch.float32),
    req_pool_indices=torch.tensor([0], device='npu', dtype=torch.int64),
    write_start=torch.tensor([0], device='npu', dtype=torch.int32),
    tail_logical_start=torch.tensor([0], device='npu', dtype=torch.int32),
    write_loc=torch.tensor([[0]], device='npu', dtype=torch.int64),
    out_cache_loc=torch.ones(b * n, device='npu', dtype=torch.int32),
    num_draft_tokens=n,
    impl='torch_native',
)
```
