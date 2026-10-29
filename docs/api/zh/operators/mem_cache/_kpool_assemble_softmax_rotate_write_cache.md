# kpool_assemble_softmax_rotate_write_cache

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 未验证 |
| `torch_native` | 支持 | 未验证 |

## 函数原型

```python
from typing import Optional
import torch
from sgl_kernel_npu.operators.mem_cache import kpool_assemble_softmax_rotate_write_cache

def kpool_assemble_softmax_rotate_write_cache(
    index_k_cache: torch.Tensor,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    n_from_tail: torch.Tensor,
    chunk_src_start: torch.Tensor,
    tail_logical_base: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: Optional[torch.Tensor] = None,
    *,
    num_programs: Optional[int] = None,
    impl="auto",
) -> None:
    ...
```

`auto` 固定选择 `triton`，省略 impl 效果相同；不执行运行时后端回退。

## 功能说明

令 `K=ape.shape[0]`、`T=tail_k.shape[1]`。对 plan 行 p，前 `n_from_tail[p]` 项从 `(tail_logical_base[p]+j)%T` tail 槽读取；后缀从 chunk 的 `chunk_src_start[p]+j-n_from_tail[p]` 读取。归约结果写到 `loc[p]`，`write_mask[p]=False` 时不写，省略 mask 表示全写。支持全 tail 且 chunk 为空的合法输入。仅 cache 修改，tail 与 chunk 只读。

逐通道 softmax pooling 使用 FP32，令 `a[j,d]=score[j,d]+ape[j,d]`、`m[d]=max_j a[j,d]`：

$$z_d=\operatorname{BF16}\left(\frac{\sum_j e^{a_{j,d}-m_d}k_{j,d}}{\sum_j e^{a_{j,d}-m_d}}\right).$$

不执行 Hadamard、FP8 量化或逐 slot scale 输出。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `index_k_cache` | `torch.Tensor` | 连续存储，可展平为 `[cache_rows, 128]` | `torch.bfloat16` | 必填 | 原地更新的 index cache。 |
| `chunk_k` | `torch.Tensor` | `[N, 128]` | `torch.bfloat16` | 必填 | 当前 chunk 的 key。 |
| `chunk_score` | `torch.Tensor` | 与 `chunk_k` 相同 | `torch.float16`、`torch.bfloat16` 或 `torch.float32` | 必填 | 与 `chunk_k` 配对的 score。 |
| `tail_k` | `torch.Tensor` | `[R, T, 128]` | `torch.bfloat16` | 必填 | 请求本地 key 环形缓冲；只读。 |
| `tail_score` | `torch.Tensor` | 与 `tail_k` 相同 | `torch.float16`、`torch.bfloat16` 或 `torch.float32` | 必填 | 与 `tail_k` 配对的 score。 |
| `req_pool_idx` | `torch.Tensor` | `[P]` | `torch.int64` | 必填 | `P` 个 pool 各自对应的请求/环形缓冲行。 |
| `n_from_tail` | `torch.Tensor` | `[P]` | `torch.int32` 或 `torch.int64` | 必填 | 每个 pool 从 tail 读取的前导 slot 数。 |
| `chunk_src_start` | `torch.Tensor` | `[P]` | `torch.int64` | 必填 | 每个 pool 后缀在 chunk 中的起始行。 |
| `tail_logical_base` | `torch.Tensor` | `[P]` | `torch.int32` 或 `torch.int64` | 必填 | 逻辑 tail 起始 slot；物理位置按 `T` 取模。 |
| `ape` | `torch.Tensor` | `[pool_size, 128]` | `torch.float32` | 必填 | pool 位置的加性 score；`pool_size > 0`。 |
| `loc` | `torch.Tensor` | `[P]` | `torch.int64` | 必填 | 待写入的扁平 cache 行号。 |
| `write_mask` | `torch.Tensor` 或 `None` | `[P]` 或 — | `torch.bool` | `None` | 可选的逐 pool 写使能。 |
| `num_programs` | `int` 或 `None` | — | — | `None` | 仅限关键字参数；Triton grid 大小，显式提供时必须为正且会截断到 `P`。默认使用可用 Vector Core 数并截断到 `P`。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；`auto`, `triton`, `torch_native` |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `None` | `NoneType` | — | — | 原地更新范围见功能说明。 |

所有 tensor 位于同一 NPU。上游保证索引值域、容量与读入历史有效，避免原地状态的并发写冲突；wrapper 不完整检查所有值域。要求 K>0、T>0，且 0<=n_from_tail<=K。

## triton 后端

固定通道 128，在线稳定 softmax、grid-stride 循环。cache、tail 和计划字段要求连续；chunk key/score、APE、loc、mask 可内部转连续。显式 num_programs 为正并截断到 P，默认按 Vector Core 数选择；查询不可用时使用实现的上限。

## torch_native 后端

独立 gather、FP32 softmax 与 cache 写回；只读取实际选中的源分支，支持全 tail/空 chunk。num_programs 不参与此后端运算。

公共 registry 导入时检测 NPU 架构，`torch_native` 入口也需要该环境。支持范围为已验收的 A3 用例；架构注册不代表所有芯片、shape 或 dtype 均已验证。

## 调用示例

需已安装 sgl-kernel-npu、torch_npu 和 NPU 运行环境。本例显式选择 `torch_native`。 也可选择 `triton` 或 `auto`；比较后端时需隔离各自的可变状态。

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import kpool_assemble_softmax_rotate_write_cache
cache = torch.zeros(1, 128, device='npu', dtype=torch.bfloat16)
chunk_k = torch.randn(4, 128, device='npu', dtype=torch.bfloat16)
chunk_score = torch.zeros(4, 128, device='npu', dtype=torch.float32)
tail_k = torch.randn(1, 4, 128, device='npu', dtype=torch.bfloat16)
tail_score = torch.zeros(1, 4, 128, device='npu', dtype=torch.float32)
ape = torch.zeros(4, 128, device='npu', dtype=torch.float32)
kpool_assemble_softmax_rotate_write_cache(
    cache,
    chunk_k,
    chunk_score,
    tail_k,
    tail_score,
    req_pool_idx=torch.tensor([0], device='npu', dtype=torch.int64),
    n_from_tail=torch.tensor([2], device='npu', dtype=torch.int32),
    chunk_src_start=torch.tensor([0], device='npu', dtype=torch.int64),
    tail_logical_base=torch.tensor([0], device='npu', dtype=torch.int32),
    ape=ape,
    loc=torch.tensor([0], device='npu', dtype=torch.int64),
    impl='torch_native',
)
```
