# kv_quant_sparse_attn_sharedkv_metadata_host

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `host` / `auto` | 不支持 | 支持；公共 API 重构后待重新验证 |
| `torch_native` | 不支持 | 单请求 decode 参考，待 NPU 验证 |

## 函数原型

默认 `auto` 选择 `host`。

```python
from sgl_kernel_npu.operators.attention import kv_quant_sparse_attn_sharedkv_metadata_host

def kv_quant_sparse_attn_sharedkv_metadata_host(
    num_heads_q,
    num_heads_kv,
    head_dim,
    layout_q,
    layout_kv,
    *,
    cu_seqlens_q=None,
    seqused_kv=None,
    batch_size=0,
    max_seqlen_q=0,
    max_seqlen_kv=0,
    cmp_topk=0,
    cmp_ratio=1,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    has_ori_kv=True,
    has_cmp_kv=True,
    kv_quant_mode=1,
    tile_size=64,
    rope_head_dim=64,
    impl="auto",
):
    ...
```

## 功能说明

为 DeepSeek-V4 量化共享 KV 注意力生成核间任务调度表。CPU 根据 query/KV 长度、窗口和压缩模式估算工作量并分配任务，然后将调度表复制到当前 NPU；此算子不计算注意力。

输出为固定 `int32[1024]`：36 个 cube 槽，每槽 9 个整数；随后 72 个 vector 槽，每槽 8 个整数，其余为保留空间。128 头模式将一个逻辑 cube 任务复制到相邻物理核对。Flash Decode 跨核切分未启用。

## 参数与返回值

所有长度 tensor 必须是 CPU 上连续的 int32；不接受 NPU 长度。输出设备由当前 NPU 决定。参数需与主算子一致。

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `num_heads_q` | int | `—` | — | `Required` | query 头数，64 或 128。 |
| `num_heads_kv` | int | `—` | — | `Required` | 共享 KV 头数，必须为 1。 |
| `head_dim` | int | `—` | — | `Required` | 必须为 512。 |
| `layout_q` | str | `—` | — | `Required` | BSND 或 TND。 |
| `layout_kv` | str | `—` | — | `Required` | 必须为 PA_ND。 |
| `cu_seqlens_q` | Tensor or None | `[B+1]` | CPU int32 | `None` | 连续 CPU query 累计长度，从 0 开始；TND 必需。 |
| `seqused_kv` | Tensor or None | `[B]` | CPU int32 | `None` | 连续的非负 KV 有效长度；缺省时使用 max_seqlen_kv。 |
| `batch_size` | int | `—` | — | `0` | 0 表示由长度向量推断 B，否则使用显式 batch size。 |
| `max_seqlen_q` | int | `—` | — | `0` | BSND 必须为正，TND 表示最大 query 长度。 |
| `max_seqlen_kv` | int | `—` | — | `0` | 最大 KV 长度；未提供 seqused_kv 时必需。 |
| `cmp_topk` | int | `—` | — | `0` | 压缩调度选择数量：0、512 或 1024。 |
| `cmp_ratio` | int | `—` | — | `1` | has_cmp_kv=True 时使用 4 或 128。 |
| `ori_mask_mode` | int | `—` | — | `4` | 原始 KV 滑动窗口模式。 |
| `cmp_mask_mode` | int | `—` | — | `3` | 压缩 KV mask 模式。 |
| `ori_win_left` | int | `—` | — | `127` | 非负 int32 左窗口范围，需与主算子一致。 |
| `ori_win_right` | int | `—` | — | `0` | 非负 int32 右窗口范围，需与主算子一致。 |
| `has_ori_kv` | bool | `—` | — | `True` | 是否存在原始 KV。 |
| `has_cmp_kv` | bool | `—` | — | `True` | 是否存在压缩 KV；仅使用原始 KV 时显式传 False。 |
| `kv_quant_mode` | int | `—` | — | `1` | 仅支持 1。 |
| `tile_size` | int | `—` | — | `64` | 仅支持 64。 |
| `rope_head_dim` | int | `—` | — | `64` | 仅支持 64。 |
| `impl` | str | `—` | — | `"auto"` | auto、host 或 torch_native；auto 选择 host。 |

| 返回值 | 类型 | Shape | Dtype | 描述 |
| --- | --- | --- | --- | --- |
| `metadata` | Tensor | `[1024]` | NPU int32 | 新建调度表，不修改输入。 |

## host 后端

使用 C++ CPU 调度器，支持多请求、不等长请求、prefill 及压缩模式；返回前发起固定格式调度表的 NPU 拷贝。每次调用会分配输出。不能在 NPU Graph 内执行；在重放前生成并复制到稳定缓冲区。NPU task profiler 不包含 CPU 调度时间，测 host 总耗时还需主机计时与同步。

## torch_native 后端

独立构造单请求、单 query 的 decode 调度表。要求 `max_seqlen_q=1`、正 KV 长度、默认窗口 127/0、原始 KV 存在；支持 64/128 头，其中压缩模式仅支持 64 头。TND 需要 CPU `cu_seqlens_q=[0,1]`。其他模式明确抛出 `NotImplementedError`，不会回退到被测 host 实现。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import (
    kv_quant_sparse_attn_sharedkv_metadata_host,
)

kwargs = dict(
    num_heads_q=64, num_heads_kv=1, head_dim=512,
    layout_q="TND", layout_kv="PA_ND",
    cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device="cpu"),
    seqused_kv=torch.tensor([128], dtype=torch.int32, device="cpu"),
    batch_size=1, max_seqlen_q=1, max_seqlen_kv=128,
    has_ori_kv=True, has_cmp_kv=False,
)
metadata = kv_quant_sparse_attn_sharedkv_metadata_host(**kwargs, impl="host")
reference = kv_quant_sparse_attn_sharedkv_metadata_host(**kwargs, impl="torch_native")
torch.testing.assert_close(metadata, reference, rtol=0, atol=0)
# Before graph replay: stable_metadata.copy_(metadata).
# Generate metadata outside graph capture.
```
