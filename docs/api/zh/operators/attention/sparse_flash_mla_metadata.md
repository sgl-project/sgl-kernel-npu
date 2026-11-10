# sparse_flash_mla_metadata

下文描述 CANN 9.2 原生接口；公共 API 同时适配 CANN 9.1。
保留的主仓 CANN 9.2 绑定返回 2048 个 int32 元素；CANN 9.1 仍为 1024 个。分配图缓冲区前应调用 `get_sparse_flash_mla_metadata_size()`。

上游原始文档：[Sparse Flash MLA Torch API](https://gitcode.com/cann/ops-transformer/blob/e7da18105f08b06e01ef0fff78f439098d483746/attention/sparse_flash_mla/docs/torchapi_sparse_flash_mla.md)。


## CANN 9.1 兼容

`cann_ops_transformers_cann920` 是随包 CANN 9.2 实现的注册名。
CANN 9.1 的窗口输入走 `ops_transformers_cann910`；显式原始 KV 稀疏输入
使用 `ascendc` 计算；metadata 的 `impl="auto"` 在内部选择 CPU 调度路径，
该适配 API 不再单独注册 `host` 实现名。显式稀疏输入不能同时提供压缩 KV。
CPU metadata 需要已有的 CPU int32 `cu_seqlens_q_cpu`、
`seqused_ori_kv_cpu` 镜像。生产绑定均随 wheel 打包，无需独立算子包。
A5 使用单独的 mixed-quant sparse API。

迁移测试中的 `torch_native` metadata 返回独立参考描述符，不生成原生 tiling
字节；配对测试分别构造参考描述符。主仓计算参考也接受不透明的原生 metadata，
并覆盖 SWA/CSA/HCA 与 BSND。原生 metadata 的字节布局取决于 SDK。

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` | 支持 | 不支持 |
| `auto`（按 SDK 和输入选择） | 支持 | 不支持 |
| `ops_transformers_cann910`（CANN 9.1 窗口输入） | 支持 | 不支持 |
| `torch_native`（参考描述符） | 支持 | 支持 |

## 函数原型

在 arch22 上，`auto` 根据当前 SDK 选择 CANN 9.1 或 9.2 实现。下列完整参数说明对应 CANN 9.2；未传参数使用所选绑定的默认值。

```python
from sgl_kernel_npu.operators.attention import sparse_flash_mla_metadata

def sparse_flash_mla_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_ori_kv=None,
    max_seqlen_cmp_kv=None,
    ori_topk=None,
    cmp_topk=None,
    cmp_ratio=None,
    ori_mask_mode=None,
    cmp_mask_mode=None,
    ori_win_left=None,
    ori_win_right=None,
    layout_q=None,
    layout_kv=None,
    has_ori_kv=None,
    has_cmp_kv=None,
    impl="auto",
):
    ...
```

## 功能说明

为 [sparse_flash_mla](sparse_flash_mla.md) 生成 AI Core 任务切分所需的 metadata。典型调用顺序为：准备 `q`、`ori_kv`、可选的 `cmp_kv`、序列长度、稀疏索引和 block table；调用本接口得到 `metadata`；再把 `metadata` 传入主算子。

两次调用中参与任务切分的参数必须成对传入且取值完全一致，否则行为未定义。特别是本接口的 `cmp_ratio`、`ori_mask_mode`、`cmp_mask_mode`、`ori_win_left`、`ori_win_right` 默认为 `None`，而主算子对应默认值为 `1`、`0`、`0`、`-1`、`-1`，因此应显式传入这几个参数，不要依赖默认值。

返回一个不透明的 `int32[2048]` 张量，内容由实现决定，调用方不应解析。

## 参数与返回值

| 参数 | 类型 / Shape / Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- |
| `num_heads_q` | `int` | 必填 | query 头数；Atlas A2/A3 取 1、2、4、8、16、32、64、128。 |
| `num_heads_kv` | `int` | 必填 | KV 头数；仅支持 1。 |
| `head_dim` | `int` | 必填 | 头维度；仅支持 512。 |
| `cu_seqlens_q`、`cu_seqlens_ori_kv`、`cu_seqlens_cmp_kv` | 可选 Tensor `[B+1]`；int32 | `None` | TND 布局的打包边界，首元素为 0、整体非递减。 |
| `seqused_q`、`seqused_ori_kv`、`seqused_cmp_kv` | 可选 Tensor `[B]`；int32 | `None` | 每个请求的有效长度。 |
| `cmp_residual_kv` | Tensor `[B]`；int32 | `None` | 压缩余数，每个元素小于 `cmp_ratio`；`cmp_mask_mode=3` 且 `cmp_ratio!=1` 时必传。 |
| `ori_topk_length` | 可选 Tensor；int32 | `None` | `ori_sparse_indices` 的有效长度。 |
| `cmp_topk_length` | 可选 Tensor；int32；TND 为 `[T,N2]`，BSND 为 `[B,S,N2]` | `None` | `cmp_sparse_indices` 的有效长度。 |
| `batch_size` | `int` | `None` | BSND 场景的批大小。 |
| `max_seqlen_q`、`max_seqlen_ori_kv`、`max_seqlen_cmp_kv` | `int` | `None` | 各输入的最大有效长度。 |
| `ori_topk` | `int` | `None` | `ori_kv` 稀疏 token 数；Atlas A2/A3 仅支持 0。 |
| `cmp_topk` | `int` | `None` | 压缩侧 TopK 长度；CSA 必传且等于 `cmp_sparse_indices` 末维。 |
| `cmp_ratio` | `int` | `None` | 压缩倍率；SWA 为 1，CSA 为 1/2/4，HCA 为 128。 |
| `ori_mask_mode` | `int` | `None` | `0` 无掩码；`3` 因果；`4` 滑动窗口。 |
| `cmp_mask_mode` | `int` | `None` | `0` 无掩码；`3` 因果。 |
| `ori_win_left`、`ori_win_right` | `int` | `None` | 滑动窗口边界；Atlas A3 要求 127 和 0。 |
| `layout_q` | `str` | `None` | `BSND` 或 `TND`。 |
| `layout_kv` | `str` | `None` | `BSND`、`TND` 或 `PA_BBND`。 |
| `has_ori_kv`、`has_cmp_kv` | `bool` | `None` | 与主算子实际传入 `ori_kv`、`cmp_kv` 的状态一致。 |
| `impl` | `str`，仅关键字传入 | `"auto"` | `auto`、`ops_transformers_cann910`、`cann_ops_transformers_cann920`或 `torch_native`；CANN 9.1 显式稀疏输入须使用 `auto`。 |

返回 `metadata`：Tensor，`int32`，shape 为 `[2048]`，与输入同设备。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import sparse_flash_mla_metadata

boundaries = torch.tensor([0, 16], dtype=torch.int32, device="npu")
seqused_ori = torch.tensor([128], dtype=torch.int32, device="npu")

metadata = sparse_flash_mla_metadata(
    8, 1, 512,
    cu_seqlens_q=boundaries,
    seqused_ori_kv=seqused_ori,
    max_seqlen_q=16,
    max_seqlen_ori_kv=128,
    ori_topk=0,
    cmp_topk=0,
    cmp_ratio=1,
    ori_mask_mode=4,
    cmp_mask_mode=0,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="TND",
    layout_kv="PA_BBND",
    has_ori_kv=True,
    has_cmp_kv=False,
)
print(metadata.shape, metadata.dtype)  # [2048], int32
```
