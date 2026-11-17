# flash_mla

该算子是 CANN 9.2.0 版本引入的非正式算子。

上游原始文档：[Flash MLA Torch API](https://gitcode.com/cann/ops-transformer/blob/37c98efe33ee7faf7d17882532dc1305221c727e/attention/flash_mla/docs/torchapi_flash_mla.md)。

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` / `auto` | 不支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

默认 `auto` 在 CANN 9.2.0、arch35 上选择 `cann_ops_transformers_cann920`。

```python
from sgl_kernel_npu.operators.attention import flash_mla

def flash_mla(
    q,
    k_cache,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_q=None,
    seqused_q=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    head_dim_v=512,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="PA_BBND",
    layout_out="BSND",
    return_softmax_lse=False,
    *,
    impl="auto",
):
    ...
```

## 功能说明

使用 Query 和分页共享 KV 缓存计算非量化 MLA 注意力。每个请求通过 `block_table` 读取有效缓存，其中 Key 使用全部 576 个通道，Value 使用前 512 个通道。

对每个 query head，设分页读取后的 Key 为 $K$、Value 为 $V=K[:, :D_v]$、缩放系数为 $\alpha$、加性掩码为 $M$：

$$
A_{ij}=\alpha Q_iK_j^\top+M_{ij},\qquad
Z_i=\sum_j e^{A_{ij}}+\delta,\qquad
O_i=\frac{\sum_j e^{A_{ij}}V_j}{Z_i},\qquad
\mathrm{LSE}_i=\log Z_i.
$$

无 sink 时 $\delta=0$；存在第 $n$ 个 head 的 sink logit $s_n$ 时，$\delta=e^{s_n}$。允许访问的位置掩码为零，屏蔽位置为 $-\infty$。模式 3 是右对齐因果注意力：query 位置 $i$ 可以访问的 key 位置满足 $j\le L_{kv}-L_q+i$。

## 参数与返回值

所有 tensor 使用相同设备；CANN 实现要求 NPU tensor。下文 `B` 为请求数，`T` 为打包后的 query token 总数，`S` 为填充后的 query 长度，`N` 为本地 query 头数，`P` 为物理页数，`L` 为页表宽度，`D_v` 为 Value 维度。CANN 使用 QK 维度 576、Value 维度 512、单个 KV head 和页大小 128。

| 参数 | 类型 / Shape / Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- |
| `q` | Tensor；`[T,N,576]`（TND）、`[B,S,N,576]`（BSND）或 `[B,N,S,576]`（BNSD）；BF16/FP16 | 必填 | 非空 Query，`1 <= N <= 128`。 |
| `k_cache` | Tensor；`[P,128,1,576]`（PA_BBND）、`[P,1,128,576]`（PA_BNBD）或 `[P,1,36,128,16]`（PA_NZ）；与 `q` 同 dtype | 必填 | 共享 Key/Value 缓存，物理布局必须与声明一致。 |
| `block_table` | 可选 Tensor `[B,L]`；int32 | `None` | 分页注意力必需；有效条目为合法物理页编号，每行覆盖该请求的有效 KV 长度。 |
| `cache_seqlens` | 可选 Tensor `[B]`；int32 | `None` | 必需的有效 KV 长度，不超过引用缓存的容量。 |
| `cu_seqlens_q` | 可选 Tensor `[B+1]`；int32 | `None` | TND 必填，其他 Query 布局不传；首项为零、末项为 `T`，相邻差值为物理 Query 长度。 |
| `seqused_q` | 可选 Tensor `[B]`；int32 | `None` | 实际 Query 长度，不超过对应物理分段长度；省略时使用完整分段。 |
| `sinks` | 可选 Tensor `[N]`；float32 | `None` | 每个 head 的 softmax sink logit；CANN 也接受空 tensor 表示关闭。 |
| `attn_mask` | 可选 Tensor `[2048,2048]`；int8 | `None` | 模式 3/4 必填，使用对角线上方为 1 的标准压缩上三角掩码；模式 0 不传。 |
| `metadata` | 可选 Tensor `[M]`；int32 | `None` | [flash_mla_metadata](flash_mla_metadata.md) 生成的调度信息；与 Query 同设备，生成参数需与本次调用匹配。 |
| `head_dim_v` | `int` | `512` | Value/输出维度；CANN 要求为 512。 |
| `softmax_scale` | `float` | `1.0` | QK 分数的缩放系数，不会根据 head 维度自动推断。 |
| `mask_mode` | `int` | `0` | `0`：无掩码；`3`：右对齐因果；`4`：右对齐滑动窗口。 |
| `win_left`、`win_right` | `int` | `-1` | 模式 4 的窗口范围，`-1` 表示无界；其他模式必须为 `-1`。 |
| `max_seqlen_q`、`max_seqlen_kv` | `int` | `-1` | 上游最大长度属性；`-1` 使用上游默认值，否则为非负值。 |
| `layout_q` | `str` | `"BSND"` | `BSND`、`BNSD` 或 `TND`。 |
| `layout_kv` | `str` | `"PA_BBND"` | `PA_BBND`、`PA_BNBD` 或 `PA_NZ`。 |
| `layout_out` | `str` | `"BSND"` | 必须与 `layout_q` 相同，TND 输入额外允许 NTD 输出。 |
| `return_softmax_lse` | `bool` | `False` | 是否返回 FP32 log-sum-exp。 |
| `impl` | `str`，仅关键字传入 | `"auto"` | `auto`、`cann_ops_transformers_cann920` 或 `torch_native`。 |

返回 `(out, lse)`，不修改输入。

| 返回值 | 类型 / Shape / Dtype | 描述 |
| --- | --- | --- |
| `out` | Tensor；按 `layout_out` 为 `[T,N,D_v]`、`[N,T,D_v]`、`[B,S,N,D_v]` 或 `[B,N,S,D_v]`；与 `q` 同 dtype、同设备 | 注意力输出。 |
| `lse` | Tensor；float32，与 `q` 同设备；TND 输入为 `[N,T]`，其他 Query 布局为 `[B,N,S]` | 启用时为 log-sum-exp；关闭时为空 tensor `[0]`。 |

## cann_ops_transformers_cann920 后端

需要 CANN 9.2.0，并安装包含 Flash MLA 算子的 sgl-kernel-npu wheel。首次调用可能编译适配代码，请保留 C++ 编译环境。

支持参数表中的布局、sink logit 和模式 0/3/4。调用前应准备好符合布局要求的缓存，以及有效的页表、序列长度和 metadata。

## torch_native 后端

用于精度对比，在输入设备上使用 FP32 计算，输出转换为 `q.dtype`。支持 FP16/BF16/FP32、TND Query、单头 PA_BBND 缓存及 TND/NTD 输出；Query 分段和 KV 长度必须为正。

支持模式 0 和 3。模式 3 要求标准因果掩码，且每个请求的 Query 长度不超过 KV 长度。`seqused_q` 省略或等于对应 Query 分段的完整长度，`sinks=None`，窗口和最大长度参数均为 `-1`。

该实现只使用输入数据计算，忽略 `metadata`，不适合图捕获。与 CANN 实现可能存在浮点舍入差异。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import flash_mla, flash_mla_metadata

device = "npu"
q = torch.randn((1, 64, 576), device=device, dtype=torch.bfloat16) * 0.1
cache = torch.randn((1, 128, 1, 576), device=device, dtype=torch.bfloat16) * 0.1
table = torch.tensor([[0]], device=device, dtype=torch.int32)
lengths = torch.tensor([128], device=device, dtype=torch.int32)
boundaries = torch.tensor([0, 1], device=device, dtype=torch.int32)

metadata = flash_mla_metadata(
    lengths, 64, 1, cu_seqlens_q=boundaries, layout_q="TND"
)
out, lse = flash_mla(
    q, cache, block_table=table, cache_seqlens=lengths,
    cu_seqlens_q=boundaries, metadata=metadata,
    softmax_scale=576**-0.5,
    layout_q="TND", layout_kv="PA_BBND", layout_out="NTD",
    return_softmax_lse=True, impl="cann_ops_transformers_cann920",
)
print(out.shape, lse.shape)  # [64, 1, 512], [64, 1]
```
