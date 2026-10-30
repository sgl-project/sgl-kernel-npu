# kv_quant_sparse_attn_sharedkv

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` / `auto` | 不支持 | 支持；公共 API 重构后待重新验证 |
| `torch_native` | 不支持 | 受限参考实现，待 NPU 验证 |

## 函数原型

默认 `auto` 与 `ascendc` 指向相同原生入口。

```python
from sgl_kernel_npu.operators.attention import kv_quant_sparse_attn_sharedkv

def kv_quant_sparse_attn_sharedkv(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    metadata=None,
    kv_quant_mode=1,
    tile_size=64,
    rope_head_dim=64,
    softmax_scale=1.0,
    cmp_ratio=1,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="PA_ND",
    return_softmax_lse=False,
    impl="auto",
):
    ...
```

## 功能说明

面向 DeepSeek-V4 共享 KV 注意力与 DSpArk 草稿路径，读取分页量化 KV，完成解量化、QK、softmax 与加权求和。所有 query 头共享一个 KV 头；输入稀疏索引由调用方提供，算子不执行 top-k 选择。

$$ z_{h,j} = \mathrm{softmax\_scale}\; q_h^T k_j,\qquad o_h = \frac{\sum_{j\in\mathcal{S}} e^{z_{h,j}} v_j}{e^{s_h}+\sum_{j\in\mathcal{S}} e^{z_{h,j}}} $$

其中 $s_h$ 为 sink logit，$\mathcal{S}$ 由窗口、分页映射或稀疏索引决定，共享 KV 的解码向量同时用作 K 和 V。窗口相对位置为 `kv_length - query_length + query_index`。

每行 640 字节按 `[64 BF16 RoPE (128 字节), 448 FP8 NoPE, 7 E8M0 scale, padding]` 打包。每组 64 个 NoPE 值乘以 `2**(scale_byte-127)` 后舍入为 BF16，再重排为 `[NoPE, RoPE]`；不能传普通的 640 维 FP8 向量代替此结构。输出为 BF16；归约顺序可能引入数值误差。

## 参数与返回值

除明确标为 CPU 的字段外，所有已提供的主算子 tensor 必须连续且与 `q` 位于相同 NPU。`None` 默认值不表示原生实现允许缺少必需字段。

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `q` | Tensor | `[B,S,H,512] / [T,H,512]` | BF16 | `Required` | 连续 NPU query；H=64 或 128；对应 BSND 或 TND。 |
| `ori_kv` | Tensor or None | `[blocks,page_size,1,640]` | FP8 E4M3FN | `None` | 必需的原始 KV 缓存；page_size 是 16 的倍数且不超过 1024。 |
| `cmp_kv` | Tensor or None | `[blocks,page_size,1,640]` | FP8 E4M3FN | `None` | 可选压缩缓存；需要 cmp_block_table 和 cmp_ratio=4 或 128。 |
| `ori_sparse_indices` | Tensor or None | `[T,1,K]` | int32 | `None` | TND 物理槽号；K=512 或 1024，有效前缀以 -1 结束；需要 cmp_kv=None。 |
| `cmp_sparse_indices` | Tensor or None | `[T,1,K]` | int32 | `None` | 可选的压缩缓存索引，需要 cmp_kv。 |
| `ori_block_table` | Tensor or None | `[B,ceil(max_KV/page_size)]` | int32 | `None` | 必需的逻辑页到物理页映射；提供物理稀疏索引时也需传入。 |
| `cmp_block_table` | Tensor or None | `[B,compressed_pages]` | int32 | `None` | 传入 cmp_kv 时必需。 |
| `cu_seqlens_q` | Tensor or None | `[B+1]` | int32 | `None` | 从 0 开始的 query 累计长度；TND 必需。 |
| `cu_seqlens_ori_kv` | Tensor or None | `—` | — | `None` | 当前不支持，保持 None。 |
| `cu_seqlens_cmp_kv` | Tensor or None | `—` | — | `None` | 当前不支持，保持 None。 |
| `seqused_q` | Tensor or None | `[B]` | int32 | `None` | 可选的 query 有效长度。 |
| `seqused_kv` | Tensor or None | `[B]` | int32 | `None` | 必需的原始 KV 有效长度。 |
| `sinks` | Tensor or None | `[H]` | float32 | `None` | 必需的每头 softmax sink logit；对应 value 贡献为零。 |
| `metadata` | Tensor or None | `[1024]` | int32 | `None` | 必需的调度表，由配套 A5 metadata 算子生成。 |
| `kv_quant_mode` | int | `—` | — | `1` | 仅支持 1。 |
| `tile_size` | int | `—` | — | `64` | NoPE 量化分组大小；仅支持 64。 |
| `rope_head_dim` | int | `—` | — | `64` | BF16 RoPE 维度；仅支持 64。 |
| `softmax_scale` | float | `—` | — | `1.0` | QK logit 的缩放系数，调用方通常传 1/sqrt(512)。 |
| `cmp_ratio` | int | `—` | — | `1` | 原始 KV 路径用 1；压缩路径用 4 或 128。 |
| `ori_mask_mode` | int | `—` | — | `4` | 仅支持滑动窗口模式 4。 |
| `cmp_mask_mode` | int | `—` | — | `3` | 仅支持模式 3。 |
| `ori_win_left` | int | `—` | — | `127` | 非负 int32 左窗口范围。 |
| `ori_win_right` | int | `—` | — | `0` | 非负 int32 右窗口范围。 |
| `layout_q` | str | `—` | — | `"BSND"` | BSND 或 TND。 |
| `layout_kv` | str | `—` | — | `"PA_ND"` | 仅支持 PA_ND。 |
| `return_softmax_lse` | bool | `—` | — | `False` | 不支持 True；第二个输出是空占位 tensor。 |
| `impl` | str | `—` | — | `"auto"` | auto、ascendc 或 torch_native；auto 选择 ascendc。 |

| 返回值 | 类型 | Shape | Dtype | 描述 |
| --- | --- | --- | --- | --- |
| `attention_out` | Tensor | 与 q 相同 | BF16 | 新分配输出，不修改输入。 |
| `softmax_lse` | Tensor | `[0]` | float32 | 同设备空占位。 |

## ascendc 后端

在 Ascend 950 启动原生 AscendC kernel；支持 BSND/TND、普通分页窗口及受约束的稀疏/压缩路径。需安装包含 A5 kernel 的 wheel。metadata 必须与主算子的窗口、长度及压缩参数一致。

主算子可在 NPU Graph 中捕获；host metadata 需在捕获外生成，重放前复制到地址稳定的 NPU 缓冲区。

## torch_native 后端

独立解包 KV 并在 CPU 用 PyTorch 计算，再将结果复制回 query 设备。仅支持 BF16 BSND、未压缩的分页窗口；TND、显式稀疏索引、压缩 KV 和图捕获不受支持。这是精度参考，性能不代表设备 kernel。

## 调用示例

```python
import math

import torch
import torch_npu

from sgl_kernel_npu.operators.attention import (
    kv_quant_sparse_attn_sharedkv,
    kv_quant_sparse_attn_sharedkv_metadata_host,
)

lengths_cpu = torch.tensor([128], dtype=torch.int32, device="cpu")
metadata = kv_quant_sparse_attn_sharedkv_metadata_host(
    num_heads_q=64, num_heads_kv=1, head_dim=512,
    layout_q="BSND", layout_kv="PA_ND",
    seqused_kv=lengths_cpu, batch_size=1,
    max_seqlen_q=1, max_seqlen_kv=128,
    has_cmp_kv=False, impl="host",
)
# Valid packed zero KV: RoPE/NoPE values are zero; E8M0 scale 127 is 1.0.
rows = torch.zeros((1, 128, 1, 640), dtype=torch.uint8, device="cpu")
rows[..., 576:583] = 127
kv = rows.view(torch.float8_e4m3fn).to("npu")
q = torch.randn(1, 1, 64, 512, dtype=torch.bfloat16, device="npu")
output, lse = kv_quant_sparse_attn_sharedkv(
    q, ori_kv=kv,
    ori_block_table=torch.zeros((1, 1), dtype=torch.int32, device="npu"),
    seqused_kv=lengths_cpu.to("npu"),
    sinks=torch.zeros(64, dtype=torch.float32, device="npu"),
    metadata=metadata, softmax_scale=1 / math.sqrt(512), impl="ascendc",
)
assert output.shape == q.shape and lse.numel() == 0
```
