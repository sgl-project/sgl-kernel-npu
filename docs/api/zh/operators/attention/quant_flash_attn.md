# quant_flash_attn

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `cann_ops_transformer`（`auto`） | 不支持 | 支持 |
| `torch_native` | 未注册 | 支持 |

arch35 后端需要 CANN 9.2.0 及以上版本和匹配的 PyTorch/torch_npu。导出 QFA 两个接口的 `cann_ops_transformer` Python 扩展及底层 CANN 算子需与 `sgl_kernel_npu` 分别安装。该后端已在 CANN 9.2.0 上验证。

## 函数原型

```python
from sgl_kernel_npu.operators.attention import quant_flash_attn

def quant_flash_attn(
    q,
    k,
    v,
    q_descale,
    k_descale,
    v_descale,
    quant_mode,
    *,
    block_table=None,
    p_scale=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_q_descale="BSND",
    layout_kv="BSND",
    layout_out="BSND",
    return_softmax_lse=False,
    impl="auto",
):
    ...
```

`auto` 选择 `cann_ops_transformer`，两者指向同一实现，不做自动回退。下述 MXFP8 路径需显式传入 TND/PA_BBND 布局；默认值保持上游接口约定。

## 功能说明

使用 Q 和分页 KV cache 计算量化 GQA attention。对于反量化后的 $Q_d$、$K_d$、$V_d$，attention 的计算目标为：

$$
S = \mathrm{softmax\_scale}\, Q_d K_d^T + M,
\qquad O = \mathrm{softmax}(S)V_d.
$$

MXFP8 模式（`quant_mode=1`）使用 FP8 E4M3 Q/K/V 和 E8M0 反量化 scale。Q/K 沿通道维每 32 个元素共享一个 scale；V 沿序列维每 32 个位置、每个通道共享一个 scale。末尾大小为 2 的维度保存相邻两组 scale。E8M0 字节 $e$ 表示 $2^{e-127}$，字节 255 表示 NaN。ModelSlim 静态 V scale 通过在这些序列分组上重复 per-channel scale 表示。

量化计算还通过 `p_scale` 将 softmax 中间量 P 舍入到 FP8 E4M3。参考实现包含分块 online softmax、带 scale 的指数和 P 的 FP8 转换，并参考 CANN MXFP8 golden 对最大值进行 ln(2) 对齐。仅对 Q/K/V 反量化后调用普通高精度 attention 无法覆盖该舍入过程。浮点累加顺序仍可能与设备实现不同。

causal 模式下，位置 `i` 的 query 可读取 `j <= kv_length - query_length + i` 的 key。该右下对齐方式使 chunked prefill 可以读取历史前缀。所有输入（包括 cache 和 metadata）均不修改。

## 参数与返回值

下表说明独立参考实现覆盖的 MXFP8 子集。其他 CANN 模式和布局原样传给已安装的 CANN 实现，不属于本次参考实现的验证范围。

记 `T` 为 Q 的总存储位置数，`B` 为请求数，`Nq/Nkv` 为本卡 Q/KV head 数，`G=Nq/Nkv`，`D` 为 head dim，`P` 为物理页数，`S` 为页大小，`L` 为每个请求的页表容量。参考实现支持的 head dim 为 64/128/256。Qwen3.6-27B 用例使用 `D=256`、`S=512`，TP1 和 TP2 的本地 Q/KV head 数分别为 24/4 和 12/2。`Nq` 必须可被 `Nkv` 整除。所有 Tensor 位于同一 NPU；长度、Q、Q scale 和页表使用连续 Tensor。分页 cache Tensor 应遵循 CANN 的 stride 约束，包括各维 stride 不为零。

| 参数 | 类型／默认值 | 描述及约束 |
| --- | --- | --- |
| `q` | Tensor；必选 | FP8 E4M3 `[T, Nq, D]`。 |
| `k`、`v` | Tensor；必选 | FP8 E4M3 PA_BBND `[P, S, Nkv, D]`。 |
| `q_descale` | Tensor；必选 | E8M0；TND 为 `[T, Nq, D/64, 2]`，N2TGD 为 `[Nkv, T, G, D/64, 2]`。 |
| `k_descale` | Tensor；必选 | E8M0 `[P, S, Nkv, D/64, 2]`。 |
| `v_descale` | Tensor；必选 | E8M0 `[P, S/64, Nkv, D, 2]`；末维对应相邻的两组 32-token scale。 |
| `quant_mode` | int；必选 | MXFP8 参考实现使用 `1`。CANN 后端原样接收模式值。 |
| `block_table` | Tensor 或 None；`None` | Int32 `[B, L]` 物理页号；分页 KV 必须提供。 |
| `p_scale` | Tensor 或 None；`None` | FP32 `[1]` P 量化 scale；应为有限正值。SGLang 集成使用 1。 |
| `cu_seqlens_q`、`cu_seqlens_kv` | Tensor 或 None；`None` | Int32 `[B+1]` 累积长度，首元素为零。TND Q 必须提供 Q 边界；参考实现可由 KV 边界差推导未提供的 `seqused_kv`。 |
| `seqused_q`、`seqused_kv` | Tensor 或 None；`None` | Int32 `[B]` 有效长度。`seqused_q=None` 时使用 Q 边界差；分页 KV 应提供有效 KV 长度。 |
| `sinks` | Tensor 或 None；`None` | 上游可选输入；MXFP8 参考实现不支持。 |
| `attn_mask` | Tensor 或 None；`None` | CANN causal mask 模板，int8 `[2048, 2048]`，上三角为屏蔽位置。decode 的 mode 0 使用 None。 |
| `metadata` | Tensor 或 None；`None` | 来自 `quant_flash_attn_metadata` 的 int32 `[2, max_schedule_size]`；CANN QFA 路径需要传入。数值参考不读取硬件调度结果。 |
| `softmax_scale` | float；`1.0` | QK 分数乘数；常规 attention 显式传 `D**-0.5`。 |
| `mask_mode` | int；`0` | `0`：无 mask；`3`：右下对齐 causal。 |
| `win_left`、`win_right` | int；`-1`、`-1` | 上游窗口属性；参考实现覆盖默认全窗口。 |
| `max_seqlen_q`、`max_seqlen_kv` | int；`-1`、`-1` | CANN 长度上界；`-1` 表示未指定。需与 metadata 准备一致。 |
| `layout_q`、`layout_out` | str；`"BSND"` | 本参考实现要求两者均为 `"TND"`。 |
| `layout_q_descale` | str；`"BSND"` | 本参考实现使用 `"TND"` 或 `"N2TGD"`。 |
| `layout_kv` | str；`"BSND"` | SGLang 使用 `"PA_BBND"`。使用 `"PA_BNBD"` 时，交换 K、V 和两份 KV scale 的第 1、2 轴（从 0 开始）。 |
| `return_softmax_lse` | bool；`False` | True 返回 LSE；False 时第二个返回值为空 Tensor。 |
| `impl` | str；`"auto"` | 仅限关键字参数，可选 `"auto"`、`"cann_ops_transformer"`、`"torch_native"`。 |

| 返回值 | 类型／dtype | shape 与含义 |
| --- | --- | --- |
| `attn_output` | Tensor／BF16 | `[T, Nq, D]`，新分配在 Q 所在设备。 |
| `softmax_lse` | Tensor／FP32 | 开启时为 `[Nq, T]`，关闭时为 `[0]`。 |

参考实现对 padding Q、零长度 KV 和完全屏蔽的行输出零，开启的 LSE 对应位置为 `+inf`。

## cann_ops_transformer 后端

首次调用时从 `cann_ops_transformer.ops` 导入 `quant_flash_attn`，缓存 callable 后透传参数。校验、输出分配和执行由 CANN 负责。wrapper 不增加 Tensor 复制、cast、host 长度读取或重复算子注册。

metadata 必须使用与主算子一致的量化模式、mask、布局、head 和长度配置生成。两个接口无法分别校验全部跨调用一致性条件。完整上游约束见 [CANN 9.2 API](https://gitcode.com/cann/ops-transformer/blob/v9.2.0/attention/quant_flash_attn/docs/torchapi_quant_flash_attn.md)。

## torch_native 后端

提供 mode 1、TND Q/output、PA_BBND 或 PA_BNBD cache、TND/N2TGD Q scale 和 mask mode 0/3 的独立 eager 数值参考。通过页表还原逻辑 KV 序列、展开 scale，再执行分块 attention。causal 模式按长度推导标准因果关系，不支持任意自定义 mask。参考实现未覆盖的模式会明确拒绝；其范围不会限制 CANN 后端的更大接口。

该后端用于正确性对比，不用于服务。它将输入复制到 CPU 执行 PyTorch 参考计算，再将输出放回 Q 所在设备，以避免依赖 NPU FP8 索引 kernel。默认 CANN 路径不执行这些复制。metadata 和最大长度提示不改变参考计算的数学结果。本次测试不包含动态图重放。

## 调用示例

在具备上述依赖的 arch35 NPU 上运行。示例使用 Qwen3.6-27B TP2 的本地 head 数和 head dim，单个 query 对 512-token cache 执行 decode。E8M0 字节 127 表示 scale 1。

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import (
    quant_flash_attn,
    quant_flash_attn_metadata,
)

device = "npu"
q = torch.ones((1, 12, 256), device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
k = torch.ones((1, 512, 2, 256), device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
v = k.clone()

def scale(shape):
    return torch.full(shape, 127, dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu)

q_descale = scale((2, 1, 6, 4, 2))  # Nkv, T, G, D/64, 2
k_descale = scale((1, 512, 2, 4, 2))
v_descale = scale((1, 8, 2, 256, 2))
cu_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
used_kv = torch.tensor([512], dtype=torch.int32, device=device)
pages = torch.tensor([[0]], dtype=torch.int32, device=device)
layout = dict(layout_q="TND", layout_q_descale="N2TGD", layout_kv="PA_BBND", layout_out="TND")
plan = quant_flash_attn_metadata(
    12, 2, 256, 1, cu_seqlens_q=cu_q, seqused_kv=used_kv,
    max_seqlen_q=1, mask_mode=0, **layout,
)
kwargs = dict(
    block_table=pages, p_scale=torch.ones(1, dtype=torch.float32, device=device),
    cu_seqlens_q=cu_q, seqused_kv=used_kv, metadata=plan,
    softmax_scale=256**-0.5, max_seqlen_q=1, mask_mode=0, **layout,
)
out, lse = quant_flash_attn(q, k, v, q_descale, k_descale, v_descale, 1, **kwargs)
reference, _ = quant_flash_attn(
    q, k, v, q_descale, k_descale, v_descale, 1, impl="torch_native", **kwargs,
)
print(out.shape, lse.shape)  # torch.Size([1, 12, 256]), torch.Size([0])
```
