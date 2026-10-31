# mlaprolog_v4

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `cann` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

**CANN 后端版本要求：CANN 9.2.0。**

两个实现均注册到 `Generation.ARCH22` 和 `Generation.ARCH35`；`auto` 是 `cann` 的别名。

arch22 支持 BF16+C16 和动态 INT8+C16，两种场景均支持有 RoPE、无 RoPE；MXFP8+C16 和 MXFP8+C8 仍仅限 arch35。更完整的 CANN V4 支持范围见[上游文档](https://gitcode.com/cann/ops-transformer/blob/d462156cf5d97712fcfeef95070d649f47d3bb6e/attention/mla_prolog_v3/docs/aclnnMlaPrologV4WeightNz.md)。

## 函数签名

```python
from typing import Optional, Tuple

import torch

from sgl_kernel_npu.operators.attention import mlaprolog_v4

def mlaprolog_v4(
    token_x: torch.Tensor,
    weight_dq: torch.Tensor,
    weight_uq_qr: torch.Tensor,
    weight_uk: torch.Tensor,
    weight_dkv_kr: torch.Tensor,
    rmsnorm_gamma_cq: torch.Tensor,
    rmsnorm_gamma_ckv: torch.Tensor,
    kv_cache: torch.Tensor,
    kr_cache: torch.Tensor,
    *,
    rope_sin: Optional[torch.Tensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    cache_index: Optional[torch.Tensor] = None,
    dequant_scale_x: Optional[torch.Tensor] = None,
    dequant_scale_w_dq: Optional[torch.Tensor] = None,
    dequant_scale_w_uq_qr: Optional[torch.Tensor] = None,
    dequant_scale_w_dkv_kr: Optional[torch.Tensor] = None,
    quant_scale_ckv: Optional[torch.Tensor] = None,
    quant_scale_ckr: Optional[torch.Tensor] = None,
    smooth_scales_cq: Optional[torch.Tensor] = None,
    actual_seq_len: Optional[torch.Tensor] = None,
    k_nope_clip_alpha: Optional[torch.Tensor] = None,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
    cache_mode: str = "PA_BSND",
    query_norm_flag: bool = False,
    weight_quant_mode: int = 0,
    kv_cache_quant_mode: int = 0,
    query_quant_mode: int = 0,
    ckvkr_repo_mode: int = 0,
    quant_scale_repo_mode: int = 0,
    tile_size: int = 128,
    qc_qr_scale: float = 1.0,
    kc_scale: float = 1.0,
    token_x_dtype: Optional[int] = None,
    weight_dq_dtype: Optional[int] = None,
    weight_uq_qr_dtype: Optional[int] = None,
    weight_dkv_kr_dtype: Optional[int] = None,
    kv_cache_dtype: Optional[int] = None,
    impl: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

`impl="auto"` 和 `impl="cann"` 调用公开接口 `cann_ops_transformer.mla_prolog`，其底层为 `aclnnMlaPrologV4WeightNz`。其余参数保留该接口的名称和默认值。这是**返回五个输出、原地更新缓存**的接口，不是返回七个输出的 functional 版本。

`rope_sin`、`rope_cos` 同时为非空张量时启用 RoPE；两者均未提供或为空时关闭 RoPE。若只有一个为非空张量，公开入口在后端分发前抛出 `ValueError`。

## 功能说明

完成 attention 之前的 MLA 投影和归一化，可选地对 R 维 Query/Key 分量执行 RoPE。返回 absorb 后的 Query（`query`）和 Query 分量（`query_rope`），并**原地更新**共享 KV latent 缓存（`kv_cache`）和 R 维 Key 缓存（`kr_cache`）。不执行 attention 分数、softmax 或最终的 Value 投影。

**单独接入的原因：**
Kimi K3 的 MLA 不使用 RoPE。截至 2026-10-28 核对的 [torch_npu V3 实现](https://github.com/Ascend/op-plugin/blob/master/op_plugin/ops/opapi/MlaPrologV3KernelNpuOpApi.cpp)调用 `aclnnMlaPrologV3WeightNz`，该入口[固定启用 RoPE](https://gitcode.com/cann/ops-transformer/blob/d462156cf5d97712fcfeef95070d649f47d3bb6e/attention/mla_prolog_v3/op_api/aclnn_mla_prolog_v3_weight_nz.cpp)。这里通过 `cann_ops_transformer.mla_prolog` 接入 V4，统一提供有 RoPE 和原生无 RoPE 两种能力。

下文中，`H` 为输入宽度，`Rq` 为 Query latent 宽度，`C` 为 KV latent 宽度，`N` 为本地 Query 头数，`D` 为 absorb 前的 Query 宽度，`R` 为 Query/Key 投影分量的宽度。

启用 RoPE 时，相邻的投影通道组成旋转对；旋转后的偶数通道存放在 `query_rope` 和 `kr_cache` 的前半部分，奇数通道存放在后半部分。正弦、余弦张量采用相同的前后半区顺序，每个 token 的各个 head 共用一组参数；传入普通正弦值，不要提前对前半部分取负。旋转使用投影结果进行 FP32 运算（mode 0 的投影结果为 BF16，mode 2 为反量化后的 FP32，mode 3 为 FP32），再将旋转结果转为 BF16。mode 2 在 RoPE 前不将反量化的投影结果提前转为 BF16。关闭 RoPE 时，投影分量保留原通道顺序。

在 `PA_BSND` 中，`s=cache_index[t]` 对应的逻辑写入位置是 `[s // P, s % P, 0, :]`，`P` 是 page size。`PA_NZ` 在分块存储中写入相同的逻辑 slot。只更新索引指定的位置；两个 cache 的生命周期和容量由调用方管理。

独立参考实现覆盖下表中的 BF16、动态 INT8 和 MXFP8 场景，均支持有 RoPE 和无 RoPE。`C16` 表示两份缓存均为 BF16；`C8` 表示 `kv_cache` 为 FP8、`kr_cache` 为 BF16。

| 场景 | `weight_quant_mode` | `kv_cache_quant_mode` | `query_quant_mode` | `query` / `kv_cache` / `kr_cache` |
| --- | --- | --- | --- | --- |
| BF16 投影，C16 | `0` | `0` | `0` | BF16 / BF16 / BF16 |
| 动态 INT8 投影，C16 | `2` | `0` | `0` | BF16 / BF16 / BF16 |
| MXFP8 投影，C16 | `3` | `0` | `0` | BF16 / BF16 / BF16 |
| MXFP8 投影，C8 | `3` | `1` | `1` | FP8 E4M3FN / FP8 E4M3FN / BF16 |

Mode 2 接收 INT8 激活和三份 INT8 投影权重，分别携带 FP32 逐 token 激活 scale 和逐输出通道权重 scale。矩阵乘使用 INT32 累加，再反量化为 FP32 参与归一化。归一化后的 Query latent 直接逐 token 动态量化为 INT8，采用最近偶数舍入，中间不先转为 BF16。`weight_uk`、absorb 后的 Query 和两份缓存仍为 BF16。调用方在进入算子前量化 `token_x`，不涉及 INT8 KV cache。

Mode 3 使用带块级 scale 的 FP8 输入和权重。归一化后的 Query latent 按 `CAST_ROUND` 转为 BF16（舍入到最近值，中点时远离零），再以每组 32 个元素和 E8M0 scale 量化。FP8 转换采用最近偶数舍入，并饱和到 `[-448, 448]`。参考实现遵循这些 CANN 转换规则；融合计算的归约顺序仍可能带来数值差异。

C8 中的 `quant_scale_ckv` 是**乘法量化系数**，即 KV 反量化 scale 的倒数。Query 按 token 和 head 动态量化。在可选 RoPE 和 BF16 转换之后，`query_rope` 根据 Q/KV scale 做逆尺度补偿，供后续量化 attention 使用；调用方不要再次做同样补偿。`kr_cache` 保持 BF16。全零 Query 行产生零 Q scale；参考实现遵循 CANN，不将这个 scale 替换为 1，也不额外加入 epsilon。

`query_norm_flag=True` 时，mode 2 返回 INT8 `query_norm` 及其 FP32 逐 token scale；mode 3 **即使使用 C16 缓存，也返回 FP8 Query latent 及其 E8M0 group scale**。attention Query 为 BF16 并不意味着 `query_norm` 也是 BF16。

## 参数和返回值

CANN 后端的所有张量应位于同一 NPU。下表为逻辑 shape；`FRACTAL_NZ` 是物理格式，不表示调用方应手工 reshape 权重。参考实现接受 CPU 或 NPU 张量，在 CPU 上计算，将输出送回 `token_x.device`，并把缓存更新复制回原 buffer。

参考实现覆盖的场景使用 `C=512`、`R=64`、`D∈{128,192}`、`Rq∈{1536,2048}`，输入宽度和本地头数应符合 CANN 约束；独立双缓存布局为 `PA_BSND` 或 `PA_NZ`。CANN 后端保留更完整的上游接口约定；参数允许某个 mode 不意味所有 mode 组合均有效。

| 参数 | 类型 / 逻辑 shape / dtype | 默认值 | 含义和约束 |
| --- | --- | --- | --- |
| `token_x` | Tensor，`[T,H]` 或 `[B,S,H]`；mode 0 为 BF16，mode 2 为 INT8，mode 3 为 FP8 E4M3FN | 必填 | 连续 ND 输入；按展平形式描述时 `T=B*S`。 |
| `weight_dq` | Tensor，`[H,Rq]`；BF16、INT8 或 FP8 E4M3FN | 必填 | Query 下投影。CANN 要求 FRACTAL_NZ，保留逻辑二维 shape。 |
| `weight_uq_qr` | Tensor，`[Rq,N*(D+R)]`；BF16、INT8 或 FP8 E4M3FN | 必填 | 每个 head 的 `q_c` 与 `q_r` 上投影拼接权重；CANN 要求 FRACTAL_NZ。 |
| `weight_uk` | Tensor，`[N,D,C]`；BF16 | 必填 | 用于 absorb 的 Key 上投影，连续 ND。 |
| `weight_dkv_kr` | Tensor，`[H,C+R]`；BF16、INT8 或 FP8 E4M3FN | 必填 | 将输入 `X` 投影为 KV latent `c_0` 与 Key 分量 `k_r`；CANN 要求 FRACTAL_NZ。 |
| `rmsnorm_gamma_cq` | Tensor，`[Rq]`；BF16 | 必填 | Query latent 的 RMSNorm 权重，连续 ND。 |
| `rmsnorm_gamma_ckv` | Tensor，`[C]`；BF16 | 必填 | KV latent 的 RMSNorm 权重，连续 ND。 |
| `kv_cache` | Tensor，`[pages,P,1,C]`；C16 为 BF16，C8 为 FP8 E4M3FN | 必填 | 可变的共享 latent 缓存。`PA_NZ` 使用对应的物理 tile 顺序。 |
| `kr_cache` | Tensor，`[pages,P,1,R]`；BF16 | 必填 | 存储 R 维 Key 投影分量 `k_r` 的可变缓存。CANN 专有的合并存储模式另有布局要求。 |
| `rope_sin`、`rope_cos` | 可选 BF16 Tensor，`[T,R]` 或 `[B,S,R]` | `None` | 连续 ND，两者 shape 相同，token 维度与 `token_x` 一致。同时非空时启用 RoPE，均未提供或为空时关闭；仅一个非空时抛出 `ValueError`。 |
| `cache_index` | 可选 Tensor，`[T]` 或 `[B,S]`；int32/int64 | `None` | 本文 PA 场景必填；每个元素是有效逻辑 slot。参考对比时使用不重复的 slot。 |
| `dequant_scale_x` | mode 2：FP32 `[T,1]`；mode 3：E8M0 `[T,H/32]` | `None` | mode 2/3 必填，分别为逐 token 或块级输入反量化 scale。 |
| `dequant_scale_w_dq` | mode 2：FP32 `[1,Rq]`；mode 3：E8M0 `[Rq,H/32]` | `None` | mode 2/3 必填的 Query 下投影权重 scale，连续 ND。 |
| `dequant_scale_w_uq_qr` | mode 2：FP32 `[1,N*(D+R)]`；mode 3：E8M0 `[N*(D+R),Rq/32]` | `None` | mode 2/3 必填的 Query 上投影权重 scale，连续 ND。 |
| `dequant_scale_w_dkv_kr` | mode 2：FP32 `[1,C+R]`；mode 3：E8M0 `[C+R,H/32]` | `None` | mode 2/3 必填的 `weight_dkv_kr` 反量化 scale，连续 ND。 |
| `quant_scale_ckv` | 可选 Tensor，C8 为 `[1]` FP32 | `None` | 正的乘法量化系数 `1/s_c`，C8 场景必填。CANN per-channel 模式使用 `[1,C]`。 |
| `quant_scale_ckr` | 可选 Tensor，`[1,R]`；FP32 | `None` | CANN `kr_cache` 的 per-channel 量化参数；参考实现要求 `None`。 |
| `smooth_scales_cq` | 可选 Tensor，`[1,Rq]`；FP32 | `None` | CANN 对应模式的 Query latent 平滑参数；参考实现要求 `None`。 |
| `actual_seq_len` | 可选 Tensor，`[B]`；int32 | `None` | CANN 对应 block-indexed 缓存模式的累计序列长度；参考实现要求 `None`。 |
| `k_nope_clip_alpha` | 可选 Tensor，`[1]`；FP32 | `None` | CANN 对应分组量化模式的裁剪系数；参考实现要求 `None`。 |
| `rmsnorm_epsilon_cq`、`rmsnorm_epsilon_ckv` | `float` | `1e-5` | Query/KV RMSNorm epsilon。 |
| `cache_mode` | `str` | `"PA_BSND"` | 参考实现支持 `PA_BSND`、`PA_NZ`；CANN 还提供其文档列出的其他布局。 |
| `query_norm_flag` | `bool` | `False` | 是否返回归一化后的 Query latent，以及量化时的 scale。 |
| `weight_quant_mode` | `int` | `0` | CANN：`0` BF16，`1` 部分 INT8，`2` 动态 INT8，`3` MXFP8，`4` FP8，`5` HiFloat8。参考实现覆盖上表中的 `0`、`2`、`3`。 |
| `kv_cache_quant_mode` | `int` | `0` | CANN：`0` 非量化，`1` per-tensor，`2` per-channel，`3` per-token/group。参考实现为 `0`，或配合 weight/query mode `3/1` 的 `1`。 |
| `query_quant_mode` | `int` | `0` | `0` 非量化，`1` per-token/head 量化；支持的组合见上表。 |
| `ckvkr_repo_mode`、`quant_scale_repo_mode` | `int` | `0` | CANN 存储方式：分开存储 `0`、合并存储 `1`。参考实现要求 `0`。 |
| `tile_size` | `int` | `128` | CANN 对应 KV 量化模式的分组 tile 大小；不是 MXFP8 投影的分组大小，也不是缓存 page size。 |
| `qc_qr_scale`、`kc_scale` | `float` | `1.0` | Query 与共享 latent 的尺度校正系数。 |
| `token_x_dtype`、`weight_dq_dtype`、`weight_uq_qr_dtype`、`weight_dkv_kr_dtype`、`kv_cache_dtype` | 可选 `int` | `None` | CANN 逻辑 dtype 覆盖参数，例如按字节存储的 HiFloat8；参考实现要求 `None`。 |
| `impl` | `str` | `"auto"` | 仅限关键字参数；可选 `auto`、`cann`、`torch_native`。 |

五个返回值均为 Tensor。未启用的可选输出为空 Tensor，返回元组不额外包含两份 cache。

| 返回值 | Shape | 参考实现覆盖场景的 dtype | 含义 |
| --- | --- | --- | --- |
| `query` | `[T,N,C]` 或 `[B,S,N,C]` | C16 为 BF16，C8 为 FP8 E4M3FN | absorb 后的 Query。 |
| `query_rope` | `[T,N,R]` 或 `[B,S,N,R]` | BF16 | 完成可选 RoPE 的 R 维 Query 分量；C8 时带有尺度补偿。 |
| `dequant_scale_q_nope` | `[T,N,1]` 或 `[B*S,N,1]`，否则为空 | FP32 | C8 场景的 per-token/head Query 反量化 scale。 |
| `query_norm` | `[T,Rq]` 或 `[B,S,Rq]`，否则为空 | mode 0 为 BF16，mode 2 为 INT8，mode 3 为 FP8 E4M3FN | 请求输出时的归一化 Query latent。 |
| `dequant_scale_q_norm` | mode 2：`[T,1]`；mode 3：`[T,Rq/32]`；三维输入时 `T=B*S` | mode 2 为 FP32，mode 3 为 E8M0 | INT8 Query latent 的逐 token scale，或 MXFP8 的 group scale；mode 0 或未请求输出时为空。 |

## cann 后端

延迟导入 `cann_ops_transformer.mla_prolog` 并透传参数。本文依据 ops-transformer 提交 [`d462156cf5d97712fcfeef95070d649f47d3bb6e`](https://gitcode.com/cann/ops-transformer/commit/d462156cf5d97712fcfeef95070d649f47d3bb6e)（2026-10-20，UTC+8）编写。

## torch_native 后端

使用独立的 PyTorch CPU 计算作为精度参考，覆盖上表场景的有 RoPE、无 RoPE 版本及 `PA_BSND`/`PA_NZ` 缓存。输出返回输入设备，并原地更新传入的两份 cache。

## 使用示例

以下 arch35 示例使用 BF16 投影和两份 BF16 缓存。尺寸采用 [Kimi K3 配置](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json)，attention TP16 下本地 Q heads 为 6；张量数值随机生成。`use_rope=False` 对应 Kimi 的无 RoPE 行为；设为 `True` 可用同样的尺寸验证 RoPE 路径。

精度测试保留本地 Q heads 96、24、6（attention TP1/4/16），按下表选择代表性组合。四种模式为 BF16+C16、动态 INT8+C16、MXFP8+C16、MXFP8+C8。除明确注明外，各组均覆盖开启/关闭 RoPE，不再对全部参数做笛卡尔积。

| 分组 | 本地 Q heads | Token 数 | 布局 | 模式 | 用例数 |
| --- | --- | --- | --- | --- | ---: |
| 性能代表形状 | 24、6 | 1、4、16、32、64 | PA_BSND | 全部四种 | 80 |
| NZ 布局 | 6 | 1、16、64 | PA_NZ | 全部四种 | 24 |
| TP1 形状 | 96 | 4 | PA_BSND | 全部四种 | 8 |
| 关闭 normalized-query 输出 | 6 | 17 | PA_BSND | 三种 C16 模式 | 6 |
| 负数 padding slot | 6 | 5 | PA_BSND、PA_NZ | 全部四种 | 16 |
| 其余 token 长度，无 RoPE | 6 | 2、3、7、8、12、24、31、40、48、56、63 | PA_BSND | BF16+C16 | 11 |

矩阵包含 145 条数值/cache 用例和 16 条 Direct-CANN parity 用例；每组被测/参考实现共收集 161 条。其中 arch35 适用 123 条，arch22 适用 87 条，其余架构不适用的用例会跳过。动态 INT8+C16 用例限定为 arch22，MXFP8 用例限定为 arch35。成对用例生成相同的输入和权重。

性能选择保持不变：TP4/TP16、token 数 `1, 4, 16, 32, 64`、`PA_BSND`、两种 RoPE 状态以及对应架构的模式，arch35 共 60 个配置，arch22 共 40 个。仅检查精度的 CI 使用 `--enable-perf=false`。

BF16/MXFP8 精度标准保持不变。INT8 `query_norm` 要求全部元素最多相差一个整数码值；FP32 scale 单独使用 `atol=1e-5、rtol=1e-5` 检查。这是本用例的 INT8 判定标准，不套用 FP8 混合容差，也不宣称为 CANN 官方阈值。公开接口与直接 CANN 的一致性测试仍要求逐字节相同。

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import mlaprolog_v4

device = "npu"
dtype = torch.bfloat16
T, H, Rq, N, D, C, R, P = 1, 7168, 1536, 6, 128, 512, 64, 128

def nz_weight(shape):
    weight = torch.randn(shape, device=device, dtype=dtype) * 0.01
    return torch_npu.npu_format_cast(weight, 29)

x = torch.randn(T, H, device=device, dtype=dtype)
w_dq = nz_weight((H, Rq))
w_uq_qr = nz_weight((Rq, N * (D + R)))
w_uk = torch.randn(N, D, C, device=device, dtype=dtype) * 0.01
w_dkv_kr = nz_weight((H, C + R))
gamma_q = torch.ones(Rq, device=device, dtype=dtype)
gamma_k = torch.ones(C, device=device, dtype=dtype)
slots = torch.tensor([0], device=device, dtype=torch.int64)
kv_cache = torch.zeros(1, P, 1, C, device=device, dtype=dtype)
kr_cache = torch.zeros(1, P, 1, R, device=device, dtype=dtype)

use_rope = False
rope_sin = rope_cos = None
if use_rope:
    positions = torch.arange(1, T + 1, dtype=torch.float32)
    inv_freq = 10000.0 ** (-torch.arange(0, R, 2, dtype=torch.float32) / R)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    rope_sin = angles.sin().to(device=device, dtype=dtype)
    rope_cos = angles.cos().to(device=device, dtype=dtype)

args = (x, w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_q, gamma_k)
kwargs = dict(
    rope_sin=rope_sin,
    rope_cos=rope_cos,
    cache_index=slots,
    cache_mode="PA_BSND",
    weight_quant_mode=0,
    kv_cache_quant_mode=0,
    query_quant_mode=0,
    query_norm_flag=True,
)

# auto 选择 CANN V4 入口，并更新传入的两份 cache。
query, query_rope, q_scale, q_norm, q_norm_scale = mlaprolog_v4(
    *args, kv_cache, kr_cache, **kwargs, impl="auto"
)
print(query.shape, query_rope.shape, q_norm.shape)

# 接口会修改 cache，因此为参考计算使用新的 buffer。
reference = mlaprolog_v4(
    *args,
    torch.zeros_like(kv_cache),
    torch.zeros_like(kr_cache),
    **kwargs,
    impl="torch_native",
)
print(reference[0].shape, q_scale.numel(), q_norm_scale.numel())
```
