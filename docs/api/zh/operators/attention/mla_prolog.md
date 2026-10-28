# mla_prolog

## 硬件支持

| 实现 | arch22 | arch35（A5） |
| --- | --- | --- |
| `cann` | 未注册支持 | 未验证；已注册的目标架构 |
| `torch_native` | 未注册支持 | 未验证；已注册的目标架构 |

两个实现均注册到 `Generation.ARCH35`。此表不表示已经完成设备验证或性能测量。`auto` 是 `cann` 的别名，不执行后端回退。

## 函数签名

```python
from typing import Optional, Tuple

import torch

from sgl_kernel_npu.operators.attention import mla_prolog

def mla_prolog(
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

## 功能说明

完成 attention 之前的 MLA 投影和归一化，返回 absorb 后的 Query 及其旁路分量，并**原地更新**共享 latent 与旁路缓存。不执行 attention 分数、softmax 或最终的 Value 投影。

记 `H` 为输入宽度，`Rq` 为 Query latent 宽度，`C` 为 KV latent 宽度，`N` 为本地 Query 头数，`D` 为 absorb 前的 Query 宽度，`R` 为旁路宽度。非量化计算为：

$$
\operatorname{RMSNorm}(x;\gamma,\epsilon)
=\gamma\odot x\,/\sqrt{\operatorname{mean}(x^2)+\epsilon}.
$$

$$
c_q=\alpha_q\operatorname{RMSNorm}(XW_{dq};\gamma_q,\epsilon_q),
\qquad [q_c,q_r]=\operatorname{reshape}(c_qW_{uq,qr}),
\qquad q_n=q_cW_{uk}.
$$

$$
[c_0,k_r]=XW_{dkv,kr},
\qquad c=\alpha_k\operatorname{RMSNorm}(c_0;\gamma_k,\epsilon_k).
$$

其中 `alpha_q=qc_qr_scale`，`alpha_k=kc_scale`。成对传入 RoPE 张量时，CANN 对 `q_r` 和 `k_r` 执行对应旋转；两者均为 `None` 或空张量时跳过旋转，但**旁路投影和 `kr_cache` 仍然存在**。Kimi K3 MLA 使用这种无 RoPE 语义。

在 `PA_BSND` 中，`s=cache_index[t]` 对应的逻辑写入位置是 `[s // P, s % P, 0, :]`，`P` 是 page size。`PA_NZ` 在分块存储中写入相同的逻辑 slot。只更新索引指定的位置；两个 cache 的生命周期和容量由调用方管理。

独立参考实现覆盖下列无 RoPE 场景。`C16` 表示 BF16 缓存；`C8` 表示 FP8 共享 latent 缓存配合 BF16 旁路缓存。

| 场景 | `weight_quant_mode` | `kv_cache_quant_mode` | `query_quant_mode` | Query / latent cache / side cache |
| --- | --- | --- | --- | --- |
| BF16 投影，C16 | `0` | `0` | `0` | BF16 / BF16 / BF16 |
| MXFP8 投影，C16 | `3` | `0` | `0` | BF16 / BF16 / BF16 |
| MXFP8 投影，C8 | `3` | `1` | `1` | FP8 E4M3FN / FP8 E4M3FN / BF16 |

Mode 3 对带块级 scale 的 FP8 输入和权重进行反量化，将归一化后的 Query latent 按 `CAST_ROUND` 转为 BF16（舍入到最近值，中点时远离零），再以每组 32 个元素重新量化。设 `a` 为**完成这次 BF16 转换后**一组的最大绝对值。对于元素均有限且至少包含一个 BF16 正规数的组，MX scale 为 `2**max(floor(log2(a)) - 8, -127)`。仅包含零或 BF16 次正规数的组是例外：E8M0 scale 字节为 `0`（scale 为 `2**-127`），整组量化结果为零。正规数与次正规数混合的组统一按共享 scale 量化，不额外逐元素清零次正规数。FP8 转换采用最近偶数舍入，并饱和到 `[-448, 448]`。中间 BF16 转换不能与所有默认 PyTorch BF16 转换视为相同。这些规则依据 CANN 源码；融合计算的归约顺序仍可能带来数值差异。

C8 中的 `quant_scale_ckv` 是**乘法量化系数** `1/s_c`，不是 KV 反量化 scale `s_c`：

$$
c_8=\operatorname{FP8}(c/s_c),\qquad
s_q=\max|q_n|/448,\qquad q_8=\operatorname{FP8}(q_n/s_q).
$$

BF16 Query 旁路补偿为 `q_r / (s_q * s_c)`。下游 attention 算子对合并后的点积乘以 `s_q * s_c` 时，旁路项才能恢复原来的尺度。`kr_cache` 保持 BF16。不要再次对返回的 `query_rope` 做同样补偿。全零 Query 行产生 `s_q=0`；参考实现遵循 CANN，不将这个 scale 替换为 1，也不额外加入 epsilon。

`query_norm_flag=True` 时，mode 3 **即使使用 C16 缓存，也返回 FP8 Query latent 及其 E8M0 group scale**。attention Query 为 BF16 并不意味着 `query_norm` 也是 BF16。

## 参数和返回值

CANN 后端的所有张量应位于同一 NPU。下表为逻辑 shape；`FRACTAL_NZ` 是物理格式，不表示调用方应手工 reshape 权重。参考实现接受 CPU 或 NPU 张量，在 CPU 上计算，将输出送回 `token_x.device`，并把缓存更新复制回原 buffer。

参考实现覆盖的场景使用 `C=512`、`R=64`、`D∈{128,192}`、`Rq∈{1536,2048}`，输入宽度和本地头数应符合 CANN 约束；独立双缓存布局为 `PA_BSND` 或 `PA_NZ`。CANN 后端保留更完整的上游接口约定；参数允许某个 mode 值不意味着所有 mode 组合均有效。

| 参数 | 类型 / 逻辑 shape / dtype | 默认值 | 含义和约束 |
| --- | --- | --- | --- |
| `token_x` | Tensor，`[T,H]` 或 `[B,S,H]`；mode 0 为 BF16，mode 3 为 FP8 E4M3FN | 必填 | 连续 ND 输入；按展平形式描述时 `T=B*S`。 |
| `weight_dq` | Tensor，`[H,Rq]`；BF16 或 FP8 E4M3FN | 必填 | Query 下投影。CANN 要求 FRACTAL_NZ，保留逻辑二维 shape。 |
| `weight_uq_qr` | Tensor，`[Rq,N*(D+R)]`；BF16 或 FP8 E4M3FN | 必填 | 每个 head 的 Query 与旁路上投影拼接权重；CANN 要求 FRACTAL_NZ。 |
| `weight_uk` | Tensor，`[N,D,C]`；BF16 | 必填 | 用于 absorb 的 Key 上投影，连续 ND。 |
| `weight_dkv_kr` | Tensor，`[H,C+R]`；BF16 或 FP8 E4M3FN | 必填 | 共享 latent 与旁路投影；CANN 要求 FRACTAL_NZ。 |
| `rmsnorm_gamma_cq` | Tensor，`[Rq]`；BF16 | 必填 | Query latent 的 RMSNorm 权重，连续 ND。 |
| `rmsnorm_gamma_ckv` | Tensor，`[C]`；BF16 | 必填 | KV latent 的 RMSNorm 权重，连续 ND。 |
| `kv_cache` | Tensor，`[pages,P,1,C]`；C16 为 BF16，C8 为 FP8 E4M3FN | 必填 | 可变的共享 latent 缓存。`PA_NZ` 使用对应的物理 tile 顺序。 |
| `kr_cache` | Tensor，`[pages,P,1,R]`；BF16 | 必填 | 可变的旁路缓存，无 RoPE 时也不能删除。CANN 专有的合并存储模式另有布局要求。 |
| `rope_sin`、`rope_cos` | 可选 Tensor，`[T,R]` 或 `[B,S,R]`；BF16 | `None` | 必须成对提供或同时不提供。参考实现要求两者均缺省或为空。 |
| `cache_index` | 可选 Tensor，`[T]` 或 `[B,S]`；int32/int64 | `None` | 本文 PA 场景必填；每个元素是有效逻辑 slot。参考对比时使用不重复的 slot。 |
| `dequant_scale_x` | 可选 Tensor，`[T,H/32]`；mode 3 为 E8M0 | `None` | Mode 3 输入块级反量化 scale。其他 CANN 模式使用各自的上游 scale 布局。 |
| `dequant_scale_w_dq` | 可选 Tensor，`[Rq,H/32]`；mode 3 为 E8M0 | `None` | Mode 3 Query 下投影权重 scale，连续 ND。 |
| `dequant_scale_w_uq_qr` | 可选 Tensor，`[N*(D+R),Rq/32]`；mode 3 为 E8M0 | `None` | Mode 3 Query 上投影权重 scale，连续 ND。 |
| `dequant_scale_w_dkv_kr` | 可选 Tensor，`[C+R,H/32]`；mode 3 为 E8M0 | `None` | Mode 3 KV/旁路投影权重 scale，连续 ND。 |
| `quant_scale_ckv` | 可选 Tensor，C8 为 `[1]` FP32 | `None` | 正的乘法量化系数 `1/s_c`，C8 场景必填。CANN per-channel 模式使用 `[1,C]`。 |
| `quant_scale_ckr` | 可选 Tensor，`[1,R]`；FP32 | `None` | CANN 旁路缓存 per-channel 量化参数；参考实现要求 `None`。 |
| `smooth_scales_cq` | 可选 Tensor，`[1,Rq]`；FP32 | `None` | CANN 对应模式的 Query latent 平滑参数；参考实现要求 `None`。 |
| `actual_seq_len` | 可选 Tensor，`[B]`；int32 | `None` | CANN 对应 block-indexed 缓存模式的累计序列长度；参考实现要求 `None`。 |
| `k_nope_clip_alpha` | 可选 Tensor，`[1]`；FP32 | `None` | CANN 对应分组量化模式的裁剪系数；参考实现要求 `None`。 |
| `rmsnorm_epsilon_cq`、`rmsnorm_epsilon_ckv` | `float` | `1e-5` | Query/KV RMSNorm epsilon。 |
| `cache_mode` | `str` | `"PA_BSND"` | 参考实现支持 `PA_BSND`、`PA_NZ`；CANN 还提供其文档列出的其他布局。 |
| `query_norm_flag` | `bool` | `False` | 是否返回归一化后的 Query latent，以及量化时的 scale。 |
| `weight_quant_mode` | `int` | `0` | CANN：`0` BF16，`1` 部分 INT8，`2` 动态 INT8，`3` MXFP8，`4` FP8，`5` HiFloat8。参考实现只覆盖上表中的 `0`、`3`。 |
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
| `query_rope` | `[T,N,R]` 或 `[B,S,N,R]` | BF16 | Query 旁路；无 RoPE 时不旋转，C8 时带有尺度补偿。 |
| `dequant_scale_q_nope` | `[T,N,1]` 或 `[B*S,N,1]`，否则为空 | FP32 | C8 场景的 per-token/head Query 反量化 scale。 |
| `query_norm` | `[T,Rq]` 或 `[B,S,Rq]`，否则为空 | mode 0 为 BF16，mode 3 为 FP8 E4M3FN | 请求输出时的归一化 Query latent。 |
| `dequant_scale_q_norm` | `[T,Rq/32]` 或 `[B*S,Rq/32]`，否则为空 | mode 3 为 E8M0 | MXFP8 Query latent 的 group scale；mode 0 或未请求输出时为空。 |

## cann 后端

延迟导入规范包名下的 `cann_ops_transformer.mla_prolog`，透传完整参数。支持范围和输入校验遵循已安装的 CANN 实现；包装层不做版本能力探测，也不替换为其他算子。

### 依赖安装

只有使用此算子的应用需要安装这项依赖。kernel-npu wheel 不包含 CANN toolkit、native MLAProlog 算子或外部 Python 绑定。准备 CANN 9.2/A5 环境，其中应包含提供 `aclnnMlaPrologV4WeightNz` 的配套 V4 native 算子、头文件和库，并使用兼容的 PyTorch 与 torch_npu。Python 绑定使用 JIT 编译，因此还需要相应的编译器和构建依赖。仅有 toolkit 版本信息不能证明 Python 绑定已安装。

本集成依据 ops-transformer 提交 `d462156cf5d97712fcfeef95070d649f47d3bb6e` 的 API。从该版本构建并安装**普通算子的完整 Python wheel**：

```bash
git clone https://gitcode.com/cann/ops-transformer.git
cd ops-transformer
git checkout d462156cf5d97712fcfeef95070d649f47d3bb6e
bash build.sh --torch_extension
python -m pip install --no-deps build_out/cann_ops_transformer-*.whl
cd ..
```

这个步骤只打包 Python 绑定，不安装或替换 native CANN 算子。Native 算子与 Python 绑定必须匹配。不要给上述命令添加 `--ops=mla_prolog_v3`：选定算子的子包使用 `cann_ops_transformer_<vendor>` 命名，默认是 `cann_ops_transformer_custom`，而本后端导入规范整包名 `cann_ops_transformer`。这里不搜索其他 namespace，也不自动回退。`cann_ops_transformer>=1.0.0` 这样的通用依赖声明无法确定所需源码版本。

Native 接口和构建依赖参考上游 [MLA Prolog 接口文档](https://gitcode.com/cann/ops-transformer/blob/d462156cf5d97712fcfeef95070d649f47d3bb6e/attention/mla_prolog_v3/docs/torchapi_mla_prolog.md)与[扩展打包说明](https://gitcode.com/cann/ops-transformer/blob/d462156cf5d97712fcfeef95070d649f47d3bb6e/torch_extension/README.md)。验证环境时应记录实际 CANN build 以及 PyTorch/torch_npu 版本；这里不指定未经核实的 wheel 发布版本，也不声明性能测量结果。

### 验证

构建并安装 kernel-npu 后，在 Ascend 950 上**从 kernel-npu 仓库根目录**运行公开 API 测试。第一条命令检查精度，第二条还采集选定性能用例的 profiling。这些测试会执行 native 算子，包括原地更新 cache；只检查 import 是否成功不够。

```bash
python -m pytest tests/python/operators/attention/test_mla_prolog.py \
    --skip-acc-check=false --enable-perf=false -v -s -x -rs
python -m pytest tests/python/operators/attention/test_mla_prolog.py \
    --skip-acc-check=false --prof-dir=results/profilings \
    --csv-path=results/mla_prolog.csv -v -s
```

## torch_native 后端

- 使用独立的 PyTorch CPU 计算，覆盖上表三种无 RoPE 场景。不通过 MLAProlog、CANN 后端或厂商量化算子产生参考结果。
- 支持逻辑二维投影权重、二维或三维 `token_x`、分开存储的 `PA_BSND`/`PA_NZ` cache。NPU FRACTAL_NZ 权重复制到逻辑 CPU Tensor 后参与计算。
- 不覆盖可选平滑、裁剪、block-indexed 序列元数据、合并缓存、逻辑 dtype 覆盖或其他量化组合。这些限制属于参考实现，CANN 入口仍保留其自身支持的组合。
- NPU 与 CPU 之间存在数据复制，因此该实现用于准确性参考，不是性能回退后端，也不能用于 NPU 图捕获。输出返回输入设备，原有的两份 cache 接收参考计算的更新。

## 使用示例

在具备上述依赖的 A5 环境运行。以下是用于说明 API 的合法合成 shape，不是采集的 Kimi 生产 shape，也不代表准确率或性能结果。示例使用 BF16 投影、无 RoPE 和两份 BF16 缓存。

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import mla_prolog

device = "npu"
dtype = torch.bfloat16
T, H, Rq, N, D, C, R, P = 1, 4096, 1536, 8, 192, 512, 64, 128

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

args = (x, w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_q, gamma_k)
kwargs = dict(
    rope_sin=None,
    rope_cos=None,
    cache_index=slots,
    cache_mode="PA_BSND",
    weight_quant_mode=0,
    kv_cache_quant_mode=0,
    query_quant_mode=0,
    query_norm_flag=True,
)

# auto 选择 CANN V4 入口，并更新传入的两份 cache。
query, side, q_scale, q_norm, q_norm_scale = mla_prolog(
    *args, kv_cache, kr_cache, **kwargs, impl="auto"
)
print(query.shape, side.shape, q_norm.shape)

# 接口会修改 cache，因此为参考计算使用新的 buffer。
reference = mla_prolog(
    *args,
    torch.zeros_like(kv_cache),
    torch.zeros_like(kr_cache),
    **kwargs,
    impl="torch_native",
)
print(reference[0].shape, q_scale.numel(), q_norm_scale.numel())
```
