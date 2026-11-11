# mla_preprocess

## 芯片支持

| 实现 | arch22（2201） | arch35（3510） |
| --- | --- | --- |
| `ascendc` | 支持 | 不支持 |
| `torch_native` | 支持 | 支持 |

CANN 实现与构建只允许 arch22；`torch_native` 可在任意架构作为精度参考运行。

## 函数原型

```python
from sgl_kernel_npu.operators.attention import mla_preprocess

def mla_preprocess(
    hidden_state,
    gamma0,
    beta0,
    wdqkv,
    descale0,
    gamma1,
    beta1,
    wuq,
    descale1,
    gamma2,
    cos,
    sin,
    wuk,
    kv_cache,
    kv_cache_rope,
    slotmapping,
    *,
    quant_scale0,
    quant_offset0,
    bias0,
    quant_scale1,
    quant_offset1,
    bias1,
    ctkv_scale=None,
    q_nope_scale=None,
    cache_mode="krope_ctkv",
    quant_mode="per_tensor_quant_asymm",
    q_out0,
    kv_cache_out0,
    q_out1,
    kv_cache_out1,
    impl="auto",
):
    ...
```

`auto` 默认使用 `ascendc`；`torch_native` 使用 PyTorch 运算。
原生实现不可用时报错，不自动切换实现。

四个输出/缓存张量由调用方分配并原地写入，同时按相同顺序返回：
`(q_out0, kv_cache_out0, q_out1, kv_cache_out1)`。

## 功能说明

该算子把 MLA（Multi-head Latent Attention）推理前的整段数据准备融合为一次调用：
RMSNorm、静态量化、量化矩阵乘、RoPE，以及分页 KV Cache 写入。

一次调用的计算顺序为：

```text
q1        = round_fp16(hidden_state / quant_scale0 + quant_offset0)          # INT8
mm1       = dequant(wdqkv @ q1) + bias0                                      # [N, 2112]
latent, q = split(mm1, [576, 1536])
k_nope    = latent[:, :512]
k_pe      = latent[:, 512:]
q         = rmsnorm(q, gamma0) * gamma1 + beta1                              # 1536 维
k_nope    = rmsnorm(k_nope, gamma2)                                          # 512 维
q2        = round_fp16(q / quant_scale1 + quant_offset1)                     # INT8
mm2       = dequant(wuq @ q2) + bias1                                        # [N, headNum*192]
q_nope, q_pe = split(mm2.view(N, headNum, 192), [128, 64])
q_nope    = bmm(q_nope, wuk)                                                 # [N, headNum, 512]
q_pe      = rope(q_pe, cos, sin)
k_pe      = rope(k_pe, cos, sin)
```

`rmsnorm` 使用 `eps = 1e-6`。RoPE 采用 half 旋转布局：输入按前后两半拆分，
输出为 `[x1*cos - x2*sin, x1*sin + x2*cos]`。

`cache_mode` 决定缓存布局与输出切分：

| `cache_mode` | 缓存布局 | `q_out0` | `q_out1` |
| --- | --- | --- | --- |
| `kvcache` | ND，nope 512 + rope 64 合并为 576 | `[N, headNum, 576]` | 不使用 |
| `krope_ctkv` | ND，nope 512 + rope 64 分开写 | `[N, headNum, 512]` | `[N, headNum, 64]` |
| `int8_nzcache` | NZ，nope INT8 按 32 元素分组 | `[N, headNum, 512]` INT8 | `[N, headNum, 64]` |
| `nzcache` | NZ，nope/rope 按 16 元素分组 | `[N, headNum, 512]` | `[N, headNum, 64]` |

NZ 布局为 `[blockNum, groups, blockSize, C0]`，其中 `int8_nzcache` 的 nope
`C0 = 32`，其余为 `16`；`blockSize` 必须为 `128`。

`quant_mode` 当前仅实现 `per_tensor_quant_asymm`（`hidden_state` 做 per-tensor
非对称量化）。其他取值不会产生有效结果。

## 参数与返回值

`N` 为 token 数，`H` 为 `hidden_state` 维度，`headNum` 为 query 头数，
`blockNum`、`blockSize` 为分页缓存参数。

| 参数 | Shape | Dtype / 默认值 | 说明 |
| --- | --- | --- | --- |
| `hidden_state` | `[N, H]` | FP16 / BF16 | 输入隐状态；`N` 取值 `[1, 1024]`，`H` 取值 `[2048, 8192]`。 |
| `gamma0`、`beta0` | `[H]` | 同输入 | 第一次 RMSNorm 的权重与偏置。 |
| `wdqkv` | 逻辑 `[2112, H]` | INT8 | 第一次量化矩阵乘权重，NPU 上为 FRACTAL_NZ。 |
| `descale0` | `[2112]` | FP16 输入为 int64，BF16 输入为 FP32 | 第一次矩阵乘的反量化系数。 |
| `gamma1`、`beta1` | `[1536]` | 同输入 | 第二次 RMSNorm 的权重与偏置。 |
| `wuq` | 逻辑 `[headNum*192, 1536]` | INT8 | 第二次量化矩阵乘权重，NPU 上为 FRACTAL_NZ。 |
| `descale1` | `[headNum*192]` | FP16 输入为 int64，BF16 输入为 FP32 | 第二次矩阵乘的反量化系数。 |
| `gamma2` | `[512]` | 同输入 | 对 `k_nope` 做 RMSNorm 的权重。 |
| `cos`、`sin` | `[N, 64]` | 同输入 | RoPE 三角函数表。 |
| `wuk` | `[headNum, 128, 512]` | 同输入 | Q 吸收权重。 |
| `kv_cache` | 见 `cache_mode` | FP16 / BF16 / INT8 | KV nope 缓存；NPU 上的物理布局由 `cache_mode` 决定。 |
| `kv_cache_rope` | `[blockNum, blockSize, 1, 64]` | 同输入 | KV rope 缓存；NZ 模式下物理布局为 NZ。 |
| `slotmapping` | `[N]` | int32，必填 | 每个 token 写入的分页槽位。 |
| `quant_scale0` | `[1]` | 同输入 | 第一次量化缩放。 |
| `quant_offset0` | `[1]` | INT8 | 第一次量化偏移。 |
| `bias0` | `[2112]` | INT32 | 第一次矩阵乘偏置。 |
| `quant_scale1` | `[1]` | 同输入 | 第二次量化缩放。 |
| `quant_offset1` | `[1]` | INT8 | 第二次量化偏移。 |
| `bias1` | `[headNum*192]` | INT32 | 第二次矩阵乘偏置。 |
| `ctkv_scale` | `[1]` | 同输入 / `None` | `int8_nzcache` 下对 `k_nope` 的量化缩放，其余模式忽略。 |
| `q_nope_scale` | `[headNum]` | 同输入 / `None` | `int8_nzcache` 下对 `q_nope` 的量化缩放，其余模式忽略。 |
| `cache_mode` | — | `"krope_ctkv"` | 可选 `kvcache`、`krope_ctkv`、`int8_nzcache`、`nzcache`。 |
| `quant_mode` | — | `"per_tensor_quant_asymm"` | 仅实现 per-tensor 非对称量化。 |
| `q_out0` | 见 `cache_mode` | 同输入 / INT8 | `q_nope` 输出，原地写入。 |
| `kv_cache_out0` | 同 `kv_cache` | 同 `kv_cache` | 与 `kv_cache` 相同的张量。 |
| `q_out1` | `[N, headNum, 64]` | 同输入 | `q_pe` 输出，原地写入。 |
| `kv_cache_out1` | 同 `kv_cache_rope` | 同 `kv_cache_rope` | 与 `kv_cache_rope` 相同的张量。 |
| `impl` | — | `"auto"` | 可选 `auto`、`ascendc`、`torch_native`，仅限关键字。 |

| 返回值 | Shape | Dtype |
| --- | --- | --- |
| `q_out0` | 见 `cache_mode` | 同输入 / INT8 |
| `kv_cache_out0` | 同 `kv_cache` | 同 `kv_cache` |
| `q_out1` | `[N, headNum, 64]` | 同输入 |
| `kv_cache_out1` | 同 `kv_cache_rope` | 同 `kv_cache_rope` |

使用时请注意：

- 所有张量必须位于同一 NPU，且 `slotmapping` 长度等于 `N`。
- FP16 输入只支持 `cache_mode` 的 `krope_ctkv`；`int8_nzcache` 与 `nzcache`
  需要 BF16 输入。
- `int8_nzcache` 与 `nzcache` 要求 `blockSize = 128`；`int8_nzcache` 还要求
  `ctkv_scale`、`q_nope_scale` 非空。
- 权重 `wdqkv`、`wuq` 需要以 FRACTAL_NZ 格式传入；调用方可使用
  `torch_npu.npu_format_cast(weight, 29)` 转换，并保持
  `torch_npu.npu.config.allow_internal_format = True`。
- 计算逻辑与旧算子一致：`torch_native` 是精度对比参考，按 FP32 完成中间计算，
  并按旧测试的取整顺序转换到目标 dtype。
