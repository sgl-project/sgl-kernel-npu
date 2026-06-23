# Fused MoE API



<div align="center">

[![Mode](https://img.shields.io/badge/Mode-Fused-purple)]()
[![Platform](https://img.shields.io/badge/Platform-A3%20only-red)]()
[![Quant](https://img.shields.io/badge/Quantization-INT8-yellow)]()

English | [中文](#中文)

</div>

> [!IMPORTANT]
> **A3 only.** This API is NOT available on A2 or A5. A5 FP8/MXFP8 support is planned for future release.

---

## English

The `fused_deep_moe` API fuses dispatch + expert FFN computation + combine into a single operator call, significantly reducing communication overhead and end-to-end latency.

Two fuse modes are available via the `FuseMode` enum, selected by the `fuse_mode` parameter. **Both modes share the same Python interface** — the only difference is which internal C++ kernel is invoked.

### Fuse Modes Comparison

| | `FUSED_DEEP_MOE` (default) | `DISPATCH_FFN_COMBINE` |
|:---|:---|:---|
| **FuseMode value** | `1` | `2` |
| **How to call** | `fuse_mode=FuseMode.FUSED_DEEP_MOE` or omit (default) | `fuse_mode=FuseMode.DISPATCH_FFN_COMBINE` |
| **C++ kernel** | `fused_deep_moe` (single fused kernel via Act library) | `dispatch_ffn_combine` (Catlass library) |
| **`num_max_dispatch_tokens_per_rank` meaning** | Max tokens to **dispatch** per rank | Max tokens **received** after dispatch phase (i.e., `max_output_size`) |
| **Internal structure** | Single kernel: dispatch → GMM1 → SwiGLU → quant/dequant → GMM2 → combine | Separate dispatch handling, different routing/buffering strategy internally |
| **Scale tensor dtype** | `torch.float32` (native) | `torch.int64` (bitcast of float32 — Python layer handles conversion) |
| **Shared expert support** | ✅ Yes (via `share_expert_num`/`share_expert_rank_num` attrs) | ❌ No |
| **When to choose** | Default choice for most scenarios | Experimental alternative; may offer different latency characteristics for specific batch sizes |

> [!NOTE]
> For most users, `FUSED_DEEP_MOE` (default) is the recommended mode. `DISPATCH_FFN_COMBINE` is an alternative implementation that may behave differently for specific workload patterns. If you experience issues with the default mode, try `DISPATCH_FFN_COMBINE` as a fallback.

### `fused_deep_moe`

```python
output, ep_recv_count = buffer.fused_deep_moe(
    x: torch.Tensor,                                  # [bs, hidden], bfloat16
    topk_idx: torch.Tensor,                           # [bs, num_topk], int64
    topk_weights: torch.Tensor,                       # [bs, num_topk], float32
    gmm1_permuted_weight: torch.Tensor,
    gmm1_permuted_weight_scale: torch.Tensor,
    gmm2_weight: torch.Tensor,
    gmm2_weight_scale: torch.Tensor,
    num_max_dispatch_tokens_per_rank: int,
    num_experts: int,
    quant_mode: int = 1,
    fuse_mode: FuseMode = FuseMode.FUSED_DEEP_MOE,
)
```

#### Parameters

| Parameter | Type | Shape | Description |
|:---|:---|:---|:---|
| `x` | `Tensor` (bfloat16) | `[bs, hidden]` | Input tokens. **bs** ∈ [1, 256]. **hidden** ∈ [512, 7168], divisible by 32. |
| `topk_idx` | `Tensor` (int64) | `[bs, num_topk]` | Expert indices (`-1` = none). |
| `topk_weights` | `Tensor` (float32) | `[bs, num_topk]` | Expert weights. |
| `gmm1_permuted_weight` | `Tensor` | `[G, 7168, 4096]` | First-stage (up-projection) weights, permuted for Grouped MatMul. |
| `gmm1_permuted_weight_scale` | `Tensor` (float32) | `[G, 4096]` | Quantization scale for first-stage weights. |
| `gmm2_weight` | `Tensor` | `[G, 7168, 2048]` | Second-stage (down-projection) weights. |
| `gmm2_weight_scale` | `Tensor` (float32) | `[G, 7168]` | Quantization scale for second-stage weights. |
| `num_max_dispatch_tokens_per_rank` | `int` | — | Max tokens per rank for buffer allocation. **FUSED_DEEP_MOE**: max tokens to dispatch per rank. **DISPATCH_FFN_COMBINE**: max tokens received after dispatch (max_output_size). |
| `num_experts` | `int` | — | Total expert count. |
| `quant_mode` | `int` | — (default `1`) | `1` = INT8 quantization. FP8 planned for A5. |
| `fuse_mode` | `FuseMode` | — (default `FUSED_DEEP_MOE`) | Fusion mode. See comparison table above. |

#### Returns

| Return | Type | Shape | Description |
|:---|:---|:---|:---|
| `output` | `Tensor` | `[bs, hidden]` | Fused expert output. |
| `ep_recv_count` | `Tensor` (int32) | `[num_local_experts]` | Token count per local expert in EP communication domain. |

---

<a id="中文"></a>

## 中文

> [!IMPORTANT]
> **仅 A3。** 此 API 在 A2 和 A5 上不可用。A5 FP8/MXFP8 计划在未来版本支持。

`fused_deep_moe` API 将 dispatch + 专家 FFN 计算 + combine 融合为单次算子调用，显著降低通信开销和端到端延迟。

两种融合模式通过 `FuseMode` 枚举选择，由 `fuse_mode` 参数控制。**两种模式共享同一个 Python 接口**——唯一区别是内部调用的 C++ kernel 不同。

### 融合模式对比

| | `FUSED_DEEP_MOE`（默认） | `DISPATCH_FFN_COMBINE` |
|:---|:---|:---|
| **FuseMode 值** | `1` | `2` |
| **调用方式** | `fuse_mode=FuseMode.FUSED_DEEP_MOE` 或省略（默认） | `fuse_mode=FuseMode.DISPATCH_FFN_COMBINE` |
| **C++ kernel** | `fused_deep_moe`（Act 库单融合 kernel） | `dispatch_ffn_combine`（Catlass 库） |
| **`num_max_dispatch_tokens_per_rank` 含义** | 每 rank 最大**分发** token 数 | dispatch 阶段最大**接收** token 数（即 `max_output_size`） |
| **内部结构** | 单 kernel：dispatch → GMM1 → SwiGLU → 量化/反量化 → GMM2 → combine | dispatch 分离处理，内部路由/缓冲策略不同 |
| **Scale tensor dtype** | `torch.float32`（原生） | `torch.int64`（float32 bitcast——Python 层自动处理转换） |
| **共享专家支持** | ✅ 支持（`share_expert_num`/`share_expert_rank_num`） | ❌ 不支持 |
| **何时选择** | 大多数场景的默认选择 | 实验性替代方案；可能对特定 batch 大小有不同的延迟表现 |

> [!NOTE]
> 大多数用户推荐使用 `FUSED_DEEP_MOE`（默认模式）。`DISPATCH_FFN_COMBINE` 是另一种实现，可能在特定负载模式下表现不同。如果默认模式遇到问题，可尝试 `DISPATCH_FFN_COMBINE` 作为备选。

### 参数

| 参数 | 类型 | 形状 | 说明 |
|:---|:---|:---|:---|
| `x` | `Tensor` (bfloat16) | `[bs, hidden]` | 输入 token。**bs** ∈ [1, 256]，**hidden** ∈ [512, 7168]，须为 32 的整数倍。 |
| `topk_idx` | `Tensor` (int64) | `[bs, num_topk]` | 专家索引，`-1` 不分发。 |
| `topk_weights` | `Tensor` (float32) | `[bs, num_topk]` | 专家权重。 |
| `gmm1_permuted_weight` | `Tensor` | `[G, 7168, 4096]` | 第一阶段权重（已 permute）。 |
| `gmm1_permuted_weight_scale` | `Tensor` (float32) | `[G, 4096]` | 第一阶段量化 scale。 |
| `gmm2_weight` | `Tensor` | `[G, 7168, 2048]` | 第二阶段权重。 |
| `gmm2_weight_scale` | `Tensor` (float32) | `[G, 7168]` | 第二阶段量化 scale。 |
| `num_max_dispatch_tokens_per_rank` | `int` | — | 每 rank 缓冲区分配最大 token 数。**FUSED_DEEP_MOE**：每 rank 最大分发 token 数。**DISPATCH_FFN_COMBINE**：dispatch 后最大接收 token 数（max_output_size）。 |
| `num_experts` | `int` | — | 全局专家总数。 |
| `quant_mode` | `int` | 默认 `1` | `1` = INT8 量化。FP8 计划未来 A5 版本支持。 |
| `fuse_mode` | `FuseMode` | 默认 `FUSED_DEEP_MOE` | 融合模式。见对比表。 |

### 返回值

| 返回值 | 类型 | 形状 | 说明 |
|:---|:---|:---|:---|
| `output` | `Tensor` | `[bs, hidden]` | 融合专家输出。 |
| `ep_recv_count` | `Tensor` (int32) | `[num_local_experts]` | EP 通信域各卡收到的 token 数。 |
