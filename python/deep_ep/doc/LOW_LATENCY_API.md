# Low-Latency Mode API

<div align="center">

[![Mode](https://img.shields.io/badge/Mode-Low--Latency-orange)]()
[![Latency](https://img.shields.io/badge/Latency-<150us-red)]()
[![Platforms](https://img.shields.io/badge/Platforms-A2%20%7C%20A3%20%7C%20A5-green)]()

English | [中文](#中文)

</div>

Buffer class low_latency_dispatch/combine API for decode-phase inference with sub-150us latency.

---

## English

### `low_latency_dispatch`

Low-latency token dispatch for decode phase.

```python
recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
    x: torch.Tensor,                                  # [num_tokens, hidden], bfloat16
    topk_idx: torch.Tensor,                           # [num_tokens, num_topk], int64
    num_max_dispatch_tokens_per_rank: int,
    num_experts: int,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    use_fp8: bool = True,
    round_scale: bool = False,
    use_ue8m0: bool = False,
    async_finish: bool = False,
    return_recv_hook: bool = False,
    topk_weights: Optional[torch.Tensor] = None,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `x` | `Tensor` (bfloat16) `[num_tokens, hidden]` | — | Input tokens. Only `torch.Tensor` (no tuple). Quantization controlled by `use_fp8`/`use_ue8m0` flags. |
| `topk_idx` | `Tensor` (int64) `[num_tokens, num_topk]` | — | Expert indices (`-1` = none). |
| `num_max_dispatch_tokens_per_rank` | `int` | — | Max tokens per rank. All ranks must agree. |
| `num_experts` | `int` | — | Total expert count. |
| `use_fp8` | `bool` | `True` | Enable quantization. See [Quantization](#quantization-modes). |
| `round_scale` | `bool` | `False` | Round scales to power of 2. **Independent** from `use_ue8m0` — not required for MXFP8. |
| `use_ue8m0` | `bool` | `False` | Use E8M0 scale format (MXFP8 per-block). **A5/C310 only.** Requires `use_fp8=True`. Does NOT require `round_scale=True`. |

<details>
<summary><b>Platform-specific Parameters</b></summary>

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `cumulative_local_expert_recv_stats` | `Optional[Tensor]` `[num_local_experts]` (int) | `None` | Expert receive stats for load balancing monitoring. Unused on Ascend. |
| `async_finish` | `bool` | `False` | No-op on Ascend. |
| `return_recv_hook` | `bool` | `False` | No-op on Ascend. |
| `topk_weights` | `Optional[Tensor]` | `None` | Required when using ops strategy `comm_alg="hierarchy"`. |

</details>

#### Returns

| Return | Type | Description |
|:---|:---|:---|
| `recv_x` | `Tensor` or `(Tensor, Tensor)` | Received tokens. See [Quantization](#quantization-modes). |
| `recv_count` | `Tensor` (int) `[num_local_experts]` | Valid token count per expert. |
| `handle` | `Tuple` | Communication handle for `low_latency_combine`. |
| `event` | `EventOverlap` | NPU event (no-op on Ascend). |
| `hook` | `Callable` | Receive hook (no-op on Ascend). |

---

### `low_latency_combine`

Reduces tokens from low_latency_dispatch across ranks.

```python
combined_x, event, hook = buffer.low_latency_combine(
    x: torch.Tensor,          # [num_local_experts, num_max * num_ranks, hidden], bfloat16
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    handle: tuple,
    zero_copy: bool = False,
    async_finish: bool = False,
    return_recv_hook: bool = False,
    out: Optional[torch.Tensor] = None,
)
```

#### Returns

| Return | Type | Description |
|:---|:---|:---|
| `combined_x` | `Tensor` (bfloat16) `[num_combined_tokens, hidden]` | Reduced tokens. |
| `event` | `EventOverlap` | NPU event (no-op on Ascend). |
| `hook` | `Callable` | Receive hook (no-op on Ascend). |

<details>
<summary><b>Parameters (no-op on Ascend)</b></summary>

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `zero_copy` | `bool` | `False` | No-op on Ascend. |
| `async_finish` | `bool` | `False` | No-op on Ascend. |
| `return_recv_hook` | `bool` | `False` | No-op on Ascend. |
| `out` | `Optional[Tensor]` | `None` | No-op on Ascend. |

</details>

---

### Quantization Modes

Three quantization modes controlled by `use_fp8` and `use_ue8m0` flags:

| Mode | quant_mode | Flags | Output Format | Platform |
|:---|:---:|:---|:---|:---|
| **BF16** (no quant) | `0` | `use_fp8=False` | `bfloat16` `[experts, max_tokens × ranks, hidden]` | A2 ✅ A3 ✅ A5 ✅ |
| **INT8 per-token** | `2` | `use_fp8=True, use_ue8m0=False` (**default**) | `(int8_tensor, float32_scales)` — scales `[experts, max_tokens × ranks]` | A2 ✅ A3 ✅ A5 ✅ |
| **MXFP8 per-block** | `3` | `use_fp8=True, use_ue8m0=True` | `(float8_e4m3fn, float8_e8m0fnu)` — scales packed `[experts, max_tokens × ranks, hidden // 512]` (int) | A2 ❌ A3 ❌ **A5 ✅** |

> [!NOTE]
> Despite the name `use_fp8`, the **default mode** (`use_fp8=True, use_ue8m0=False`) produces **INT8 per-token** data, not FP8. The quantization formula is: `quant_mode = 3 if use_ue8m0 else 2 if use_fp8 else 0`.

> [!IMPORTANT]
> `use_ue8m0` does NOT require `round_scale=True`. These are independent parameters — `round_scale` rounds scales to powers of 2 for specific hardware optimizations, but MXFP8 per-block works regardless of `round_scale` setting.

---

### Strategy & `comm_alg` Options

| Strategy | `DEEP_USE_MODE` | Platforms | Notes |
|:---|:---|:---|:---|
| DefaultLowLatencyCommStrategy | `default` | A2, A3, A5 | Custom ops. Internally: A2=fullmesh, A3=fullmesh_v1, A5=ccu/fullmesh_v1. |
| OpsLowLatencyCommStrategy | `ops` | A2, A3, A5 | torch_npu ops. Supports `comm_alg`. |
| AllToAllLowLatencyCommStrategy | `alltoall` | A2, A3, A5 | torch.distributed alltoall. |

**`comm_alg` for ops strategy:**

| `comm_alg` | A2 | A3 | A5 | Notes |
|:---|:---:|:---:|:---:|:---|
| `""` | ✅ | ✅ | ✅ | Default |
| `hierarchy` | ❌ | ✅ | ❌ | A3 only. Requires `topk_weights`. |
| `fullmesh_v1` | ✅ | ✅ | ✅ | Standard full-mesh. |
| `fullmesh_v2` | ❌ | ✅ | ❌ | A3 only. Has restrictions. |
| `ccu` | ❌ | — | ✅ | A5 CCU engine. |

> [!WARNING]
> `comm_alg` is hardcoded to `"hierarchy"` in `Buffer.__init__`. To use a different `comm_alg`, modify the source code or pass a `low_latency_strategy` parameter — though note that `Buffer.__init__` currently overrides strategy parameters with `DEEP_USE_MODE`.

---

### Platform Constraints

| | A2 | A3 | A5 (C310) |
|:---|:---|:---|:---|
| `num_tokens` | ≤ 512 | ≤ 512 | ≤ 512 |
| `hidden` | (0, 7168], ÷ 32 | [1024, 7168] | — |
| `num_experts` | (0, 512] | (0, 512] | (0, 512] |
| INT8 per-token | ✅ | ✅ | ✅ |
| MXFP8 per-block | ❌ | ❌ | ✅ |

---

<a id="中文"></a>

## 中文

### `low_latency_dispatch`

Decode 阶段的低时延 token 分发。

```python
recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
    x: torch.Tensor,                                  # [num_tokens, hidden], bfloat16
    topk_idx: torch.Tensor,                           # [num_tokens, num_topk], int64
    num_max_dispatch_tokens_per_rank: int,
    num_experts: int,
    use_fp8: bool = True,
    round_scale: bool = False,
    use_ue8m0: bool = False,
)
```

#### 参数

| 参数 | 类型 | 默认 | 说明 |
|:---|:---|:---|:---|
| `x` | `Tensor` (bfloat16) `[num_tokens, hidden]` | — | 输入 token，仅 Tensor（不支持 tuple）。量化由 `use_fp8`/`use_ue8m0` 控制。 |
| `topk_idx` | `Tensor` (int64) `[num_tokens, num_topk]` | — | 专家索引，`-1` 无选中。 |
| `num_max_dispatch_tokens_per_rank` | `int` | — | 每 rank 最大 token 数，所有 rank 一致。 |
| `num_experts` | `int` | — | 专家总数。 |
| `use_fp8` | `bool` | `True` | 启用量化。详见[量化模式](#量化模式-1)。 |
| `round_scale` | `bool` | `False` | 缩放因子四舍五入为 2 的幂。**独立于** `use_ue8m0`——MXFP8 不需要此参数。 |
| `use_ue8m0` | `bool` | `False` | E8M0 格式缩放因子 → MXFP8 per-block（**仅 A5/C310**）。需 `use_fp8=True`，**不需要** `round_scale=True`。 |

<details>
<summary><b>平台相关参数</b></summary>

| 参数 | 类型 | 默认 | 说明 |
|:---|:---|:---|:---|
| `cumulative_local_expert_recv_stats` | `Optional[Tensor]` | `None` | 负载均衡统计，Ascend 上未使用。 |
| `async_finish` | `bool` | `False` | Ascend 上无实际作用。 |
| `return_recv_hook` | `bool` | `False` | Ascend 上无实际作用。 |
| `topk_weights` | `Optional[Tensor]` | `None` | ops 策略 `comm_alg="hierarchy"` 时必填。 |

</details>

#### 返回值

| 返回值 | 类型 | 说明 |
|:---|:---|:---|
| `recv_x` | `Tensor` 或 `(Tensor, Tensor)` | 接收 token。格式见[量化模式](#量化模式-1)。 |
| `recv_count` | `Tensor` (int) `[num_local_experts]` | 每个专家的有效 token 数。 |
| `handle` | `Tuple` | 供 `low_latency_combine` 使用。 |

---

### `low_latency_combine`

对 low_latency_dispatch 返回的 token 进行归约。

#### 返回值

| 返回值 | 类型 | 说明 |
|:---|:---|:---|
| `combined_x` | `Tensor` (bfloat16) `[num_combined_tokens, hidden]` | 归约后的 token。 |

---

### 量化模式

三种量化模式由 `use_fp8` 和 `use_ue8m0` 控制：

| 模式 | quant_mode | 标志 | 输出格式 | 平台 |
|:---|:---:|:---|:---|:---|
| **BF16**（不量化） | `0` | `use_fp8=False` | `bfloat16` `[experts, max_tokens × ranks, hidden]` | A2 ✅ A3 ✅ A5 ✅ |
| **INT8 per-token** | `2` | `use_fp8=True, use_ue8m0=False`（**默认**） | `(int8_tensor, float32_scales)` — scales `[experts, max_tokens × ranks]` | A2 ✅ A3 ✅ A5 ✅ |
| **MXFP8 per-block** | `3` | `use_fp8=True, use_ue8m0=True` | `(float8_e4m3fn, float8_e8m0fnu)` — scales packed `[experts, max_tokens × ranks, hidden // 512]` | A2 ❌ A3 ❌ **A5 ✅** |

> [!NOTE]
> 尽管参数名 `use_fp8`，**默认模式**（`use_fp8=True, use_ue8m0=False`）产出的是 **INT8 per-token** 数据，不是 FP8。量化公式为：`quant_mode = 3 if use_ue8m0 else 2 if use_fp8 else 0`。

> [!IMPORTANT]
> `use_ue8m0` **不需要** `round_scale=True`。这两个参数是独立的——`round_scale` 将缩放因子四舍五入为 2 的幂用于特定硬件优化，但 MXFP8 per-block 无论 `round_scale` 设置如何均可工作。

---

### 策略与 `comm_alg` 选项

| 策略 | `DEEP_USE_MODE` | 平台 | 说明 |
|:---|:---|:---|:---|
| DefaultLowLatencyCommStrategy | `default` | A2/A3/A5 | 自定义算子。内部：A2=fullmesh, A3=fullmesh_v1, A5=ccu。 |
| OpsLowLatencyCommStrategy | `ops` | A2/A3/A5 | torch_npu 算子，支持 `comm_alg`。 |
| AllToAllLowLatencyCommStrategy | `alltoall` | A2/A3/A5 | torch.distributed alltoall。 |

**ops 策略 `comm_alg`：**

| `comm_alg` | A2 | A3 | A5 | 说明 |
|:---|:---:|:---:|:---:|:---|
| `""` | ✅ | ✅ | ✅ | 默认 |
| `hierarchy` | ❌ | ✅ | ❌ | 仅 A3，需 `topk_weights`。 |
| `fullmesh_v1` | ✅ | ✅ | ✅ | 标准 full-mesh。 |
| `fullmesh_v2` | ❌ | ✅ | ❌ | 仅 A3，有额外限制。 |
| `ccu` | ❌ | — | ✅ | A5 CCU 引擎。 |

> [!WARNING]
> `comm_alg` 在 `Buffer.__init__` 中硬编码为 `"hierarchy"`。要使用其他 `comm_alg` 需修改源码或传入 `low_latency_strategy` 参数——但注意 `Buffer.__init__` 目前会用 `DEEP_USE_MODE` 覆盖策略参数。

---

### 平台约束

| | A2 | A3 | A5 (C310) |
|:---|:---|:---|:---|
| `num_tokens` | ≤ 512 | ≤ 512 | ≤ 512 |
| `hidden` | (0, 7168]，÷ 32 | [1024, 7168] | — |
| `num_experts` | (0, 512] | (0, 512] | (0, 512] |
| INT8 per-token | ✅ | ✅ | ✅ |
| MXFP8 per-block | ❌ | ❌ | ✅ |
