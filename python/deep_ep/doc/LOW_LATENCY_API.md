# Low-Latency Mode API

<div align="center">

[![Mode](https://img.shields.io/badge/Mode-Low--Latency-orange)]()
[![Platform](https://img.shields.io/badge/Platform-A2%20%7C%20A3%20%7C%20A5-green)]()

English | [中文](#中文)

</div>

> **File**: `buffer.py`
> **Core class**: `Buffer`
> **Dependencies**: `torch`, `deep_ep_cpp`
> **Purpose**: Efficiently perform **low-latency Token Dispatch** and **Token Combine** (dispatch-reduce) operations in **multi-NPU (Intranode)** and **cross-node (Internode)** environments.

---

## `low_latency_dispatch`

### Interface

```python
def low_latency_dispatch(
    self,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    num_max_dispatch_tokens_per_rank: int,
    num_experts: int,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    use_fp8: bool = True,
    round_scale: bool = False,
    use_ue8m0: bool = False,
    use_mxfp4: bool = False,
    async_finish: bool = False,
    return_recv_hook: bool = False,
    topk_weights: Optional[torch.Tensor] = None,
    quant_mode: Optional[str] = None,
    use_mxfp8: bool = False,
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple, EventOverlap, Callable
]
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| **x** | `torch.Tensor` (`bfloat16`) | Yes | – | Shape `[num_tokens, hidden]`. The number of tokens to be dispatched must be less than `num_max_dispatch_tokens_per_rank`. A2: `0 < hidden <= 7168` and `hidden % 32 == 0`. |
| **topk_idx** | `torch.Tensor` (`int64`) | Yes | – | Shape `[num_tokens, num_topk]`, expert indices selected per token. `-1` means no expert selected. |
| **num_max_dispatch_tokens_per_rank** | `int` | Yes | – | Maximum number of tokens to dispatch per rank. All ranks must hold the same value. |
| **num_experts** | `int` | Yes | – | Total number of experts. |
| **cumulative_local_expert_recv_stats** | `Optional[torch.Tensor]` (`int`) | No | `None` | Shape `[num_local_experts]`, cumulative expert count for online EP load balance monitoring. Not needed on DeepEP-Ascend. |
| **use_fp8** | `bool` | No | `True` | When `quant_mode` is omitted, selects `pertoken_fp8_e4m3` on A5 and falls back to `int8` with a warning on A2/A3. |
| **round_scale** | `bool` | No | `False` | Whether to round scaling factors into powers of 2. Used together with `use_ue8m0`. |
| **use_ue8m0** | `bool` | No | `False` | Legacy MXFP8 selector. When `quant_mode` is omitted, `use_fp8=True, use_ue8m0=True` is treated as `use_mxfp8=True`. |
| **use_mxfp4** | `bool` | No | `False` | When `quant_mode` is omitted, selects `mx_fp4_e2m1` on A5. Raises `NotImplementedError` on A2/A3. |
| **async_finish** | `bool` | No | `False` | If set, the current stream will not wait for the communication kernel to finish. Not needed on DeepEP-Ascend. |
| **return_recv_hook** | `bool` | No | `False` | If set, returns a receiving hook. The kernel will only issue RDMA requests without actually receiving data; you must call the hook to ensure data arrival. Not needed on DeepEP-Ascend. |
| **topk_weights** | `Optional[torch.Tensor]` (`float`) | No | `None` | Top-k weights corresponding to `topk_idx`. |
| **quant_mode** | `Optional[str]` | No | `None` | Explicit quantization mode and highest-priority source for the `default` strategy. Supported values: `"bf16"`, `"int8"`, `"mx_fp8_e4m3"`, `"mx_fp8_e5m2"`, `"pertoken_fp8_e4m3"`, and `"mx_fp4_e2m1"`. Bool flags and the environment fallback are ignored when set. |
| **use_mxfp8** | `bool` | No | `False` | When `quant_mode` is omitted, selects `mx_fp8_e4m3` on A5. Raises `NotImplementedError` on A2/A3. Appended after `quant_mode` to preserve positional-call compatibility. |

### Return Values

| Return Value | Type | Description |
|--------------|------|-------------|
| **recv_x** | `Tuple[torch.Tensor, torch.Tensor]` or `torch.Tensor` | Received tokens. Format depends on the resolved mode:<br>- **BF16**: single tensor `[num_max_tokens, hidden]`, dtype `torch.bfloat16`.<br>- **INT8**: tuple `(int8_data, float32_scales)`, with one scale per token.<br>- **Per-token FP8** (`pertoken_fp8_e4m3`, A5): tuple `(float8_e4m3fn_data, float32_scales)`, with one scale per token.<br>- **MXFP8** (A5): tuple `(float8_data, float8_e8m0fnu_scales)`, with one scale per 32 elements.<br>- **MXFP4** (A5): tuple `(float4_e2m1fn_x2_data, float8_e8m0fnu_scales)`, with one scale per 32 elements.<br>Not all tokens are valid; only the first `recv_count` tokens per expert contain meaningful data. |
| **recv_count** | `torch.Tensor` (`int64`) | Shape `[num_local_experts]`, number of tokens each expert actually received. |
| **handle** | `Tuple` | Communication handle for `low_latency_combine`. Must be passed unchanged. |
| **event** | `EventOverlap` | Event after kernel execution (valid only if `async_finish=True`). Not needed on DeepEP-Ascend. |
| **hook** | `Callable` | Receiving hook function (valid only if `return_recv_hook=True`). Not needed on DeepEP-Ascend. |

### Constraints

- **num_tokens**: `num_tokens <= 512`, must be less than `num_max_dispatch_tokens_per_rank`.
- **hidden**: A2 series: `0 < hidden <= 7168` and `hidden % 32 == 0`.
- **num_topk**: A2 series internode: `[2, 16]`; intranode: `(0, 16]`; A3 series: `(0, 16]`.
- **num_experts**: `(0, 512]`.
- **HCCL_BUFFSIZE**: Check before calling. Default 200 MB. Minimum required size (non-layered): `(bs × ep_world_size × min(num_local_experts, topk) × hidden × 2B + 2MB) × 2`. For layered (A2 dual-node): `num_experts × bs × (hidden × 2B + 4 × topk × 4B) + 4MB + 800MB`. A5 subtracts 1MB state zone from the configured value.
- **HCCL_INTRA_PCIE_ENABLE / HCCL_INTRA_ROCE_ENABLE**: A2 series internode: set `HCCL_INTRA_PCIE_ENABLE=1` and `HCCL_INTRA_ROCE_ENABLE=0`.

### Quantization Selection Priority

For the `default` strategy, `Buffer._resolve_low_latency_quant_mode()` resolves the mode in this order:

1. Explicit `quant_mode`. It is validated and returned directly; bool flags and the environment variable are not consulted.
2. Bool flags in the order `use_mxfp4` > `use_mxfp8` > `use_fp8`:
   - `use_mxfp4=True`: A5 → `mx_fp4_e2m1`; A2/A3 → `NotImplementedError`.
   - `use_mxfp8=True`: A5 → `mx_fp8_e4m3`; A2/A3 → `NotImplementedError`.
   - `use_fp8=True`: A5 → `pertoken_fp8_e4m3`; A2/A3 → `int8` with a warning.
   - The legacy `use_fp8=True, use_ue8m0=True` combination is treated as `use_mxfp8=True`.
3. `DEEP_NORMAL_MODE_USE_INT8_QUANT=1`, as a deprecated compatibility fallback.
4. `None`, meaning BF16 without quantization.

`use_fp8` defaults to `True`, so callers must pass `use_fp8=False` before the environment-variable or BF16 fallback can be reached. Explicit `quant_mode="bf16"` is converted to the backend's `None` representation after resolution. The `ops` and `alltoall` strategies retain their legacy boolean behavior.

---

## `low_latency_combine`

### Interface

```python
def low_latency_combine(
    self,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    handle: tuple,
    zero_copy: bool = False,
    async_finish: bool = False,
    return_recv_hook: bool = False,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, EventOverlap, Callable]
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| **x** | `torch.Tensor` (`bfloat16`) | Yes | – | Shape `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]`. Local computed tokens to be sent back to the original rank and reduced. |
| **topk_idx** | `torch.Tensor` (`int64`) | Yes | – | Shape `[num_combined_tokens, num_topk]`. Expert indices selected by dispatched tokens. `-1` supported. `num_combined_tokens` equals the number of dispatched tokens. Must match dispatch indices. |
| **topk_weights** | `torch.Tensor` (`float`) | Yes | – | Shape `[num_combined_tokens, num_topk]`. Top-k weights for reduction. |
| **handle** | `Tuple` | Yes | – | Communication handle from the corresponding `low_latency_dispatch`. Must be passed unchanged. |
| **zero_copy** | `bool` | No | `False` | Whether the tensor is already in the RDMA buffer. Should be used with `get_next_low_latency_combine_buffer`. Not needed on DeepEP-Ascend. |
| **async_finish** | `bool` | No | `False` | If set, the current stream will not wait for the communication kernel to finish. Not needed on DeepEP-Ascend. |
| **return_recv_hook** | `bool` | No | `False` | If set, returns a receiving hook. The kernel will only issue RDMA requests without actually receiving data. Not needed on DeepEP-Ascend. |
| **out** | `Optional[torch.Tensor]` (`bfloat16`) | No | `None` | In-place output tensor. If set, the kernel writes the result directly to this tensor and returns it. Not needed on DeepEP-Ascend. |

### Return Values

| Return Value | Type | Description |
|--------------|------|-------------|
| **combined_x** | `torch.Tensor` (`bfloat16`) | Shape `[num_combined_tokens, hidden]`. Reduced token tensor (weighted sum across experts). |
| **event** | `EventOverlap` | Event after kernel execution (valid only if `async_finish=True`). Not needed on DeepEP-Ascend. |
| **hook** | `Callable` | Receiving hook function (valid only if `return_recv_hook=True`). Must be called to ensure data arrival. Not needed on DeepEP-Ascend. |

### Constraints

- `low_latency_dispatch` and `low_latency_combine` must be used together.
- **HCCL_BUFFSIZE**: Check before calling. Default 200 MB. Minimum required size (non-layered): `(bs × ep_world_size × min(num_local_experts, topk) × hidden × 2B + 2MB) × 2`. For layered (A2 dual-node): `num_experts × bs × (hidden × 2B + 4 × topk × 4B) + 4MB + 800MB`. A5 subtracts 1MB state zone from the configured value.
- **HCCL_INTRA_PCIE_ENABLE / HCCL_INTRA_ROCE_ENABLE**: A2 series internode: set `HCCL_INTRA_PCIE_ENABLE=1` and `HCCL_INTRA_ROCE_ENABLE=0`.

---

<a id="中文"></a>

## 中文

> **文件**：`buffer.py`
> **核心类**：`Buffer`
> **依赖**：`torch`, `deep_ep_cpp`
> **目的**：在 **多 NPU（Intranode）** 与 **跨节点（Internode）** 环境下，高效完成 低时延**Token Dispatch** 与 **Token Combine**（即分发/归约）操作。

---

## `low_latency_dispatch`

### 接口原型

```python
def low_latency_dispatch(
    self,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    num_max_dispatch_tokens_per_rank: int,
    num_experts: int,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    use_fp8: bool = True,
    round_scale: bool = False,
    use_ue8m0: bool = False,
    use_mxfp4: bool = False,
    async_finish: bool = False,
    return_recv_hook: bool = False,
    topk_weights: Optional[torch.Tensor] = None,
    quant_mode: Optional[str] = None,
    use_mxfp8: bool = False,
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple, EventOverlap, Callable
]
```

### 参数说明

| 参数 | 类型 | 必要 | 默认 | 说明 |
|------|------|------|------|------|
| **x** | `torch.Tensor` (`bfloat16`) | ✅ | – | 形状 `[num_tokens, hidden]`。要分发的 token 数必须小于 `num_max_dispatch_tokens_per_rank`。A2 算子实现要求 `0 < hidden <= 7168` 且 `hidden % 32 == 0`。 |
| **topk_idx** | `torch.Tensor` (`int64`) | ✅ | – | 形状 `[num_tokens, num_topk]`，每个 token 选中的 expert 索引。`-1` 表示不选择任何 expert。 |
| **num_max_dispatch_tokens_per_rank** | `int` | ✅ | – | 每个 rank 最大分发 token 数，所有 rank 必须相同。 |
| **num_experts** | `int` | ✅ | – | 专家总数。 |
| **cumulative_local_expert_recv_stats** | `Optional[torch.Tensor]` (`int`) | ❌ | `None` | 形状 `[num_local_experts]`，累计 expert 接收统计，用于在线 EP 负载均衡监控。DeepEP-Ascend 不需要。 |
| **use_fp8** | `bool` | ❌ | `True` | 未传 `quant_mode` 时，A5 上选择 `pertoken_fp8_e4m3`，A2/A3 上带 warning 回退到 `int8`。 |
| **round_scale** | `bool` | ❌ | `False` | 是否将缩放因子四舍五入为 2 的次幂。与 `use_ue8m0` 配合使用。 |
| **use_ue8m0** | `bool` | ❌ | `False` | 旧版 MXFP8 选择参数。未传 `quant_mode` 时，`use_fp8=True, use_ue8m0=True` 等价于 `use_mxfp8=True`。 |
| **use_mxfp4** | `bool` | ❌ | `False` | 未传 `quant_mode` 时，A5 上选择 `mx_fp4_e2m1`；A2/A3 上抛出 `NotImplementedError`。 |
| **async_finish** | `bool` | ❌ | `False` | 若设置，当前 stream 不会等待通信 kernel 完成。DeepEP-Ascend 不需要。 |
| **return_recv_hook** | `bool` | ❌ | `False` | 若设置，返回接收钩子；kernel 只发 RDMA 请求不接收数据，必须调用钩子确保数据到达。DeepEP-Ascend 不需要。 |
| **topk_weights** | `Optional[torch.Tensor]` (`float`) | ❌ | `None` | 对应 `topk_idx` 的 top-k 权重。 |
| **quant_mode** | `Optional[str]` | ❌ | `None` | `default` 策略的显式量化模式，也是最高优先级来源。支持 `"bf16"`、`"int8"`、`"mx_fp8_e4m3"`、`"mx_fp8_e5m2"`、`"pertoken_fp8_e4m3"` 和 `"mx_fp4_e2m1"`。设置后忽略布尔标志和环境变量回退。 |
| **use_mxfp8** | `bool` | ❌ | `False` | 未传 `quant_mode` 时，A5 上选择 `mx_fp8_e4m3`；A2/A3 上抛出 `NotImplementedError`。该参数追加在 `quant_mode` 之后，以保持位置参数调用兼容。 |

### 返回值说明

| 返回值 | 类型 | 说明 |
|--------|------|------|
| **recv_x** | `Tuple[torch.Tensor, torch.Tensor]` 或 `torch.Tensor` | 接收的 token，格式取决于解析后的模式：<br>- **BF16**：单个 `[num_max_tokens, hidden]` 的 `bfloat16` tensor。<br>- **INT8**：tuple `(int8_data, float32_scales)`，每个 token 一个 scale。<br>- **Per-token FP8**（`pertoken_fp8_e4m3`，仅 A5）：tuple `(float8_e4m3fn_data, float32_scales)`，每个 token 一个 scale。<br>- **MXFP8**（仅 A5）：tuple `(float8_data, float8_e8m0fnu_scales)`，每 32 个元素一个 scale。<br>- **MXFP4**（仅 A5）：tuple `(float4_e2m1fn_x2_data, float8_e8m0fnu_scales)`，每 32 个元素一个 scale。<br>并非所有 token 都有效，仅每个 expert 前 `recv_count` 个 token 含有意义数据。 |
| **recv_count** | `torch.Tensor` (`int64`) | 形状 `[num_local_experts]`，每个 expert 实际接收的 token 数。 |
| **handle** | `Tuple` | 供 `low_latency_combine` 使用的通信句柄，必须原样传递。 |
| **event** | `EventOverlap` | kernel 执行后的事件（仅在 `async_finish=True` 时有效）。DeepEP-Ascend 不需要。 |
| **hook** | `Callable` | 接收钩子函数（仅在 `return_recv_hook=True` 时有效）。DeepEP-Ascend 不需要。 |

### 约束说明

- **num_tokens**：`num_tokens <= 512`，必须小于 `num_max_dispatch_tokens_per_rank`。
- **hidden**：A2 算子实现要求 `0 < hidden <= 7168` 且 `hidden % 32 == 0`。
- **num_topk**：A2 系列双机 `[2, 16]`；单机 `(0, 16]`；A3 系列 `(0, 16]`。
- **num_experts**：`(0, 512]`。
- **HCCL_BUFFSIZE**：调用接口前需检查，默认 200 MB。非分层最小需求：`(bs × ep_world_size × min(num_local_experts, topk) × hidden × 2B + 2MB) × 2`；分层（A2双机）：`num_experts × bs × (hidden × 2B + 4 × topk × 4B) + 4MB + 800MB`。A5 从配置值中扣除 1MB 状态区。
- **HCCL_INTRA_PCIE_ENABLE / HCCL_INTRA_ROCE_ENABLE**：A2 系列双机场景需配置 `HCCL_INTRA_PCIE_ENABLE=1` 和 `HCCL_INTRA_ROCE_ENABLE=0`。

### 量化模式选择优先级

对于 `default` 策略，`Buffer._resolve_low_latency_quant_mode()` 按以下顺序解析：

1. 显式 `quant_mode`：校验后原样返回，不读取布尔标志和环境变量。
2. 按 `use_mxfp4` > `use_mxfp8` > `use_fp8` 的顺序处理布尔标志：
   - `use_mxfp4=True`：A5 → `mx_fp4_e2m1`；A2/A3 → `NotImplementedError`。
   - `use_mxfp8=True`：A5 → `mx_fp8_e4m3`；A2/A3 → `NotImplementedError`。
   - `use_fp8=True`：A5 → `pertoken_fp8_e4m3`；A2/A3 → 带 warning 的 `int8`。
   - 旧式组合 `use_fp8=True, use_ue8m0=True` 等价于 `use_mxfp8=True`。
3. `DEEP_NORMAL_MODE_USE_INT8_QUANT=1`，作为已弃用的兼容回退。
4. `None`，表示不量化的 BF16。

`use_fp8` 默认值为 `True`，因此调用方必须传入 `use_fp8=False` 才能进入环境变量或 BF16 回退。显式 `quant_mode="bf16"` 会在解析完成后转换为底层使用的 `None` 表示。`ops` 和 `alltoall` 策略保留原有的布尔参数行为。

---

## `low_latency_combine`

### 接口原型

```python
def low_latency_combine(
    self,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    handle: tuple,
    zero_copy: bool = False,
    async_finish: bool = False,
    return_recv_hook: bool = False,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, EventOverlap, Callable]
```

### 参数说明

| 参数 | 类型 | 必要 | 默认 | 说明 |
|------|------|------|------|------|
| **x** | `torch.Tensor` (`bfloat16`) | ✅ | – | 形状 `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]`。本地计算后需发送回原始 rank 进行 reduce 的 token。 |
| **topk_idx** | `torch.Tensor` (`int64`) | ✅ | – | 形状 `[num_combined_tokens, num_topk]`。dispatched token 选中的 expert 索引，支持 `-1`。`num_combined_tokens` 等于分发 token 数。必须与 dispatch 时的索引匹配。 |
| **topk_weights** | `torch.Tensor` (`float`) | ✅ | – | 形状 `[num_combined_tokens, num_topk]`。归约时使用的 top-k 权重。 |
| **handle** | `Tuple` | ✅ | – | 由对应 `low_latency_dispatch` 返回的通信句柄，必须原样传递。 |
| **zero_copy** | `bool` | ❌ | `False` | tensor 是否已在 RDMA 缓冲区，需与 `get_next_low_latency_combine_buffer` 配合使用。DeepEP-Ascend 不需要。 |
| **async_finish** | `bool` | ❌ | `False` | 若设置，当前 stream 不会等待通信 kernel 完成。DeepEP-Ascend 不需要。 |
| **return_recv_hook** | `bool` | ❌ | `False` | 若设置，返回接收钩子；kernel 只发 RDMA 请求不接收数据。DeepEP-Ascend 不需要。 |
| **out** | `Optional[torch.Tensor]` (`bfloat16`) | ❌ | `None` | 原地输出 tensor，若设置则 kernel 将结果直接写入此 tensor 并返回。DeepEP-Ascend 不需要。 |

### 返回值说明

| 返回值 | 类型 | 说明 |
|--------|------|------|
| **combined_x** | `torch.Tensor` (`bfloat16`) | 形状 `[num_combined_tokens, hidden]`。归约后的 token tensor（加权求和）。 |
| **event** | `EventOverlap` | kernel 执行后的事件（仅在 `async_finish=True` 时有效）。DeepEP-Ascend 不需要。 |
| **hook** | `Callable` | 接收钩子函数（仅在 `return_recv_hook=True` 时有效），必须调用以确保数据到达。DeepEP-Ascend 不需要。 |

### 约束说明

- `low_latency_dispatch` 和 `low_latency_combine` 必须配套使用。
- **HCCL_BUFFSIZE**：调用接口前需检查，默认 200 MB。非分层最小需求：`(bs × ep_world_size × min(num_local_experts, topk) × hidden × 2B + 2MB) × 2`；分层（A2双机）：`num_experts × bs × (hidden × 2B + 4 × topk × 4B) + 4MB + 800MB`。A5 从配置值中扣除 1MB 状态区。
- **HCCL_INTRA_PCIE_ENABLE / HCCL_INTRA_ROCE_ENABLE**：A2 系列双机场景需配置 `HCCL_INTRA_PCIE_ENABLE=1` 和 `HCCL_INTRA_ROCE_ENABLE=0`。
