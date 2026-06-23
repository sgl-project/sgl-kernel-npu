# Normal Mode API

<div align="center">

[![Mode](https://img.shields.io/badge/Mode-Normal-blue)]()
[![Platforms](https://img.shields.io/badge/Platforms-A2%20%7C%20A3%20%7C%20A5-green)]()

English | [中文](#中文)

</div>

Buffer class dispatch/combine API for high-throughput token distribution in training and prefill phases.

---

## English

### `dispatch`

Dispatches local tokens to other ranks based on top-k routing, returns received tokens and a handle for `combine`.

```python
recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens, handle, event = buffer.dispatch(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    num_tokens_per_rank: Optional[torch.Tensor] = None,
    num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
    is_token_in_rank: Optional[torch.Tensor] = None,
    num_tokens_per_expert: Optional[torch.Tensor] = None,
    topk_idx: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    expert_alignment: int = 1,
    num_worst_tokens: int = 0,
    config: Optional[Config] = None,
    previous_event: Optional[EventOverlap] = None,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False,
    dispatch_wait_recv_cost_stats: Optional[torch.Tensor] = None,
)
```

#### Parameters

| Parameter | Type | Required | Description |
|:---|:---|:---:|:---|
| `x` | `Tensor` or `(Tensor, Tensor)` | ✅ | Input tokens. Plain `bfloat16` tensor for BF16 mode; tuple for quantized mode — see [Quantization](#quantization-modes). |
| `num_tokens_per_rank` | `Tensor` (int32) `[num_ranks]` | ✅* | Token count per destination rank. |
| `num_tokens_per_rdma_rank` | `Tensor` | ✅* | Token count per RDMA rank. |
| `is_token_in_rank` | `Tensor` (int) `[num_tokens, num_ranks]` | ✅ | Whether each token is sent to each rank. |
| `num_tokens_per_expert` | `Tensor` (int) `[num_experts]` | ✅ | Token count per expert. |
| `topk_idx` | `Tensor` (int64) `[num_tokens, num_topk]` | ✅ | Expert indices (`-1` = none). |
| `topk_weights` | `Tensor` (float) `[num_tokens, num_topk]` | ✅ | Expert weights per token. |
| `expert_alignment` | `int` | — | Alignment granularity (default `1`). |

<details>
<summary><b>Advanced Parameters</b></summary>

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `num_worst_tokens` | `int` | `0` | Currently unused. |
| `config` | `Config` | `None` | Performance tuning config (currently unused). |
| `previous_event` | `EventOverlap` | `None` | Event to wait before kernel execution. |
| `async_finish` | `bool` | `False` | If True, stream won't block on completion. |
| `allocate_on_comm_stream` | `bool` | `False` | Currently unused. |
| `dispatch_wait_recv_cost_stats` | `Tensor` (int64) `[num_ranks]` | `None` | Per-rank receive timing statistics. |

</details>

*Required for intranode / internode routing respectively.

#### Returns

| Return | Type | Description |
|:---|:---|:---|
| `recv_x` | `Tensor` or `(Tensor, Tensor)` | Received tokens. See [Quantization](#quantization-modes) for format. |
| `recv_topk_idx` | `Optional[Tensor]` (int64) | Received top-k indices. |
| `recv_topk_weights` | `Optional[Tensor]` (float) | Received top-k weights. |
| `num_recv_tokens_per_expert_list` | `List[int]` | Per-local-expert received token count. |
| `handle` | `Tuple` | Communication handle — **must pass unchanged to `combine`**. |
| `event` | `EventOverlap` | NPU event for async synchronization. |

#### Internal routing

`DefaultNormalCommStrategy` auto-selects:
- **Intranode** when `num_rdma_ranks <= 1` (A3, A5, A2 single)
- **Internode** when `num_rdma_ranks > 1` (A2 dual-node)

> [!WARNING]
> `AlltoAllNormalCommStrategy` (DEEP_USE_MODE=alltoall) does NOT support tuple quantization input. When using alltoall strategy, only plain `bfloat16` tensor or INT8 via `DEEP_NORMAL_MODE_USE_INT8_QUANT` env var is supported.

---

### `combine`

Reduces tokens from dispatch across ranks (weighted addition).

```python
recv_x, recv_topk_weights, event = buffer.combine(
    x: torch.Tensor,          # [num_tokens, hidden], bfloat16
    handle: Tuple,            # from dispatch — must be unchanged
    topk_weights: Optional[torch.Tensor] = None,
    bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
    config: Optional[Config] = None,
    previous_event: Optional[EventOverlap] = None,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False,
    combine_send_cost_stats: Optional[torch.Tensor] = None,
)
```

#### Returns

| Return | Type | Description |
|:---|:---|:---|
| `recv_x` | `Tensor` (bfloat16) `[recv_token_cnt, hidden]` | Reduced tokens. |
| `recv_topk_weights` | `Optional[Tensor]` (float) | Reduced top-k weights. |
| `event` | `EventOverlap` | NPU event for async synchronization. |

---

### Quantization Modes

| Mode | quant_mode | Trigger | Output Format | Platform |
|:---|:---:|:---|:---|:---|
| **BF16** (no quant) | `0` | Plain `bfloat16` tensor as `x` | `bfloat16` tensor `[recv_tokens, hidden]` | A2 ✅ A3 ✅ A5 ✅ |
| **INT8 per-token** | `2` | Tuple `(bf16_data, int8_empty_tensor)` or env var `DEEP_NORMAL_MODE_USE_INT8_QUANT=1` ⛔ deprecated | `(int8_tensor, float32_scales)` — scales `[recv_tokens]` | A2 ✅ A3 ✅ A5 ✅ |
| **MXFP8 per-block** | `3` | Tuple `(bf16_data, fp8_e4m3fn_empty_tensor)` — **second element dtype determines quant type** | `(float8_e4m3fn, float8_e8m0fnu_scales)` — scales `[recv_tokens, hidden // 32]` | A2 ❌ A3 ❌ **A5 ✅** |

> [!IMPORTANT]
> The second tuple element is a **type discriminator** — an empty tensor whose dtype selects the quantization mode:
> - `torch.int8` → INT8 per-token (quant_mode=2)
> - `torch.float8_e4m3fn` → MXFP8 per-block (quant_mode=3)
>
> The first element is always **BF16 data** — the kernel performs quantization internally.

> [!WARNING]
> - A2 dual-node internode does NOT support any quantization in normal mode.
> - MXFP8 per-block tuple input is only supported in **intranode** dispatch (not internode).
> - AlltoAll strategy does NOT support tuple quantization input.

**INT8 tuple input example:**
```python
x_tuple = (x_bf16, torch.tensor([], dtype=torch.int8, device="npu"))
recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens, handle, event = buffer.dispatch(x=x_tuple, ...)
# recv_x is a tuple: (int8_tensor, float32_scales)
```

**MXFP8 tuple input example:**
```python
x_tuple = (x_bf16, torch.tensor([], dtype=torch.float8_e4m3fn, device="npu"))
recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens, handle, event = buffer.dispatch(x=x_tuple, ...)
# recv_x is a tuple: (float8_e4m3fn_data, float8_e8m0fnu_scales)
```

---

### Platform Constraints

| Parameter | A2 Single | A2 Dual | A3 | A5 (C310) |
|:---|:---|:---|:---|:---|
| `num_tokens` | (0, 8192] | (0, 4096] | (0, 8192] / (0, 32k] long-seq | — |
| `hidden` | (0, 7168], ÷ 32 | (0, 7168], ÷ 32 | [1024, 7168] | — |
| `num_topk` | (0, 16] | [2, 16] | (0, 16] | — |
| `num_experts` | (0, 512] | (0, 512] | (0, 512] | (0, 512] |
| Quantization (intranode) | INT8 ✅ | ❌ None | INT8 ✅, MXFP8 ✅ | INT8 ✅, MXFP8 ✅ |
| Quantization (internode) | INT8 ✅ | ❌ None | INT8 ✅ | — |
| Strategies | default | default | default, alltoall | default, alltoall |

---

<a id="中文"></a>

## 中文

### `dispatch`

将本地 token 按 top-k 路由分发到其他 rank，返回接收到的 token 和供 `combine` 使用的通信句柄。

```python
recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens, handle, event = buffer.dispatch(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    num_tokens_per_rank: Optional[torch.Tensor] = None,
    num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
    is_token_in_rank: Optional[torch.Tensor] = None,
    num_tokens_per_expert: Optional[torch.Tensor] = None,
    topk_idx: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    expert_alignment: int = 1,
)
```

#### 参数

| 参数 | 类型 | 必要 | 说明 |
|:---|:---|:---:|:---|
| `x` | `Tensor` 或 `(Tensor, Tensor)` | ✅ | 输入 token。普通 `bfloat16` tensor 用于 BF16 模式；tuple 用于量化模式——见[量化模式](#量化模式-1)。 |
| `num_tokens_per_rank` | `Tensor` (int32) `[num_ranks]` | ✅* | 每个目标 rank 接收的 token 数。 |
| `num_tokens_per_rdma_rank` | `Tensor` | ✅* | 每个 RDMA rank 接收的 token 数。 |
| `is_token_in_rank` | `Tensor` (int) `[num_tokens, num_ranks]` | ✅ | 每个 token 是否发送到对应 rank。 |
| `num_tokens_per_expert` | `Tensor` (int) `[num_experts]` | ✅ | 每个专家的 token 数。 |
| `topk_idx` | `Tensor` (int64) `[num_tokens, num_topk]` | ✅ | 专家索引，`-1` 表示无选中。 |
| `topk_weights` | `Tensor` (float) `[num_tokens, num_topk]` | ✅ | 专家权重。 |
| `expert_alignment` | `int` | — | 对齐粒度（默认 `1`）。 |

<details>
<summary><b>高级参数</b></summary>

| 参数 | 类型 | 默认 | 说明 |
|:---|:---|:---|:---|
| `num_worst_tokens` | `int` | `0` | 当前未使用。 |
| `async_finish` | `bool` | `False` | 若 True，不阻塞等待通信完成。 |

</details>

*intranode / internode 路由分别必填。

#### 返回值

| 返回值 | 类型 | 说明 |
|:---|:---|:---|
| `recv_x` | `Tensor` 或 `(Tensor, Tensor)` | 接收的 token。格式见[量化模式](#量化模式-1)。 |
| `recv_topk_idx` | `Optional[Tensor]` (int64) | 接收的 top-k 索引。 |
| `handle` | `Tuple` | 通信句柄，**必须原样传给 `combine`**。 |

#### 内部路由

`DefaultNormalCommStrategy` 自动选择：
- **Intranode**：`num_rdma_ranks <= 1`（A3、A5、A2 单机）
- **Internode**：`num_rdma_ranks > 1`（A2 双机）

> [!WARNING]
> `AlltoAllNormalCommStrategy`（DEEP_USE_MODE=alltoall）**不支持** tuple 量化输入。使用 alltoall 策略时，仅支持普通 `bfloat16` tensor 或通过 `DEEP_NORMAL_MODE_USE_INT8_QUANT` 环境变量启用 INT8。

---

### `combine`

对 dispatch 返回的 token 进行归约（加权相加）。

#### 返回值

| 返回值 | 类型 | 说明 |
|:---|:---|:---|
| `recv_x` | `Tensor` (bfloat16) `[recv_token_cnt, hidden]` | 归约后的 token。 |
| `recv_topk_weights` | `Optional[Tensor]` (float) | 归约后的权重。 |

---

### 量化模式

| 模式 | quant_mode | 触发方式 | 输出格式 | 平台 |
|:---|:---:|:---|:---|:---|
| **BF16**（不量化） | `0` | 普通 `bfloat16` tensor | `bfloat16` tensor `[recv_tokens, hidden]` | A2 ✅ A3 ✅ A5 ✅ |
| **INT8 per-token** | `2` | tuple `(bf16_data, int8空tensor)` 或环境变量 `DEEP_NORMAL_MODE_USE_INT8_QUANT=1` ⛔ 已弃用 | `(int8_tensor, float32_scales)` — scales `[recv_tokens]` | A2 ✅ A3 ✅ A5 ✅ |
| **MXFP8 per-block** | `3` | tuple `(bf16_data, fp8_e4m3fn空tensor)` — **第二个元素 dtype 决定量化类型** | `(float8_e4m3fn, float8_e8m0fnu_scales)` — scales `[recv_tokens, hidden // 32]` | A2 ❌ A3 ❌ **A5 ✅** |

> [!IMPORTANT]
> 第二个 tuple 元素是**类型标识**——空 tensor，其 dtype 选择量化模式：
> - `torch.int8` → INT8 per-token（quant_mode=2）
> - `torch.float8_e4m3fn` → MXFP8 per-block（quant_mode=3）
>
> 第一个元素始终是 **BF16 数据**——内核内部执行量化。

> [!WARNING]
> - A2 双机 internode 在 normal 模式下**不支持任何量化**。
> - MXFP8 per-block tuple 输入仅在 **intranode** dispatch 支持（不支持 internode）。
> - AlltoAll 策略**不支持** tuple 量化输入。

**INT8 tuple 示例：**
```python
x_tuple = (x_bf16, torch.tensor([], dtype=torch.int8, device="npu"))
recv_x, ..., handle, event = buffer.dispatch(x=x_tuple, ...)
# recv_x 为 tuple: (int8_tensor, float32_scales)
```

**MXFP8 tuple 示例：**
```python
x_tuple = (x_bf16, torch.tensor([], dtype=torch.float8_e4m3fn, device="npu"))
recv_x, ..., handle, event = buffer.dispatch(x=x_tuple, ...)
# recv_x 为 tuple: (float8_e4m3fn_data, float8_e8m0fnu_scales)
```

---

### 平台约束

| 参数 | A2 单机 | A2 双机 | A3 | A5 (C310) |
|:---|:---|:---|:---|:---|
| `num_tokens` | (0, 8192] | (0, 4096] | (0, 8192] / (0, 32k] 长序列 | — |
| `hidden` | (0, 7168]，÷ 32 | (0, 7168]，÷ 32 | [1024, 7168] | — |
| `num_topk` | (0, 16] | [2, 16] | (0, 16] | — |
| `num_experts` | (0, 512] | (0, 512] | (0, 512] | (0, 512] |
| 量化（intranode） | INT8 ✅ | ❌ 不支持 | INT8 ✅，MXFP8 ✅ | INT8 ✅，MXFP8 ✅ |
| 量化（internode） | INT8 ✅ | ❌ 不支持 | INT8 ✅ | — |
| 策略 | default | default | default, alltoall | default, alltoall |
