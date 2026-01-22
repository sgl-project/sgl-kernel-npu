> **文件**：`buffer.py`
> **核心类**：`Buffer`
> **依赖**：`torch`, `deep_ep_cpp`
> **目的**：在 **多 NPU（Intranode）** 与 **跨节点（Internode）** 环境下，高效完成 **Token Dispatch** 与 **Token Combine**（即分发‑归约）操作。

---

## `dispatch`

### 功能说明

将本地 token 按 **top‑k** 选择结果分发到其他 rank（包括同节点 intra‑node 与跨节点 inter‑node 两种模式），并返回收到的 token、对应的 top‑k 信息以及用于后续 **combine** 的通信句柄。

### 接口原型

```python
dispatch(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    handle: Optional[Tuple] = None,
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
) -> Tuple[
    Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    List[int],
    Tuple,
    EventOverlap
]
```

### 参数说明

| 参数 | 类型 | 必要 | 默认 | 说明 |
|------|------|------|------|------|
| **x** | `torch.Tensor` 或 `(torch.Tensor, torch.Tensor)` | ✅ | – | Shape为 `[num_tokens, hidden]`，dtype=`torch.bfloat16`。当前仅支持 `torch.Tensor` 类型。|
| **handle** | `Optional[Tuple]` | ❌ | `None` | 预先创建的通信句柄（目前仅支持 `None`）。|
| **num_tokens_per_rank** | `torch.Tensor` (`int32`) | ✅（intranode） | `None` | Shape为 `[num_ranks]`，每个 rank 将接收的 token 数。 |
| **num_tokens_per_rdma_rank** | `torch.Tensor` | ✅（internode） | `None` | Shape为 `[num_rdma_ranks]`，跨节点（RDMA）时每个 remote rank 接收的 token 数。 |
| **is_token_in_rank** | `torch.Tensor` (`int`) | ✅ | `None` | `[num_tokens, num_ranks]` 指明每个 token 是否需要发送到对应 rank。 |
| **num_tokens_per_expert** | `torch.Tensor` (`int`) | ✅ | `None` | `[num_experts]`，当前rank发送给每个expert的 token 数。 |
| **topk_idx** | `torch.Tensor` (`int64`) | ✅ | `None` | `[num_tokens, num_topk]`，每个 token 选中的 expert 索引，`-1` 表示无选中。 |
| **topk_weights** | `torch.Tensor` (`float`) | ✅ | `None` | `[num_tokens, num_topk]`，对应的权重。 |
| **expert_alignment** | `int` | ❌ | `1` | 对每个本地 expert 接收的 token 数进行对齐的粒度。 |
| **num_worst_tokens** | `int` | ❌ | `0` | 当前未使用。 |
| **config** | `deep_ep_cpp.Config` | ❌ | `None` | 当前未使用。 |
| **previous_event** | `EventOverlap` | ❌ | `None` | 在执行 kernel 前必须等待的前置事件。 |
| **async_finish** | `bool` | ❌ | `False` | 若 `True`，当前 stream 不会阻塞等待通信完成，返回的 `event` 可用于后续同步。 |
| **allocate_on_comm_stream** | `bool` | ❌ | `False` | 当前未使用。 |
| **dispatch_wait_recv_cost_stats** | `torch.Tensor` (`int64`) | ❌ | `None` | Shape为 `[num_ranks]`，记录当前 rank 从每个 rank 收到全部 token 所耗时间（统计信息）。 |

> **内部逻辑**
>
> 1. **模式判定**：`self.runtime.get_num_rdma_ranks() > 1` → **Internode**，否则 **Intranode**。
> 2. **返回的 `handle`**：内部保存了后续 `combine` 所需的所有索引/前缀矩阵等信息，**必须原样传递**给 `combine`。

### 返回值说明

| 返回值 | 类型 | 说明 |
|--------|------|------|
| **recv_x** | `torch.Tensor` 或 `(torch.Tensor, torch.Tensor)` | 接收到的 token。<br>若开启 int8 量化，则返回 `(int8_tensor, scales_float_tensor)`；否则直接返回 `bfloat16` tensor。 |
| **recv_topk_idx** | `Optional[torch.Tensor]` (`int64`) | 接收到的 top‑k expert 索引（形状 `[recv_token_cnt, num_topk]`），若未使用 top‑k 则为 `None`。 |
| **recv_topk_weights** | `Optional[torch.Tensor]` (`float`) | 对应的 top‑k 权重，形状同上。 |
| **num_recv_tokens_per_expert_list** | `List[int]` | 每个 **本地 expert** 实际收到的 token 数（已对齐）。<br>若 `num_worst_tokens>0`，列表为空（因为不做同步）。 |
| **handle** | `Tuple` | 供 `combine` 使用的通信句柄。 |
| **event** | `EventOverlap` | 若 `async_finish=True`，返回的 NPU 事件对象，可用于后续 `event.wait()` 同步。 |

### 约束说明

- 参数里Shape使用的变量如下：
    - num_tokens: 表示batch sequence size，即本卡输入输出的token数量。(当输入num_tokens=0时，会经过padding到1)
        - A2系列双机取值范围：(0, 4096]；单机取值范围：(0, 8192]；
        - A3系列取值范围，不开蚂蚁搬家：(0, 8192]，开蚂蚁搬家：(0, 32k]；
    - hidden: 表示hidden size隐藏层大小。
        - A2系列取值范围：(0, 7168]，且保证是32的整数倍；
        - A3系列取值范围：[1024, 7168]；
    - num_experts：表示专家数量，取值范围：(0, 512]。
    - num_topk：表示选取topk个专家。
        - A2系列双机取值范围：[2, 16]；单机取值范围：(0, 16]；
        - A3系列取值范围：(0, 16]。
- HCCL_BUFFSIZE: 调用接口前需检查HCCL_BUFFSIZE环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。
- HCCL_INTRA_PCIE_ENABLE和HCCL_INTRA_ROCE_ENABLE：
    - A2系列双机场景需要配置，`HCCL_INTRA_PCIE_ENABLE=1` 和 `HCCL_INTRA_ROCE_ENABLE=0`；
- 量化：设置环境变量 `DEEP_NORMAL_MODE_USE_INT8_QUANT=1` 时，会把 `x` 量化为 `int8` 并返回 `(tensor, scales)`。

---

## `combine`

### 功能说明

对 `dispatch` 之后收到的 token 进行 归约，即把同一 token 在不同 rank 上的副本整合（乘权重再相加）。

### 接口原型

```python
combine(
    x: torch.Tensor,
    handle: Tuple,
    topk_weights: Optional[torch.Tensor] = None,
    bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
    config: Optional[Config] = None,
    previous_event: Optional[EventOverlap] = None,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False,
    combine_send_cost_stats: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    EventOverlap
]
```

### 参数说明

| 参数 | 类型 | 必要 | 默认 | 说明 |
|------|------|------|------|------|
| **x** | `torch.Tensor` (`bfloat16`) | ✅ | – | 本 rank 需要发送回原始 rank 的 token，形状 `[num_tokens, hidden]`。 |
| **handle** | `Tuple` | ✅ | – | `dispatch` 返回的 **handle**（必须保持不变）。 |
| **topk_weights** | `torch.Tensor` (`float`) | ❌ | `None` | 若在 `dispatch` 时使用了 top‑k 权重，则在 combine 时把权重一起归约。 |
| **bias** | `torch.Tensor` 或 `(Tensor, Tensor)` | ❌ | `None` | 预留参数（目前未在实现里使用）。 |
| **config** | `deep_ep_cpp.Config` | ❌ | `None` | 性能调优配置，目前未使用。 |
| **previous_event** | `EventOverlap` | ❌ | `None` | 在执行 kernel 前需要等待的前置事件。 |
| **async_finish** | `bool` | ❌ | `False` | 同 `dispatch`，若为 `True`，返回的 `event` 用于手动同步。 |
| **allocate_on_comm_stream** | `bool` | ❌ | `False` | 是否把临时 tensor 放在通信 stream。 |
| **combine_send_cost_stats** | `torch.Tensor` (`int64`) | ❌ | `None` | 长度 `[num_ranks]`，记录本 rank 向其他 rank 发送所有 token 所耗时间（统计信息）。 |

### 返回值说明

| 返回值 | 类型 | 说明 |
|--------|------|------|
| **recv_x** | `torch.Tensor` (`bfloat16`) | 归约后的 token，形状 `[recv_token_cnt, hidden]`。 |
| **recv_topk_weights** | `Optional[torch.Tensor]` (`float`) | 若 `topk_weights` 不为 `None`，则返回归约后的权重；否则为 `None`。 |
| **event** | `EventOverlap` | 同 `dispatch`，仅在 `async_finish=True` 时有意义。 |

### 约束说明

- `dispatch`和`combine`必须配套使用。
- HCCL_BUFFSIZE: 调用接口前需检查HCCL_BUFFSIZE环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。
- HCCL_INTRA_PCIE_ENABLE和HCCL_INTRA_ROCE_ENABLE：
    - A2系列双机场景需要配置，`HCCL_INTRA_PCIE_ENABLE=1` 和 `HCCL_INTRA_ROCE_ENABLE=0`；
