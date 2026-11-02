# get_dispatch_layout API

<div align="center">

[![Mode](https://img.shields.io/badge/Mode-Normal-blue)]()
[![Platform](https://img.shields.io/badge/Platform-A2%20%7C%20A3%20%7C%20A5-green)]()

English | [中文](#中文)

</div>

> **File**: `buffer.py`
> **Core class**: `Buffer`
> **Dependencies**: `torch`, `deep_ep_cpp`
> **Purpose**: Data preprocessing before dispatch and combine in Normal mode.

---

## `get_dispatch_layout`

### Description

Computes the local copy of parameters needed by subsequent Dispatch and Combine operations in Normal mode, based on the input `topk_idx`.

### Interface

```python
def get_dispatch_layout(
        self,
        topk_idx: torch.Tensor,
        num_experts: int,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, EventOverlap
    ]:
```

```cpp
std::tuple<
torch::Tensor,                      // num_tokens_per_rank
std::optional[torch::Tensor](torch::Tensor),      // num_tokens_per_rdma_rank (reserved)
torch::Tensor,                      // num_tokens_per_expert
torch::Tensor,                      // is_token_in_rank
std::optional<EventHandle>         // output_event (currently unused)
>

Buffer::get_dispatch_layout(
const torch::Tensor& topk_idx,
int num_experts,
std::optional<EventHandle>& previous_event,
bool async,
bool allocate_on_comm_stream
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `topk_idx` | `torch::Tensor` (`int64`, `[num_tokens, num_topk]`) | Top-k expert indices per token (must be a contiguous 2D tensor). Second dimension range: [1, 16]. |
| `num_experts` | `int` | Total number of experts. Range: [1, 512], must be divisible by `num_ranks`. |
| `previous_event` | `std::optional<EventHandle>&` | Pre-event for async execution (currently unused, pass `std::nullopt`). |
| `async` | `bool` | Whether to enable async mode (currently unused). |
| `allocate_on_comm_stream` | `bool` | Whether to allocate memory on the communication stream (currently unused). |

### Constraints

- **num_tokens**: Batch sequence size, i.e., the number of input/output tokens on this rank. Reflected as the first dimension of `topk_idx`.
    - A2 series internode: (0, 4096]; intranode: (0, 8192];
    - A3 series: without "ant moving home": (0, 8192], with "ant moving home": (0, 32k];

### Return Values

| Return Value | Type | Description |
|--------------|------|-------------|
| `num_tokens_per_rank` | `torch::Tensor` (`int32`, `[num_ranks]`) | Number of tokens assigned to each rank. |
| `num_tokens_per_rdma_rank` | `std::optional<torch::Tensor>` | Reserved field, currently always `std::nullopt`. |
| `num_tokens_per_expert` | `torch::Tensor` (`int32`, `[num_experts]`) | Number of tokens received by each expert. |
| `is_token_in_rank` | `torch::Tensor` (`bool`, `[num_tokens, num_ranks]`) | Indicates whether each token belongs to a given rank. |
| `output_event` | `std::optional<EventHandle>` | Reserved field, currently `std::nullopt`. |

### Internal Logic

1. Copy `topk_idx` to each core's UB buffer; first pass computes partial parameters;
2. Use DataCopy to move results to GM, aggregate data across cores via atomic add or address-partitioned transfer;
3. Move partial GM results (e.g., prefix sums) back to UB; second pass computes remaining parameters;
4. Move final result tensors back to GM.

### Multi-Core Strategy

- Partition the 0th dimension of `topk_idx` (i.e., token count) as evenly as possible across cores. If token count < core count, only the first `num_tokens` cores are used, achieving data parallelism.

### Example Usage

```cpp
auto topk_idx = torch::randint(0, 256, {4096, 8}, torch::dtype(torch::kInt64).device(torch::kCUDA));
int num_experts = 256;

std::optional<EventHandle> dummy_event = std::nullopt;

auto [tokens_per_rank, _, tokens_per_expert, token_in_rank, _] =
    buffer.get_dispatch_layout(topk_idx, num_experts, dummy_event, false, false);
```

### Notes

- `topk_idx` must be `int64` type and located on NPU;
- Maximum supported `num_ranks` is 384;
- `async`, `previous_event`, `allocate_on_comm_stream` parameters are currently unused;
- RDMA, async communication, or event scheduling requires extending this interface;
- If `num_experts` is not divisible by `num_ranks`, a logic error will occur;
- All returned tensors are located on the same device as the input tensor by default;
- The layout implementation differs between A3 and A2, but both compute the parameters needed for subsequent operations. The operator is configured to select based on the environment, but ensure the correct platform-specific operator is used.

### Future Extensions

- Implement `async` execution and pre-event dependencies (improve pipeline parallelism);
- Complete `num_tokens_per_rdma_rank` output after RDMA rank support is implemented.

---

<a id="中文"></a>

## 中文

> **文件**：`buffer.py`
> **核心类**：`Buffer`
> **依赖**：`torch`, `deep_ep_cpp`
> **目的**：normal 模式下 dispatch 和 combine 之前的数据预处理。

### 接口功能简述

根据传入的 `topk_idx` 计算 Normal 模式下后续的 Dispatch 和 Combine 需要的参数的本地副本。

### 接口定义

```python
def get_dispatch_layout(
        self,
        topk_idx: torch.Tensor,
        num_experts: int,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, EventOverlap
    ]:
```

```cpp
std::tuple<
torch::Tensor,                      // num_tokens_per_rank
std::optional[torch::Tensor](torch::Tensor),      // num_tokens_per_rdma_rank (预留字段)
torch::Tensor,                      // num_tokens_per_expert
torch::Tensor,                      // is_token_in_rank
std::optional<EventHandle>         // output_event (暂未使用)
>

Buffer::get_dispatch_layout(
const torch::Tensor& topk_idx,
int num_experts,
std::optional<EventHandle>& previous_event,
bool async,
bool allocate_on_comm_stream
)
```

### 输入参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `topk_idx` | `torch::Tensor` (`int64`, `[num_tokens, num_topk]`) | 每个 token 的 top-k expert 索引（必须是连续的二维张量），第二维大小取值范围 [1, 16] |
| `num_experts` | `int` | 系统中总的 expert 数量，取值范围 [1, 512]，且能被 `num_ranks` 整除 |
| `previous_event` | `std::optional<EventHandle>&` | 异步执行用的前置事件（当前未使用，传入 `std::nullopt`） |
| `async` | `bool` | 是否启用异步模式（当前未使用） |
| `allocate_on_comm_stream` | `bool` | 是否在通信流上分配内存（当前未使用） |

### 输入约束

- num_tokens: 表示 batch sequence size，即本卡输入输出的 token 数量，在输入中体现为 topk_idx 的第一维。
    - A2 系列双机取值范围：(0, 4096]；单机取值范围：(0, 8192]；
    - A3 系列取值范围，不开蚂蚁搬家：(0, 8192]，开蚂蚁搬家：(0, 32k]；

### 返回值说明

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `num_tokens_per_rank` | `torch::Tensor` (`int32`, `[num_ranks]`) | 每个 rank 中被分配的 token 数量 |
| `num_tokens_per_rdma_rank` | `std::optional<torch::Tensor>` | 保留字段，当前始终为 `std::nullopt` |
| `num_tokens_per_expert` | `torch::Tensor` (`int32`, `[num_experts]`) | 每个 expert 接收到的 token 数量 |
| `is_token_in_rank` | `torch::Tensor` (`bool`, `[num_tokens, num_ranks]`) | 指示每个 token 是否属于某个 rank |
| `output_event` | `std::optional<EventHandle>` | 保留字段，当前为 `std::nullopt` |

### 内部逻辑简述

1. 将 `topk_idx` 搬到每个核的 UB buffer 上，第一遍遍历计算需要的部分参数；
2. 使用 DataCopy 将计算结果搬到 GM 上，利用原子加或者分地址传输的方法聚合各核的数据；
3. 将部分计算完的 GM 数据（如前缀和等）搬回 UB，第二遍遍历计算剩下的参数；
4. 计算出的结果 tensor 搬回 GM。

### 多核策略

- 直接将 `topk_idx` 的第 0 维，即 token 数目按照核数目进行尽可能平均的划分，如果 token 数小于核数，则只使用前 token 数目个核，以此来实现数据并行。

### 示例用法

```cpp
auto topk_idx = torch::randint(0, 256, {4096, 8}, torch::dtype(torch::kInt64).device(torch::kCUDA));
int num_experts = 256;

std::optional<EventHandle> dummy_event = std::nullopt;

auto [tokens_per_rank, _, tokens_per_expert, token_in_rank, _] =
    buffer.get_dispatch_layout(topk_idx, num_experts, dummy_event, false, false);
```

### 注意事项

- `topk_idx` 必须是 `int64` 类型并位于 NPU 上；
- 当前支持的运行卡数 `num_ranks` 最大值为 384；
- 当前实现不使用 `async`、`previous_event`、`allocate_on_comm_stream` 等参数；
- 若需要使用 `RDMA`、`异步通信` 或 `事件调度`，需扩展本接口；
- 若 `num_experts` 不能被 `num_ranks` 整除，会导致逻辑错误；
- 返回的所有 tensor 默认与输入 tensor 位于相同设备上；
- A3 机器和 A2 机器上 layout 实现并不完全相同，但都是计算后续需要的参数，算子中配置了根据环境选择，但仍要确保使用对应机器的算子。

### 扩展建议

- 实现 `async` 执行和前置事件依赖（提高流水线并行度）；
- RDMA rank 支持后完善 `num_tokens_per_rdma_rank` 输出。
