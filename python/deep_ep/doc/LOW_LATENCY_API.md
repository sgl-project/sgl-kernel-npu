**文件**：`buffer.py`

**核心类**：`Buffer`

**依赖**：`torch`, `deep_ep_cpp`

**目的**：在 **多 NPU（Intranode）** 与 **跨节点（Internode）** 环境下，高效完成 低时延**Token Dispatch** 与 **Token Combine**（即分发?归约）操作。

# low_latency_dispatch

## python侧接口

```python
# noinspection PyTypeChecker
def low_latency_dispatch(self, x: torch.Tensor, topk_idx: torch.Tensor,
                         num_max_dispatch_tokens_per_rank: int, num_experts: int,
                         cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                         use_fp8: bool = True, round_scale: bool = False, use_ue8m0: bool = False,
                         async_finish: bool = False, return_recv_hook: bool = False) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple, EventOverlap, Callable]:
"""
        A low-latency implementation for dispatch.

Arguments:
	x：带有`torch.bfloat16`的`torch.Tensor`，形状为`[num_tokens，隐藏的]`，只有几个隐藏的形状是支持，要分发的令牌数必须小于`num_max_dispatch_tokens_per_rank`。
	topk_idx：带有`torch.int64`的`torch.Tensor`，形状为`[num_tokens, num_topk]`，只有几个top-k形状都支持。支持`-1`个索引（不选择任何专家）。
	num_max_dispatch_tokens_per_rank：要分发的令牌的最大数量，所有Rank必须持有相同的值。
	num_experts：所有专家的个数。
	accumulation_local_expert_recv_stats：用于统计的累积专家计数张量，它应该具有形状`[num_local_experts]`并键入为`torch.int`。这对于在线服务EP负载平衡监控非常有用。
	use_fp8：是否启用FP8铸造，有了此，接收的数据将是FP8张量和缩放因子的元组。
	Round_scale：是否将缩放因子四舍五入为2的次幂。
	use_ue8m0：是否使用UE8M0作为缩放因子格式（仅适用于`round_scale=True`）。
	async_finish：如果设置了，当前流不会等待通信内核完成。
	return_recv_hook：如果设置了，则返回接收钩子。如果设置，内核将只处理RDMA请求问题，但是并没有实际接收到数据，你必须调用接收的钩子来确保数据的到达。如果不设置此标志，内核将确保数据的到达。

Returns:
	recv_x：一个张量或元组，包含每个专家的接收令牌。当 `use_fp8=True` 时：第一个元素是一个形状为 `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` 的 `torch.Tensor`，使用 `torch.float8_e4m3fn` 类型。第二个张量是第一个元素的相应比例，形状为 `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden // 128]`，使用 `torch.float` 类型，如果 `use_ue8m0=False`。如果 `use_ue8m0=True`，第二个张量是打包的，形状为 `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden // 512]`，类型为 `torch.int`。注意，比例张量的最后两个维度是列主序的，以兼容 TMA。当 `use_fp8=False` 时，结果将是一个形状为 `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` 的张量，使用 `torch.bfloat16` 类型。此外，并非所有令牌都是有效的，只有部分 `num_max_dispatch_tokens_per_rank * num_ranks` 是有效的，因为我们没有将 CPU 接收计数与 GPU 同步（即使同步也不会与 CUDA 图不兼容）。
	recv_count：一个形状为 `[num_local_experts]` 的张量，类型为 `torch.int`，表示每个专家接收的令牌数量。如前所述，`recv_x` 中并非所有令牌都是有效的。
	handle：在 `low_latency_combine` 函数中使用的通信句柄。
	event：执行内核后的事件（仅在 `async_finish` 设置时有效）。
	hook：接收钩子函数（仅在 `return_recv_hook` 设置时有效）。
"""
```

| **参数类别** | **参数名**                           | **类型**                 | **默认值** | **详细描述**                                                 | **是否必要** | **注意事项**                                                 |
| ------------ | ------------------------------------ | ------------------------ | ---------- | ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ |
| **核心输入** | `x`                                  | `torch.Tensor`           | -          | 输入token数据，形状`[num_tokens, hidden]`，类型`torch.bfloat16`。`num_tokens <= 512`| 是        | token数必须小于`num_max_dispatch_tokens_per_rank`, A2 算子实现要求 0 < hidden <= 7168 and hidden % 32 = 0           |
|              | `topk_idx`                           | `torch.Tensor`           | -          | 专家索引，形状`[num_tokens, num_topk]`，类型`torch.int64`，支持`-1`（不选择任何专家） | 是        | 决定token路由到哪个专家                                      |
| **配置参数** | `num_max_dispatch_tokens_per_rank`   | `int`                    | -          | 每个rank最大分发token数，所有rank必须相同                    | 是        | 影响内存分配和性能上限                                       |
|              | `num_experts`                        | `int`                    | -          | 专家总数                                                     | 是        | 用于路由决策和负载均衡                                       |
| **统计监控** | `cumulative_local_expert_recv_stats` | `Optional[torch.Tensor]` | `None`     | 累计专家接收统计，形状`[num_local_experts]`，类型`torch.int` | -          | 用于在线服务EP负载均衡监控。DeepEp-Ascend不需要              |
| **精度控制** | `use_fp8`                            | `bool`                   | `True`     | 是否启用FP8量化（A3/A2芯片只支持INT8量化），若use_fp8=True，算子内部会先把token从bfloat16转化为INT8类型的tensor后再通信，以降低通信时延 | -        | 显著减少通信带宽                                             |
|              | `round_scale`                        | `bool`                   | `False`    | 是否将缩放因子四舍五入为2的幂                                | -          | 与`use_ue8m0`配合使用                                        |
|              | `use_ue8m0`                          | `bool`                   | `False`    | 是否使用UE8M0作为缩放因子格式（仅在`round_scale=True`时有效） | -          | 优化缩放因子存储格式                                         |
| **异步控制** | `async_finish`                       | `bool`                   | `False`    | 如果设置，当前流不会等待通信内核完成                         | -          | 提高GPU利用率，需手动同步。DeepEp-Ascend不需要               |
|              | `return_recv_hook`                   | `bool`                   | `False`    | 如果设置，返回接收钩子，内核只发RDMA请求不接收数据           | -          | 实现真正的异步通信，必须调用钩子确保数据到达。DeepEp-Ascend不需要 |
| **返回值**   | `recv_x`                             | `Tuple/Tensor`           | -          | 接收的token数据：<br>- `use_fp8=True`: `(INT8_tensor, scales)`<br>- `use_fp8=False`: `bfloat16_tensor` | 是        | 并非所有token都有效，需结合`recv_count`使用                  |
|              | `recv_count`                         | `torch.Tensor`           | -          | 每个专家接收的token数量，形状`[num_local_experts]`，类型`torch.int` | 是        | 指示`recv_x`中有效token数量                                  |
|              | `handle`                             | `tuple`                  | -          | 通信句柄，包含`(src_info, layout_range, num_max_dispatch_tokens_per_rank, hidden, num_experts, packed_recv_count)` | 是        | 必须传递给`low_latency_combine`                              |
|              | `event`                              | `EventOverlap`           | -          | 内核执行后的事件（仅在`async_finish=True`时有效）            | -          | 用于事件同步和记录。DeepEp-Ascend不需要                      |
|              | `hook`                               | `Callable`               | -          | 接收钩子函数（仅在`return_recv_hook=True`时有效）            | -          | 调用以确保数据到达。DeepEp-Ascend不需要                      |

# low_latency_combine

## Python 侧接口

```python
def low_latency_combine(self, x: torch.Tensor,
                        topk_idx: torch.Tensor,
                        topk_weights: torch.Tensor,
                        handle: tuple,
                        zero_copy: bool = False,
                        async_finish: bool = False,
                        return_recv_hook: bool = False,
                        out: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor,
                  EventOverlap,
                  Callable]:
"""
Combine算子的低时延实现

参数：
	x：数据类型为 torch.bfloat16、形状为 [num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden] 的张量，指需发送至原始 rank 进行 reduce 运算的本地 token。
	topk_idx：数据类型为torch.int64、形状为[num_combined_tokens, num_topk]的张量，代表由调度令牌选中的专家索引。支持-1索引（表示不选中任何专家）。需注意，num_combined_tokens等于dispatched token 的数量。
	topk_weights：数据类型为 torch.float、形状为 [num_tokens, num_topk] 的张量，指需发送至初始 rank 进行 reduce 运算的token 的 Top-K 权重。接收的 token 将通过该张量中的权重进行归约。
	handle：由 dispatch 函数提供的通信句柄。
	zero_copy：表示张量是否已复制到 RDMA（远程直接内存访问）缓冲区中，需与get_next_low_latency_combine_buffer协同使用。
	async_finish：若设置为 True，当前 stream 将不会等待通信核心运算完成（即采用异步执行方式）。
	return_recv_hook：若设为 True，将返回接收钩子（receiving hook）。此时，内核仅会发起 RDMA 请求，不会实际接收数据。必须调用接收钩子，以确保数据到达。若不设置此标志，内核将确保数据到达。
	out：原地（in-place）输出张量。若设置该参数，内核会将结果写入此张量，并直接返回该张量。

返回值（Returns）：
	combined_x：归约后的 token 张量，形状为[num_combined_tokens, hidden]，数据类型为torch.bfloat16。
	event：执行内核后的事件（仅当async_finish设为 True 时有效）。
	hook：接收钩子函数（仅当return_recv_hook设为 True 时有效）。
"""
```

| **参数类别** | **参数名**         | **类型**                 | **默认值** | **详细描述**                                                 | **是否必要** | **注意事项**                                |
| ------------ | ------------------ | ------------------------ | ---------- | ------------------------------------------------------------ | ---------- | ------------------------------------------- |
| **核心输入** | `x`                | `torch.Tensor`           | -          | 本地计算的token，形状`[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]`，类型`torch.bfloat16` | 是        | 每个专家处理后的结果                        |
|              | `topk_idx`         | `torch.Tensor`           | -          | 专家索引，形状`[num_combined_tokens, num_topk]`，类型`torch.int64`，`num_combined_tokens`等于分发token数 | 是        | 必须与dispatch时的索引匹配                  |
|              | `topk_weights`     | `torch.Tensor`           | -          | 专家权重，形状`[num_combined_tokens, num_topk]`，类型`torch.float`，用于归约时加权 | 是        | 决定最终token的加权结果                     |
| **通信控制** | `handle`           | `tuple`                  | -          | 由dispatch函数返回的通信句柄，包含路由信息和统计信息         | 是        | **必须**从对应的dispatch调用获取            |
| **优化控制** | `zero_copy`        | `bool`                   | `False`    | 张量是否已复制到RDMA缓冲区，需与`get_next_low_latency_combine_buffer`配合使用 | -          | 减少内存拷贝开销。DeepEp-Ascend不需要       |
| **异步控制** | `async_finish`     | `bool`                   | `False`    | 如果设置，当前流不会等待通信内核完成                         | -          | 提高GPU利用率。DeepEp-Ascend不需要          |
|              | `return_recv_hook` | `bool`                   | `False`    | 如果设置，返回接收钩子，内核只发RDMA请求不接收数据           | -          | 实现真正的异步通信。DeepEp-Ascend不需要     |
| **输出控制** | `out`              | `Optional[torch.Tensor]` | `None`     | 原地输出张量，如果设置，结果直接写入此张量                   | -          | 避免额外内存分配。DeepEp-Ascend不需要       |
| **返回值**   | `combined_x`       | `torch.Tensor`           | -          | 归约后的token张量，形状`[num_combined_tokens, hidden]`，类型`torch.bfloat16` | 是        | 最终的专家混合结果                          |
|              | `event`            | `EventOverlap`           | -          | 内核执行后的事件（仅在`async_finish=True`时有效）            | -          | 用于事件同步和记录。DeepEp-Ascend不需要     |
|              | `hook`             | `Callable`               | -          | 接收钩子函数（仅在`return_recv_hook=True`时有效）            | -          | 必须调用以确保数据到达。DeepEp-Ascend不需要 |
