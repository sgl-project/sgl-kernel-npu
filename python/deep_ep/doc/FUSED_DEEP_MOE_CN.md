## DeepEP-DeepFusedMoE

### 介绍
在 MoE（Mixture of Experts，混合专家模型）中，fused_deep_moe 算子实现 Dispatch + Experts FFN (2×GMM) + Combine 的超融合功能。

该算子在一次调用中完成 token 分发、专家计算（矩阵乘、激活、量化/反量化）以及结果合并操作，相比传统多算子实现显著降低通信开销和端到端时延。

通信时长（Batch size = 32 / 155μs，Dispatch = 80μs，Combine = 75μs）降低到85μs以内，单层通信时长降低70μs，推理端到端时延降低4ms。

- 在MoE类大模型中，每个token（一个向量，所有token长度一致）需要交给多个专家处理，并将处理后的结果收回并累加到一起。不同专家分布在不同的NPU卡上，每张卡支持部署多个专家。
- token交给多个专家的操作/算子被称为dispatch（分发）。当前CANN中已有对应的alcnn算子。
- 专家处理主要是一系列计算动作，依次为矩阵乘、激活、矩阵乘，处理后得到的新token长度不变。
  - 由于一张卡上可能部署多个专家，一个计算算子会同时处理多个专家，因此单卡的计算动作依次为分组矩阵乘（Grouped MatMul）、激活、分组矩阵乘。
  - 为减少显存开销、加速计算，通常会引入量化-反量化操作，完整计算流程为：分组矩阵乘 → 反量化 → 激活 → 量化 → 分组矩阵乘 → 反量化。
  - 当前ATB已提供大计算算子GmmDepSwigluQuantGmmDep，可一次性完成上述所有计算动作。
- 将处理后的结果收回并累加到一起的操作/算子，被称为combine（合并）。当前CANN中已有对应的alcnn算子。

### Python-API
```python
def fused_deep_moe(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    gmm1_permuted_weight: torch.Tensor,
    gmm1_permuted_weight_scale: torch.Tensor,
    gmm2_weight: torch.Tensor,
    gmm2_weight_scale: torch.Tensor,
    num_max_dispatch_tokens_per_rank: int,
    num_experts: int,
    quant_mode: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]
```

### 参数说明
| 参数 | 类型 | 形状                    | 说明                                                                                                                                                                                                                         |
|------|------|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **x** | `torch.Tensor` | `[bs, hidden]`        | 输入 token 表示，每行一个 token 的隐藏向量（常用 `bfloat16`）。<br><br>**bs**（batch size）取值范围为 **[1, 256]**。<br>**hidden**  表示隐藏维度大小，通常取决于模型隐层宽度（如 2048、4096、6144、7168 等）。取值范围 **[512, 7168]**，且必须能被 **32** 整除，以满足底层矩阵乘与通信对齐要求。 |
| **topk_idx** | `torch.Tensor` | `[bs, num_topk]`      | 每个 token 的专家索引，`int64` 类型。若值为 `-1` 表示该 token 不分发。                                                                                                                                                                          |
| **topk_weights** | `torch.Tensor` | `[bs, num_topk]`      | 合并专家输出的加权系数（`float32`）。                                                                                                                                                                                                    |
| **gmm1_permuted_weight** | `torch.Tensor` | 例如 `[G, 7168, 4096]` | 第一阶段（上投）专家权重，已做 permute 以适配 Grouped MatMul。                                                                                                                                                                                |
| **gmm1_permuted_weight_scale** | `torch.Tensor` | 例如 `[G, 4096]`       | 第一阶段权重量化 scale，量化模式下必需（`float32`）。                                                                                                                                                                                         |
| **gmm2_weight** | `torch.Tensor` | 例如 `[G, 7168, 2048]` | 第二阶段（下投）专家权重。                                                                                                                                                                                                              |
| **gmm2_weight_scale** | `torch.Tensor` | 例如 `[G, 7168]`       | 第二阶段权重量化 scale。                                                                                                                                                                                                            |
| **num_max_dispatch_tokens_per_rank** | `int` | 标量                    | 每个 rank 最多分发的 token 数，用于 buffer/内存分配。                                                                                                                                                                                      |
| **num_experts** | `int` | 标量                    | 全局专家总数。                                                                                                                                                                                                                    |
| **quant_mode** | `int` | 标量，默认 `1`             | 表示量化模式：<br>`1`： 表示int8；<br>后续A5支持fp8。                                                                                                                                                                                              |


### 返回值
| 参数                              | 类型             | 形状                         | 说明                                   |
|---------------------------------| -------------- | -------------------------- |--------------------------------------|
| **output**                      | `torch.Tensor` | `[bs, hidden]`             | 融合专家输出。                              |
| **ep_recv_count**               | `torch.Tensor` | `[num_local_experts]`           | 表示EP通信域各卡收到的token数量，用于后续通信同步或负载均衡统计。 |
