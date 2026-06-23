# Fused Deep MoE API

> **English** | [中文](#中文)
>
> **适用平台**：A3 only（仅 A3）

---

## DeepEP-DeepFusedMoE


### Introduction
In Mixture of Experts (MoE) models, the fused_deep_moe operator implements the hyper-fusion of Dispatch + Experts FFN (2×GMM) + Combine functionalities.
This operator completes token distribution, expert computation (matrix multiplication, activation, quantization/dequantization), and result aggregation in a single call. Compared with traditional multi-operator implementations, it significantly reduces communication overhead and end-to-end latency.
The communication latency (Batch size = 32 / 155μs, Dispatch = 80μs, Combine = 75μs) is reduced to less than 85μs, with a 70μs reduction in single-layer communication latency and a 4ms reduction in inference end-to-end latency.

* In MoE-based large models, each token (a vector with consistent length across all tokens) needs to be processed by multiple experts, and the processed results are collected and accumulated. Different experts are distributed across different NPU cards, and each card supports deploying multiple experts.

* The operation/operator for distributing tokens to multiple experts is called dispatch. The corresponding alcnn operator is already available in CANN.
* Expert processing mainly consists of computational operations: matrix multiplication, activation, and matrix multiplication in sequence, resulting in new tokens with unchanged length after processing.
  * Since multiple experts may reside on a single card, a single computation operator processes multiple experts simultaneously. Thus, the computational steps on one card are grouped matrix multiplication, activation, and grouped matrix multiplication in sequence.
  * To reduce memory overhead and accelerate computation, quantization-dequantization operations are typically introduced. The complete computational flow becomes: grouped matrix multiplication → dequantization → activation → quantization → grouped matrix multiplication → dequantization.
  * ATB currently provides a large computation operator GmmDepSwigluQuantGmmDep that can complete all the above computational steps in one go.
* The operation/operator for collecting and accumulating processed results is called combine. The corresponding alcnn operator is already available in CANN.

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

### Parameter Description
| Parameter | Type | Shape | Description                                                                                                                                                                                                                                                                                                                                                                                                                                |
|-----------|------|-------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **x** | `torch.Tensor` | `[bs, hidden]` | Input token representations, where each row is the hidden vector of a token (commonly `bfloat16`).<br><br>**bs** (batch size): Range **[1, 256]**.<br>**hidden**: Represents the hidden dimension size, typically determined by the model's hidden layer width (e.g., 2048, 4096, 6144, 7168). Range **[512, 7168]**, and must be divisible by **32** to meet the alignment requirements of underlying matrix multiplication and communication. |
| **topk_idx** | `torch.Tensor` | `[bs, num_topk]` | Expert indices for each token, `int64` type. A value of `-1` indicates the token is not dispatched.                                                                                                                                                                                                                                                                                                                                        |
| **topk_weights** | `torch.Tensor` | `[bs, num_topk]` | Weighting coefficients for aggregating expert outputs (`float32`).                                                                                                                                                                                                                                                                                                                                                                         |
| **gmm1_permuted_weight** | `torch.Tensor` | e.g., `[G, 7168, 4096]` | First-stage (up-projection) expert weights, permuted to fit Grouped MatMul.                                                                                                                                                                                                                                                                                                                                                                |
| **gmm1_permuted_weight_scale** | `torch.Tensor` | e.g., `[G, 4096]` | Quantization scale for first-stage weights, required in quantization mode (`float32`).                                                                                                                                                                                                                                                                                                                                                     |
| **gmm2_weight** | `torch.Tensor` | e.g., `[G, 7168, 2048]` | Second-stage (down-projection) expert weights.                                                                                                                                                                                                                                                                                                                                                                                             |
| **gmm2_weight_scale** | `torch.Tensor` | e.g., `[G, 7168]` | Quantization scale for second-stage weights.                                                                                                                                                                                                                                                                                                                                                                                               |
| **num_max_dispatch_tokens_per_rank** | `int` | Scalar | Maximum number of tokens to dispatch per rank, used for buffer/memory allocation.                                                                                                                                                                                                                                                                                                                                                          |
| **num_experts** | `int` | Scalar | Total number of global experts.                                                                                                                                                                                                                                                                                                                                                                                                            |
| **quant_mode** | `int` | Scalar, default `1` | Indicates quantization mode:<br>`1`: int8;<br>fp8 will be supported in A5 release.                                                                                                                                                                                                                                                                                                                                                         |

### Return Values
| Parameter                     | 	Type             | 	Shape                         | Description                                                                     |
|-------------------------------| -------------- | -------------------------- |------------------------------------------------------------------------|
| **output**                      | `torch.Tensor` | `[bs, hidden]`             | Fused expert outputs.                                                 |
| **ep_recv_count**             | `torch.Tensor` | `[num_local_experts]`           | Number of tokens received by each card in the EP communication domain, which is used for subsequent communication synchronization or load balancing statistics.
|

---

<a id="中文"></a>

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
