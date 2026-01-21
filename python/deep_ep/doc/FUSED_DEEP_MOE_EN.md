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
