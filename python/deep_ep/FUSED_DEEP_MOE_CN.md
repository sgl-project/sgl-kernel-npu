<h2 align="left">
DeepEP-DeepFusedMoE
</h2>


## 介绍
DeepFusedMoE基于Dispatch+Combine+2*GMM超融合算子，通信时长(BS=32/155us，Dispatch=80us, Combine=75us)降低到85us以内，单层通信时长降低70us，推理端到端时延降低4ms。

* 在MoE类大模型中，每个token（一个向量，所有token长度是一致）需要交给多个专家处理，并将处理后的结果收回并累加到一起。不同专家分布在不同的NPU卡上，每张卡支持部署多个专家。

* token交给多个专家的操作/算子被称为dispatch。当前CANN中已有对应的alcnn算子。
* 专家处理主要是一些计算动作，依次为矩阵乘、激活、矩阵乘，处理后得到的新token长度不变。
  * 由于一张卡上可能有多个专家，一个计算算子会同时处理多个专家，所以一张卡的计算动作依次为分组矩阵乘、激活、分组矩阵乘。
  * 为了减少显存开销、加速计算，通常会引入量化-反量化操作，所以计算动作依次为分组矩阵乘、反量化、激活、量化、分组矩阵乘、反量化。
  * 当前ATB已有一个大计算算子GmmDepSwigluQuantGmmDep，可以一次性完成上述所有计算动作。
* 将处理后的结果收回并累加到一起的操作/算子，被称为combine。当前CANN中已有对应的alcnn算子。

## Python-API
```python
def fused_deep_moe(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    gmm1PermutedWeight: torch.Tensor,
    gmm1PermutedWeightScale: torch.Tensor,
    gmm2Weight: torch.Tensor,
    gmm2WeightScale: torch.Tensor,
    num_max_dispatch_tokens_per_rank: int,
    num_experts: int,
    use_fp8: bool = True,
) -> Tuple[torch.Tensor, EventOverlap, Callable]
```

### 参数说明

- `x` — `torch.Tensor` — shape `[bs, hidden]`（例如 `[32,1024]`） — 输入的 token 表示（通常为 `torch.bfloat16`），每行一个 token 的隐藏向量。

  `topk_idx` — `torch.Tensor` — shape `[bs, num_topk]` — 每个 token 对应的专家索引（`int64`），支持 `-1` 表示该 token 不路由到任何专家。

  `topk_weights` — `torch.Tensor` — shape `[num_combined_tokens, num_topk]`（实际行数为被接收/合并的 tokens，通常 ≤ `bs`） — 合并专家输出时用的权重（`float32`），用于对接收方的权重加权汇总。

  `gmm1PermutedWeight` — `torch.Tensor` — 形状依实现而定（示例 `[G, 7168, 4096]`） — 第一阶段（上投/分组 matmul）使用的专家权重，已按 grouped-matmul 要求做过重排/permute，dtype/device 应与 kernel 要求一致（例如 `bfloat16`/`npu`）。

  `gmm1PermutedWeightScale` — `torch.Tensor` — 形状与 `gmm1PermutedWeight` 对应的 scale（示例 `[G, 4096]`） — 第一阶段权重的量化 scale（通常 `float32`），用于 FP8/量化路径；若不量化可传相应空或单位 scale。

  `gmm2Weight` — `torch.Tensor` — 形状依实现而定（示例 `[G, 7168, 2048]`） — 第二阶段（下投/FFN 输出）使用的专家权重，同样可能已重排以适配内核。

  `gmm2WeightScale` — `torch.Tensor` — 形状与 `gmm2Weight` 对应的 scale（示例 `[G, 7168]`） — 第二阶段权重的量化 scale（通常 `float32`），用于量化/反量化操作。

  `num_max_dispatch_tokens_per_rank` — `int` — 标量 — 每个 rank 上允许分发（dispatch）的最大 token 数（用于 buffer/内存分配）。

  `num_experts` — `int` — 标量 — 全局专家总数（logical experts），用于内核属性与路由计算。

  `use_fp8` — `bool` — 标量（默认 `True`） — 是否启用 FP8 路径；开启后内核会走 FP8 量化/伸缩相关分支，需同时提供对应的 scale 张量。

### 返回值

- **output**: `[bs, hidden]`, 融合专家输出，通常为 `torch.bfloat16`。
- **event**: `EventOverlap`，内核执行后的事件句柄。
- **hook**: `Callable`，异步回调函数。

