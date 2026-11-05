# MoE 门控 Top-K

## 硬件支持

| 实现 | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `torch_npu` | 已声明，需验证 SDK 内核 | 已声明，需验证 SDK 内核 |
| `torch_native` | 已声明，需验证 SDK 内核 | 已声明，需验证 SDK 内核 |

## 接口

```python
from sgl_kernel_npu.operators.moe import moe_gating_top_k

def moe_gating_top_k(x, k, *, bias=None, input_ids=None, tid2eid=None, k_group=1, group_count=1, group_select_mode=0, renorm=0, norm_type=0, out_flag=False, routed_scaling_factor=1.0, eps=1e-20, impl='auto'):
    ...
```

`auto` 根据初始化的 CANN SDK 选择生产路径：9.1 使用旧绑定，9.2 使用正式绑定；不在执行失败后重试其他后端。

## 说明

输入 FP32/FP16/BF16 [T,E]，支持分组或哈希路由；A3 sqrt-softplus 使用 group_count=1。

采用 softmax/sigmoid/sqrt-softplus 打分；bias 只影响选择，不影响被收集的权重。选择专家后按需归一化并乘 routed_scaling_factor；哈希 input_ids 和 tid2eid 必须同时提供。

## 参数

| 参数 | 类型 | 默认值 | 约束 |
| --- | --- | --- | --- |
| x | Tensor | 必填 | Input activation; shape described below; production device NPU |
| k | Tensor / int | 必填 | Indexer: paged [pages,page_size,1,D]; MoE: requested expert count |
| bias | Tensor / None | None | Optional correction bias [E], affects selection only |
| input_ids | Tensor / None | None | Optional token hash IDs, paired with tid2eid |
| tid2eid | Tensor / None | None | Token-to-expert hash mapping |
| k_group | int | 1 | Number of selected groups |
| group_count | int | 1 | Number of expert groups |
| group_select_mode | int | 0 | Vendor group score selection mode |
| renorm | int / bool | 0 | Whether selected weights are renormalized |
| norm_type | int | 0 | 0 softmax, 1 sigmoid, 2 sqrt-softplus |
| out_flag | bool | False | Whether to return full normalized scores |
| routed_scaling_factor | float | 1.0 | Multiplier after optional renormalization |
| eps | float | 1e-20 | Numerical stability epsilon |
| impl | str | 'auto' | Registered implementation; auto selects the SDK path; never falls back to torch_native |

## 返回值

| 名称 | 类型 | 约束 |
| --- | --- | --- |
| weights | Tensor | [T,k], x.dtype. |
| indices | int32 Tensor | [T,k] expert IDs. |
| scores | FP32 Tensor | [T,E] if out_flag=True, otherwise empty. |

## torch_npu

保留现有注册名；包装器内部按 SDK 选择原始或正式绑定。metadata/计算配套检查和 A5 KV reader/writer 格式检查在分配或执行前完成。

## torch_native

独立 eager PyTorch 参考；仅用于正确性检查，不自动替代生产实现，也不保证图捕获。元数据描述对象不等价于原生 tiling 字节。上述参考支持域外的模式会明确拒绝。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.moe import moe_gating_top_k

x = torch.randn(8, 256, device="npu", dtype=torch.float32)
weights, ids, scores = moe_gating_top_k(x, 8, norm_type=0, renorm=1, impl="auto")
```

先构建安装当前 wheel 并初始化 SDK。上例只演示接口，不能替代 NPU 精度、性能或端到端验证。

```bash
python3 -m pytest tests/python/operators/moe/test_moe_gating_top_k.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[上游 torch_npu schema](https://github.com/Ascend/op-plugin/blob/master/op_plugin/config/op_plugin_functions.yaml)
