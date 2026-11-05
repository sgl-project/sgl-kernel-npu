# 融合 KV 压缩器

## 硬件支持

| 实现 | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformer` | 已声明，需验证 SDK 内核 | 已声明，需验证 SDK 内核 |
| `torch_native` | 已声明，需验证 SDK 内核 | 已声明，需验证 SDK 内核 |

## 接口

```python
from sgl_kernel_npu.operators.attention import compressor

def compressor(x, wkv, wgate, state_cache, ape, norm_weight, *, impl='auto', **kwargs):
    ...
```

`auto` 根据初始化的 CANN SDK 选择生产路径：9.1 使用旧绑定，9.2 使用正式绑定；不在执行失败后重试其他后端。

## 说明

保留融合 RMSNorm/RoPE 的计算边界，不能替换为仅压缩核心的正式算子；state 原地更新。

投影 KV 和 gate，通过带位置偏置的归一化 gate 权重聚合窗口，再执行 RMSNorm 和部分 RoPE；输出已完成压缩行并保留未完成及重叠状态。

## 参数

| 参数 | 类型 | 默认值 | 约束 |
| --- | --- | --- | --- |
| x | Tensor | 必填 | Input activation; shape described below; production device NPU |
| wkv | Tensor | 必填 | [coff*D,C] KV projection weight |
| wgate | Tensor | 必填 | [coff*D,C] gate projection weight |
| state_cache | Tensor | 必填 | Mutable FP32 [pages,slots,2*coff*D] history |
| ape | Tensor | 必填 | [cmp_ratio,coff*D] positional gate bias |
| norm_weight | Tensor | 必填 | [D] RMSNorm weight |
| impl | str | 'auto' | Registered implementation; auto selects the SDK path; never falls back to torch_native |

### 关键字参数

| 名称 | 类型 | 约束 |
| --- | --- | --- |
| rope_sin / rope_cos | FP32 Tensor | Rotary factors for compressed output rows. |
| rope_head_dim / cmp_ratio | int | Rotary width / compression ratio, required. |
| state_block_table | int32 Tensor | Required: paged table in mode 1; A3 explicit slots or A5 request-bank IDs in mode 2. |
| cu_seqlens / seqused / start_pos | int32 Tensor / None | Cumulative TH offsets (None for BSH), per-request counts, absolute starts. |
| coff / norm_eps | int / float | Defaults 1 / 1e-6; reference coff is 1 or 2. |
| rotary_mode / cache_mode | int | Defaults 2 / 2; reference supports 1 or 2. |

## 返回值

| 名称 | 类型 | 约束 |
| --- | --- | --- |
| output | Tensor | TH: [capacity,D]; BSH: [B,capacity,D]. Only completed compressed rows are valid; padding is unspecified. state_cache is also mutated. |

## cann_ops_transformer

保留现有注册名；包装器内部按 SDK 选择原始或正式绑定。metadata/计算配套检查和 A5 KV reader/writer 格式检查在分配或执行前完成。

`torch.ops.npu.compressor` supplies the fused contract; legacy A5 uses `torch.ops.custom.compressor`.

## torch_native

独立 eager PyTorch 参考；仅用于正确性检查，不自动替代生产实现，也不保证图捕获。元数据描述对象不等价于原生 tiling 字节。上述参考支持域外的模式会明确拒绝。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import compressor

x = torch.randn(4, 1024, device="npu", dtype=torch.bfloat16)
wkv = torch.randn(128, 1024, device="npu", dtype=x.dtype) * 0.02
wgate = torch.randn_like(wkv) * 0.02
state = torch.zeros(2, 16, 256, device="npu", dtype=torch.float32)
ape = torch.zeros(4, 128, device="npu", dtype=torch.float32)
norm = torch.ones(128, device="npu", dtype=torch.float32)
i32 = lambda values: torch.tensor(values, device="npu", dtype=torch.int32)
y = compressor(x, wkv, wgate, state, ape, norm,
    rope_sin=torch.zeros(2, 64, device="npu"),
    rope_cos=torch.ones(2, 64, device="npu"), rope_head_dim=64, cmp_ratio=4,
    state_block_table=i32([[1]]), cu_seqlens=i32([0, 4]),
    seqused=i32([4]), start_pos=i32([0]), cache_mode=1, impl="torch_native")
```

先构建安装当前 wheel 并初始化 SDK。上例只演示接口，不能替代 NPU 精度、性能或端到端验证。

```bash
python3 -m pytest tests/python/operators/attention/test_compressor.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[上游算子仓库](https://github.com/Ascend/ops-transformer)
