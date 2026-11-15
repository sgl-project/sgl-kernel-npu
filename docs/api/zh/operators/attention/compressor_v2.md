# compressor_v2

目标为 **CANN 9.2.0 / A3（arch22）**，接入 experimental CompressorV2。
`auto` 与 `cann_ops_transformers_cann920` 是同一包内后端，不自动回退。

上游原始文档: [CompressorV2 Torch API](https://gitcode.com/cann/ops-transformer/blob/e7da18105f08b06e01ef0fff78f439098d483746/experimental/attention/compressor_v2/docs/torchapi_compressor.md).

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` / `auto` | 支持 | 不支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.attention import compressor_v2

def compressor_v2(
    x, wkv, wgate, state_cache, cmp_ratio=4, *,
    state_block_table=None, cu_seqlens=None, seqused=None, start_pos=None,
    impl="auto",
): ...
```

## 功能说明

先计算 `K = x.float() @ wkv.float().T`、`G = x.float() @ wgate.float().T`，
对绝对位置上已完成的每组 cmp_ratio 个 token，逐 feature 沿 token 轴计算
`sum(softmax(G, token_axis) * K)`，再转换为 x.dtype。首组未对齐部分从历史 cache 读取，
未完成组不输出。每个 batch 只将新输入最后 `min(seqused[b],C)` 个 token 写入
`state_cache[block_id, absolute_position % C, :]`，前 D 通道为 K，后 D 通道为 G；
未写位置保持原值。固定 coff=1、cache_mode=2，无 APE 输入。

**state_cache 原地修改，返回值只有 cmp_kv。** BSH 输出 shape 为
`[B,ceil(S/cmp_ratio),D]`，各 batch 前
`floor(((start_pos[b] % cmp_ratio)+seqused[b])/cmp_ratio)` 个槽有效。
TH 输出 shape 为 `[min(T,T//cmp_ratio+B),D]`，各 batch 有效压缩组依 batch 顺序紧凑拼接，
之后为 padding。padding 未定义，不应用于精度比较。续接调用必须使用更新后的 cache
及正确 start_pos。

## 参数与返回值

| 参数 | dtype | shape / 约束 |
| --- | --- | --- |
| x | fp16/bf16 | `[B,S,H]` 或 packed `[T,H]`；H=1024..10240，512 对齐 |
| wkv、wgate | 与 x 相同 | `[D,H]`，D=128/512 |
| state_cache | fp32 | `[P,C,2*D]`，P>=B、P>0、C=1..1024；原地更新 |
| state_block_table | int32 | `[B]`，A3 必传；物理块号在 `[0,P)` 内且全局唯一，0 为有效块号 |
| cu_seqlens | int32 | TH：`[B+1]`，从 0 到 T 的非递减累计边界；BSH：None |
| seqused | int32 | 可选 `[B]`，0..各序列长度；None 使用完整序列 |
| start_pos | int32 | 可选 `[B]`，非负绝对位置；None 视为全 0 |
| cmp_ratio | int | 2/4/8/16/32/64/128，默认 4 |

所有 tensor 位于同一设备。x、权重及索引 tensor 连续，state_cache 仅第 0 轴允许非连续。
B/S/T 可为 0，权重与 cache 仍须非空。上游 Python 签名的 state_block_table 默认 None，
但 A3 tiling 要求存在，本接口提前拒绝 None。launch 前只检查元信息，不读回 tensor 数值；
块号、长度、位置及有效历史由调用方保证。

## cann_ops_transformers_cann920 后端

使用 `ops_transformer` profile 构建，其中 `additional_builds` 声明 CompressorV2 的独立构建，
两条上游命令均带 `--experimental`。
该依赖使用独立 vendor 和 Python 包，可与普通 profile 一起随 wheel 分发，保留完整
vendor、LICENSE、manifest；实验依赖不含 AICPU。首次调用可能 JIT 编译 C++ 绑定，
需要编译器及匹配的 CANN、Torch、torch_npu。

名称映射：构建项/OpType 为 `compressor_v2` / `CompressorV2`，包内绑定导出为
`_third_party.sgl_ops_transformer_experimental.ops.ds41.compressor`，Torch 名为
`cann_ops_transformer::ds41.compressor`，ACLNN 为 `aclnnCompressorV2`。

## torch_native 后端

`torch_native` 独立使用 CPU FP32 计算，支持 BSH/TH、循环 cache 和历史续接，要求
C>=cmp_ratio；会读取主机 metadata，不用于性能路径。固定版本上游 custom_op 声明
`mutates_args=()`，与实际 cache 写入不一致；本次面向 eager 推理，未验证 functionalization、
torch.compile、TorchAir 图捕获或 autograd。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import compressor_v2

x = torch.randn(1, 16, 1024, dtype=torch.bfloat16, device="npu")
wkv = torch.randn(128, 1024, dtype=x.dtype, device=x.device)
wgate = torch.randn_like(wkv)
cache = torch.zeros(1, 16, 256, dtype=torch.float32, device=x.device)
blocks = torch.tensor([0], dtype=torch.int32, device=x.device)
y = compressor_v2(x, wkv, wgate, cache, 4, state_block_table=blocks)
assert y.shape == (1, 4, 128)
```

上游版权：Huawei Technologies Co., Ltd.；CANN Open Software License Agreement Version 2.0。
