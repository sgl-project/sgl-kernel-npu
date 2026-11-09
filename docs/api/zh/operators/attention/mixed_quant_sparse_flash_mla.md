# mixed_quant_sparse_flash_mla

## 芯片支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| cannbot_dsl | 不支持 | 支持 |
| torch_native | 不支持 | 支持 |

## 函数原型

`from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla`

```python
def mixed_quant_sparse_flash_mla(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    quant_mode,
    softmax_scale=None,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_softmax_lse=False,
    out,
    lse,
    impl="auto"
): ...
```

## 功能说明

混合量化稀疏 MLA：对 BF16 Query、FP8 原始 KV 和可选 FP4 压缩 KV 执行稀疏注意力。

`q` 为 BF16 TND `[T1,64,512]`；`ori_kv` 为 UINT8 PA_BBND `[pages,page_size,1,544]`，`cmp_kv` 可选，格式为 `[pages,page_size,1,320]`。两侧需分别提供稀疏索引、页表与可选有效 TopK 长度。`metadata` 来自 `mixed_quant_sparse_flash_mla_metadata`；`quant_mode=1`。`out` 和 `lse` 由调用方预分配。

计算语义（量化边界使用 BF16）：

```text
z_i = softmax_scale * dot(q, dequant(k_i)); out = sum_i exp(z_i) * dequant(v_i) / (exp(sink) + sum_i exp(z_i)); lse = log(exp(sink) + sum_i exp(z_i))
```

原始 KV 使用 E4M3FN，每 32 个特征共用一个 BF16 scale；压缩 KV 使用带符号 E2M1，每 16 个特征共用一个 BF16 scale，低半字节在前。两者均按 `dequant(value, scale) = value * scale` 解量化，无 zero point，结果舍入到 BF16。PV 权重在与 value 相乘前舍入到 BF16。sink 参与 softmax 分母和 LSE，value 贡献为零。

## 参数

T 为查询行数，B 为请求数，P 为物理页数，S 为每页槽位数。

| 参数 | 默认值 | 类型、形状与含义 |
| --- | --- | --- |
| `q` | 必填 | BF16 [T,64,512]，attention 查询。 |
| `ori_kv` | `None` | UINT8 [P,S,1,544]，FP8 KV 字节与 BF16 分组 scale；cannbot_dsl 必需。 |
| `cmp_kv` | `None` | 可选 UINT8 [P,S,1,320]，打包 FP4 KV 字节与 BF16 分组 scale。 |
| `ori_sparse_indices` | `None` | 必需的 INT32 [T,1,Kori] 原始 KV 逻辑 token ID；-1 为填充。 |
| `cmp_sparse_indices` | `None` | 可选 INT32 [T,1,Kcmp]，压缩 KV 逻辑 token ID；-1 为填充。 |
| `ori_block_table` | `None` | 必需的 INT32 [B,max_ori_blocks] 原始 KV 页映射。 |
| `cmp_block_table` | `None` | 可选 INT32 [B,max_cmp_blocks]，压缩 KV 页映射。 |
| `cu_seqlens_q` | `None` | cannbot_dsl 必需的 INT32 [B+1] 查询累计边界，末项为 T。 |
| `seqused_q` | `None` | 可选 INT32 [B]；提供时必须等于 cu_seqlens_q 的相邻差值，不支持填充查询行。 |
| `seqused_ori_kv` | `None` | 可选 INT32 [B]，原始 KV 长度；保留该兼容参数。 |
| `seqused_cmp_kv` | `None` | 可选 INT32 [B]，压缩 KV 长度；保留该兼容参数。 |
| `ori_topk_length` | `None` | 可选 INT32 [T,1] 有效稀疏前缀长度；None 使用原始索引的全部列。 |
| `cmp_topk_length` | `None` | 可选 INT32 [T,1] 有效稀疏前缀长度；None 使用压缩索引的全部列。无 cmp_kv 时省略。 |
| `sinks` | `None` | FP32 [64]，attention sink 的 logit（不贡献 value）。 |
| `metadata` | `None` | cannbot_dsl 必需的 INT32 [1024] 设备调度数据；torch_native 不使用该参数。 |
| `quant_mode` | 必填 | int，当前支持的量化配置使用 1。 |
| `softmax_scale` | `None` | 可选 float，默认 1/sqrt(512)。 |
| `layout_q` | `'TND'` | str，使用 TND（展平查询 token）。 |
| `layout_kv` | `'PA_BBND'` | str，使用 PA_BBND 打包分页 KV。 |
| `return_softmax_lse` | `False` | bool，True 时写入 lse。 |
| `out` | 必填 | 可写 BF16 [T,64,512] 输出张量。 |
| `lse` | 必填 | return_softmax_lse=True 时为可写 FP32 [1,T,64]；否则传入空的 FP32 [0] 张量。 |
| `impl` | `'auto'` | str，auto 选择 cannbot_dsl；torch_native 是独立计算参考。 |

## 返回值

| 值 | 含义 |
| --- | --- |
| None | 结果写入 out；return_softmax_lse=True 时写入 lse。 |

## cannbot_dsl

`auto` 默认选择此后端。需要 A5 和环境中安装的兼容 CANNBot-DSL 运行时，算子入口不强制检查版本。输入在同一设备，首次调用 JIT 编译；图捕获前预热实际形状。

## torch_native

独立的 CPU Torch 精度参考，结果复制回输入设备，保留原地输出语义。无需 DSL；不适合性能执行或 NPU graph capture。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import (
    mixed_quant_sparse_flash_mla,
    mixed_quant_sparse_flash_mla_metadata,
)

q = torch.randn(1, 64, 512, device="npu", dtype=torch.bfloat16)
kv = torch.zeros(1, 128, 1, 544, device="npu", dtype=torch.uint8)
kv[..., 512:] = torch.ones(1, 128, 1, 16, device="npu", dtype=torch.bfloat16).view(
    torch.uint8
)
cu = torch.tensor([0, 1], device="npu", dtype=torch.int32)
length = torch.tensor([[128]], device="npu", dtype=torch.int32)
metadata = mixed_quant_sparse_flash_mla_metadata(
    length,
    torch.zeros_like(length),
    cu_seqlens_q=cu,
    num_heads_q=64,
    num_heads_kv=1,
    head_dim=512,
    quant_mode=1,
    has_cmp_kv=False,
)
out = torch.empty_like(q)
lse = torch.empty(1, 1, 64, device="npu")
mixed_quant_sparse_flash_mla(
    q,
    ori_kv=kv,
    ori_sparse_indices=torch.arange(128, device="npu", dtype=torch.int32).reshape(
        1, 1, -1
    ),
    ori_block_table=torch.zeros(1, 1, device="npu", dtype=torch.int32),
    cu_seqlens_q=cu,
    ori_topk_length=length,
    sinks=torch.zeros(64, device="npu"),
    metadata=metadata,
    quant_mode=1,
    out=out,
    lse=lse,
    return_softmax_lse=True,
)
```

## 设备测试

```bash
python -m pytest tests/python/operators/attention/test_mixed_quant_sparse_flash_mla.py --npu-device=0 --enable-perf=false -v -s
```
