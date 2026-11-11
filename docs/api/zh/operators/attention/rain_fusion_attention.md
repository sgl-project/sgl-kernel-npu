# rain_fusion_attention

RainFusionAttention 块稀疏融合注意力算子。

## 硬件支持

| 实现 | arch22 (2201) | arch35 (3510) |
| --- | --- | --- |
| `cann_ops_transformer` | 支持 | 不支持 |
| `torch_native` | 支持 | 不支持 |

## 签名

```python
from sgl_kernel_npu.operators.attention import rain_fusion_attention

def rain_fusion_attention(
    query,
    key,
    value,
    select_idx,
    select_num_idx,
    blockshape,
    attn_mask=None,
    actual_seq_qlen=None,
    actual_seq_kvlen=None,
    block_table=None,
    q_input_layout="TND",
    kv_input_layout="TND",
    head_num=1,
    mask_type=0,
    scale=1.0,
    inner_precise=1,
    block_size=0,
    impl="auto",
):
    ...
```

`auto` 在 arch22 上选择 `cann_ops_transformer`。没有回退路径：在 arch35 上请求该算子会直接报错。
同时导出旧名字 `rainfusionattention` 作为别名。

## 功能说明

按 Q 块选中的 KV 块执行块稀疏注意力，只有被选中的块参与
`query @ key^T`、softmax 和 `probs @ value` 的计算：

```text
attention_out = softmax(scale * (query @ key^T)) @ value
```

稀疏模式由 `select_idx` / `select_num_idx` 给出，块大小由 `blockshape` 指定。

## 参数与返回值

`B` 为 batch，`N` 为头数，`S` 为 query 序列长度，`D` 为 head 维度。

| 参数 | 形状 | 数据类型 / 默认值 | 说明 |
| --- | --- | --- | --- |
| `query` | TND `[T, N, D]` 或 BNSD `[B, N, S, D]` | FP16 / BF16 | 布局由 `q_input_layout` 决定。 |
| `key`、`value` | TND `[Tkv, Nkv, D]` 或 BNSD | 与 `query` 相同 | 布局由 `kv_input_layout` 决定。 |
| `select_idx` | `[q_blocks, N, kv_blocks]` | INT64 | 有效 KV 块索引在前并按升序排列，无效位置填 `-1`。 |
| `select_num_idx` | `[q_blocks, N]` | INT64 | 每个 Q 块、每个头实际选中的 KV 块数量。 |
| `blockshape` | `[block_rows, block_cols]` | 必填 | Host 侧整数块大小。 |
| `attn_mask` | — | `None` | 作为可选 ACLNN tensor 透传，本层不做校验。 |
| `actual_seq_qlen`、`actual_seq_kvlen` | `[B]` | `None` | TND 打包输入的 Host 整数序列边界。 |
| `block_table` | — | `None` | PagedAttention 场景透传的可选 ACLNN tensor。 |
| `q_input_layout`、`kv_input_layout` | — | str，`"TND"` | 只接受 `TND` 和 `BNSD`，其他取值直接报错。 |
| `head_num` | — | int，1 | 原样传给 ACLNN 的 `numKeyValueHeads` 槽位。 |
| `mask_type` | — | int，0 | ACLNN mask 类型。 |
| `scale` | — | float，1.0 | 分数缩放系数，通常取 `D ** -0.5`。 |
| `inner_precise` | — | int，1 | Softmax 精度模式。 |
| `block_size` | — | int，0 | PagedAttention 的 block 大小；0 表示不使用。 |
| `impl` | — | `"auto"` | 仅关键字参数：`auto`、`cann_ops_transformer`。 |

| 返回值 | 形状 | 数据类型 |
| --- | --- | --- |
| `attention_out` | 与 `query` 相同 | 与 `query` 相同 |
| `softmax_lse` | `query` 的前三个维度 | 与 `query` 相同 |

## cann_ops_transformer 后端

- 调用 CANN toolkit 自带的 `aclnnRainFusionAttention`，符号解析顺序与其他算子一致：
  先查 wheel 内置的 `libcust_opapi.so`，未找到时回退到 CANN toolkit 的
  `libopapi.so`。不需要额外 vendor 包，也不需要设置 `ASCEND_CUSTOM_OPP_PATH`。
- 两个输出在下发前先用 query 的 options 和 query 的 NPU format 分配，与旧绑定一致。
- `softmax_lse` 仅为保持签名兼容而分配。当前 CANN 算子可能不写入该张量，
  其内容视为未定义，请只校验 `attention_out`。
- 序列长度和 `blockshape` 都是 Host 侧整数序列，不能传 NPU tensor。
  其中 `blockshape[1]`（KV 方向块大小）必须是 128 的倍数，否则 tiling 直接报错。
- 本层不额外增加校验、类型转换或连续性拷贝；输入必须已经满足算子契约。

## torch_native 后端

- 纯 PyTorch FP32 参考实现，用于精度对照；结果按 query dtype 转回。
- 只实现 `mask_type=0`（无 mask），不支持 `attn_mask` 和 `block_table`。
- `select_idx` / `select_num_idx` 按 `[Q 块, 头]` 展平读取；GQA 下 query 头按
  连续分组映射到 KV 头，与内核的 `head // (query_heads / kv_heads)` 一致。
- 设备内核可能不写 `softmax_lse`，参考实现返回同形状零张量；测试只比较
  `attention_out`。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import rain_fusion_attention

batch, heads, seq, dim, block = 1, 2, 256, 128, 128
query = torch.randn(batch * seq, heads, dim, device="npu", dtype=torch.float16)
key = torch.randn(batch * seq, heads, dim, device="npu", dtype=torch.float16)
value = torch.randn(batch * seq, heads, dim, device="npu", dtype=torch.float16)

q_blocks, kv_blocks = seq // block, seq // block
select_idx = torch.arange(kv_blocks, dtype=torch.int64, device="npu")
select_idx = select_idx.repeat(q_blocks, heads, 1)
select_num_idx = torch.full((q_blocks, heads), kv_blocks, dtype=torch.int64, device="npu")

out, _ = rain_fusion_attention(
    query,
    key,
    value,
    select_idx,
    select_num_idx,
    [block, block],
    actual_seq_qlen=[seq] * batch,
    actual_seq_kvlen=[seq] * batch,
    head_num=heads,
    scale=dim**-0.5,
)
print(out.shape)  # [256, 2, 128]
```
