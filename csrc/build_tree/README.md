# torch.ops.build_tree_kernel_efficient


## Function Description | 功能描述

### English:
This is the AscendC version `build_tree_kernel_efficient` kernel function, which organizes the draft model’s multi-step top-k candidate tokens into a verification tree。

Adapted from [CUDA Implementation](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/speculative/eagle_utils.cu)

For each sample it concurrently builds

- tree_mask – which nodes must be verified by the target model
- positions – absolute position of each node in the full sequence
- retrive_* linked lists – allow O(1) navigation to children & siblings

### 中文:
这是AscendC版本的`build_tree_kernel_efficient`内核方法，它将draft模型产生的多步top-k候选token组织成验证树（verification tree）

引用自 [CUDA 实现](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/speculative/eagle_utils.cu)

内核为每个样本并行构造

- 树掩码 tree_mask（标记哪些节点需要被大模型验证）
- 位置编码 positions（节点在完整序列中的位置）
- 检索链表 retrive_*（支持 O(1) 找到子节点与兄弟节点）


## Interface Prototype | 接口原型

### Python Binding Definition
```python
import sgl_kernel_npu

torch.ops.npu.build_tree_kernel_efficient(
    parent_list: torch.Tensor,           # int64, [batch_size, topk*(depth-1)+1]
    selected_index: torch.Tensor,        # int64, [batch_size, draft_token_num-1]
    verified_seq_len: torch.Tensor,      # int64, [batch_size]
    tree_mask: torch.Tensor,             # bool,  [draft_token*(seq_len[0]+draft_token) | draft_token*(seq_len[1]+draft_token) | ..] = [sum(verified_seq_len)*draft_token+bs*draft_token*draft_token]
    positions: torch.Tensor,             # int64, [batch_size, draft_token_num]
    retrive_index: torch.Tensor,         # int64, [batch_size, draft_token_num]
    retrive_next_token: torch.Tensor,    # int64, [batch_size, draft_token_num]
    retrive_next_sibling: torch.Tensor,  # int64, [batch_size, draft_token_num]
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: int
) -> None
```

### Kernel Definition | 核函数定义
```C++
extern "C" __global__ __aicore__ void build_tree_efficient(GM_ADDR parent_list,
    GM_ADDR selected_index,
    GM_ADDR verified_seq_len,
    GM_ADDR tree_mask,
    GM_ADDR positions,
    GM_ADDR retrive_index,
    GM_ADDR retrive_next_token,
    GM_ADDR retrive_next_sibling,
    GM_ADDR workspace_in,
    GM_ADDR tiling_in)
```

## Parameter Description | 参数说明

| Parameter Name (参数名称) | DataType (数据类型) | Description                               | 说明                         |
|:----------------------|:----------------|:------------------------------------------|:---------------------------|
| `parent_list`         | `torch.Tensor`  | parent id of every draft token            | 每个 draft token 的父节点 id     |
| `selected_index`      | `torch.Tensor`  | flat index of sampled token in top-k list | 采样的 token 在 top-k 列表中的扁平索引 |
| `verified_seq_len`    | `torch.Tensor`  | length of already-verified prefix         | 当前已验证序列长度                  |
| `topk`                | `int`           | branching factor per step                 | 每步分支数                      |
| `depth`               | `int`           | maximum speculative depth                 | 最大投机深度                     |
| `draft_token_num`     | `int`           | total #draft tokens per sample            | 单样本 draft token 总数         |
| `tree_mask_mode`      | `int`           | mask layout mode (1=FULL\_MASK)           | 掩码布局模式（1=FULL\_MASK）       |


## Output Description | 输出说明

| Parameter Name (参数名称)  | DataType (数据类型) | Description                        | 说明                 |
|:-----------------------|:----------------|:-----------------------------------|:-------------------|
| `tree_mask`            | `torch.Tensor`  | true → node must be verified       | true → 该节点需被验证     |
| `positions`            | `torch.Tensor`  | absolute position in full sequence | 节点在完整序列中的绝对位置      |
| `retrive_index`        | `torch.Tensor`  | node → flat index for quick lookup | 快速检索：节点→扁平索引       |
| `retrive_next_token`   | `torch.Tensor`  | first child id (-1 = none)         | 第一个子节点 id（-1 表示无）  |
| `retrive_next_sibling` | `torch.Tensor`  | next sibling id (-1 = none)        | 下一个兄弟节点 id（-1 表示无） |


## Constraints | 约束说明

### English:
`TreeMaskMode.QLEN_ONLY_BITPACKING = 2` is not implemented

### 中文:
`TreeMaskMode.QLEN_ONLY_BITPACKING = 2` 暂未实现

## Example | 调用示例

```python
import math
import sgl_kernel_npu
import torch
import torch_npu

device = torch.device('npu:0')

topk = 4
depth = 4
num_verify_tokens = 8

parent_list=...
top_scores_index=...
seq_lens=...

bs = seq_lens.numel()

tree_mask_mode = TreeMaskMode.FULL_MASK
if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
    tree_mask = torch.full(
        (num_verify_tokens * bs * num_verify_tokens,),
        True,
        dtype=torch.bool,
        device=device,
    )
elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
    packed_dtypes = [torch.uint8, torch.uint16, torch.uint32]
    packed_dtype_idx = int(math.ceil(math.log2((num_verify_tokens + 7) // 8)))
    tree_mask = torch.zeros(
        (num_verify_tokens * bs,),
        dtype=packed_dtypes[packed_dtype_idx],
        device=device,
    )
elif tree_mask_mode == TreeMaskMode.FULL_MASK:
    tree_mask = torch.full(
        (
            seq_lens_sum * num_verify_tokens
            + num_verify_tokens * num_verify_tokens * bs,
        ),
        True,
        device=device,
    )
else:
    raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")

retrive_index = torch.full(
    (bs, num_verify_tokens), -1, device=device, dtype=torch.long
)
retrive_next_token = torch.full(
    (bs, num_verify_tokens), -1, device=device, dtype=torch.long
)
retrive_next_sibling = torch.full(
    (bs, num_verify_tokens), -1, device=device, dtype=torch.long
)

positions = torch.empty(
    (bs * num_verify_tokens,), device=device, dtype=torch.long
)

torch.ops.npu.build_tree_kernel_efficient(
    parent_list,
    top_scores_index,
    seq_lens,
    tree_mask,
    positions,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    topk,
    depth,
    num_verify_tokens,
    tree_mask_mode,
)
```
