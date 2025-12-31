# torch.ops.npu.sparse_flash_attention<a name="ZH-CN_TOPIC_0000001979260730"></a>

## Product Support Status <a name="zh-cn_topic_0000001832267083_section14441124184110"></a>
| Product                                                         | Supported |
| ------------------------------------------------------------ | :-------: |
|<term>Atlas A3 Inference Product Series</term>   | √  |

## Function Description<a name="zh-cn_topic_0000001832267083_section14441124184110"></a>

`SparseFlashAttention` implements a sparse variant of the Flash Attention algorithm optimized for NPU hardware. It computes attention scores with sparse computation patterns to improve efficiency for long sequences. The operation supports various layouts, RoPE (Rotary Positional Encoding), and PageAttention scenarios.

Given query $Q \in \R^{B \times S_q \times N_q \times D}$, key $K \in \R^{B \times S_k \times N_k \times D}$, value $V \in \R^{B \times S_k \times N_k \times D}$, and sparse indices specifying which attention blocks to compute, the operation computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} \odot M_{\text{sparse}}\right)V
$$

where $M_{\text{sparse}}$ is a sparse mask defined by `sparse_indices` and `sparse_mode`, and $d$ is the head dimension.

## Function Prototype<a name="zh-cn_topic_0000001832267083_section45077510411"></a>

```
torch.ops.npu.sparse_flash_attention(query, key, value, sparse_indices, scale_value, sparse_block_size, block_table=None, actual_seq_lengths_query=None, actual_seq_lengths_kv=None, query_rope=None, key_rope=None, attention_out, layout_query=None, layout_kv=None, sparse_mode=None) -> Tensor
```

## Parameter Description<a name="zh-cn_topic_0000001832267083_section112637109429"></a>

>**Note:**<br>
>
>- Dimension meanings for query, key, and value parameters: B (Batch Size) represents the batch size of input samples, S (Sequence Length) represents the sequence length of input samples, N (Head Num) represents the number of attention heads, D (Head Dim) represents the dimension of each attention head.
>- S_q represents the S dimension in query shape, S_k represents the S dimension in key/value shape, N_q represents the N dimension in query shape, N_k represents the N dimension in key/value shape.

-   **query** (`Tensor`): Required parameter, non-contiguous tensors not supported. Data layout supports 'BSND' and 'TND' formats. Data types supported: `bfloat16` and `float16`.

-   **key** (`Tensor`): Required parameter, non-contiguous tensors not supported. Data layout supports 'BSND', 'TND', and 'PA_BSND' (PageAttention) formats. Data types supported: `bfloat16` and `float16`.

-   **value** (`Tensor`): Required parameter, non-contiguous tensors not supported. Data layout supports 'BSND', 'TND', and 'PA_BSND' (PageAttention) formats. Data types supported: `bfloat16` and `float16`.

-   **sparse_indices** (`Tensor`): Required parameter, specifies which attention blocks to compute. Data type supported: `int32`. Defines the sparse computation pattern.

- <strong>*</strong>: Represents that parameters before it are position-dependent and must be provided in order (required parameters); parameters after it are keyword arguments, position-independent, and optional (default values will be used if not provided).

-   **block_table** (`Tensor`): Optional parameter, represents the block mapping table used for KV storage in PageAttention. Data type supported: `int32`. Required for PageAttention scenarios.

-   **actual_seq_lengths_query** (`Tensor`): Optional parameter, represents the number of valid tokens for `query` in different batches. Data type supported: `int32`. If sequence length is not specified, None can be passed, indicating it's the same as the S dimension length of `query`'s shape.

-   **actual_seq_lengths_kv** (`Tensor`): Optional parameter, represents the number of valid tokens for `key`/`value` in different batches. Data type supported: `int32`. If sequence length is not specified, None can be passed, indicating it's the same as the S dimension length of key's shape.

-   **query_rope** (`Tensor`): Optional parameter, Rotary Positional Encoding (RoPE) for query. Data types supported: `bfloat16` and `float16`.

-   **key_rope** (`Tensor`): Optional parameter, Rotary Positional Encoding (RoPE) for key. Data types supported: `bfloat16` and `float16`.

-   **scale_value** (`float`): Required parameter, scaling factor for attention scores. Typically $1/\sqrt{d}$ where $d$ is the head dimension.

-   **sparse_block_size** (`int`): Required parameter, size of sparse blocks for computation. Defines the granularity of sparse attention.

-   **layout_query** (`str`): Optional parameter, identifies the data layout format of input `query`. Currently supports: 'BSND', 'TND'. Default value: "BSND".

-   **layout_kv** (`str`): Optional parameter, identifies the data layout format of input `key`/`value`. Currently supports: 'PA_BSND', 'BSND', 'TND'. Default value: "BSND". In non-PageAttention scenarios, this parameter value should be consistent with **layout_query**.

-   **sparse_mode** (`int`): Optional parameter, specifies the sparse attention mode. Supports values 0/3. Default value: 3.
    -   When sparse_mode is 0, it represents defaultMask mode.
    -   When sparse_mode is 3, it represents rightDownCausal mode mask, corresponding to the lower triangular scenario divided by the right vertex.

## Return Value Description<a name="zh-cn_topic_0000001832267083_section22231435517"></a>

-   **attention_out** (`Tensor`): Output attention tensor, same as input `attention_out` parameter. Data types supported: `bfloat16` and `float16`.

## Constraints<a name="zh-cn_topic_0000001832267083_section12345537164214"></a>

-   This interface supports inference scenarios.
-   This interface supports graph mode.
-   When used with PyTorch, the versions of CANN-related packages and PyTorch-related packages must be compatible.
-   Parameter D (head dimension) in query, key, and value must be equal and support typical values like 128.
-   Data types of parameters query, key, value, query_rope, and key_rope must be consistent.
-   Supports various sequence lengths and batch sizes optimized for NPU hardware.
-   Sparse block size must be compatible with the hardware constraints and sequence dimensions.

## Usage Example<a name="zh-cn_topic_0000001832267083_section14459801435"></a>

```python
import torch
import torch_npu
import numpy as np

# Configuration
batch, s_q, s_kv, n_q, n_kv, head_dim, rope_dim = 4, 1, 4096, 128, 1, 512, 64
sparse_block_size = 1  # Granularity of sparse computation
sparse_block_count = 2048

# Initialize tensors on NPU
query = torch.randn(batch, s_q, n_q, head_dim, dtype=torch.float16).npu()
key = torch.randn(batch, 8192, n_kv, head_dim, dtype=torch.float16).npu()
value = key.clone()

# Create sparse indices: (batch, s_q, n_kv, sparse_block_count)
# Specifies which blocks in the KV sequence each query head attends to
idxs = torch.randint(0, s_kv, (sparse_block_count,), dtype=torch.int32).npu()
sparse_indices = idxs.repeat(batch * s_q * n_kv).reshape(batch, s_q, n_kv, sparse_block_count)

# Optional RoPE and sequence lengths
query_rope = torch.randn(batch, s_q, n_q, rope_dim, dtype=torch.float16).npu()
key_rope = torch.randn(batch, 8192, n_kv, rope_dim, dtype=torch.float16).npu()
act_seq_q = torch.tensor([s_q] * batch, dtype=torch.int32).npu()
act_seq_kv = torch.tensor([s_kv] * batch, dtype=torch.int32).npu()

# Execute Sparse Flash Attention
scale_value = 1.0 / (head_dim ** 0.5)
output = torch.ops.npu.sparse_flash_attention(
    query, key, value, sparse_indices, scale_value, sparse_block_size,
    actual_seq_lengths_query=act_seq_q,
    actual_seq_lengths_kv=act_seq_kv,
    query_rope=query_rope,
    key_rope=key_rope,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=3
)

print(f"BSND Output shape: {output.shape}")
```

For more detailed examples and test cases, refer to the test files in the repository.
