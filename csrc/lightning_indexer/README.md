# torch.ops.npu.lightning_indexer<a name="ZH-CN_TOPIC_0000001979260729"></a>

## Product Support Status <a name="zh-cn_topic_0000001832267082_section14441124184110"></a>
| Product                                                         | Supported |
| ------------------------------------------------------------ | :-------: |
|<term>Atlas A3 Inference Product Series</term>   | √  |

## Function Description<a name="zh-cn_topic_0000001832267082_section14441124184110"></a>

`LightningIndexer` computes the Top-$k$ positions corresponding to each token based on a series of operations. For an Index Query $Q_{index}\in\R^{g\times d}$ corresponding to a certain token, given the context Index Key $K_{index}\in\R^{S_{k}\times d},W\in\R^{g\times 1}$, where $g$ is the group size for GQA, $d$ is the dimension of each head, and $S_{k}$ is the context length, the specific calculation formula for `LightningIndexer` is as follows:
$$
\text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right]\right\}
$$

## Function Prototype<a name="zh-cn_topic_0000001832267082_section45077510411"></a>

```
torch.ops.npu.lightning_indexer(query, key, weights, actual_seq_lengths_query=None, actual_seq_lengths_key=None, block_table=None, layout_query='BSND', layout_key='BSND', sparse_count=2048, sparse_mode=3) -> Tensor
```

## Parameter Description<a name="zh-cn_topic_0000001832267082_section112637109429"></a>

>**Note:**<br>
>
>- Dimension meanings for query, key, and weights parameters: B (Batch Size) represents the batch size of input samples, S (Sequence Length) represents the sequence length of input samples, H (Head Size) represents the size of the hidden layer, N (Head Num) represents the number of attention heads, D (Head Dim) represents the smallest unit dimension of the hidden layer, satisfying D=H/N, T represents the cumulative sum of sequence lengths for all batch input samples.
>- S1 represents the S dimension in query shape, S2 represents the S dimension in key shape, N1 represents the N dimension in query shape, N2 represents the N dimension in key shape.

-   **query** (`Tensor`): Required parameter, non-contiguous tensors not supported. Data layout supports ND format. Data types supported: `bfloat16` and `float16`.

-   **key** (`Tensor`): Required parameter, non-contiguous tensors not supported. Data layout supports ND format. Data types supported: `bfloat16` and `float16`. When layout_key is 'PA_BSND', the shape is [block_count, block_size, N2, D], where block_count is the total number of blocks in PageAttention, and block_size is the number of tokens in one block.

-   **weights** (`Tensor`): Required parameter, non-contiguous tensors not supported. Data layout supports ND format. Data types supported: `bfloat16` and `float16`. Supported input shapes: [B,S1,N1], [T,N1].

- <strong>*</strong>: Represents that parameters before it are position-dependent and must be provided in order (required parameters); parameters after it are keyword arguments, position-independent, and optional (default values will be used if not provided).

-   **actual_seq_lengths_query** (`Tensor`): Optional parameter, represents the number of valid tokens for `query` in different batches. Data type supported: `int32`. If sequence length is not specified, None can be passed, indicating it's the same as the S dimension length of `query`'s shape.
    -   The number of valid tokens for each batch in this parameter must not exceed the S dimension size in `query`. Supports a 1D tensor of length B. When `query`'s input_layout is 'TND', this parameter must be provided, and the number of elements in this parameter is used as the B value. Each element's value in this parameter represents the cumulative sum of tokens for the current batch and all previous batches (prefix sum), so the value of a later element must be >= the value of the previous element. Negative values are not allowed.

-   **actual_seq_lengths_key** (`Tensor`): Optional parameter, represents the number of valid tokens for `key` in different batches. Data type supported: `int32`. If sequence length is not specified, None can be passed, indicating it's the same as the S dimension length of key's shape. Supports a 1D tensor of length B.

-   **block_table** (`Tensor`): Optional parameter, represents the block mapping table used for KV storage in PageAttention. Data layout supports ND format. Data type supported: `int32`.
    -   In PageAttention scenarios, block_table must be 2D, with the first dimension length equal to B, and the second dimension length not less than maxBlockNumPerSeq (maxBlockNumPerSeq is the maximum number of blocks corresponding to actual_seq_lengths_key for each batch).

-   **layout_query** (`str`): Optional parameter, identifies the data layout format of input `query`. Currently supports: 'BSND', 'TND'. Default value: "BSND".

-   **layout_key** (`str`): Optional parameter, identifies the data layout format of input `key`. Currently supports: 'PA_BSND', 'BSND', 'TND'. Default value: "BSND". In non-PageAttention scenarios, this parameter value should be consistent with **layout_query**.

-   **sparse_count** (`int`): Optional parameter, represents the number of blocks to retain during the topK phase. Supports values 1-2048. Data type supported: `int32`.

-   **sparse_mode** (`int`): Optional parameter, specifies the sparse mode. Supports values 0/3. Data type supported: `int32`.

    -   When sparse_mode is 0, it represents defaultMask mode.
    -   When sparse_mode is 3, it represents rightDownCausal mode mask, corresponding to the lower triangular scenario divided by the right vertex.

## Return Value Description<a name="zh-cn_topic_0000001832267082_section22231435517"></a>

-   **out** (`Tensor`): Output from the formula, data type supported: `int32`. Data layout supports ND format.

## Constraints<a name="zh-cn_topic_0000001832267082_section12345537164214"></a>

-   This interface supports inference scenarios.
-   This interface supports graph mode.
-   When used with PyTorch, the versions of CANN-related packages and PyTorch-related packages must be compatible.
-   Parameter N in query supports 64, parameter N in key supports 1.
-   Parameter D in query and parameter D in key must be equal to 128.
-   Data types of parameters query, key, and weights must be consistent.
-   Supports block_size values that are multiples of 16, with maximum support up to 1024.

## Usage Example<a name="zh-cn_topic_0000001832267082_section14459801435"></a>

-   See details in [test_lightning_indexer.py](../../tests/python/sgl_kernel_npu/test_lightning_indexer.py)
