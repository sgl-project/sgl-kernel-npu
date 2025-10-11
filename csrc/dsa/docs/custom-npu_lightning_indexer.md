# custom.npu\_lightning\_indexer<a name="ZH-CN_TOPIC_0000001979260729"></a>

## 产品支持情况 <a name="zh-cn_topic_0000001832267082_section14441124184110"></a>
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 推理系列产品</term>   | √  |

## 功能说明<a name="zh-cn_topic_0000001832267082_section14441124184110"></a>

`LightningIndexer`基于一系列操作得到每一个 token 对应的 Top-$k$ 个位置。对于某个 token 对应的 Index Query $Q_{index}\in\R^{g\times d}$，给定上下文 Index Key $K_{index}\in\R^{S_{k}\times d},W\in\R^{g\times 1}$，其中 $g$ 为 GQA 对应的 group size，$d$ 为每一个头的维度，$S_{k}$ 是上下文的长度，`LightningIndexer`的具体计算公式如下：
$$
\text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right]\right\}
$$

## 函数原型<a name="zh-cn_topic_0000001832267082_section45077510411"></a>

```
custom.npu_lightning_indexer(query, key, weights, *, actual_seq_lengths_query=None, actual_seq_lengths_key=None, block_table=None, layout_query='BSND', layout_key='PA_BSND', sparse_count=2048, sparse_mode=3) -> Tensor
```

## 参数说明<a name="zh-cn_topic_0000001832267082_section112637109429"></a>

>**说明：**<br> 
>
>- query、key、weights参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
>- S1表示query shape中的S，S2表示key shape中的S，N1表示query shape中的N，N2表示key shape中的N。

-   **query**（`Tensor`）：必选参数，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`。
    
-   **key**（`Tensor`）：必选参数，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，layout\_key为PA_BSND时shape为[block\_count, block\_size, N2, D]，其中block\_count为PageAttention时block总数，block\_size为一个block的token数。
    
-   **weights**（`Tensor`）：必选参数，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，支持输入shape[B,S1,N1,1]、[T,N1,1]。
    
- <strong>*</strong>：代表其之前的参数是位置相关的，必须按照顺序输入，属于必选参数；其之后的参数是键值对赋值，与位置无关，属于可选参数（不传入会使用默认值）。

-   **actual\_seq\_lengths\_query**（`Tensor`）：可选参数，表示不同Batch中`query`的有效token数，数据类型支持`int32`。如果不指定seqlen可传入None，表示和`query`的shape的S长度相同。
    -   该入参中每个Batch的有效token数不超过`query`中的维度S大小。支持长度为B的一维tensor。当`query`的input\_layout为TND时，该入参必须传入，且以该入参元素的数量作为B值，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值。不能出现负值。

-   **actual\_seq\_lengths\_key**（`Tensor`）：可选参数，表示不同Batch中`key`的有效token数，数据类型支持`int32`。如果不指定seqlen可传入None，表示和key的shape的S长度相同。支持长度为B的一维tensor。

-   **block\_table**（`Tensor`）：可选参数，表示PageAttention中KV存储使用的block映射表，数据格式支持ND，数据类型支持`int32`。
    -   PageAttention场景下，block\_table必须为二维，第一维长度需要等于B，第二维长度不能小于maxBlockNumPerSeq(maxBlockNumPerSeq为每个batch中最大actual\_seq\_lengths\_key对应的block数量)

-   **layout\_query**（`str`）：可选参数，用于标识输入`query`的数据排布格式，当前支持BSND、TND，默认值"BSND"。

-   **layout\_key**（`str`）：可选参数，用于标识输入`key`的数据排布格式，当前支持PA_BSND，默认值"PA_BSND"。

-   **sparse\_count**（`int`）：可选参数，代表topK阶段需要保留的block数量，支持1-2048，数据类型支持`int32`。

-   **sparse\_mode**（`int`）：可选参数，表示sparse的模式，支持0/3，数据类型支持`int32`。
    
    -   sparse\_mode为0时，代表defaultMask模式。
    -   sparse\_mode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。

## 返回值说明<a name="zh-cn_topic_0000001832267082_section22231435517"></a>

-   **out**（`Tensor`）：公式中的输出，数据类型支持`bfloat16`。数据格式支持ND。

## 约束说明<a name="zh-cn_topic_0000001832267082_section12345537164214"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式。
-   该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
-   参数query中的N支持64，key中的N支持1。
-   参数query中的D和参数key中的D值相等为128。
-   支持block_size取值为16的整数倍，最大支持到1024。

## 调用示例<a name="zh-cn_topic_0000001832267082_section14459801435"></a>

-   详见[test_npu_lightning_indexer.py](../examples/test_npu_lightning_indexer.py)