## Introduction
A top-k, top-p and min-p sampling implementation for ascend.

## Sheet 1: Parameters
| Parameter    | Dimension                | Data Type            | Format | Description                                      |
|--------------|--------------------------|----------------------|--------|--------------------------------------------------|
| probs        | [batch_size, vocab_size] | float32/float16/bf16 | ND     | Probabilities for sampling.<br>The probabilities should be sorted in descending order. |
| k            | [batch_size]             | int32                | ND     | Representing the threshold for top-k sampling.   |
| p            | [batch_size]             | float32/float16/bf16 | ND     | Representing the threshold for top-p sampling.   |
| min_p        | [batch_size]             | float32/float16/bf16 | ND     | Representing the threshold for min-p sampling.<br>When min_p is nullptr, the min-p sampling will be skipped.  |
| sampled_res  | [batch_size, vocab_size] | float32/float16/bf16 | ND     | The result after sampling.<br>The DataType of sampled_res should be same as probs. |

## Calculation Formula
$$
sampled\_res[b][v] =
\begin{cases}
0 & \text{v >= k[b]} \\
probs[b][v] & \text{v < k[b]}
\end{cases}
$$
$$probs\_sum = cumsum(sampled\_res, dim=-1)$$
$$top\_p\_mask[b][v] = probs\_sum[b][v] - sampled\_res[b][v] > p[b]$$
$$
sampled\_res[b][v] =
\begin{cases}
0 & \text{top\_p\_mask = True} \\
sampled\_res[b][v] & \text{top\_p\_mask = False}
\end{cases}
$$
$$min\_p\_mask[b][v] = sampled\_res[b][v] < sampled\_res[b][0] * min\_p[b]$$
$$
sampled\_res[b][v] =
\begin{cases}
0 & \text{min\_p\_mask = True} \\
sampled\_res[b][v] & \text{min\_p\_mask = False}
\end{cases}
$$
Where $0 \le b \lt batch\_size$, and $0 \le v \lt vocab\_size$.

## Restrictions
1. Only support Ascend A2/A3.
2. $0 \lt k[b] \le vocab\_size$, where $0 \le b \lt batch\_size$, if $k[b] \lt 0$ or $k[b] \gt vocab\_size$, the $k[b]$ will regarded as vocab\_size.
2. $0 \le p[b] \le 1$, where $0 \le b \lt batch\_size$.

## Sample Code
```python
import numpy as np
import torch
import torch_npu
import sgl_kernel_npu

dtype = torch.float16
batch_size = 4
vocab_size = 128

logits = torch.tensor(np.random.uniform(-10, 10, (batch_size, vocab_size))).to(dtype).npu()
k = torch.tensor(np.random.randint(1, vocab_size, (batch_size))).to(torch.int32).npu()
p = torch.tensor(np.random.uniform(0, 1, (batch_size))).to(dtype).npu()
min_p = torch.tensor(np.random.uniform(0, 1, (batch_size))).to(dtype).npu()

probs = torch.softmax(logits, dim=-1)
probs_sort, probs_idx = probs.sort(dim=-1, descending=True, stable=True)

torch.ops.npu.apply_top_k_top_p_min_p(probs_sort, k, p, min_p=min_p)
```
