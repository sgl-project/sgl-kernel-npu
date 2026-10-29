# DSV4 DSpark Tree 四算子说明

本文说明 `sgl-kernel-npu` 中 DSV4 DSpark tree 路径的四个算子分别做什么，
以及为什么从 chain 切换到 tree 后不能只复用原来的 chain 算子。

四个算子不是四个独立模型，而是同一个 DSpark verify 流程在“树形候选”
场景下拆出来的四个阶段：

```text
compressor_tree_project   (project)
        ↓
compressor_tree           (compressor)
        ↓
quant_lightning_indexer_tree (indexer)
        ↓
sparse_attn_sharedkv_tree (attention)
```

---

## 1. 背景：chain 与 tree 的根本区别

### 1.1 Chain

DSpark chain（`topk=1`）的每个 request 只有一条候选路径：

```text
prefix -> draft token 1 -> draft token 2 -> ... -> draft token gamma
```

目标 verify 时，root/prefix 后面是一条连续序列：

- 每行 query 只看到自己的前缀；
- KV 可以按线性 token 顺序追加；
- 不需要“某个节点属于哪条分支”的信息；
- 标准 causal/band mask 就足够。

因此 chain 可以使用线性序列上的标准 compressor、indexer、attention。

### 1.2 Tree

DSpark tree（`topk>1`）的同一个 request 在 root 之后有多个候选节点：

```text
                    ┌─ node A ── node C
root/prefix ────────┤
                    └─ node B ── node D
```

每个节点有自己的：

- hidden state；
- `prefix`（该 request 的历史起点）；
- `depth`（树深度）；
- `ancestors`（root-first、包含自身的祖先链）；
- 自己的 compressed KV / node KV；
- 自己的 attention 可见集合。

关键点：

1. `node A` 不能看到 `node B`，即使它们在同一层；
2. `node C` 只能看到 root、A、C 自己的祖先链；
3. tree 的 query 行数由 `1 + budget` 决定，不再是 chain 的 `gamma + 1`；
4. 候选节点在 verify 后只有被接受的路径会提交，未被接受的节点不能污染正式 KV cache。

所以 tree 不是“把 chain 的输出 topk 改大”，而是**改变 verify 行的组织和可见性**。

---

## 2. 四个算子总览

| 算子 | 角色 | 主要输入 | 主要输出 |
|---|---|---|---|
| `compressor_tree_project` | 把每个 node 的 hidden 投影成 node 级 KV/score 两半 | node hidden、wkv/wgate、APE、residue | FP32 `nodes` 池 |
| `compressor_tree` | 沿每个 node 的历史窗口 + 祖先链做分支感知压缩 | paged state、projected nodes、cos/sin、tree 几何 | 每个 node 的 BF16 compressed KV |
| `quant_lightning_indexer_tree` | 为每个 node 选择 compressed KV 候选 slot | INT8 Q/K、node K/scale、page table、ancestors | `[n,1,topk]` int32 logical indices |
| `sparse_attn_sharedkv_tree` | 对原始窗口、压缩 KV、tree node KV 做稀疏 attention | Q、ori/cmp KV、topk indices、node KV、ancestors | attention 输出 + LSE |

注意两个不同的 `topk`：

- tree 配置里的 `topk` 是树的分叉因子；
- indexer 的 `topk` 是每个 node 选择的 compressed slot 数。

两者不是同一个概念。

---

## 3. 数据流

```text
每个 node 的 hidden[T, H]
        │
        ▼
compressor_tree_project
        │  nodes [n+1, 1, 2*c*d]  (FP32)
        ▼
compressor_tree
        │  node_kv [n, d]          (BF16)
        ├───────────────┐
        │               │
        ▼               ▼
quantize_indexer_nodes   sparse_attn_sharedkv_tree
        │  int8 nodes / fp16 scale       ▲
        ▼                                 │
quant_lightning_indexer_tree              │
        │  topk indices [n,1,K]           │
        └─────────────────────────────────┘
```

其中：

- `compressor_tree_project` 的 `nodes` 是**投影后的 FP32 中间池**；
- `compressor_tree` 的 `node_kv` 是**分支感知压缩后的 BF16 KV**；
- indexer 使用 `node_kv` 动态量化后的 INT8 `nodes` 和 FP16 `node_scale`；
- attention 使用未量化的 `node_kv` 作为 `cmp_tree_kv`。

---

## 4. `compressor_tree_project`

### 4.1 做什么

对每个树节点独立做两个投影：

```text
kv    = x @ wkv.T
score = x @ wgate.T + ape[(prefix + depth) % ratio]
```

然后写入该节点的池化行：

```text
nodes[node + 1] = [ kv(c*d) | score(c*d) ]
```

`nodes[0]` 是保留行，不写。

### 4.2 为什么 tree 需要它

Chain 的压缩器可以直接沿连续 token 序列做投影和聚合。Tree 的每个节点：

- 有自己的 hidden；
- 有自己的 depth/prefix；
- 后面要按祖先链而不是按 token 顺序被聚合。

所以必须先把每个 node 的 hidden 投影成独立的 node 行，供后续
`compressor_tree` 按分支选择。

### 4.3 它不做什么

这个算子：

- 不做窗口聚合；
- 不做 RMSNorm；
- 不做 RoPE；
- 不产生最终 compressed KV。

它只做 dense 投影和 APE 加法。把这一步单独拆出来，可以让
project 保持规则 GEMM tile，而把复杂的祖先选择留给 compressor。

---

## 5. `compressor_tree`

### 5.1 做什么

对每个 node，沿它的局部窗口做 branch-aware 压缩：

```text
对窗口中的每个位置 p:
    if p < prefix:
        key/score 来自 paged state
    else:
        step = p - prefix
        ancestor = ancestors[node, step]
        key/score 来自 nodes[ancestor + 1]

value = online_softmax(score) 加权平均 key
value = RMSNorm(value)
value = RoPE(value, ropePos = pos + 1 - ratio)
```

只对 `(pos + 1) % ratio == 0` 的 node 发射结果；其他 node 输出零行。

C4 (`ratio=4`) 使用两个 interleaved half；C128 (`ratio=128`) 使用单窗口。

### 5.2 为什么 tree 需要它

这是 tree 与 chain 差异最大的地方。

Chain 的压缩窗口是一条线性历史：

```text
prefix 前面的 token 和当前 token 的顺序固定
```

Tree 的窗口必须按节点解析：

- 同一个 request 的不同 node 有不同祖先；
- 不能把兄弟节点的 KV 混进来；
- `p < prefix` 要走 page table；
- `p >= prefix` 要走 ancestor -> node pool；
- 无效祖先或未来位置必须跳过；
- 没有有效 key 的 node 必须输出全零。

如果用 chain compressor，node A 会看到 node B 的 KV，或者把
不同分支的窗口混在一起，压缩结果直接错误。

### 5.3 为什么和 project 分开

Project 是规则 GEMM；compressor 是：

- 数据相关的祖先选择；
- paged/tree 两种来源混合；
- online softmax；
- RMSNorm；
- RoPE。

两者的 tiling、UB 占用和输入布局完全不同。拆开后：

- project 可以保持 BF16 dense tile；
- compressor 可以按 node 一个 program 做小规模归约；
- 单个 kernel 不需要同时承担大 GEMM 和分支 gather。

这也是 vendor 原本就把它拆成 `compressor_tree_project` 和
`compressor_tree` 两个算子的原因。

---

## 6. `quant_lightning_indexer_tree`

### 6.1 做什么

对每个 node 的 query，与 compressed history 和该 node 的祖先 node K
做 INT8 QK，算出每个 compressed slot 的分数，然后选 top-k：

```text
QK[g] = int8_dot(query[node, g, :], key)
S[g]  = fp16(relu(QK[g]) * 2^-10)
Wq[g] = fp16(weights[node, g] * query_scale[node, g])
score = sum_g float(S[g]) * float(Wq[g]) * float(key_scale)
```

候选 slot 的来源：

```text
k < prefix / 4:
    paged history

否则:
    step = (k + 1) * 4 - 1 - prefix
    node = ancestors[node, step]
    key 来自 tree node pool
```

无效 slot：

```text
score = -inf
index = -1
```

最后输出：

```text
[n, 1, topk] int32
```

### 6.2 为什么 tree 需要它

Chain indexer 的候选是线性历史的连续 slot，query 也只有一个固定位置。

Tree indexer 必须解决：

- 每个 node 的 query 不同；
- 每个 node 的祖先链不同；
- 同一个 logical slot 对不同 node 可能映射到不同 node K；
- 某些 slot 对某个 node 不可见，必须变成 `-inf`；
- 不能把兄弟节点的 key 选进来。

所以 tree indexer 不是“把 chain topk 换成更大的 topk”，而是增加了
**按 node 解析 logical slot 到 history/tree 的映射**。

### 6.3 当前 Triton 实现

当前实现分成：

```text
_gather_kernel
    每个 slot 解析一次，写 contiguous keys/scales/valid

_score_kernel
    读 contiguous key 矩阵做 QK/降精度/head reduce

_topk_merge_kernel
    每行做稳定 top-k
```

这样避免在 score kernel 内对每个 head、每个 D chunk 重复做 page/ancestor
gather。

---

## 7. `sparse_attn_sharedkv_tree`

### 7.1 做什么

对每个 tree node 的 query，在以下 key 集合上做 sparse attention：

```text
原始 SWA 窗口
+ compressed KV 中 indexer 选出的 top-k
+ 该 node 的祖先 tree node KV
```

每个 node 的可见性由：

- `ori_sparse_indices`：原始 KV 的物理 slot；
- `cmp_sparse_indices`：compressed KV 的 logical top-k；
- `tree_ancestors`：root-first 祖先链；
- `tree_prefix`：节点的历史起点；
- `cmp_tree_kv`：node KV 池

共同决定。

计算：

```text
scores = qk * softmax_scale
online softmax over valid slots
out = Acc / S
LSE = ln(S) + M
```

可选 sink 作为零值 key，初始权重 1；空行返回：

```text
out = 0
lse = 0
```

### 7.2 为什么 tree 需要它

Chain attention 只需要一条 causal/band mask：

```text
query 看到 prefix + 自己之前的连续 token
```

Tree attention 必须处理：

- 当前 node 只能看到自己的祖先，不看兄弟/后代；
- 原始窗口和 compressed top-k 是不同 key 源；
- compressed top-k 是 logical position，不是物理 cache index；
- tree node KV 不在普通 page table 中；
- 不同 node 的可见集合不同；
- 空行、无效祖先、超出范围的 page 必须安全跳过。

如果直接使用 chain attention，兄弟节点会互相可见，attention 结果错误。

### 7.3 为什么和 chain attention 分开

除了 mask 不同，tree attention 还需要：

- 解析三路 key：原始 KV、compressed KV、tree node KV；
- 对 `cmp_sparse_indices` 做 logical -> paged/tree 的二次解析；
- 维护祖先节点 KV 的行偏移；
- 对无效 slot 做安全地址处理。

因此它是一个独立的 tree 版本，而不是 chain attention 的一个 mask 参数。

---

## 8. 为什么不是“一个 fused 算子”

逻辑上可以把它看成：

```text
tree compressor = project + compressor
tree verify = compressor + indexer + attention
```

但不建议融成一个 kernel，原因有四个：

### 8.1 数学阶段不同

- project：dense GEMM + APE
- compressor：branch gather + online softmax + RMSNorm + RoPE
- indexer：INT8 QK + relu/dequant + top-k
- attention：QK/PV + online softmax + LSE

每一段的数值路径和输出类型都不同。

### 8.2 数据布局不同

- project 是规则 `[T,H] x [H,2cd]`
- compressor 是 paged state + ancestor node pool
- indexer 是 page table + ancestor -> INT8 key
- attention 是原始 KV + compressed KV + node KV

强行融合会让每个 program 同时持有多种布局，UB 和同步都会失控。

### 8.3 Tree 分支需要显式边界

Tree 的正确性依赖：

- `ancestors` 的 root-first 顺序；
- `prefix` / `depth` 的绝对位置；
- 无效祖先的 `-1`；
- node KV 的独立池；
- 只有被接受路径最终提交。

这些边界在四个小算子里更容易验证；融成一个 kernel 后，一处 mask 错误
会污染整条 verify。

### 8.4 测试、回退和性能定位

四个算子可以分别做：

- CPU reference parity；
- 单算子 NPU parity；
- 单算子性能 benchmark；
- 单算子回退 vendor。

这对当前迁移阶段尤其重要。

---

## 9. Chain vs Tree 对照

| 维度 | Chain | Tree (`topk>1`) |
|---|---|---|
| 每个 request 的 verify 行 | `gamma + 1` 连续行 | `1 + budget` 个节点 |
| 路径 | 单一路径 | 多分支，共享祖先 |
| 隐状态 | 连续 token hidden | 每个节点独立 hidden |
| 压缩 | 线性窗口 | 按 node 的祖先链 + 历史窗口 |
| indexer | 单 query / 连续 slot | 每 node top-k + logical slot 解析 |
| attention | causal/band mask | band + ancestor + compressed top-k |
| KV 生命周期 | 直接 append | node KV 临时池，接受路径才提交 |
| 需要的算子 | 标准 compressor/indexer/attention | project + compressor + tree indexer + tree attention |

---

## 10. 实现位置

Triton 实现：

```text
python/sgl_kernel_npu/sgl_kernel_npu/tree/
├── compressor_tree_project.py
├── compressor_tree.py
├── quant_lightning_indexer_tree.py
└── sparse_attn_sharedkv_tree.py
```

SGLang 适配：

```text
python/sglang/srt/speculative/dspark_components/tree/
├── node_pool.py       # project + compressor
├── indexer.py         # tree indexer
└── attention.py       # tree attention
```

SGLang 的全量后端开关：

```bash
SGLANG_DSPARK_TREE_BACKEND=vendor
SGLANG_DSPARK_TREE_BACKEND=sgl_kernel_npu
SGLANG_DSPARK_TREE_BACKEND=shadow
```

其中：

- `vendor`：四个算子都使用 `_C_ascend`
- `sgl_kernel_npu`：四个算子都使用 Triton
- `shadow`：indexer 做 vendor/Triton 对比，compressor/attention 保持 vendor

---

## 11. 一句话总结

Chain 是一条连续路径，只需要线性 KV、线性压缩、线性 indexer 和 causal
attention；Tree 有多个共享祖先但互相不可见的候选节点，所以必须增加：

1. 每节点投影（project）
2. 分支感知压缩（compressor）
3. 每节点祖先感知的 top-k（tree indexer）
4. 带祖先 mask 和多路 KV 的 sparse attention（tree attention）

这四步共同保证：**每个 tree 节点只看自己的历史窗口和祖先链，兄弟节点互不污染，
未被接受的候选节点不会进入正式 KV cache。**
