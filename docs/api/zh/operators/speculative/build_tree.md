# build_tree

```python
from sgl_kernel_npu.operators.speculative import build_tree

build_tree(parent_list, selected_index, verified_seq_len, tree_mask, positions,
           retrive_index, retrive_next_token, retrive_next_sibling,
           topk, depth, draft_token_num, tree_mask_mode, *, impl="auto") -> None
```

构造验证树，原地写入调用方提供的五个输出缓冲区。`auto` 固定选择
`ascendc`，失败时不自动回退；`torch_native` 是独立 PyTorch 实现。
旧 `torch.ops.npu.build_tree_kernel_efficient` 接口与旧实现均保持原样。

## 参数与副作用

记批量为 `B`，`D=draft_token_num`，`k=topk`，请求 b 的已验证前缀长度为
`L_b`。所有张量位于同一设备，CANN 路线要求 NPU。预期布局为连续 ND：

| 参数 | dtype 与 shape | 含义 |
|---|---|---|
| `parent_list` | int64 `[B,k*(depth-1)+1]` | 父候选节点表 |
| `selected_index` | int64 `[B,D-1]` | 不含根的已选候选索引 |
| `verified_seq_len` | int64 `[B]` | 已验证前缀长度 |
| `tree_mask` | bool，一维 | 原地掩码，长度见下文 |
| `positions` | int64 `[B*D]` 或连续 `[B,D]` | 原地绝对位置 |
| `retrive_index` | int64 `[B,D]` | 原地全局节点索引 |
| `retrive_next_token` | int64 `[B,D]` | 原地首个子节点，缺失为 -1 |
| `retrive_next_sibling` | int64 `[B,D]` | 原地下一个兄弟节点，缺失为 -1 |

返回 `None`。五个输出保留调用方原有存储，只读输入不修改。接入层不创建
连续副本，也不增加结果回写：继续按旧 Kernel 的连续裸指针方式解释存储，
这不表示支持一般 strided 语义。输出之间须独立且容量充足。独立参考针对
已选索引唯一、包含所有祖先的合法树；非法树和不足的缓冲区没有已定义的旧行为。
Kernel 中 D 与 mask 长度的 uint16 转换原样保留，不据此声明新的已验证 shape 上限。

## 计算与掩码

节点 0 是根。节点 `i>0` 的 `q=selected_index[b,i-1]//k`：q 为 0 时父节点
为根；否则在 `selected_index[b]` 中查找 `parent_list[b,q]`，查找位置加 1
得到父节点索引。`positions[b,i]=L_b+节点深度`，`retrive_index[b,i]=b*D+i`。
子节点与兄弟链按子节点索引升序排列。数学掩码仅允许节点自身及其祖先。

* `tree_mask_mode=0`（FULL_MASK）：顺次拼接每个请求的 `[D,L_b+D]`，总长度
  为 `D*sum(L_b)+B*D*D`。只写每行最后 D 列；前缀列保留调用方初值。
* `tree_mask_mode=1`（QLEN_ONLY）：预期为展平的 `[B,D,D]`，总长 `B*D*D`。
  **本轮明确保留已知问题：**旧、新 Ascend C 对每个节点均使用
  `b*D*D+b*D` 行地址，而非 `b*D*D+i*D`，导致反复覆盖同一区域、其他行保留
  初值，某些 batch 下还可能越过缓冲区。本轮没有修正或声称其数学正确；
  `torch_native` 按正确的逐节点行地址计算，精确对照会暴露该问题。
* `tree_mask_mode=2`（QLEN_ONLY_BITPACKING）：未实现，报错。

接口不会自动选择模式或后端。其他整数沿用旧非 FULL 分支，不构成新增的
已验证模式。空批量在旧版没有已定义执行行为；新 CANN tiler 对零 block
返回失败，避免 Host 除零。

## 验证状态

| 后端 | 源码声明 | 本轮设备验证 |
|---|---|---|
| `ascendc` / `auto` | ARCH22 / 2201、ARCH35 / 3510 | 已报告测试环境下 FULL_MASK 通过，完整架构矩阵仍待验证 |
| `torch_native` | 同一注册代际，独立 PyTorch 计算 | 参考实现，不代表设备计算已通过 |

正式公开接口测试原样复用旧 fixture：B=2、k=4、depth=4、D=8、L=[5,10]。
五个整数/bool 输出使用公共精确比较，每条路线独立准备可变缓冲区，并检查
只读输入与存储身份。该 fixture 来自旧测试，不冒充生产用例。

同一测试文件提供分环境回放：旧安装用 `--impl legacy --save PATH` 保存结果，
新安装用 `--impl ascendc --compare-legacy PATH` 先做新旧直接比较，再与独立
数学参考比较。两次均增加 `--mode qlen` 可显式运行已知问题诊断；数学比较
失败不会被 xfail、跳过或放宽规则掩盖。默认 pytest 仅运行 FULL_MASK
新版本/参考对照，不代表旧版回放已经完成。新生产用例、图捕获及完整架构矩阵
仍属于后续验收。

## 性能测试

`perf_shape` 选择原有 fixture，不引入新的维度。精度检查通过后，公共性能框架
预热 5 次、正式采样 5 次。FULL_MASK 重复调用会覆盖树区域、保留前缀列，因此
直接复用输出缓冲区，不添加重置或复制 Kernel。

在仓库根目录、已安装新版 wheel 的 Python 环境中运行：

```bash
SGL_TEST_ROOT="$(pwd -P)"
python3 -m pytest "$SGL_TEST_ROOT/tests/python/operators/speculative/test_build_tree.py" \
  --impl ascendc --impl-ref torch_native --skip-acc-check=false \
  --enable-perf=true --csv-path "$SGL_TEST_ROOT/test-results/build_tree.csv" \
  --prof-dir "$SGL_TEST_ROOT/test-results/profilings" -sv -ra
```

`--enable-perf=false` 关闭采样。跳过精度检查时结果标记为 `accuracy_checked=False`。
CSV 记录平均 NPU 设备任务耗时，不是主机端到端延迟。实际性能采样及与旧实现的
性能回归对照仍需上机完成；已报告的 FULL_MASK 通过仅代表精度结果。
