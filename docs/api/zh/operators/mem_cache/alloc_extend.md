# alloc_extend

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |
| `cpu` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.mem_cache import alloc_extend

def alloc_extend(pre_lens, seq_lens, last_loc, free_pages, page_size, out_indices, values, *, impl="auto"):
    ...
```

## 功能说明

为分页请求扩展生成紧凑排列的 token slot。原地写入 `out_indices`，有条件地写入
`values[0]`，返回 `None`。`auto` 直接选择 `ascendc`，不自动回退；
`torch_native` 在输入设备计算；`cpu` 在 CPU 计算，再写回调用方输出缓冲区。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `pre_lens` / `seq_lens` / `last_loc` | `torch.Tensor` | `[B]` | int64 | 必填 | 已有长度、目标长度及已有末尾槽位；稠密向量。 |
| `free_pages` | `torch.Tensor` | `[P]` | int64 | 必填 | 按分配顺序排列的空闲页 ID。 |
| `page_size` | `int` | — | — | 必填 | 正页长，Tiling 中按 int32 表示。 |
| `out_indices` | `torch.Tensor` | `[sum(seq_lens-pre_lens)]` | int64 | 必填 | 调用方分配的稠密输出，原地写入。 |
| `values` | `torch.Tensor` | 至少 1 个元素 | int64 | 必填 | 条件写入 values[0]，其他元素保留。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；auto / ascendc / torch_native / cpu。auto 选择 ascendc，不自动回退。 |

返回 `None`，输出通过调用方提供的缓冲区原地写入。

B 必须大于零；prefix 不超过 sequence，空闲页及存储容量足够，last_loc 有效。

公共入口只检查前四个 Tensor 与 out_indices 的 int64 类型，不增加 shape、连续性
或 values 类型检查。直接调用 `torch.ops.npu.alloc_extend` 不经过 Python 检查。
CANN 声明接受 int64 Tensor；AscendC adapter 不自动连续化或转换输入类型。

每个请求先复用已有不完整页中 last_loc 后面的槽位，再从 free_pages 填入完整页及
末尾不完整页。请求的扩展长度及新增页数的前缀和决定输出位置及空闲页起点。
算子不修改输入，也不从 allocator 的空闲页表中删除已使用的页。

**values 不是通用的页数查询输出。** 仅当最后一个请求进入新分配的不完整页时，
写入整个批次的新增页数；最后请求在已有页中结束或恰好填满整页时，values 保持输入值。
两个参考实现都保留这个条件写入行为，测试用哨兵检查，不将未写入解释为零。

## 后端说明

CANN 类型为 SglAllocExtend，两个输出与输入同名，生成 Ref 参数。
ACLNN query 接收六个 Tensor 描述符再接 page_size 属性；Torch 公共参数顺序仍是
四个输入、page_size、两个输出。adapter 保留当前流及排队执行所需的输入、描述符和
workspace 生命周期。Tiling 使用 min(AIV核数,B) 和四字段结构，Kernel计算未改。

CPU 参考遵循 CPU golden 的分配语义，但写入接口规定的 int64 输出并保留 values
条件写入；torch_native 使用输入设备上的向量化索引。没有 NumPy 或浮点升精度。
CPU 传输和 native 元数据判断可能同步。
包导入仍依赖 NPU runtime 及安装的原生库，选择 cpu 不代表可在无 NPU 环境独立导入包。

## 构建与测试

```bash
python3 setup.py --make_clean --arch-version 2201 --ops alloc_extend --artifact wheel
python3 -m pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl
python3 -m pytest tests/python/operators/mem_cache/test_alloc_extend.py \
  --impl=ascendc --impl-ref=cpu --skip-acc-check=false --enable-perf=false -v -s
python3 -m pytest tests/python/operators/mem_cache/test_alloc_extend.py \
  --impl=torch_native --impl-ref=cpu --skip-acc-check=false --enable-perf=false -v -s
```

arch35 使用3510。测试显式固定 impl_ref=cpu，12组输入比较独立输出、检查只读输入不变；
省略性能开关即可测选定的长prefill和B100负载。按 --ops 构建的wheel仅包含所选原生算子；
测试同名 torch.ops 时应确认实际安装产物来源。
