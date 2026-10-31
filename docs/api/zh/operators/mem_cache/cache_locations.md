# cache_loc_assign / cache_loc_update

在请求到 token 的映射表与打包的位置数组之间复制缓存槽位编号。
`cache_loc_assign` 将编号写入映射表，`cache_loc_update` 从映射表收集编号；
两者均不搬运 KV 向量数据。

## 架构支持

代际名称沿用公共 operators 框架的 `arch22`、`arch35`，分别对应架构编号
`2201`、`3510`。构建声明和真实硬件验证状态分别列出。

| 实现 | arch22 / 2201 | arch35 / 3510 |
| --- | --- | --- |
| `ascendc` | 已实现；CANN 编译与 NPU 验证待完成 | 已实现；CANN 编译与 NPU 验证待完成 |
| `torch_native` | 已实现；NPU 验证待完成 | 已实现；NPU 验证待完成 |

## 接口

```python
from sgl_kernel_npu.operators.mem_cache import cache_loc_assign, cache_loc_update

cache_loc_assign(req_pool_indices, token_pool, start_offset, end_offset,
                 out_cache_loc, impl="auto")
cache_loc_update(req_pool_indices, token_pool, start_offset, end_offset,
                 out_cache_loc, impl="auto")
```

两个接口均接收五个 Tensor，`impl` 仅限关键字，默认 `auto`。
`auto` 与 `ascendc` 调用对应的 `torch.ops.npu.cache_loc_assign` 或
`torch.ops.npu.cache_loc_update`，没有自动回退；`torch_native` 显式选择
PyTorch 实现。原生 Torch 接口仍可直接调用。

## 参数与语义

| 参数 | 形状、类型与约束 |
| --- | --- |
| `req_pool_indices` | NPU Tensor `[B]`，int32 或 int64，表示请求在 `token_pool` 中的行索引。 |
| `token_pool` | NPU int32 Tensor `[pool_rows, row_width]`，使用稠密存储布局，保存每个请求的缓存槽位编号。 |
| `start_offset` | NPU int64 Tensor `[B]`，表示各请求区间的起始位置，包含该位置。 |
| `end_offset` | NPU int64 Tensor `[B]`，表示各请求区间的结束位置，不包含该位置。 |
| `out_cache_loc` | 调用方分配的连续 NPU int32 Tensor，按一维打包数组访问，元素数至少为各区间长度之和。连续多维 Tensor 保留原 shape。assign 将其作为输入，update 将其作为输出。 |
| `impl` | `auto`、`ascendc`、`torch_native`；默认 `auto`。 |

记 `length[b] = end_offset[b] - start_offset[b]`，
`offset[b] = sum(length[:b])`。各请求的数据按请求顺序连续排列在
`out_cache_loc` 中，不要求每个请求的区间长度相同。

### cache_loc_assign

对每个 `0 <= j < length[b]`，执行：

```text
token_pool[req_pool_indices[b], start_offset[b] + j] = out_cache_loc.flat[offset[b] + j]
```

**原地修改 `token_pool`，返回与其共享存储的 Tensor**，shape、dtype 和 device
保持不变。未选中的行和目标区间之外的位置保持原值；其余四个输入不变。
AscendC 实现在读取 token pool 前跳过零长度区间。

### cache_loc_update

对每个 `0 <= j < length[b]`，执行：

```text
out_cache_loc.flat[offset[b] + j] = token_pool[req_pool_indices[b], start_offset[b] + j]
```

**原地修改 `out_cache_loc`，返回与其共享存储的 Tensor**，shape、dtype 和 device
保持不变。有效打包长度之后的尾部保持原值；其余四个输入不变。

两个接口均不自动连续化、转换索引类型或裁剪索引。公共入口检查上述 dtype
组合和 `out_cache_loc` 连续性；直接调用 `torch.ops.npu` 时，调用方须自行保证
这些条件，因为 adapter 不重复执行公共入口的检查。

## 实现与使用限制

CANN 类型分别为 `SglCacheLocAssign`、`SglCacheLocUpdate`，通过 ACLNN
在当前 NPU stream 上执行。assign 沿用全部 AIV 核的任务分配，update 沿用单核调度。
两个算子共享计算实现和 16 字段 Tiling，位置数组大小取其逻辑 `numel()`。

AscendC 的区间长度上限为 `MAX_STEP=16`，还受 UB 容量及对齐读取限制。
正常测试使用唯一请求行、有效索引和不重叠的输入输出存储。空 batch、重复请求
冲突、非法索引和输入输出存储重叠尚未验证。固定 16 元素读取要求对应存储可读；
行末数值测试不能替代分配边界的内存安全诊断。UB 超限通过 CANN/ACLNN 错误路径上报。

`torch_native` 在输入设备上执行 PyTorch slice copy；为 Python 循环读取索引
元信息可能产生同步。整数复制不需要浮点升精度。图捕获和自动求导尚未验证。

## 构建与测试

两个算子成对构建。通过 `setup.py --ops` 选择任意一个 ID 都会包含两个接口；
直接使用 CMake 时须选择两个 ID 或 `ALL`。

```bash
python3 setup.py --make_clean --arch-version 2201 \
  --ops cache_loc_assign,cache_loc_update --artifact wheel
# 在匹配的 Linux/CANN/Torch 环境中安装 output/ 内本次生成的 wheel。
python3 -m pytest tests/python/operators/mem_cache/test_cache_locations.py \
  --impl=ascendc --impl-ref=torch_native --skip-acc-check=false --enable-perf=false -v -s
```

arch35 环境使用 `--arch-version 3510`。测试共 52 项，包含单算子精度、原地写入、
只读输入、未修改区域和改变源值后的 roundtrip 检查。删除 `--enable-perf=false`
可运行用例指定的性能场景。

包在导入时加载原生库；缺少原生产物时，其他 backend 尚未实现完整隔离。
