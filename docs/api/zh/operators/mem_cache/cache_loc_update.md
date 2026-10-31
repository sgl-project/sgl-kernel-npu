# cache_loc_update

## 签名

```python
from sgl_kernel_npu.operators.mem_cache import cache_loc_update

result = cache_loc_update(req_pool_indices, token_pool, start_offset, end_offset,
                         out_cache_loc, impl="ascendc")
```

五个Tensor为位置参数；`impl`仅限关键字，默认`auto`。`auto`直接选择
`ascendc`，没有缺失时自动回退；`torch_native`为独立PyTorch实现。
原生`torch.ops.npu.cache_loc_update`接口保留。

## 参数与行为

| 参数 | 合同 |
| --- | --- |
| `req_pool_indices` | NPU请求行索引`[B]`，int32或int64。 |
| `token_pool` | NPU int32池`[pool_rows,row_width]`，使用稠密存储布局。 |
| `start_offset/end_offset` | NPU int64数组`[B]`，区间为`[start,end)`。 |
| `out_cache_loc` | 调用方分配的连续NPU int32 Tensor，按flat packed布局访问；连续多维view保留原shape。 |
| `impl` | `auto`、`ascendc`或`torch_native`。 |

记`length[b]=end_offset[b]-start_offset[b]`，`offset[b]=sum(length[:b])`。
对每个`0<=j<length[b]`：

```text
out_cache_loc.flat[offset[b]+j] = token_pool[req_pool_indices[b], start_offset[b]+j]
```

原地修改并返回**out_cache_loc的Tensor alias**，保留存储、dtype、shape和device。
请求索引和offset不改。
token_pool只读；packed有效长度之后的out_cache_loc尾部保持原值。
不额外分配输出、不连续化、不转换索引dtype、不clamp。

公共入口检查dtype组合和out_cache_loc连续性条件。
其他shape／索引／存储前提不扩展为新的校验规则。

CANN adapter不重复公共入口的dtype／连续性检查。直接调用
`torch.ops.npu.cache_loc_update`时，调用者须保证这些条件。
Torch接口的实现由加载的库提供。

## 原生实现与支持范围

CANN类型为`SglCacheLocUpdate`，Torch adapter通过生成的ACLNN两阶段
API在当前NPU stream执行。update使用单核调度。
两算子共享Kernel计算体与16字段Tiling。cacheLocSize使用L的逻辑numel，
不使用B*16或view背后的allocation大小。

Kernel使用MAX_STEP=16、UB容量限制及对齐读取。正常测试使用唯一req、无src/dst
存储重叠、长度不超过16的区间。随机start用例覆盖接近行末的
位置，16元素读取要求相应存储可读；数值覆盖不能证明allocation末端
的读取安全，需独立NPU诊断。
不新增空batch、重复req冲突、非法索引或存储alias的支持承诺或自动修复。

UB超限通过CANN/ACLNN错误路径上报。

| 实现 | arch22 / 2201 | arch35 / 3510 |
| --- | --- | --- |
| `ascendc` | 已声明；CANN编译与真实NPU验证待完成 | 已声明；CANN编译与真实NPU验证待完成 |
| `torch_native` | 已实现；真实NPU验证待完成 | 已实现；真实NPU验证待完成 |

native使用输入设备上的PyTorch slice copy；Python循环读取索引元信息可能
产生同步。它不是CPU NumPy golden，也不替代AscendC延迟基线。整数复制无需
浮点升精度。不宣称已验证graph capture或autograd。

## 构建与测试

```bash
python3 setup.py --make_clean --arch-version 2201 \
  --ops cache_loc_assign,cache_loc_update --artifact wheel
# 在匹配的Linux/CANN/Torch环境安装output/中的准确wheel文件。
python3 -m pytest tests/python/operators/mem_cache/test_cache_locations.py \
  --impl=ascendc --impl-ref=torch_native --skip-acc-check=false --enable-perf=false -v -s
```

子集wheel仅提供构建时选中的native算子。根包在导入时加载native库，
未实现缺产物时其他backend的隔离。

两个cache-location算子成对构建。通过`setup.py --ops`选择任意一个ID都会包含两个接口；直接使用CMake时须选择两个ID或`ALL`。
