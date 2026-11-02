# assign_cache_op

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.mem_cache import assign_cache_op

def assign_cache_op(dst, src, dst_start, dst_end, src_start, src_end, *, impl="auto"):
    ...
```

## 功能说明

`dst` 为连续 `[batch, width]`，`src` 为一维拼接数据，四个索引数组为 int64。第 i 行从 `src[src_start[i]:src_end[i]]` 拷贝到 `dst[i, dst_start[i]:...]`；长度由源区间决定，调用方保证两侧区间长度一致、容量足够。索引数组可以带终止偏移或未使用的后缀。成功入队返回 True；索引不足或零维输入返回 False，不执行写入。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `dst` | `torch.Tensor` | `[B, W]` | 1/2/4/8 字节元素 | 必填 | 稠密目标，原地写入；按元素位模式复制。 |
| `src` | `torch.Tensor` | `[N]` | 与目标复制长度兼容 | 必填 | 一维拼接源数据。 |
| `dst_start` / `dst_end` | `torch.Tensor` | 至少 B 个元素 | int64 | 必填 | 目标区间边界；长度须与源区间匹配。 |
| `src_start` / `src_end` | `torch.Tensor` | 至少 B 个元素 | int64 | 必填 | 源区间边界，决定每行复制长度。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；auto / ascendc / torch_native。auto 选择 ascendc，不自动回退。 |

返回 `bool`：成功入队为 `True`，公共入口拒绝无效维度或索引数组不足时为 `False`。

按目标元素字节数分发：1/2/4/8 字节使用等宽整数承载按位拷贝，不设业务 dtype 白名单。adapter 仅把目标描述为等宽整数 ACL Tensor、源描述为字节缓冲区，不复制存储、不改变用户 Tensor dtype。其他宽度没有拷贝分支，返回 True 且不写入；例如16字节 complex128 不属于支持的拷贝操作。入口不隐式转换 dtype 或连续化。AscendC 使用 32 字节取整的 DMA 访问，调用方须保证末尾源区间等位置的实际存储足够。

## 后端说明

ARCH22/ARCH35 的 auto 选择 ascendc，不自动回退。公共入口直接调用 torch.ops；CANN 类型为 SglAssignCache，目标为 Ref 输出，adapter 分配并保活清零的同步 Tensor。torch_native 在输入设备上向量化拷贝。参考计算源于 NPU 测试，因此不额外注册 CPU backend。

## 构建与测试

12 条工作负载用例覆盖四种整数及 `(gap,width)=(1,1024)/(2,1024)/(50000,100000)`，保留 96 行存储中 48 行活动请求；检查完整目标、只读输入、原地写入和 True 返回，int32 三组进入框架性能测量。每组执行32轮，每轮同步并检查完整输出、只读输入和原地地址；关闭性能采集也会执行这32轮功能检查。测试不依赖其他测试目录。

```bash
python3 setup.py --make_clean --arch-version 2201 --ops assign_cache_op --artifact wheel
python3 -m pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl
python3 -m pytest tests/python/operators/mem_cache/test_assign_cache_op.py \
  --impl=ascendc --impl-ref=torch_native --enable-perf=false -v -s
```
