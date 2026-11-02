# kv_compress_epilog

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 不支持 | 支持 |
| `torch_native` | 支持 | 支持 |
| `cpu` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog

def kv_compress_epilog(kv_compress_cache, x, slot_mapping, quant_group_size, quant_mode, round_scale_flag, layout, *, impl="auto"):
    ...
```

## 功能说明

原地写 cache，返回 None。x 为正尺寸二维 BF16，slot_mapping 为逐行 int32/int64 索引，-1 表示跳过；cache 是 float8_e4m3fn/float8_e5m2 字节存储。调用方保证 kernel 所需的连续布局及存储容量，入口不额外转换 dtype 或连续化。并行写入需使用互异的有效 slot。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `kv_compress_cache` | `torch.Tensor` | 由 layout 决定 | float8_e4m3fn / float8_e5m2 | 必填 | 原地写入的打包缓存，容量及 stride 须满足布局。 |
| `x` | `torch.Tensor` | `[N, D]` | bfloat16 | 必填 | 输入行，末尾 64 列为 RoPE 部分。 |
| `slot_mapping` | `torch.Tensor` | `[N]` | int32 / int64 | 必填 | 输出槽位；-1 跳过，有效槽位应互异。 |
| `quant_group_size` | `int` | — | — | 必填 | MXFP8 使用 64 或 128；FP8 使用固定 128 列分组。 |
| `quant_mode` | `int` | — | — | 必填 | 1 为 FP8；2 为 MXFP8。 |
| `round_scale_flag` | `bool` | — | — | 必填 | MXFP8 中是否将非零偏置指数加一。 |
| `layout` | `int` | — | — | 必填 | 1 为逐槽布局；2 为 MXFP8 block 布局。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；auto / ascendc / torch_native / cpu。auto 选择 ascendc，不自动回退。 |

返回 `None`，输出通过调用方提供的缓冲区原地写入。

最后 64 列 BF16 RoPE 按字节拷贝。mode=1 对前面的列每 128 个值量化并写 float32 scale；mode=2 支持 group=64/128，写 E8M0 scale 字节，round_scale_flag 将非零偏置指数加一。PyTorch 计算保留 FP32 中间值和 kernel 使用的系数位模式。

layout=1 的 FP8 排列为 `[quant | rope | scale | 零padding]`，MXFP8 为 `[rope | quant | scale | 零padding]`，行间距为 cache.stride(0)，支持 `[slots,1,width]`。layout=2 仅支持 MXFP8，一个 block 的 values 区后接 scales 区；adapter 将四维 `[blocks,block_size,1,width]` squeeze 成三维。未写入 slot 和尾部 block 存储保持不变。

## 后端说明

ascendc/auto 仅注册 ARCH35，不自动回退。torch_native 在输入设备计算；cpu 在 CPU 打包并写回原输出，两者复用打包算法。测试显式固定 impl_ref="cpu"；根包仍依赖 NPU 运行环境。

CANN 类型 SglKvCompressEpilog 以内部属性接收 cache stride，cache 为 Ref 输出，Host 根据核数与 UB 容量计算分块。

## 构建与测试

15 条数值用例覆盖两种 FP8、int32/int64 slot、D=192/320/512、group=64/128、round 开关、跳过/打散 slot、零输入、4096 行、尾部 block、三/四维 cache 和行间距；另有一条 FP8 layout2 拒绝测试。quant 使用 FP8/MXFP8 容差，float32 scale 为 rtol=1e-5/atol=1e-7，RoPE、有效 E8M0 指数、layout 1 填充及未写 slot 逐字节检查。layout 2 已写 slot 在最后一个有效指数之后的对齐字节没有规定值。原地写入使用独立参考和实际缓冲区；性能接 make_executor/check_performance。

```bash
python3 setup.py --make_clean --arch-version 3510 --ops kv_compress_epilog --artifact wheel
python3 -m pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl
python3 -m pytest tests/python/operators/mem_cache/test_kv_compress_epilog.py \
  --impl=ascendc --impl-ref=cpu --enable-perf=false -v -s
```
