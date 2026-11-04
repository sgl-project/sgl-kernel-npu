# Split-KV Host/Device 搬运

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `aicpu` | MF A3/910C SDMA 路径，待实机回归 | MF A5/950 URMA 路径，待实机回归 |
| `aiv` | 已开放注册，待实机验证 | 声明支持，本轮 PF/LF 改动待实机回归 |
| `torch_native` | 未验证 | 同步正确性参考 |

包装层不再按硬件代际限制注册，实际可用性取决于匹配的 MemFabric 后端和运行环境。已有初版 URMA 用例通过不等于本轮重构已经通过实机测试。

## 签名

公共导入路径：`sgl_kernel_npu.operators.mem_cache`。

```python
def transfer_kv_per_layer_kv_split_pf_lf(
    device_indices, host_indices, device_k, host_k, device_v, host_v,
    device_index_k=None, host_index_k=None,
    device_index_k_scale=None, host_index_k_scale=None, page_size=128,
    *, device_layer_id, host_layer_id, device_index_layer_id=None, host_index_layer_id=None,
    impl="auto",
) -> None: ...

def transfer_kv_all_layer_kv_split_lf_pf(
    device_indices, host_indices, device_k, host_k, device_v, host_v,
    device_index_k=None, host_index_k=None,
    device_index_k_scale=None, host_index_k_scale=None, page_size=128,
    *, host_layer_start=0,
    impl="auto",
) -> None: ...

def transfer_kv_per_layer_kv_split(
    device_indices, host_indices, device_k, host_k, device_v, host_v,
    device_index_k=None, host_index_k=None,
    device_index_k_scale=None, host_index_k_scale=None, page_size=128,
    *, device_layer_id, host_layer_id, device_index_layer_id=None, host_index_layer_id=None,
    impl="auto",
) -> None: ...

def transfer_kv_all_layer_kv_split(
    device_indices, host_indices, device_k, host_k, device_v, host_v,
    device_index_k=None, host_index_k=None,
    device_index_k_scale=None, host_index_k_scale=None, page_size=128,
    *, host_layer_start=0,
    impl="auto",
) -> None: ...


```

`auto` 默认在逐层 H2D 入口选择 `aicpu`，在全层 D2H 入口选择 `aiv`，与 MemFabric scene 无关。显式 `impl` 可选 `aicpu`、`aiv`、`torch_native`；两个生产实现均保留 H2D/D2H 能力。

## 行为

Device 布局始终为 `[layer,page,token,head,dim]`。带 `pf_lf` 的逐层入口做 H2D；带 `lf_pf` 的全层入口做 D2H；两者 Host 均为 PF `[page,layer,token,head,dim]`。不带布局后缀的两个入口使用 LF Host `[layer,page,token,head,dim]`。

逐层入口只搬指定 Device/Host KV 层以及显式指定的 Indexer 槽。两个 Indexer 槽号均为 None 时不搬 Indexer；只提供一端槽号时报错。全层入口把当前 Device 池的所有 KV 层搬到从 `host_layer_start` 开始的 Host 层；Indexer 使用自己独立的完整层空间。

按 indices 的顺序配对 Device/Host 页。每页载荷逐字节复制，不做量化或反量化。目的页应彼此不重叠；输入必须由连续、按页对齐的完整 token 槽位段组成，每段有 `page_size` 项。生产内核读取每段的第一项确定页号，调用者负责段内其余 token 的连续性。

## 参数和返回值

| 参数 | 类型、默认值与约束 |
| --- | --- |
| `device_indices` / `host_indices` | 一维 int64 Tensor；位于与 K 相同的 NPU；长度相同且为 page_size 的倍数；值是 token 槽位，不是页号或地址；页必须在各组件容量范围内 |
| `device_k` / `host_k` | 非空五维 Tensor；Device 为 NPU LF，Host 为 CPU PF/LF；dtype 和每页 `[T,H,D]` 相同 |
| `device_v` / `host_v` | Tensor 对；非空 Device V 使用相同层选择；空 Device V 表示 packed-K 无独立 V 载荷，Host V 可以保留未使用占位空间 |
| `device_index_k` / `host_index_k` | 可选 Tensor，默认 None；仅两端同时存在时搬运，保留旧接口行为；独立 Indexer 层数可少于 KV 层数 |
| `device_index_k_scale` / `host_index_k_scale` | 可选 FP32 scale，默认 None；两端同时存在时原样搬运；Device 支持 `[I,P,T,1]`，Host 为 `[P,I,T,1,1]` 或 `[I,P,T,1,1]` |
| `page_size` | 正整数，默认 128；须匹配所有参与组件的页 token 维 |
| `device_layer_id` / `host_layer_id` | 逐层入口必填的非负池内层号，可不相同 |
| `device_index_layer_id` / `host_index_layer_id` | 逐层入口可选的非负 Indexer 槽号，默认 None；两端必须同时指定或同时省略 |
| `host_layer_start` | 全层入口的 Host KV 起始层，默认 0；须容纳当前 Device 池的全部层 |
| `impl` | 关键字参数，默认 auto；见各后端说明 |
| 返回值 | None；H2D 原地修改 Device，D2H 原地修改 Host |

每个 `(page,layer)` 内的载荷必须连续，允许 Host 层切片形成的跨页间隙。Host 缓冲区须由调用者通过匹配的 MemFabric 分配/注册流程提供。

## aicpu

Triton 使用一次 launch 为全部有效组件构造最终地址表，再调用一次五参数 `offload.sparse_copy(src, dst, lengths, count, device)`。复用原来的二维 grid 和编译期 LAYOUTS；组件列表按 K、V、Indexer K、Indexer scale 固定四个位置，缺失项为 None。地址、页容量平铺为按组件命名的运行时标量，不构造组件字典或角色到槽位的映射。仅在 kernel 内构造 tuple，绕开有缺陷的 Ascend launcher 运行时 tuple 展开路径。缺失组件在 LAYOUTS 中层数为零，编译期跳过，grid 仅覆盖有效组件。对运行时地址、页容量及页数禁用值特化。各组件直接写最终表，无需临时表或 cat。MF 初始化使用 `Scene.AICPU`、`flags=0`，`reserve_size` / `alloc_size` 非零且按 GiB 对齐后相等。MF 在 Ascend910C（A3）选择 AICPU + SDMA，在 Ascend950（A5）选择 AICPU + URMA。

地址表填写 MF 池的原始 Host 地址和 NPU payload 地址，不替换成 AIV DVA。MF 执行时按每条记录哪一端属于 Host 池判断 H2D/D2H；包装层仍根据方向构造源/目的表。支持最多 65536 条记录，每条长度为正 int32；两端 indices 都须提前位于 KV buffer 所在的 NPU。不再依赖旧 `LOCAL_URMA`、`CopyDirection` 或私有 `_get_scene` 接口。

Host 库、Python binding、launch 扩展及 MF AICPU 算子包须配套更新。A5 当前每次 sparse_copy 固定提交九个搬运分块及 Notify Wait，再提交状态检查 kernel，小批次也相同；A3 提交搬运 kernel、Notify Wait 和状态检查。性能比较应采集完整提交序列。

## aiv

通过 `host_memory.resolve_host_dva` 获取已注册 Host 的设备地址，然后经仓内 `kv_exchange` CANN 算子（`torch.ops.npu.kv_exchange`）提交：C++ adapter 构造驻留设备的 44 槽 metadata（实际层/页 stride、单页字节数、PF/LF 遍历模式和每组件层区间，对齐 aicpu `build_kv_sparse_metadata` 的契约）并在调用方流上启动 AIV kernel。承载 metadata 上传的 pinned staging 会返回给 wrapper，由完成事件保活。

AIV 路径仍需要 MF 的 `host_register` / `host_get_device_pointer`。当前 AICPU 专用 MF 仓库没有导出这两个接口；AIV 或混合 `auto` 对接须使用具有兼容共享 Host 池注册契约的 MF 构建。

## torch_native

同步独立参考：直接用原始 Tensor 的层/页索引执行 Torch copy，不复用生产视图选择或地址生成。用于正确性对照，不用于评估搬运性能。

## 流与生命周期

生产实现提交到调用者当前 NPU stream，不创建搬运 stream，也不执行同步。调用者负责生产者依赖及消费完成事件，并在传输结束前禁止改写/复用相关存储。包装保留原始输入、sparse-copy 地址表和 AIV pinned staging，直到事件完成。同流分配、使用的新连续索引由分配器的流顺序保证生命周期。sparse_copy 包装跨流串行提交，以保护共享资源；AIV 的跨流数据依赖由调用者提供。释放 MemFabric entity 前须同步并结束所有使用该 entity 的传输。

## NPU Graph 捕获

两个方向都显式使用 `impl="aicpu"`，在捕获前初始化支持入图的 MF 资源，并按相同搬运形状预热。两端 indices 都须提前放在 KV buffer 所在的 NPU 上。捕获后地址、shape、页数、层选择和方向固定；每次 replay 前可以原地更新 indices 和载荷内容。

地址表在捕获期间通过 `torch.empty` 分配，由 Graph 私有内存池持有。捕获路径跳过 eager 的事件查询、回收和共享资源事件记账。调用者在图生命周期内持有静态输入 Tensor 和 Host/MF 存储，并在图外显式安排共享 MF 资源的全部 replay 与 eager 搬运顺序，包括跨流执行。释放 MF 资源前须等待 replay 完成并销毁图。本轮不增加 AIV 入图支持。

运行 `python -m pytest tests/python/operators/mem_cache/test_kv_transfer.py -k graph --impl aicpu --enable-perf=false -v -s`，验证 PF/LF、packed/独立 K/V 的往返搬运、NPU indices 内容更新、图内消费者以及图外排序的交替流 replay。仍需在实机执行验证。

## 示例

```python
import torch
import torch_npu
from memfabric_hybrid import offload
from sgl_kernel_npu.operators.mem_cache import (
    transfer_kv_all_layer_kv_split_lf_pf as backup,
    transfer_kv_per_layer_kv_split_pf_lf as load,
)

torch.npu.set_device(0)
config = offload.OffloadConfig()
config.device_id = 0
config.reserve_size = config.alloc_size = 1 << 30
config.flags = 0
config.scene = offload.Scene.AICPU
assert offload.initialize(config) == 0
ps = 4
original = torch.arange(3 * 4 * ps * 8, device="cpu").reshape(3, 4, ps, 1, 8).to(torch.bfloat16)
dev = original.npu()
host = offload.empty([6, 5, ps, 1, 8], dtype=torch.bfloat16)
empty_dev, empty_host = torch.empty(0, device="npu"), torch.empty(0, device="cpu")
source = torch.arange(ps, 2 * ps, dtype=torch.int64, device="npu")
saved = torch.arange(2 * ps, 3 * ps, dtype=torch.int64, device="npu")
target = torch.arange(3 * ps, 4 * ps, dtype=torch.int64, device="npu")
backup(source, saved, dev, host, empty_dev, empty_host, page_size=ps,
       host_layer_start=1, impl="aicpu")
load(target, saved, dev, host, empty_dev, empty_host, page_size=ps,
     device_layer_id=2, host_layer_id=1, impl="aicpu")
torch.npu.synchronize()
assert torch.equal(dev[2, 3].cpu(), original[0, 1])
del host
offload.uninitialize()

```

LF Host 可改为 `[5,6,ps,1,8]`，即 `[host_layers,host_pages,ps,1,8]` 分配，并调用无布局后缀的接口。

## 验证

```bash
python -m pytest tests/python/operators/mem_cache/test_kv_transfer.py --impl aicpu --impl-ref torch_native --enable-perf=false -v -s -rs
python -m pytest tests/python/operators/mem_cache/test_kv_transfer.py --impl aiv --impl-ref torch_native --enable-perf=false -v -s -rs
python -m pytest tests/python/operators/mem_cache/test_kv_transfer.py --impl aicpu --impl-ref torch_native --prof-dir results/kv --csv-path results/kv.csv -v -s
```

测试按显式实现选择初始化 MF 资源（aiv 使用 LOCAL，其余使用 AICPU）；新版 AICPU 专用 MF 包应使用 `--impl aicpu`。auto 在 H2D 调用 sparse_copy，在 D2H 调用 kv_exchange AIV kernel，需 MF 支持且构建中包含仓内 `kv_exchange` 算子。默认仅服务尺寸用例采性能，可用 `--enable-perf` 覆盖。测试对原地修改目标使用独立缓冲区并逐字节比较，再通过 `make_executor` / `check_performance` 采集。补充地址表检查加 `-m extra_checks`。共享性能工具报告 NPU task 耗时之和的平均值，不包含全部 CPU 提交开销；与旧实现对比端到端开销时应另行计时。执行这些测试需有可用 A3/910C 或 A5/950、匹配的 CANN/Torch/MemFabric，且确认导入当前构建。
