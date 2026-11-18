# Split-KV Host/Device 搬运

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc_aiv` | 支持 | 支持 |
| `memfabric_aicpu` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

AIV 实现面向 Ascend A3（arch22）和 A5（arch35）。运行时需匹配对应硬件，并提供已注册的 Host 内存。

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

使用 ACL 注册的 Host 内存时，`auto` 在两个方向均选择 `ascendc_aiv`。使用 MemFabric 注册的 Host 内存时，逐层 H2D 选择 `memfabric_aicpu`，由 AICPU 构建地址表并提交 MemFabric `sparse_copy`。全层 D2H 选择 `ascendc_aiv`，使用 MemFabric 注册时取得的设备虚拟地址（DVA）。显式 `impl` 可选 `ascendc_aiv`、`memfabric_aicpu`、`torch_native`。

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

每个 `(page,layer)` 内的载荷必须连续，允许 Host 层切片形成的跨页间隙。Host 内存的分配和注册要求取决于所选实现。

## ascendc_aiv

通过算子目录中的 Host 内存辅助函数解析已注册 Host 的设备地址，再调用 CANN `kv_exchange` 算子（`torch.ops.npu.kv_exchange`）。Torch adapter 通过 ACLNN 属性传入组件地址、页和层的 stride 以及遍历参数。CANN tiling 将描述交给调用方流上的 AIV kernel。

A3 使用 `python3 setup.py --arch-version 2201 --ops kv_exchange` 构建，A5 使用 `--arch-version 3510`。

使用 MemFabric 时，创建 `memfabric_hybrid.offload.Offload` 实例，以 `Scene.LOCAL` 且 `alloc_size == reserve_size` 初始化，通过 `host_memory.set_memfabric_pool` 发布实例，用 `pool.empty` 分配 Host 缓冲区，并在传入切片前为每个完整分配调用一次 `host_memory.register_host_memory`。使用 ACL 时，分配 CPU 张量，并为每个完整分配调用一次 `host_memory.register_acl_host_memory`。

## memfabric_aicpu

包含 `build_kv_sparse_metadata` 的 wheel 中，四个接口均可显式选择 `impl="memfabric_aicpu"`。
在 A3 上可以用以下命令同时构建 AIV 和 AICPU 算子。

    python3 setup.py --arch-version 2201 --ops kv_exchange,build_kv_sparse_metadata --artifact wheel

默认 `--ops all` 也会发现这两个算子。环境要求见官方
[AICPU 构建说明](../../../../../python/sgl_kernel_npu/README.md#building-and-installing-aicpu-operators)。

一次 AICPU 任务读取 NPU 上实时的页起始 indices，生成 `src`、`dst`（int64 地址位模式）
及 `lengths`、`count`（int32），记录按组件、页、层排列。组件描述通过 ACLNN 整数数组
属性传递，无需暂存张量或单独的 H2D 拷贝。CANN 负责序列化并下发任务参数。

Python 将 `host_memory.set_memfabric_pool` 发布的已初始化池的 `sparse_copy`
传给 Host C++ 提交入口。该入口在调用者的当前流上提交 AICPU 构表任务，回调同一池的
`sparse_copy`，再记录 transfer 内部完成事件。没有发布池时，沿用已初始化的旧版模块级
`offload.sparse_copy`。事件管理下沉到 C++，并未删除内部事件；调用方的完成事件保持不变。

设备 AICPU 根据 NPU 上的 indices 构建地址表。已安装的 MemFabric 包决定 `sparse_copy` 的执行单元。
所选 MemFabric copy 须支持构表产生的原始 Host 虚拟地址（HVA），Host 缓冲区须来自同一个池。
超过 65,536 条记录的批次在启动前报错。发现非法、未对齐或越界的页起始值时，构表任务会
发布空批次并报告错误。

构表地址校验不需要初始化 MF entity：

    python -m pytest tests/python/operators/mem_cache/test_build_kv_sparse_metadata.py -v

验证整条搬运链路还需要兼容的 MemFabric sparse copy 和匹配的 Host 池。

## torch_native

同步独立参考：直接用原始 Tensor 的层/页索引执行 Torch copy，不复用生产视图选择或地址生成。用于正确性对照，不用于评估搬运性能。

## 流与生命周期

AIV 在调用者的当前 NPU 流上提交搬运。C++ adapter 将 NPU 张量保留到排队的算子下发完成。连续索引副本在提交流上创建并使用。调用者负责建立跨流依赖，在搬运完成前保留自己的 NPU 输入和已注册的 Host 缓冲区，并等待搬运结束后再改写或复用相关存储。释放 MemFabric 池前，须同步所有尚未完成的搬运。

AICPU 在调用者的当前 NPU 流上提交构表和 MemFabric copy。普通异步执行时，C++ adapter 在该流上记录 Device 组件和原始索引。地址表在同一流上分配和消费。普通异步提交时，Host C++ 入口在 MF copy 提交后记录内部事件，并为同一设备上的跨流 MemFabric copy 建立顺序；Python 不再创建、记录或等待内部事件。调用者负责持有 Host 缓冲区，并建立生产、搬运和消费之间的依赖。

## NPU Graph 捕获

AIV 路径要求普通异步提交。使用支持图捕获的 MemFabric 运行时时，AICPU 路径可在捕获流上提交构表和 `sparse_copy`。调用方须在捕获和重放期间保留固定地址的索引张量、KV 张量及已注册的 MF 池，并以相同形状原地更新索引值。图内搬运跳过输入的 `record_stream()` 和普通提交的逐设备完成事件，调用方负责对共享同一 MF 池的重放与其他搬运建立顺序。

## 示例

```python
import torch
import torch_npu
from memfabric_hybrid import offload
from sgl_kernel_npu.operators.mem_cache.memfabric.host_memory import (
    register_host_memory,
    set_memfabric_pool,
    uninitialize_host_memory,
)
from sgl_kernel_npu.operators.mem_cache import (
    transfer_kv_all_layer_kv_split_lf_pf as backup,
    transfer_kv_per_layer_kv_split_pf_lf as load,
)

torch.npu.set_device(0)
pool = offload.Offload()
config = offload.OffloadConfig()
config.device_id = 0
config.reserve_size = 1 << 30
config.alloc_size = 1 << 30
config.scene = offload.Scene.LOCAL
assert pool.initialize(config) == 0
set_memfabric_pool(pool)
ps = 4
original = torch.arange(3 * 4 * ps * 8, device="cpu").reshape(3, 4, ps, 1, 8).to(torch.bfloat16)
dev = original.npu()
host = pool.empty([6, 5, ps, 1, 8], dtype=torch.bfloat16)
register_host_memory(host)
empty_dev, empty_host = torch.empty(0, device="npu"), torch.empty(0, device="cpu")
source = torch.arange(ps, 2 * ps, dtype=torch.int64, device="npu")
saved = torch.arange(2 * ps, 3 * ps, dtype=torch.int64, device="npu")
target = torch.arange(3 * ps, 4 * ps, dtype=torch.int64, device="npu")
backup(source, saved, dev, host, empty_dev, empty_host, page_size=ps,
       host_layer_start=1, impl="ascendc_aiv")
load(target, saved, dev, host, empty_dev, empty_host, page_size=ps,
     device_layer_id=2, host_layer_id=1, impl="ascendc_aiv")
torch.npu.synchronize()
assert torch.equal(dev[2, 3].cpu(), original[0, 1])
del host
uninitialize_host_memory()

```

LF Host 可改为 `[5,6,ps,1,8]`，即 `[host_layers,host_pages,ps,1,8]` 分配，并调用无布局后缀的接口。

## 验证

```bash
python -m pytest tests/python/operators/mem_cache/test_kv_transfer.py --impl ascendc_aiv --impl-ref torch_native --enable-perf=false -v -s -rs
```

测试初始化 LOCAL MemFabric 池，并注册完整的 Host 分配。使用 `--impl auto` 时，测试的 H2D 走 `memfabric_aicpu`，D2H 走 `ascendc_aiv`。对原地修改目标使用独立缓冲区，逐字节比较后通过 `make_executor` / `check_performance` 采集。共享性能工具报告 NPU task 耗时之和，不包含全部 CPU 提交开销；端到端开销须另行计时。测试需要可用 NPU、匹配的 CANN、Torch 和 MemFabric，以及 `kv_exchange` 算子。使用 `--impl auto` 时还需要 `build_kv_sparse_metadata` 和兼容的 MemFabric `sparse_copy`。
