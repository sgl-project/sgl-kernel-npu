# Split-KV Host/Device 搬运

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc_aiv` | 支持 | 支持 |
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

四个入口的 `auto` 均选择 `ascendc_aiv`，支持逐层 H2D 和全层 D2H。显式 `impl` 可选 `ascendc_aiv`、`torch_native`。

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

每个 `(page,layer)` 内的载荷必须连续，允许 Host 层切片形成的跨页间隙。调用者通过 MemFabric 或 ACL 注册完整的 Host 缓冲区后，再传入其视图。

## ascendc_aiv

通过算子目录中的 Host 内存辅助函数解析已注册 Host 的设备地址，再调用 CANN `kv_exchange` 算子（`torch.ops.npu.kv_exchange`）。Torch adapter 通过 ACLNN 属性传入组件地址、页和层的 stride 以及遍历参数。CANN tiling 将描述交给调用方流上的 AIV kernel。

A3 使用 `python3 setup.py --arch-version 2201 --ops kv_exchange` 构建，A5 使用 `--arch-version 3510`。

使用 MemFabric 时，创建 `memfabric_hybrid.offload.Offload` 实例，以 `Scene.LOCAL` 且 `alloc_size == reserve_size` 初始化，通过 `host_memory.set_memfabric_pool` 发布实例，用 `pool.empty` 分配 Host 缓冲区，并在传入切片前为每个完整分配调用一次 `host_memory.register_host_memory`。使用 ACL 时，分配 CPU 张量，并为每个完整分配调用一次 `host_memory.register_acl_host_memory`。

## torch_native

同步独立参考：直接用原始 Tensor 的层/页索引执行 Torch copy，不复用生产视图选择或地址生成。用于正确性对照，不用于评估搬运性能。

## 流与生命周期

AIV 在调用者的当前 NPU 流上提交搬运。C++ adapter 对 Device 组件和传给它的两个索引张量调用 `record_stream`，由 NPU 缓存分配器保护其存储直至搬运完成。连续索引副本在提交流上创建并使用。调用者负责建立跨流依赖，保留 Host 缓冲区及其注册直至搬运完成，并等待搬运结束后再改写或复用相关存储。释放 MemFabric 池前，须同步所有尚未完成的搬运。

## NPU Graph 捕获

当前 AIV 搬运路径不支持 NPU Graph 捕获。

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

测试为 `auto` 和 `ascendc_aiv` 均初始化 LOCAL Host-memory entry，并注册完整 Host 分配。对原地修改目标使用独立缓冲区，逐字节比较后通过 `make_executor` / `check_performance` 采集。共享性能工具报告 NPU task 耗时之和，不包含全部 CPU 提交开销；端到端开销须另行计时。需要可用 NPU、匹配的 CANN/Torch/MemFabric，以及包含仓内 `kv_exchange` 算子的构建。
