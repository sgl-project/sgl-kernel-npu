# transfer_kv_dim_exchange

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `aclrt` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.data_move import TransferDirection, TransferFlag, transfer_kv_dim_exchange

def transfer_kv_dim_exchange(device_indices, host_indices, device_k, host_k, device_v, host_v, device_index_k=None, host_index_k=None, device_index_k_scale=None, host_index_k_scale=None, page_size=128, direction=TransferDirection.H2D, flags=TransferFlag.FAST2D, *, impl='auto'): ...
```

`auto` 选择 `aclrt`。

## 功能说明

按整页在 NPU 与锁页主机内存之间搬运选中的 KV cache，支持 H2D 恢复和 D2H 备份。设备缓存按 `[层, 页, 页内 token, 头, 维度]` 排列，主机缓存按 `[页, 层, 页内 token, 头, 维度]` 排列；根据两组 token 索引确定源、目标页，一次搬运该页的所有层。可只传 K，也可同时传 V、indexer key 及其 scale，未选中的页保持不变。

## 参数与返回值

| 参数 | 类型 | 形状 | 数据类型 | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `device_indices` | Tensor | [N] | int64 | 必填 | 设备槽位索引。KV 接口传 token 索引。 |
| `host_indices` | Tensor | [N] | int64 | 必填 | 与设备索引一一对应的主机索引。 |
| `device_k` | Tensor | [L,D,P,A,C] | 源与目标一致 | 必填 | 设备 K，按层、页排列。 |
| `host_k` | Tensor | [H,L,P,A,C] | 源与目标一致 | 必填 | 锁页主机 K，按页、层排列。 |
| `device_v` | Tensor | [L,D,P,A,C] | 源与目标一致 | 必填 | 设备 V，仅传 K 时使用空张量。参与拷贝时，层数、页数、页大小、head 数和 dtype 须与 K 一致，head 维度可不同。 |
| `host_v` | Tensor | [H,L,P,A,C] | 源与目标一致 | 必填 | 主机 V，仅传 K 时使用空张量。 |
| `device_index_k` | Tensor | [L,D,P,A,C] | 源与目标一致 | None | 可选的设备 indexer key。 |
| `host_index_k` | Tensor | [H,L,P,A,C] | 源与目标一致 | None | 与设备 indexer key 配对的主机缓存。 |
| `device_index_k_scale` | Tensor | [L,D,P,1] / [L,D,P,1,1] | FP32 | None | 可选的设备逐 token scale，支持四维或五维。 |
| `host_index_k_scale` | Tensor | [H,L,P,1,1] | FP32 | None | 配对的五维主机 scale。 |
| `page_size` | int | — | — | 128 | 每页 token 数；索引数须为其整数倍，每组首个索引决定页号，其余索引不参与寻址。 |
| `direction` | TransferDirection | — | — | TransferDirection.H2D | TransferDirection.H2D 或 D2H。 |
| `flags` | TransferFlag | — | — | TransferFlag.FAST2D | 二维拷贝标志 FAST2D。 |
| `impl` | str | — | — | 'auto' | 选择实现，默认为 auto。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| `None` | NoneType | 原地写入目标。在读取结果或复用源缓冲区前同步提交拷贝的流。 |

## aclrt 后端

在当前 NPU 流提交异步二维拷贝。设备与主机须具有匹配的 dtype 和页形状，主机缓冲区须连续且锁页；设备页内数据允许稠密转置，但页间、层间 pitch 须保持紧凑。提交前检查所有待搬运页的索引，ACL 提交错误直接上抛。

## torch_native 后端

复用 SGLang `MambaPoolHost` 的 PyTorch gather/scatter 参考逻辑：从每组 `page_size` 个 token 索引的首项计算页号，将整页物理存储适配为一个槽位，再按方向执行单层恢复或全层备份。K、V 和可选 indexer/scale 分别处理，不调用 `aclrt` 实现。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.data_move import transfer_kv_dim_exchange

dev = torch.zeros(2, 4, 128, 1, 8, device="npu")
host = torch.ones(4, 2, 128, 1, 8, pin_memory=True)
tokens = torch.arange(128)
transfer_kv_dim_exchange(tokens, tokens, dev, host, torch.empty(0), torch.empty(0))
torch.npu.synchronize()
print(dev[:, 0, 0])
```
