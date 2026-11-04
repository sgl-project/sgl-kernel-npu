# get_symm_buffer_for_mega_moe

通过外部 CANN provider 创建供 [mega_moe](mega_moe.md) 使用的持久化 EP 通信 buffer。

```python
from sgl_kernel_npu.operators.moe import get_symm_buffer_for_mega_moe

get_symm_buffer_for_mega_moe(*args, impl="auto", **kwargs)
```

注册名称为 `moe.get_symm_buffer_for_mega_moe`。`auto` 和 `cann` 指向同一个实现，显式注册支持 `Generation.ARCH22`、`Generation.ARCH35`。需要正常的 Ascend 运行环境及兼容的 `cann_ops_transformer`。provider 在首次调用后端时加载，并与 MegaMoE 接口共享函数缓存。

除 `impl` 外的参数原样传入 `cann_ops_transformer.ops.get_symm_buffer_for_mega_moe`，返回对象也原样返回。未知实现会在加载后端前抛出 `ImplementationNotFoundError`。provider 错误直接传播，不进行回退。

现有集成传入进程组及容量、布局参数：

```python
buffer = get_symm_buffer_for_mega_moe(
    group,
    num_experts=num_experts,
    num_max_tokens_per_rank=max_tokens,
    num_topk=topk,
    hidden=hidden,
    intermediate_hidden=intermediate_hidden,
    max_recv_token_num=0,
    dispatch_quant_mode=dispatch_quant_mode,
    dispatch_quant_out_dtype=dispatch_quant_out_dtype,
    impl="cann",
)
```

参数约束和默认值遵循所安装 provider 的契约。MegaMoE 使用期间必须保持 buffer 存活；相关操作完成后通过 provider 的 `destroy()` 方法释放。注册层不缓存 buffer 对象，也不会自动销毁它；调用方需协调各 rank 的集合通信及 buffer 生命周期。
