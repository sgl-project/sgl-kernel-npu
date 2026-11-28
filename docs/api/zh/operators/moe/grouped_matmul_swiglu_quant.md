# A2/A3 W8A8 和 W4A8 分组矩阵乘、SwiGLU 与量化融合算子

该 CANN 9.2 算子为 A2/A3 注册，复用现有 `ops_transformer` profile、
固定版本的 CANN 源码、vendor 打包和 `OpBuilder`，不依赖 vLLM 源码或安装包。

```bash
python setup.py --arch-version 2201 --ops apply_token_bitmask --third-party ops_transformer -j8
```

该 profile 增加一个隔离的 GMM vendor，同时打包 `ascend910b`、`ascend910_93` 内核。
固定版本的上游接口尚无限幅参数，因此通过补丁补齐 clipped SwiGLU，使用私有
算子名和 ACLNN 符号保留工具包原有 ABI。已有普通和 experimental vendor 的版本不变。

Torch 绑定沿用已有 CANN `OpBuilder` 流程，首次调用可能编译并写入共享 Torch 扩展缓存。
与其他 CANN 算子一样，应先预热再捕获图；图重放不会编译绑定。

```python
from sgl_kernel_npu.operators.moe import grouped_matmul_swiglu_quant
```

`grouped_matmul_swiglu_quant(x, weight, weight_scale, x_scale, group_list, swiglu_limit=0.0, *, weight_assist_matrix=None, impl="auto")`
支持 `auto`/`cann_ops_transformers_cann920`（A3 CANN）或 `torch_native`（独立 eager 参考实现）。

该算子融合分组矩阵乘、反量化、SwiGLU 和逐 token 动态量化。

- `x`：INT8 `[R,K]` 激活。
- `weight`：W8A8 使用 INT8 NZ `[E,K,N]`，W4A8 使用打包后的 INT32 NZ `[E,K,N/8]`。
- `weight_scale`：W8A8 使用 BF16 `[E,N]`。W4A8 使用 INT64 张量存储 UINT64 scale 位模式，形状为 `[E,N]` 或 `[E,K_groups,N]`。
- `weight_assist_matrix`：W4A8 需要的 FP32 `[E,N]` 激活偏移补偿矩阵。W8A8 省略此参数。
- `x_scale`：FP32 `[R]` 激活反量化 scale。
- `group_list`：设备上的 INT64 `[E]` 专家累计结束位置。
- `swiglu_limit`：正值将 gate 上界限制为 swiglu_limit、up 限制为 `[-swiglu_limit,swiglu_limit]`；0 表示不截断。

返回 INT8 `[R,N/2]` 和 FP32 `[R]` 反量化 scale。动态量化按每行绝对值最大值
除以 127 计算 scale，再舍入和饱和到 INT8。只有 `group_list[-1]` 之前的行有效，
填充行内容未定义。同一次调用的张量必须位于同一 NPU。

使用 PyTorch 准备 NZ 权重时，需在 `torch_npu.npu_format_cast(weight, 29)` 前启用
`torch.npu.config.allow_internal_format = True`。W4A8 与 SGLang ModelSlim 加载方式一致，
先把打包的 INT8 `[E,K,N/2]` 权重转换为 NZ，再将其视作 INT32。
原生算子支持图捕获及输入变化后的重放。eager 参考实现会回读专家边界，应在捕获之外运行。

数值测试比较有效行反量化输出与 eager 参考实现，W8A8 和 W4A8 的相对 RMS 容差分别
为 0.01 和 0.02。测试覆盖空专家、不均匀专家分布，以及图重放时修改激活和专家边界。
W4A8 测试覆盖 per-channel 和 per-group scale。
