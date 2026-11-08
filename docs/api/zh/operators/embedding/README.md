# Over-embedding 算子

LongCat2 的 over-embedding 路径先调用
[update_oe_token_table](update_oe_token_table.md) 将当前 token 写入请求历史，
再调用 [compute_n_gram_ids](compute_n_gram_ids.md)，从更新后的历史计算 embedding 表索引。

两个公共 API 均在 arch22 注册 `auto`、`ascendc` 和独立的 `torch_native` 参考实现。
`auto` 选择 `ascendc`，不自动回退。输入说明记录现有调用约定，不表示公共分发层
增加了相应的运行时校验。

在 arch22 Linux/CANN 环境使用仓库根目录的 REFACTOR 入口构建，再安装本次生成的 wheel：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 setup.py --arch-version 2201 \
  --ops compute_n_gram_ids,update_oe_token_table --artifact wheel
```

`2201` 对应 arch22，包含 `ascend910b` 和 `ascend910_93` 编译目标。
两个算子均沿用官方独立 CANN 版本的 host、tiling 和 kernel，经本仓库的 Torch adapter
调用 ACLNN。CANN 管理 tiling 和调度，旧直接启动路径的手写 tiling cache 不再使用。
现有 `torch.ops.npu.compute_n_gram_ids` 和 `torch.ops.npu.update_oe_token_table`
接口保留；update 继续原地修改并返回同一表。

arch22 的 `--ops all` 包含这两个算子。需要三个 LongCat2 算子时，在上述 `--ops`
列表加入 `mlp_lightning_indexer`；单独生成 CANN 安装包则使用 `--artifact run`。
子集 wheel 替换现有安装，不与旧 wheel 叠加。
安装步骤见[构建指南](../../../../../python/sgl_kernel_npu/README.md)。

架构声明表示实现范围；本次 ACLNN 迁移后的 NPU 编译、精度、性能及已有 graph
capture/replay 用例仍需在对应构建上验证。此前直接启动路径的结果不作为新链路的验收结果。
