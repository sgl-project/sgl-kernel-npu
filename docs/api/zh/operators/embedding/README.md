# Over-embedding 算子

LongCat2 的 over-embedding 路径先调用
[update_oe_token_table](update_oe_token_table.md) 将当前 token 写入请求历史，
再调用 [compute_n_gram_ids](compute_n_gram_ids.md)，从更新后的历史计算 embedding 表索引。

两个公共 API 均在 arch22 注册 `auto`、`ascendc` 和独立的 `torch_native` 参考实现。
`auto` 选择 `ascendc`，不自动回退。输入说明记录现有调用约定，不表示公共分发层
增加了相应的运行时校验。

在 arch22 Linux/CANN 环境使用 legacy 入口构建，再安装本次生成的 wheel：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh -a kernels 910
```

根目录 `setup.py --ops all` 入口不包含这两个尚未迁移的原生算子。
安装步骤见[构建指南](../../../../../python/sgl_kernel_npu/README.md)。
架构声明表示实现范围；NPU 精度、性能和 graph 结果应对应具体测试提交与环境。
