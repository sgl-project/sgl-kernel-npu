# 当前进度

- 当前分支：`feat/npu-glm5-flash`；本轮同步社区主干 `661c99a`。
- `build_pooled_page_table_64` 已迁至 `sgl_kernel_npu.operators.attention`，测试、CI 分组和双语文档已同步；合并时保留社区 attention 算子和本 PR 页表 API 的全部导出。
- 当前 PR 的 5 个独立防御性测试保留 `pytest.mark.extra_checks`：4 个非连续输入拒绝测试、1 个输出缓冲复用契约测试。
- 完全复用社区 `pytest.ini`：`addopts = -m "not extra_checks"`，本地和 CI 默认排除额外检查；`-m extra_checks` 仅运行额外检查，`-m ""` 运行全部用例。
- 移除上次添加的重复 CI `-m` 参数和仅因本地 pytest.ini 新增的 workflow 触发路径；不额外修改测试框架或 `tests/python/README.md`，后者仅随社区主干更新。
- 本机收集验证：默认 697/786，额外检查 89/786，全部 786；额外检查包含社区已有 RoPE 用例和本 PR 的 5 个用例。
- 页表 auto/native 两个入口的 10 次 CPU 用例执行、3 个空输入/布局探针及 SGLang 新路径契约检查通过。
- 全仓 pre-commit 通过；CI lint 使用的构建配置单测为 15 passed、11 skipped（本机环境）。
- 既有 CPU/静态检查不代替目标 NPU 上的精度、性能或 SGLang 端到端验收。
