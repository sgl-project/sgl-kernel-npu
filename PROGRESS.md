# 当前进度

- 当前分支：`feat/npu-glm5-flash`；已推送提交：`4b47dd4`。
- `build_pooled_page_table_64` 已迁至 `sgl_kernel_npu.operators.attention`，测试、CI 分组和双语文档已同步。
- 本轮为当前 PR 新增算子中 5 个独立防御性测试添加 `pytest.mark.extra_checks`：4 个非连续输入拒绝测试、1 个输出缓冲复用契约测试。
- 根目录 `pytest.ini` 注册标记；默认仍运行全部用例，`-m extra_checks` 仅选额外检查，`-m "not extra_checks"` 排除额外检查。
- 典型与合法边界精度测试保持常规分类；嵌入精度流程的状态不变断言保留，不改测试框架或 `tests/python/README.md`。
- 验证：`--collect-only --strict-markers -m extra_checks` 选中 5/512 个用例；反向筛选选中 507/512。AST 检查确认仅新增装饰器，正文及参数矩阵未变；全仓 pre-commit 通过。
- CI 的 kernel 测试公共脚本与 lint pytest 入口统一使用 `-m "not extra_checks"`；PR/Daily workflow 纳入 `pytest.ini` 变更触发。手工 pytest 不筛选时仍执行全部用例。
- 本轮按用户要求提交并推送 marker、注册和 CI 排除配置。
- 既有 CPU/静态检查不代替目标 NPU 上的精度、性能或 SGLang 端到端验收。
