# 公共框架 UT 实现与参数覆盖总览

代码快照：`c83ef124bcca0a34cc57f66fa495da2608c1c6d4`；基线：官方 `main@df2b9f168ffb5635c62b6dc59d22c195ae1372c6`（已包含 #869/#870）。
分支：[cl-vv-h/sgl-kernel-npu:framework-ut-main-npu](https://github.com/cl-vv-h/sgl-kernel-npu/tree/framework-ut-main-npu)。
本文件只描述当前实现和已有证据；不将参数进入过函数等同于所有取值或分支均已覆盖。

## 1. 数量、运行与证据

- 12个测试文件、92个测试函数，参数化后219条：218条UT和1条真实NPU集成测试。
- 算子框架75条；精度工具106条；性能UT37条；真实采集1条。
- 已完成静态检查与219条收集；本地替代NPU/ACL前置条件的CPU诊断为217 passed、2 deselected。
- 真实CPU/NPU设备差异及真实profiler未在本机执行；全部219条真实NPU验证仍待执行。名字中含real_npu的检测UT在本地诊断时同样替代了ACL，不是实机证据。
- 默认精度阈值是当前配置，UT不证明它们适合所有算子；固定12.5等替身耗时不是性能基线。当前没有历史性能回退判定。

```bash
# 在仓库根目录、已配置NPU/torch_npu/ACL和安装包的环境中，单进程执行。
python -m pytest tests/python/framework -m 'not npu_integration' -v -rs -x --npu-device=0
python -m pytest tests/python/framework -m npu_integration -v -s -rs -x --npu-device=0
# 不加marker筛选时运行全部219条，包括真实采集。
```

## 2. 实现结构与替身边界

```text
tests/python/framework/
  conftest.py                         环境、源码隔离、轻量算子、测量替身
  test_registry.py                   公共注册/查询
  test_generation.py                 ACL架构识别
  test_generation_registry.py        按架构的注册规则
  test_dispatch.py                   RMSNorm公共入口与薄适配
  test_imports.py                    导入、依赖和隔离
  test_standards.py                  精度标准选择和合法性
  test_accuracy.py                   输出数值/结构比较
  test_check_accuracy.py             精度执行流程与配置
  test_performance.py                性能控制和报告
  test_profile_data.py               CSV/目录/命名
  test_benchmark_npu.py               底层采集调用协议
  test_performance_integration.py     真实NPU运算及profiler链路
```

| fixture / hook | 作用及清理 |
| --- | --- |
| npu_runtime | session/autouse；要求NPU与ACL，选择设备、正常导入原生包，结束恢复显式选择前设备。 |
| package_factory | 每用例私有命名空间加载checkout真实模块，隔离注册和缓存，结束删除私有模块。 |
| operator_modules / registry / generation_module | 提供已加载的真实模块/注册表；不替换实现。 |
| registry_for_generation | 导入registry前替代硬件检测结果，验证ARCH22/ARCH35规则；不代表跨架构实机运行。 |
| rmsnorm_module / kernels | 保留公共入口与适配，只替代implementation叶子kernel。 |
| accuracy_tools / accuracy_case | 真实精度工具、真实注册和小NPU Tensor算术；临时把正常registry名称绑定到同一私有模块，结束恢复。 |
| performance_case | 复用绑定executor，临时benchmark返回12.5并记录调用；上下文、CSV、JSON、报告真实，临时cwd恢复。 |
| pytest_configure | 声明npu_integration标签；标签不会默认跳过采集。 |

`test_benchmark_npu`保持真实benchmark，替代底层profiler与同步来记录事件；`test_profile_data`使用真实临时文件。
真实集成只用accuracy_case，不用performance_case；两个测试backend是独立函数但公式相同，验证通路接通，不证明生产Triton/AscendC算法或精度参考的算法独立性。
上层conftest的自动配置只作用于operators目录；framework用例显式设置和恢复精度/性能上下文。
ACL识别固定查询可见设备0属性601；Tensor/性能设备仍由--npu-device选择，遵循主线可见设备架构一致的假设。

## 3. 每个测试函数的功能

每行是一项测试函数；“条数”是参数化展开后的pytest用例数。ID用于后续参数覆盖表，并链接到实际测试源码。

| 文件 | 函数数 | 条数 |
| --- | ---: | ---: |
| [test_registry.py](test_registry.py) | 14 | 20 |
| [test_generation.py](test_generation.py) | 6 | 10 |
| [test_generation_registry.py](test_generation_registry.py) | 11 | 23 |
| [test_dispatch.py](test_dispatch.py) | 9 | 15 |
| [test_imports.py](test_imports.py) | 6 | 7 |
| [test_standards.py](test_standards.py) | 5 | 20 |
| [test_accuracy.py](test_accuracy.py) | 9 | 49 |
| [test_check_accuracy.py](test_check_accuracy.py) | 12 | 37 |
| [test_performance.py](test_performance.py) | 7 | 12 |
| [test_profile_data.py](test_profile_data.py) | 9 | 20 |
| [test_benchmark_npu.py](test_benchmark_npu.py) | 3 | 5 |
| [test_performance_integration.py](test_performance_integration.py) | 1 | 1 |
| 合计 | 92 | 219 |

### test_registry.py：注册与查询

| ID | 测试函数 | 条数 | 验证内容 |
| --- | --- | ---: | --- |
| [REG01](test_registry.py#L9) | `test_public_registration_preserves_function` | 1 | 保留函数身份，使用显式算子名，注册不执行函数。 |
| [REG02](test_registry.py#L18) | `test_duplicate_public_function_is_rejected_without_changing_name` | 1 | 同一公共函数重复注册失败，原名称不变。 |
| [REG03](test_registry.py#L29) | `test_duplicate_public_name_is_rejected_without_registering_function` | 1 | 公共名称冲突失败，新函数不进入注册表。 |
| [REG04](test_registry.py#L43) | `test_unregistered_function_is_rejected` | 3 | 未注册的公共函数不能被查询或枚举。 |
| [REG05](test_registry.py#L54) | `test_impl_registration_and_lookup_do_not_execute_backend` | 1 | 注册和两种查询返回原实现，不执行backend。 |
| [REG06](test_registry.py#L65) | `test_impl_can_be_registered_before_public_operator` | 1 | 允许先注册实现，再注册公共入口。 |
| [REG07](test_registry.py#L75) | `test_duplicate_impl_is_rejected_without_overwriting` | 2 | 两次默认全架构声明发生重叠，拒绝覆盖；同函数和不同函数都覆盖。 |
| [REG08](test_registry.py#L89) | `test_same_impl_name_is_scoped_to_operator` | 1 | 不同算子的同名impl独立查询。 |
| [REG09](test_registry.py#L99) | `test_aliases_preserve_callable_identity` | 1 | 不同impl别名可指向同一个函数，并正常枚举。 |
| [REG10](test_registry.py#L110) | `test_list_impls_is_sorted_and_excludes_other_operators` | 1 | 列表排序、按算子隔离，修改返回列表不影响注册表。 |
| [REG11](test_registry.py#L127) | `test_registered_operator_without_impls_has_no_choices` | 1 | 仅注册公共入口时列表为空，查找实现失败。 |
| [REG12](test_registry.py#L137) | `test_unknown_name_has_no_choices` | 1 | 未知算子名列表为空，查找失败。 |
| [REG13](test_registry.py#L145) | `test_unknown_impl_reports_choices_without_fallback` | 4 | 未知名、大小写差异、空字符串和None报错，包含已声明选择。 |
| [REG14](test_registry.py#L162) | `test_public_exports_use_the_same_registry` | 1 | 公共导出的查询函数与注册函数使用同一注册表。 |

### test_generation.py：ACL架构识别

| ID | 测试函数 | 条数 | 验证内容 |
| --- | --- | ---: | --- |
| [GEN01](test_generation.py#L10) | `test_known_arch_maps_to_expected_generation` | 2 | 2201映射ARCH22，3510映射ARCH35。 |
| [GEN02](test_generation.py#L21) | `test_unknown_arch_is_not_guessed` | 3 | -1、None和不可哈希ID报错，不猜测架构。 |
| [GEN03](test_generation.py#L28) | `test_real_npu_is_identified` | 1 | Hardware结果与真实ACL设备0属性601查询一致；实际NPU运行待验证。 |
| [GEN04](test_generation.py#L37) | `test_successful_detection_is_cached` | 1 | 成功结果复用，ACL查询只发生一次，参数为(0,601)。 |
| [GEN05](test_generation.py#L52) | `test_device_query_failure_can_retry_without_caching` | 2 | Python异常和ACL非零错误码不缓存；随后可重试并缓存成功结果。 |
| [GEN06](test_generation.py#L79) | `test_unknown_arch_is_not_cached` | 1 | 未知架构失败后可以再次查询得到已知架构。 |

### test_generation_registry.py：按架构注册实现

| ID | 测试函数 | 条数 | 验证内容 |
| --- | --- | ---: | --- |
| [GREG01](test_generation_registry.py#L10) | `test_support_is_classified_against_import_time_hardware` | 2 | 导入时按当前架构分类；查询和枚举不再检测硬件；检查不支持异常字段。 |
| [GREG02](test_generation_registry.py#L68) | `test_support_and_exclusion_sets` | 6 | 支持/排除列表的空集合、重复项、当前/其他/全部架构。 |
| [GREG03](test_generation_registry.py#L85) | `test_support_options_are_mutually_exclusive` | 1 | 不能同时指定支持列表和排除列表，包括同时为空。 |
| [GREG04](test_generation_registry.py#L95) | `test_generation_strings_are_rejected_without_registration` | 2 | 列表必须使用Generation枚举，字符串无效且不留下注册。 |
| [GREG05](test_generation_registry.py#L104) | `test_disjoint_entries_select_current_architecture` | 4 | 同impl可按不重叠范围注册两个入口；两架构和两注册顺序均选择正确。 |
| [GREG06](test_generation_registry.py#L124) | `test_equivalent_support_declarations_allow_aliases` | 1 | 包含ARCH22与排除ARCH35表达相同支持集，别名均可用。 |
| [GREG07](test_generation_registry.py#L147) | `test_aliases_have_independent_support_ranges` | 2 | 同函数的不同impl名称可声明不同范围，各自判断可用性。 |
| [GREG08](test_generation_registry.py#L176) | `test_shared_callable_support_is_scoped_to_operator` | 1 | 同函数在不同算子下的支持声明独立。 |
| [GREG09](test_generation_registry.py#L190) | `test_support_declaration_is_a_snapshot` | 1 | 创建装饰器后修改原支持列表，不改变已捕获的范围。 |
| [GREG10](test_generation_registry.py#L205) | `test_overlapping_support_is_rejected_atomically` | 2 | 重叠范围被拒绝；原入口不变，未注册范围仍可继续补充。 |
| [GREG11](test_generation_registry.py#L237) | `test_empty_support_declaration_can_be_extended` | 1 | 空支持声明可枚举、暂不可用，随后可补充同名impl的架构入口。 |

### test_dispatch.py：公共入口与适配

| ID | 测试函数 | 条数 | 验证内容 |
| --- | --- | ---: | --- |
| [DIS01](test_dispatch.py#L16) | `test_explicit_impl_calls_only_selected_backend` | 2 | 显式triton或torch_native仅调用对应叶子，正确透传参数和返回值。 |
| [DIS02](test_dispatch.py#L30) | `test_default_and_auto_select_triton` | 2 | 省略impl或指定auto均选择当前默认Triton入口。 |
| [DIS03](test_dispatch.py#L42) | `test_optional_arguments_are_forwarded_without_impl` | 2 | 位置/关键字调用均正确传计算参数，impl不传给叶子kernel。 |
| [DIS04](test_dispatch.py#L72) | `test_invalid_impl_does_not_call_backends` | 2 | 未知impl或显式None在执行任何backend前失败。 |
| [DIS05](test_dispatch.py#L80) | `test_impl_is_keyword_only` | 1 | impl不能按位置传入。 |
| [DIS06](test_dispatch.py#L91) | `test_backend_error_propagates_without_fallback` | 2 | 保留选中backend的异常，不偷偷尝试其他实现。 |
| [DIS07](test_dispatch.py#L104) | `test_new_registered_backend_needs_no_dispatcher_or_directory_change` | 1 | 新增测试backend注册后可直接经现有公共入口调用。 |
| [DIS08](test_dispatch.py#L123) | `test_unsupported_impl_is_rejected_by_public_entry` | 1 | 当前架构不支持时，公共入口在执行backend前失败。 |
| [DIS09](test_dispatch.py#L149) | `test_cached_backends_do_not_cross_routes` | 2 | 两种起始顺序反复切换实现；逐次检查缓存路由、完整调用记录和新参数/返回对象。 |

### test_imports.py：导入与隔离

| ID | 测试函数 | 条数 | 验证内容 |
| --- | --- | ---: | --- |
| [IMP01](test_imports.py#L12) | `test_checked_out_public_api_uses_the_same_registry` | 1 | 源码路径、公共导出、RMSNorm注册和auto别名均来自checkout。 |
| [IMP02](test_imports.py#L28) | `test_public_import_registers_without_importing_kernel_leaves` | 1 | 公共导入完成注册，不提前导入叶子；重复普通import复用模块。 |
| [IMP03](test_imports.py#L44) | `test_hardware_detection_failure_aborts_registry_import` | 1 | 硬件检测失败时导入失败，原异常传出，无半初始化registry缓存。 |
| [IMP04](test_imports.py#L59) | `test_private_registrations_do_not_leak_to_other_modules` | 1 | 两份私有注册隔离，安装包路径和模块绑定快照不被改变。 |
| [IMP05](test_imports.py#L83) | `test_checkout_framework_does_not_require_installed_operators` | 1 | 安装包没有新版operators时，checkout私有框架仍可加载。 |
| [IMP06](test_imports.py#L100) | `test_missing_backend_is_isolated_and_failed_import_can_retry` | 2 | 某个叶子缺依赖只影响该路由；恢复后能重试，不回退。 |

### test_standards.py：精度标准

| ID | 测试函数 | 条数 | 验证内容 |
| --- | --- | ---: | --- |
| [STD01](test_standards.py#L20) | `test_default_standards_match_dtype_policy` | 6 | 六种dtype的默认atol/rtol与独立期望表一致。 |
| [STD02](test_standards.py#L26) | `test_override_changes_only_requested_dtype` | 1 | 覆盖一个dtype，其他回退默认，原配置和全局默认不被污染。 |
| [STD03](test_standards.py#L45) | `test_invalid_tolerances_are_rejected` | 8 | atol/rtol分别拒绝负数、NaN和正负Inf。 |
| [STD04](test_standards.py#L65) | `test_invalid_standard_mapping_is_rejected` | 4 | 拒绝非映射、非字符串键、错误值，包括未查询项中的非法配置。 |
| [STD05](test_standards.py#L71) | `test_unknown_dtype_has_no_default_standard` | 1 | 测试专用虚构键无默认值；显式提供标准后可查询，不依赖float64一直缺省。 |

### test_accuracy.py：输出数值与结构比较

| ID | 测试函数 | 条数 | 验证内容 |
| --- | --- | ---: | --- |
| [ACC01](test_accuracy.py#L33) | `test_tensor_tolerance_controls_pass_and_failure` | 11 | FP32绝对/相对误差内、边界和外；FP16/BF16代表场景，真实Tensor比较。 |
| [ACC02](test_accuracy.py#L48) | `test_integer_and_bool_tensors_compare_exactly` | 4 | int8/int32/int64/bool默认相同通过，差异失败。 |
| [ACC03](test_accuracy.py#L57) | `test_tensor_metadata_mismatch_is_rejected` | 4 | 拒绝shape、dtype、device及Tensor/非Tensor不一致。 |
| [ACC04](test_accuracy.py#L76) | `test_nested_outputs_are_compared_with_error_paths` | 1 | 递归比较dict/list/tuple/Tensor，字典顺序无关，深层错误给出路径。 |
| [ACC05](test_accuracy.py#L103) | `test_container_structure_mismatch_is_rejected` | 6 | 拒绝容器类型、列表/元组长度、字典缺键/多键/不同键。 |
| [ACC06](test_accuracy.py#L132) | `test_python_scalars_follow_type_and_tolerance_policy` | 16 | Python float/complex容差比较，其他简单标量精确比较，True与1等类型差异失败。 |
| [ACC07](test_accuracy.py#L146) | `test_nan_values_are_rejected` | 4 | Tensor及Python浮点，单侧或双侧NaN都失败。 |
| [ACC08](test_accuracy.py#L159) | `test_floating_tensor_requires_an_accuracy_standard` | 1 | 局部移除FP32默认标准后报错；显式标准可恢复比较，结束后还原配置。 |
| [ACC09](test_accuracy.py#L174) | `test_unsupported_output_type_is_rejected` | 2 | set和普通object即使与自身比较也报不支持类型。 |

### test_check_accuracy.py：精度执行流程

| ID | 测试函数 | 条数 | 验证内容 |
| --- | --- | ---: | --- |
| [CHK01](test_check_accuracy.py#L20) | `test_accuracy_success_returns_a_bound_executor` | 1 | ref先于dut各一次，参数和返回executor正确，再调用只执行dut。 |
| [CHK02](test_check_accuracy.py#L43) | `test_accuracy_mismatch_fails` | 1 | 真实Tensor结果不同必须精度失败，不能漏掉比较步骤。 |
| [CHK03](test_check_accuracy.py#L59) | `test_custom_standard_reaches_comparison` | 1 | 同一差异默认失败，自定义标准后通过，验证标准透传。 |
| [CHK04](test_check_accuracy.py#L119) | `test_impl_arguments_override_each_configured_side` | 7 | 省略/None补默认，显式参数分别覆盖待测与参考实现。 |
| [CHK05](test_check_accuracy.py#L135) | `test_impl_in_operator_kwargs_is_rejected` | 3 | kwargs中出现impl一律拒绝，两侧不执行。 |
| [CHK06](test_check_accuracy.py#L151) | `test_skip_accuracy_returns_an_unchecked_executor` | 1 | 跳过时不执行、不要求参考存在；executor可运行且仍未标记精度通过。 |
| [CHK07](test_check_accuracy.py#L183) | `test_accuracy_skip_context_overrides_code_setting` | 5 | 上下文显式bool优先，None使用代码值；校验覆盖后的有效值。 |
| [CHK08](test_check_accuracy.py#L199) | `test_invalid_effective_skip_flag_is_rejected` | 4 | 有效skip值为整数、字符串或None时拒绝执行。 |
| [CHK09](test_check_accuracy.py#L216) | `test_same_entry_cannot_be_used_as_reference` | 2 | 同名或同函数别名不能作为独立精度参考。 |
| [CHK10](test_check_accuracy.py#L259) | `test_implementation_resolution_controls_fail_and_skip` | 8 | 8种实现缺失/架构不支持/参考错误/跳过组合，检查失败或skip及异常来源。 |
| [CHK11](test_check_accuracy.py#L296) | `test_backend_failure_stops_accuracy_flow` | 2 | ref/dut执行失败原样传出，不重试；ref失败时dut不执行。 |
| [CHK12](test_check_accuracy.py#L309) | `test_accuracy_configuration_is_restored` | 2 | 嵌套impl/skip配置正常或异常退出后恢复。 |

### test_performance.py：性能控制与报告

| ID | 测试函数 | 条数 | 验证内容 |
| --- | --- | ---: | --- |
| [PERF01](test_performance.py#L12) | `test_disabled_performance_has_no_side_effects` | 1 | 默认关闭，即使提供路径也不采集、不执行、不生成文件。 |
| [PERF02](test_performance.py#L27) | `test_performance_context_overrides_code_switch` | 2 | 上下文True/False分别覆盖代码False/True。 |
| [PERF03](test_performance.py#L43) | `test_invalid_effective_performance_flag_is_rejected` | 1 | 整数enable_perf=1被拒绝。 |
| [PERF04](test_performance.py#L53) | `test_enabled_performance_reports_measurement_and_metadata` | 3 | 固定测量12.5及采样参数正确传递，元数据/JSON/设备/回调准确，精度状态True/False/缺失。 |
| [PERF05](test_performance.py#L92) | `test_performance_paths_follow_context_precedence` | 2 | 代码路径或上下文覆盖路径生效，实际CSV位置/内容正确，另一组路径不创建。 |
| [PERF06](test_performance.py#L123) | `test_benchmark_failure_does_not_publish_success` | 1 | 采集异常原样传出，不追加成功CSV/报告，失败产物保留。 |
| [PERF07](test_performance.py#L159) | `test_performance_context_is_restored` | 2 | 内层关闭/外层开启，正常或异常退出后恢复开关、参数、路径和回调。 |

### test_profile_data.py：采集数据与文件

| ID | 测试函数 | 条数 | 验证内容 |
| --- | --- | ---: | --- |
| [DATA01](test_profile_data.py#L26) | `test_summary_files_are_summed_without_statistics_double_counting` | 1 | 递归汇总多个summary，忽略同时存在的statistic，10+20除active2得15。 |
| [DATA02](test_profile_data.py#L34) | `test_statistics_are_used_when_summary_is_absent` | 1 | 只有statistic时读取Total Time(us)，30除active2得15。 |
| [DATA03](test_profile_data.py#L41) | `test_missing_or_unusable_reports_are_rejected` | 4 | 缺文件、缺列、只有表头、总量为0均失败；坏summary不降级到好statistic。 |
| [DATA04](test_profile_data.py#L61) | `test_invalid_task_durations_are_rejected` | 4 | 拒绝非数字、负数、NaN、Inf耗时。 |
| [DATA05](test_profile_data.py#L68) | `test_result_csv_can_be_created_and_appended` | 1 | 创建父目录和表头，追加两行，保留JSON转义字段，耗时保留三位小数。 |
| [DATA06](test_profile_data.py#L99) | `test_incompatible_csv_schema_preserves_existing_results` | 1 | 列不兼容时失败，已有文件字节不变。 |
| [DATA07](test_profile_data.py#L114) | `test_profile_directory_flattening_preserves_files` | 2 | 单host目录完整上移；同名冲突时不移动、不覆盖。 |
| [DATA08](test_profile_data.py#L143) | `test_profile_case_names_are_safe_bounded_and_unique` | 1 | 同参数生成不同名字，特殊字符被处理，单组件不超255字节。 |
| [DATA09](test_profile_data.py#L170) | `test_profile_parser_requires_positive_integer_active` | 5 | active拒绝0、bool、小数、NaN、Inf。 |

### test_benchmark_npu.py：采集调用协议

| ID | 测试函数 | 条数 | 验证内容 |
| --- | --- | ---: | --- |
| [BENCH01](test_benchmark_npu.py#L12) | `test_benchmark_separates_initial_execution_and_profile_iterations` | 1 | 真实benchmark按预执行/同步/进入/五轮同步执行step/退出的顺序，真实解析合成CSV得10。 |
| [BENCH02](test_benchmark_npu.py#L91) | `test_existing_profile_directory_is_not_overwritten` | 1 | 已有目录时执行前失败，原文件不变。 |
| [BENCH03](test_benchmark_npu.py#L114) | `test_benchmark_rejects_invalid_sampling_before_execution` | 3 | warmup0、active0或小数在创建目录/执行前被拒绝。 |

### test_performance_integration.py：真实NPU集成

| ID | 测试函数 | 条数 | 验证内容 |
| --- | --- | ---: | --- |
| [INT01](test_performance_integration.py#L14) | `test_real_npu_accuracy_and_performance_pipeline` | 1 | 轻量真实NPU表达式完成精度、真实profiler、CSV解析与报告；检查正有限耗时，不设回退阈值。 |

<details>
<summary>展开查看全部219条pytest实例（含参数化node ID）</summary>

上表逐函数说明功能；以下是完整实例清单。每条实例的功能对应所属函数，参数值可通过上表源码链接定位。

```text
tests/python/framework/test_accuracy.py::test_tensor_tolerance_controls_pass_and_failure[equal]
tests/python/framework/test_accuracy.py::test_tensor_tolerance_controls_pass_and_failure[atol-inside]
tests/python/framework/test_accuracy.py::test_tensor_tolerance_controls_pass_and_failure[atol-boundary]
tests/python/framework/test_accuracy.py::test_tensor_tolerance_controls_pass_and_failure[atol-outside]
tests/python/framework/test_accuracy.py::test_tensor_tolerance_controls_pass_and_failure[rtol-inside]
tests/python/framework/test_accuracy.py::test_tensor_tolerance_controls_pass_and_failure[rtol-boundary]
tests/python/framework/test_accuracy.py::test_tensor_tolerance_controls_pass_and_failure[rtol-outside]
tests/python/framework/test_accuracy.py::test_tensor_tolerance_controls_pass_and_failure[fp16-inside]
tests/python/framework/test_accuracy.py::test_tensor_tolerance_controls_pass_and_failure[fp16-outside]
tests/python/framework/test_accuracy.py::test_tensor_tolerance_controls_pass_and_failure[bf16-inside]
tests/python/framework/test_accuracy.py::test_tensor_tolerance_controls_pass_and_failure[bf16-outside]
tests/python/framework/test_accuracy.py::test_integer_and_bool_tensors_compare_exactly[int8]
tests/python/framework/test_accuracy.py::test_integer_and_bool_tensors_compare_exactly[int32]
tests/python/framework/test_accuracy.py::test_integer_and_bool_tensors_compare_exactly[int64]
tests/python/framework/test_accuracy.py::test_integer_and_bool_tensors_compare_exactly[bool]
tests/python/framework/test_accuracy.py::test_tensor_metadata_mismatch_is_rejected[shape]
tests/python/framework/test_accuracy.py::test_tensor_metadata_mismatch_is_rejected[dtype]
tests/python/framework/test_accuracy.py::test_tensor_metadata_mismatch_is_rejected[device]
tests/python/framework/test_accuracy.py::test_tensor_metadata_mismatch_is_rejected[tensor-type]
tests/python/framework/test_accuracy.py::test_nested_outputs_are_compared_with_error_paths
tests/python/framework/test_accuracy.py::test_container_structure_mismatch_is_rejected[list-vs-tuple]
tests/python/framework/test_accuracy.py::test_container_structure_mismatch_is_rejected[list-length]
tests/python/framework/test_accuracy.py::test_container_structure_mismatch_is_rejected[tuple-length]
tests/python/framework/test_accuracy.py::test_container_structure_mismatch_is_rejected[missing-key]
tests/python/framework/test_accuracy.py::test_container_structure_mismatch_is_rejected[extra-key]
tests/python/framework/test_accuracy.py::test_container_structure_mismatch_is_rejected[different-key]
tests/python/framework/test_accuracy.py::test_python_scalars_follow_type_and_tolerance_policy[float-inside]
tests/python/framework/test_accuracy.py::test_python_scalars_follow_type_and_tolerance_policy[float-outside]
tests/python/framework/test_accuracy.py::test_python_scalars_follow_type_and_tolerance_policy[complex-inside]
tests/python/framework/test_accuracy.py::test_python_scalars_follow_type_and_tolerance_policy[complex-outside]
tests/python/framework/test_accuracy.py::test_python_scalars_follow_type_and_tolerance_policy[int-equal]
tests/python/framework/test_accuracy.py::test_python_scalars_follow_type_and_tolerance_policy[int-different]
tests/python/framework/test_accuracy.py::test_python_scalars_follow_type_and_tolerance_policy[bool-equal]
tests/python/framework/test_accuracy.py::test_python_scalars_follow_type_and_tolerance_policy[bool-different]
tests/python/framework/test_accuracy.py::test_python_scalars_follow_type_and_tolerance_policy[str-equal]
tests/python/framework/test_accuracy.py::test_python_scalars_follow_type_and_tolerance_policy[str-different]
tests/python/framework/test_accuracy.py::test_python_scalars_follow_type_and_tolerance_policy[bytes-equal]
tests/python/framework/test_accuracy.py::test_python_scalars_follow_type_and_tolerance_policy[bytes-different]
tests/python/framework/test_accuracy.py::test_python_scalars_follow_type_and_tolerance_policy[none-equal]
tests/python/framework/test_accuracy.py::test_python_scalars_follow_type_and_tolerance_policy[none-type-mismatch]
tests/python/framework/test_accuracy.py::test_python_scalars_follow_type_and_tolerance_policy[bool-is-not-int]
tests/python/framework/test_accuracy.py::test_python_scalars_follow_type_and_tolerance_policy[int-is-not-float]
tests/python/framework/test_accuracy.py::test_nan_values_are_rejected[both-nan-tensor]
tests/python/framework/test_accuracy.py::test_nan_values_are_rejected[both-nan-scalar]
tests/python/framework/test_accuracy.py::test_nan_values_are_rejected[one-nan-tensor]
tests/python/framework/test_accuracy.py::test_nan_values_are_rejected[one-nan-scalar]
tests/python/framework/test_accuracy.py::test_floating_tensor_requires_an_accuracy_standard
tests/python/framework/test_accuracy.py::test_unsupported_output_type_is_rejected[set]
tests/python/framework/test_accuracy.py::test_unsupported_output_type_is_rejected[object]
tests/python/framework/test_benchmark_npu.py::test_benchmark_separates_initial_execution_and_profile_iterations
tests/python/framework/test_benchmark_npu.py::test_existing_profile_directory_is_not_overwritten
tests/python/framework/test_benchmark_npu.py::test_benchmark_rejects_invalid_sampling_before_execution[zero-warmup]
tests/python/framework/test_benchmark_npu.py::test_benchmark_rejects_invalid_sampling_before_execution[zero-active]
tests/python/framework/test_benchmark_npu.py::test_benchmark_rejects_invalid_sampling_before_execution[fractional-active]
tests/python/framework/test_check_accuracy.py::test_accuracy_success_returns_a_bound_executor
tests/python/framework/test_check_accuracy.py::test_accuracy_mismatch_fails
tests/python/framework/test_check_accuracy.py::test_custom_standard_reaches_comparison
tests/python/framework/test_check_accuracy.py::test_impl_arguments_override_each_configured_side[both-omitted]
tests/python/framework/test_check_accuracy.py::test_impl_arguments_override_each_configured_side[both-none]
tests/python/framework/test_check_accuracy.py::test_impl_arguments_override_each_configured_side[explicit-dut]
tests/python/framework/test_check_accuracy.py::test_impl_arguments_override_each_configured_side[explicit-reference]
tests/python/framework/test_check_accuracy.py::test_impl_arguments_override_each_configured_side[both-explicit]
tests/python/framework/test_check_accuracy.py::test_impl_arguments_override_each_configured_side[none-dut]
tests/python/framework/test_check_accuracy.py::test_impl_arguments_override_each_configured_side[none-reference]
tests/python/framework/test_check_accuracy.py::test_impl_in_operator_kwargs_is_rejected[None]
tests/python/framework/test_check_accuracy.py::test_impl_in_operator_kwargs_is_rejected[dut]
tests/python/framework/test_check_accuracy.py::test_impl_in_operator_kwargs_is_rejected[other]
tests/python/framework/test_check_accuracy.py::test_skip_accuracy_returns_an_unchecked_executor
tests/python/framework/test_check_accuracy.py::test_accuracy_skip_context_overrides_code_setting[context-skips]
tests/python/framework/test_check_accuracy.py::test_accuracy_skip_context_overrides_code_setting[context-requires-check]
tests/python/framework/test_check_accuracy.py::test_accuracy_skip_context_overrides_code_setting[code-skips]
tests/python/framework/test_check_accuracy.py::test_accuracy_skip_context_overrides_code_setting[code-requires-check]
tests/python/framework/test_check_accuracy.py::test_accuracy_skip_context_overrides_code_setting[validate-effective-value]
tests/python/framework/test_check_accuracy.py::test_invalid_effective_skip_flag_is_rejected[1]
tests/python/framework/test_check_accuracy.py::test_invalid_effective_skip_flag_is_rejected[0]
tests/python/framework/test_check_accuracy.py::test_invalid_effective_skip_flag_is_rejected[true]
tests/python/framework/test_check_accuracy.py::test_invalid_effective_skip_flag_is_rejected[None]
tests/python/framework/test_check_accuracy.py::test_same_entry_cannot_be_used_as_reference[same-name]
tests/python/framework/test_check_accuracy.py::test_same_entry_cannot_be_used_as_reference[alias]
tests/python/framework/test_check_accuracy.py::test_implementation_resolution_controls_fail_and_skip[unknown-dut]
tests/python/framework/test_check_accuracy.py::test_implementation_resolution_controls_fail_and_skip[unknown-reference]
tests/python/framework/test_check_accuracy.py::test_implementation_resolution_controls_fail_and_skip[unsupported-dut]
tests/python/framework/test_check_accuracy.py::test_implementation_resolution_controls_fail_and_skip[unsupported-reference]
tests/python/framework/test_check_accuracy.py::test_implementation_resolution_controls_fail_and_skip[reference-error-before-skip]
tests/python/framework/test_check_accuracy.py::test_implementation_resolution_controls_fail_and_skip[both-unsupported]
tests/python/framework/test_check_accuracy.py::test_implementation_resolution_controls_fail_and_skip[skip-accuracy-unknown-dut]
tests/python/framework/test_check_accuracy.py::test_implementation_resolution_controls_fail_and_skip[skip-accuracy-unsupported-dut]
tests/python/framework/test_check_accuracy.py::test_backend_failure_stops_accuracy_flow[ref]
tests/python/framework/test_check_accuracy.py::test_backend_failure_stops_accuracy_flow[dut]
tests/python/framework/test_check_accuracy.py::test_accuracy_configuration_is_restored[normal]
tests/python/framework/test_check_accuracy.py::test_accuracy_configuration_is_restored[exception]
tests/python/framework/test_dispatch.py::test_explicit_impl_calls_only_selected_backend[triton]
tests/python/framework/test_dispatch.py::test_explicit_impl_calls_only_selected_backend[torch_native]
tests/python/framework/test_dispatch.py::test_default_and_auto_select_triton[default]
tests/python/framework/test_dispatch.py::test_default_and_auto_select_triton[auto]
tests/python/framework/test_dispatch.py::test_optional_arguments_are_forwarded_without_impl[positional]
tests/python/framework/test_dispatch.py::test_optional_arguments_are_forwarded_without_impl[keyword]
tests/python/framework/test_dispatch.py::test_invalid_impl_does_not_call_backends[unknown]
tests/python/framework/test_dispatch.py::test_invalid_impl_does_not_call_backends[explicit-none]
tests/python/framework/test_dispatch.py::test_impl_is_keyword_only
tests/python/framework/test_dispatch.py::test_backend_error_propagates_without_fallback[triton]
tests/python/framework/test_dispatch.py::test_backend_error_propagates_without_fallback[torch_native]
tests/python/framework/test_dispatch.py::test_new_registered_backend_needs_no_dispatcher_or_directory_change
tests/python/framework/test_dispatch.py::test_unsupported_impl_is_rejected_by_public_entry
tests/python/framework/test_dispatch.py::test_cached_backends_do_not_cross_routes[triton-first]
tests/python/framework/test_dispatch.py::test_cached_backends_do_not_cross_routes[torch-native-first]
tests/python/framework/test_generation.py::test_known_arch_maps_to_expected_generation[2201-ARCH22]
tests/python/framework/test_generation.py::test_known_arch_maps_to_expected_generation[3510-ARCH35]
tests/python/framework/test_generation.py::test_unknown_arch_is_not_guessed[unknown-architecture]
tests/python/framework/test_generation.py::test_unknown_arch_is_not_guessed[none]
tests/python/framework/test_generation.py::test_unknown_arch_is_not_guessed[unhashable]
tests/python/framework/test_generation.py::test_real_npu_is_identified
tests/python/framework/test_generation.py::test_successful_detection_is_cached
tests/python/framework/test_generation.py::test_device_query_failure_can_retry_without_caching[exception]
tests/python/framework/test_generation.py::test_device_query_failure_can_retry_without_caching[return-code]
tests/python/framework/test_generation.py::test_unknown_arch_is_not_cached
tests/python/framework/test_generation_registry.py::test_support_is_classified_against_import_time_hardware[ARCH22]
tests/python/framework/test_generation_registry.py::test_support_is_classified_against_import_time_hardware[ARCH35]
tests/python/framework/test_generation_registry.py::test_support_and_exclusion_sets[support-none]
tests/python/framework/test_generation_registry.py::test_support_and_exclusion_sets[support-deduplicated]
tests/python/framework/test_generation_registry.py::test_support_and_exclusion_sets[exclude-none]
tests/python/framework/test_generation_registry.py::test_support_and_exclusion_sets[exclude-current]
tests/python/framework/test_generation_registry.py::test_support_and_exclusion_sets[exclude-other]
tests/python/framework/test_generation_registry.py::test_support_and_exclusion_sets[exclude-all]
tests/python/framework/test_generation_registry.py::test_support_options_are_mutually_exclusive
tests/python/framework/test_generation_registry.py::test_generation_strings_are_rejected_without_registration[supported_generations]
tests/python/framework/test_generation_registry.py::test_generation_strings_are_rejected_without_registration[excluded_generations]
tests/python/framework/test_generation_registry.py::test_disjoint_entries_select_current_architecture[forward-ARCH22]
tests/python/framework/test_generation_registry.py::test_disjoint_entries_select_current_architecture[forward-ARCH35]
tests/python/framework/test_generation_registry.py::test_disjoint_entries_select_current_architecture[reverse-ARCH22]
tests/python/framework/test_generation_registry.py::test_disjoint_entries_select_current_architecture[reverse-ARCH35]
tests/python/framework/test_generation_registry.py::test_equivalent_support_declarations_allow_aliases
tests/python/framework/test_generation_registry.py::test_aliases_have_independent_support_ranges[default-all]
tests/python/framework/test_generation_registry.py::test_aliases_have_independent_support_ranges[exclude-arch35]
tests/python/framework/test_generation_registry.py::test_shared_callable_support_is_scoped_to_operator
tests/python/framework/test_generation_registry.py::test_support_declaration_is_a_snapshot
tests/python/framework/test_generation_registry.py::test_overlapping_support_is_rejected_atomically[ARCH22]
tests/python/framework/test_generation_registry.py::test_overlapping_support_is_rejected_atomically[ARCH35]
tests/python/framework/test_generation_registry.py::test_empty_support_declaration_can_be_extended
tests/python/framework/test_imports.py::test_checked_out_public_api_uses_the_same_registry
tests/python/framework/test_imports.py::test_public_import_registers_without_importing_kernel_leaves
tests/python/framework/test_imports.py::test_hardware_detection_failure_aborts_registry_import
tests/python/framework/test_imports.py::test_private_registrations_do_not_leak_to_other_modules
tests/python/framework/test_imports.py::test_checkout_framework_does_not_require_installed_operators
tests/python/framework/test_imports.py::test_missing_backend_is_isolated_and_failed_import_can_retry[triton]
tests/python/framework/test_imports.py::test_missing_backend_is_isolated_and_failed_import_can_retry[torch_native]
tests/python/framework/test_performance.py::test_disabled_performance_has_no_side_effects
tests/python/framework/test_performance.py::test_performance_context_overrides_code_switch[False-True]
tests/python/framework/test_performance.py::test_performance_context_overrides_code_switch[True-False]
tests/python/framework/test_performance.py::test_invalid_effective_performance_flag_is_rejected
tests/python/framework/test_performance.py::test_enabled_performance_reports_measurement_and_metadata[True]
tests/python/framework/test_performance.py::test_enabled_performance_reports_measurement_and_metadata[False]
tests/python/framework/test_performance.py::test_enabled_performance_reports_measurement_and_metadata[missing]
tests/python/framework/test_performance.py::test_performance_paths_follow_context_precedence[code-paths]
tests/python/framework/test_performance.py::test_performance_paths_follow_context_precedence[context-paths]
tests/python/framework/test_performance.py::test_benchmark_failure_does_not_publish_success
tests/python/framework/test_performance.py::test_performance_context_is_restored[normal]
tests/python/framework/test_performance.py::test_performance_context_is_restored[exception]
tests/python/framework/test_performance_integration.py::test_real_npu_accuracy_and_performance_pipeline
tests/python/framework/test_profile_data.py::test_summary_files_are_summed_without_statistics_double_counting
tests/python/framework/test_profile_data.py::test_statistics_are_used_when_summary_is_absent
tests/python/framework/test_profile_data.py::test_missing_or_unusable_reports_are_rejected[missing]
tests/python/framework/test_profile_data.py::test_missing_or_unusable_reports_are_rejected[column]
tests/python/framework/test_profile_data.py::test_missing_or_unusable_reports_are_rejected[empty]
tests/python/framework/test_profile_data.py::test_missing_or_unusable_reports_are_rejected[zero]
tests/python/framework/test_profile_data.py::test_invalid_task_durations_are_rejected[invalid]
tests/python/framework/test_profile_data.py::test_invalid_task_durations_are_rejected[-1]
tests/python/framework/test_profile_data.py::test_invalid_task_durations_are_rejected[NaN]
tests/python/framework/test_profile_data.py::test_invalid_task_durations_are_rejected[Inf]
tests/python/framework/test_profile_data.py::test_result_csv_can_be_created_and_appended
tests/python/framework/test_profile_data.py::test_incompatible_csv_schema_preserves_existing_results
tests/python/framework/test_profile_data.py::test_profile_directory_flattening_preserves_files[flatten]
tests/python/framework/test_profile_data.py::test_profile_directory_flattening_preserves_files[preserve-conflict]
tests/python/framework/test_profile_data.py::test_profile_case_names_are_safe_bounded_and_unique
tests/python/framework/test_profile_data.py::test_profile_parser_requires_positive_integer_active[zero]
tests/python/framework/test_profile_data.py::test_profile_parser_requires_positive_integer_active[bool]
tests/python/framework/test_profile_data.py::test_profile_parser_requires_positive_integer_active[fraction]
tests/python/framework/test_profile_data.py::test_profile_parser_requires_positive_integer_active[nan]
tests/python/framework/test_profile_data.py::test_profile_parser_requires_positive_integer_active[inf]
tests/python/framework/test_registry.py::test_public_registration_preserves_function
tests/python/framework/test_registry.py::test_duplicate_public_function_is_rejected_without_changing_name
tests/python/framework/test_registry.py::test_duplicate_public_name_is_rejected_without_registering_function
tests/python/framework/test_registry.py::test_unregistered_function_is_rejected[get_op_name]
tests/python/framework/test_registry.py::test_unregistered_function_is_rejected[get_impl]
tests/python/framework/test_registry.py::test_unregistered_function_is_rejected[list_impls]
tests/python/framework/test_registry.py::test_impl_registration_and_lookup_do_not_execute_backend
tests/python/framework/test_registry.py::test_impl_can_be_registered_before_public_operator
tests/python/framework/test_registry.py::test_duplicate_impl_is_rejected_without_overwriting[False]
tests/python/framework/test_registry.py::test_duplicate_impl_is_rejected_without_overwriting[True]
tests/python/framework/test_registry.py::test_same_impl_name_is_scoped_to_operator
tests/python/framework/test_registry.py::test_aliases_preserve_callable_identity
tests/python/framework/test_registry.py::test_list_impls_is_sorted_and_excludes_other_operators
tests/python/framework/test_registry.py::test_registered_operator_without_impls_has_no_choices
tests/python/framework/test_registry.py::test_unknown_name_has_no_choices
tests/python/framework/test_registry.py::test_unknown_impl_reports_choices_without_fallback[missing]
tests/python/framework/test_registry.py::test_unknown_impl_reports_choices_without_fallback[TRITON]
tests/python/framework/test_registry.py::test_unknown_impl_reports_choices_without_fallback[]
tests/python/framework/test_registry.py::test_unknown_impl_reports_choices_without_fallback[None]
tests/python/framework/test_registry.py::test_public_exports_use_the_same_registry
tests/python/framework/test_standards.py::test_default_standards_match_dtype_policy[torch.float32-expected0]
tests/python/framework/test_standards.py::test_default_standards_match_dtype_policy[torch.float16-expected1]
tests/python/framework/test_standards.py::test_default_standards_match_dtype_policy[torch.bfloat16-expected2]
tests/python/framework/test_standards.py::test_default_standards_match_dtype_policy[torch.int8-expected3]
tests/python/framework/test_standards.py::test_default_standards_match_dtype_policy[torch.int32-expected4]
tests/python/framework/test_standards.py::test_default_standards_match_dtype_policy[torch.int64-expected5]
tests/python/framework/test_standards.py::test_override_changes_only_requested_dtype
tests/python/framework/test_standards.py::test_invalid_tolerances_are_rejected[negative-atol]
tests/python/framework/test_standards.py::test_invalid_tolerances_are_rejected[negative-rtol]
tests/python/framework/test_standards.py::test_invalid_tolerances_are_rejected[nan-atol]
tests/python/framework/test_standards.py::test_invalid_tolerances_are_rejected[nan-rtol]
tests/python/framework/test_standards.py::test_invalid_tolerances_are_rejected[positive-inf-atol]
tests/python/framework/test_standards.py::test_invalid_tolerances_are_rejected[positive-inf-rtol]
tests/python/framework/test_standards.py::test_invalid_tolerances_are_rejected[negative-inf-atol]
tests/python/framework/test_standards.py::test_invalid_tolerances_are_rejected[negative-inf-rtol]
tests/python/framework/test_standards.py::test_invalid_standard_mapping_is_rejected[not-a-mapping]
tests/python/framework/test_standards.py::test_invalid_standard_mapping_is_rejected[non-string-key]
tests/python/framework/test_standards.py::test_invalid_standard_mapping_is_rejected[invalid-value]
tests/python/framework/test_standards.py::test_invalid_standard_mapping_is_rejected[invalid-unused-entry]
tests/python/framework/test_standards.py::test_unknown_dtype_has_no_default_standard
```

</details>

## 4. 被测函数 × 参数 × 用例覆盖表

范围为本批UT显式验证的接口，以及经真实调用链有断言支撑的适配/辅助函数。fixture、测试专用backend和第三方库API不作为被测产品接口列入。
“直接”表示测试调用该接口并断言行为；“间接”表示经另一真实接口进入；“部分”明确标出当前没有专门断言的场景。这里不是代码覆盖率或全输入域覆盖声明。
无参数函数单列一行；self省略；dataclass/NamedTuple按用户可传字段列出；返回的注册装饰器也列出fn参数。

| 被测函数 | 参数 | 对应用例 | 已验证内容 | 方式 / 当前边界 |
| --- | --- | --- | --- | --- |
| [registry.register_operator](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/registry.py#L38) | `op_name` | [REG01](test_registry.py#L9)、[REG02](test_registry.py#L18)、[REG03](test_registry.py#L29) | 显式名不同于函数名、名称唯一性、重复失败后状态 | 直接；空字符串/非字符串名称未单独验证 |
| [registry.register_operator.decorator](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/registry.py#L41) | `fn` | [REG01](test_registry.py#L9)、[REG02](test_registry.py#L18)、[REG03](test_registry.py#L29) | 保留函数身份、不执行、同函数重复/同名新函数冲突 | 直接；非callable输入未验证 |
| [registry.get_op_name](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/registry.py#L55) | `operator` | [REG01](test_registry.py#L9)、[REG02](test_registry.py#L18)、[REG03](test_registry.py#L29)、[REG04](test_registry.py#L43) | 已注册函数精确取名，未注册函数失败 | 直接；不校验全部Python对象类型 |
| [registry.register_impl](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/registry.py#L62) | `op_name` | [REG05](test_registry.py#L54)、[REG06](test_registry.py#L65)、[REG08](test_registry.py#L89)、[GREG08](test_generation_registry.py#L176) | 先实现后入口、同impl跨算子隔离、同函数跨算子支持独立 | 直接；名称类型/空名称未单独验证 |
| [registry.register_impl](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/registry.py#L62) | `impl` | [REG07](test_registry.py#L75)、[REG08](test_registry.py#L89)、[REG09](test_registry.py#L99)、[GREG05](test_generation_registry.py#L104)、[GREG07](test_generation_registry.py#L147)、[GREG10](test_generation_registry.py#L205)、[GREG11](test_generation_registry.py#L237) | 别名、同名分架构、重叠拒绝、空声明扩展和枚举去重 | 直接；注册名称的类型/空名称未单独验证 |
| [registry.register_impl](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/registry.py#L62) | `supported_generations` | [GREG01](test_generation_registry.py#L10)、[GREG02](test_generation_registry.py#L68)、[GREG03](test_generation_registry.py#L85)、[GREG04](test_generation_registry.py#L95)、[GREG05](test_generation_registry.py#L104)、[GREG09](test_generation_registry.py#L190)、[GREG10](test_generation_registry.py#L205)、[GREG11](test_generation_registry.py#L237) | 默认全部、空、去重、枚举类型、互斥、快照、不重叠/重叠范围 | 直接；不做所有集合组合的全排列 |
| [registry.register_impl](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/registry.py#L62) | `excluded_generations` | [GREG02](test_generation_registry.py#L68)、[GREG03](test_generation_registry.py#L85)、[GREG04](test_generation_registry.py#L95)、[GREG06](test_generation_registry.py#L124)、[GREG07](test_generation_registry.py#L147) | 默认/空/排除当前或其他或全部、类型、互斥、别名独立 | 直接；部分；原排除列表事后修改的快照未单独验证 |
| [registry.register_impl.decorator](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/registry.py#L91) | `fn` | [REG05](test_registry.py#L54)、[REG07](test_registry.py#L75)、[REG09](test_registry.py#L99)、[GREG05](test_generation_registry.py#L104)、[GREG07](test_generation_registry.py#L147)、[GREG08](test_generation_registry.py#L176)、[GREG10](test_generation_registry.py#L205) | 保留函数、不执行、同/不同函数重复、别名共享和多架构不同函数 | 直接；非callable输入未验证 |
| [registry.list_impls_by_name](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/registry.py#L110) | `op_name` | [REG10](test_registry.py#L110)、[REG12](test_registry.py#L137)、[GREG05](test_generation_registry.py#L104)、[GREG11](test_generation_registry.py#L237) | 已知/未知算子、排序、不同算子隔离、同名多声明只枚举一次 | 直接；名称类型边界未验证 |
| [registry.get_impl_by_name](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/registry.py#L122) | `op_name` | [REG05](test_registry.py#L54)、[REG08](test_registry.py#L89)、[REG12](test_registry.py#L137)、[GREG08](test_generation_registry.py#L176) | 已知/未知、按算子独立查询 | 直接；非法名称类型未验证 |
| [registry.get_impl_by_name](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/registry.py#L122) | `impl` | [REG05](test_registry.py#L54)、[GREG01](test_generation_registry.py#L10)、[GREG05](test_generation_registry.py#L104)、[GREG07](test_generation_registry.py#L147)、[GREG10](test_generation_registry.py#L205)、[GREG11](test_generation_registry.py#L237) | 当前架构选择、支持/不支持、别名、重叠失败后状态 | 直接+间接；缺失/大小写/空/None主要经get_impl转调覆盖REG13 |
| [registry.get_impl](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/registry.py#L140) | `operator` | [REG04](test_registry.py#L43)、[REG05](test_registry.py#L54)、[REG14](test_registry.py#L162) | 已注册/未注册、公共导出与同注册表 | 直接；无 |
| [registry.get_impl](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/registry.py#L140) | `impl` | [REG09](test_registry.py#L99)、[REG13](test_registry.py#L145)、[GREG01](test_generation_registry.py#L10)、[CHK09](test_check_accuracy.py#L216)、[CHK10](test_check_accuracy.py#L259) | 正常、别名、未知、大小写、空、None、不支持架构 | 直接+间接；无额外自动fallback约定 |
| [registry.list_impls](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/registry.py#L145) | `operator` | [REG04](test_registry.py#L43)、[REG09](test_registry.py#L99)、[REG10](test_registry.py#L110)、[REG11](test_registry.py#L127)、[REG14](test_registry.py#L162) | 未注册、别名、排序、隔离、修改返回列表、无实现 | 直接；无 |
| [registry.UnsupportedGenerationError.__init__](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/registry.py#L15) | `op_name` | [GREG01](test_generation_registry.py#L10) | 错误字段和消息包含算子名 | 间接；不是独立构造器UT |
| [registry.UnsupportedGenerationError.__init__](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/registry.py#L15) | `impl` | [GREG01](test_generation_registry.py#L10)、[CHK10](test_check_accuracy.py#L259) | 错误字段/消息和错误来源 | 间接；不是独立构造器UT |
| [registry.UnsupportedGenerationError.__init__](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/registry.py#L15) | `hardware` | [GREG01](test_generation_registry.py#L10) | 保留Hardware及当前架构信息 | 间接；不再要求SoC名出现在消息 |
| [registry.UnsupportedGenerationError.__init__](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/registry.py#L15) | `supported_generations` | [GREG01](test_generation_registry.py#L10)、[GREG10](test_generation_registry.py#L205)、[GREG11](test_generation_registry.py#L237) | 支持集字段含实际已声明范围，包括空范围 | 间接；不单测排序展示的所有组合 |
| [generation.generation_from_arch](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/generation.py#L29) | `arch_id` | [GEN01](test_generation.py#L10)、[GEN02](test_generation.py#L21)、[GEN06](test_generation.py#L79) | 2201/3510、未知-1、None和不可哈希ID | 直接+间接；未把所有Python数字类型/可转换字符串展开 |
| [generation.get_hardware](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/generation.py#L41) | `—` | [GEN03](test_generation.py#L28)、[GEN04](test_generation.py#L37)、[GEN05](test_generation.py#L52)、[GEN06](test_generation.py#L79) | 无形参；外部ACL查询(0,601)、成功缓存、异常cause/错误码和失败重试 | 直接；NPU实测待执行；IMP03替代检测函数，只验证registry导入者的异常处理，不计作此函数覆盖 |
| [generation.Hardware](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/generation.py#L24) | `arch_id` | [GEN03](test_generation.py#L28)、[GEN04](test_generation.py#L37)、[GEN05](test_generation.py#L52) | 查询结果在Hardware.arch_id中保留 | 间接；构造器本身不做类型校验，未单测 |
| [generation.Hardware](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/generation.py#L24) | `generation` | [GEN01](test_generation.py#L10)、[GEN03](test_generation.py#L28)、[GEN04](test_generation.py#L37)、[GEN05](test_generation.py#L52)、[GREG01](test_generation_registry.py#L10) | 架构映射和注册分类使用正确枚举 | 间接；NamedTuple生成器不作为独立被测逻辑 |
| [rmsnorm_bias.rmsnorm_bias](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/norm/rmsnorm_bias.py#L36) | `input` | [DIS01](test_dispatch.py#L16)、[DIS03](test_dispatch.py#L42)、[DIS09](test_dispatch.py#L149) | 位置/关键字透传，反复切换时保持每次输入身份 | 直接；Tensor语义和布局不在分发UT中验证 |
| [rmsnorm_bias.rmsnorm_bias](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/norm/rmsnorm_bias.py#L36) | `norm_weight` | [DIS01](test_dispatch.py#L16)、[DIS03](test_dispatch.py#L42)、[DIS09](test_dispatch.py#L149) | 权重位置/关键字透传及对象身份 | 直接；数值/shape合法性属于算子测试 |
| [rmsnorm_bias.rmsnorm_bias](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/norm/rmsnorm_bias.py#L36) | `norm_bias` | [DIS01](test_dispatch.py#L16)、[DIS03](test_dispatch.py#L42)、[DIS09](test_dispatch.py#L149) | 偏置位置/关键字透传及对象身份 | 直接；数值/shape合法性属于算子测试 |
| [rmsnorm_bias.rmsnorm_bias](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/norm/rmsnorm_bias.py#L36) | `eps` | [DIS01](test_dispatch.py#L16)、[DIS03](test_dispatch.py#L42)、[DIS09](test_dispatch.py#L149) | 不同eps值正确透传 | 直接；未测生产算子的eps数值边界 |
| [rmsnorm_bias.rmsnorm_bias](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/norm/rmsnorm_bias.py#L36) | `quant_scale` | [DIS01](test_dispatch.py#L16)、[DIS03](test_dispatch.py#L42)、[DIS09](test_dispatch.py#L149) | 默认None和显式对象，位置/关键字方式 | 直接；量化数值与Tensor转换不在本层验证 |
| [rmsnorm_bias.rmsnorm_bias](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/norm/rmsnorm_bias.py#L36) | `quant_offset` | [DIS01](test_dispatch.py#L16)、[DIS03](test_dispatch.py#L42)、[DIS09](test_dispatch.py#L149) | 默认None和显式对象，位置/关键字方式 | 直接；量化数值与Tensor转换不在本层验证 |
| [rmsnorm_bias.rmsnorm_bias](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/norm/rmsnorm_bias.py#L36) | `impl` | [DIS01](test_dispatch.py#L16)、[DIS02](test_dispatch.py#L30)、[DIS04](test_dispatch.py#L72)、[DIS05](test_dispatch.py#L80)、[DIS06](test_dispatch.py#L91)、[DIS07](test_dispatch.py#L104)、[DIS08](test_dispatch.py#L123)、[DIS09](test_dispatch.py#L149) | 默认/auto/显式/未知/None、keyword-only、异常不回退、扩展和缓存路由 | 直接；不是所有生产backend集成测试 |
| [rmsnorm_bias._triton_impl](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/norm/rmsnorm_bias.py#L12) | `input` | [DIS01](test_dispatch.py#L16)、[DIS06](test_dispatch.py#L91)、[DIS09](test_dispatch.py#L149)、[IMP06](test_imports.py#L100) | 经真实适配函数透传到叶子，检查对象/值，包含默认None、缓存和失败重试 | 间接；kernel叶子是替身；不是Triton/torch_native生产数值验证 |
| [rmsnorm_bias._triton_impl](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/norm/rmsnorm_bias.py#L12) | `norm_weight` | [DIS01](test_dispatch.py#L16)、[DIS06](test_dispatch.py#L91)、[DIS09](test_dispatch.py#L149)、[IMP06](test_imports.py#L100) | 经真实适配函数透传到叶子，检查对象/值，包含默认None、缓存和失败重试 | 间接；kernel叶子是替身；不是Triton/torch_native生产数值验证 |
| [rmsnorm_bias._triton_impl](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/norm/rmsnorm_bias.py#L12) | `norm_bias` | [DIS01](test_dispatch.py#L16)、[DIS06](test_dispatch.py#L91)、[DIS09](test_dispatch.py#L149)、[IMP06](test_imports.py#L100) | 经真实适配函数透传到叶子，检查对象/值，包含默认None、缓存和失败重试 | 间接；kernel叶子是替身；不是Triton/torch_native生产数值验证 |
| [rmsnorm_bias._triton_impl](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/norm/rmsnorm_bias.py#L12) | `eps` | [DIS01](test_dispatch.py#L16)、[DIS06](test_dispatch.py#L91)、[DIS09](test_dispatch.py#L149)、[IMP06](test_imports.py#L100) | 经真实适配函数透传到叶子，检查对象/值，包含默认None、缓存和失败重试 | 间接；kernel叶子是替身；不是Triton/torch_native生产数值验证 |
| [rmsnorm_bias._triton_impl](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/norm/rmsnorm_bias.py#L12) | `quant_scale` | [DIS01](test_dispatch.py#L16)、[DIS06](test_dispatch.py#L91)、[DIS09](test_dispatch.py#L149)、[IMP06](test_imports.py#L100) | 经真实适配函数透传到叶子，检查对象/值，包含默认None、缓存和失败重试 | 间接；kernel叶子是替身；不是Triton/torch_native生产数值验证 |
| [rmsnorm_bias._triton_impl](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/norm/rmsnorm_bias.py#L12) | `quant_offset` | [DIS01](test_dispatch.py#L16)、[DIS06](test_dispatch.py#L91)、[DIS09](test_dispatch.py#L149)、[IMP06](test_imports.py#L100) | 经真实适配函数透传到叶子，检查对象/值，包含默认None、缓存和失败重试 | 间接；kernel叶子是替身；不是Triton/torch_native生产数值验证 |
| [rmsnorm_bias._torch_native_impl](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/norm/rmsnorm_bias.py#L23) | `input` | [DIS01](test_dispatch.py#L16)、[DIS06](test_dispatch.py#L91)、[DIS09](test_dispatch.py#L149)、[IMP06](test_imports.py#L100)、[DIS03](test_dispatch.py#L42) | 经真实适配函数透传到叶子，检查对象/值，包含默认None、缓存和失败重试 | 间接；kernel叶子是替身；不是Triton/torch_native生产数值验证 |
| [rmsnorm_bias._torch_native_impl](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/norm/rmsnorm_bias.py#L23) | `norm_weight` | [DIS01](test_dispatch.py#L16)、[DIS06](test_dispatch.py#L91)、[DIS09](test_dispatch.py#L149)、[IMP06](test_imports.py#L100)、[DIS03](test_dispatch.py#L42) | 经真实适配函数透传到叶子，检查对象/值，包含默认None、缓存和失败重试 | 间接；kernel叶子是替身；不是Triton/torch_native生产数值验证 |
| [rmsnorm_bias._torch_native_impl](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/norm/rmsnorm_bias.py#L23) | `norm_bias` | [DIS01](test_dispatch.py#L16)、[DIS06](test_dispatch.py#L91)、[DIS09](test_dispatch.py#L149)、[IMP06](test_imports.py#L100)、[DIS03](test_dispatch.py#L42) | 经真实适配函数透传到叶子，检查对象/值，包含默认None、缓存和失败重试 | 间接；kernel叶子是替身；不是Triton/torch_native生产数值验证 |
| [rmsnorm_bias._torch_native_impl](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/norm/rmsnorm_bias.py#L23) | `eps` | [DIS01](test_dispatch.py#L16)、[DIS06](test_dispatch.py#L91)、[DIS09](test_dispatch.py#L149)、[IMP06](test_imports.py#L100)、[DIS03](test_dispatch.py#L42) | 经真实适配函数透传到叶子，检查对象/值，包含默认None、缓存和失败重试 | 间接；kernel叶子是替身；不是Triton/torch_native生产数值验证 |
| [rmsnorm_bias._torch_native_impl](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/norm/rmsnorm_bias.py#L23) | `quant_scale` | [DIS01](test_dispatch.py#L16)、[DIS06](test_dispatch.py#L91)、[DIS09](test_dispatch.py#L149)、[IMP06](test_imports.py#L100)、[DIS03](test_dispatch.py#L42) | 经真实适配函数透传到叶子，检查对象/值，包含默认None、缓存和失败重试 | 间接；kernel叶子是替身；不是Triton/torch_native生产数值验证 |
| [rmsnorm_bias._torch_native_impl](../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/norm/rmsnorm_bias.py#L23) | `quant_offset` | [DIS01](test_dispatch.py#L16)、[DIS06](test_dispatch.py#L91)、[DIS09](test_dispatch.py#L149)、[IMP06](test_imports.py#L100)、[DIS03](test_dispatch.py#L42) | 经真实适配函数透传到叶子，检查对象/值，包含默认None、缓存和失败重试 | 间接；kernel叶子是替身；不是Triton/torch_native生产数值验证 |
| [standards.AccuracyStandard](../../../tests/python/utils/standards.py#L9) | `atol` | [STD01](test_standards.py#L20)、[STD02](test_standards.py#L26)、[STD03](test_standards.py#L45)、[ACC01](test_accuracy.py#L33) | 0/正值、负数、NaN、正负Inf；真实比较时绝对阈值边界 | 直接；非数字值及bool类型策略未单独验证 |
| [standards.AccuracyStandard](../../../tests/python/utils/standards.py#L9) | `rtol` | [STD01](test_standards.py#L20)、[STD02](test_standards.py#L26)、[STD03](test_standards.py#L45)、[ACC01](test_accuracy.py#L33) | 0/正值、负数、NaN、正负Inf；真实比较时相对阈值边界 | 直接；非数字值及bool类型策略未单独验证 |
| [standards.get_standard](../../../tests/python/utils/standards.py#L32) | `dtype` | [STD01](test_standards.py#L20)、[STD02](test_standards.py#L26)、[STD05](test_standards.py#L71) | 六种默认字符串键、未覆盖dtype回退、测试专用未知键 | 直接；传torch.dtype对象而非字符串的行为未单独验证 |
| [standards.get_standard](../../../tests/python/utils/standards.py#L32) | `standard` | [STD01](test_standards.py#L20)、[STD02](test_standards.py#L26)、[STD04](test_standards.py#L65)、[STD05](test_standards.py#L71) | None、自定义映射、局部覆盖、全配置校验、非法键/值/非映射 | 直接；更多Mapping子类不逐个展开 |
| [accuracy.assert_accuracy](../../../tests/python/utils/accuracy.py#L16) | `result` | [ACC01](test_accuracy.py#L33)、[ACC02](test_accuracy.py#L48)、[ACC03](test_accuracy.py#L57)、[ACC04](test_accuracy.py#L76)、[ACC05](test_accuracy.py#L103)、[ACC06](test_accuracy.py#L132)、[ACC07](test_accuracy.py#L146)、[ACC08](test_accuracy.py#L159)、[ACC09](test_accuracy.py#L174) | 真实Tensor、容器、标量、类型与数值错误、NaN、不支持对象 | 直接；complex Tensor未测；Python complex已测 |
| [accuracy.assert_accuracy](../../../tests/python/utils/accuracy.py#L16) | `result_ref` | [ACC01](test_accuracy.py#L33)、[ACC02](test_accuracy.py#L48)、[ACC03](test_accuracy.py#L57)、[ACC04](test_accuracy.py#L76)、[ACC05](test_accuracy.py#L103)、[ACC06](test_accuracy.py#L132)、[ACC07](test_accuracy.py#L146) | 数值误差以参考为基准；结构/元信息/类型必须一致 | 直接；非对称错误不做交换两侧的全排列 |
| [accuracy.assert_accuracy](../../../tests/python/utils/accuracy.py#L16) | `standard` | [ACC01](test_accuracy.py#L33)、[ACC02](test_accuracy.py#L48)、[ACC06](test_accuracy.py#L132)、[ACC08](test_accuracy.py#L159)、[CHK03](test_check_accuracy.py#L59) | 默认、按dtype覆盖、零/绝对/相对容差、浮点缺标准 | 直接+间接；整数Tensor自定义放宽容差未测 |
| [accuracy.assert_accuracy](../../../tests/python/utils/accuracy.py#L16) | `path` | [ACC04](test_accuracy.py#L76) | 默认output及递归路径output['aux'][1][1] | 间接；部分；显式自定义入口path值未直接传入测试 |
| [accuracy.check_accuracy](../../../tests/python/utils/accuracy.py#L79) | `operator` | [CHK01](test_check_accuracy.py#L20)、[CHK02](test_check_accuracy.py#L43)、[CHK09](test_check_accuracy.py#L216)、[CHK10](test_check_accuracy.py#L259)、[CHK11](test_check_accuracy.py#L296)、[INT01](test_performance_integration.py#L14) | 真实注册入口、结果差异、同入口、选择失败和实现异常 | 直接；未单独把未注册公共函数传入此入口 |
| [accuracy.check_accuracy](../../../tests/python/utils/accuracy.py#L79) | `args` | [CHK01](test_check_accuracy.py#L20)、[CHK02](test_check_accuracy.py#L43)、[CHK06](test_check_accuracy.py#L151) | 非空tuple、位置参数和Tensor身份、executor再次执行 | 直接；部分；默认空args和list等其他iterable未专门验证 |
| [accuracy.check_accuracy](../../../tests/python/utils/accuracy.py#L79) | `kwargs` | [CHK01](test_check_accuracy.py#L20)、[CHK05](test_check_accuracy.py#L135) | 默认None、scale关键字、原dict不变、禁止impl键 | 直接；非映射参数等格式错误未单独验证 |
| [accuracy.check_accuracy](../../../tests/python/utils/accuracy.py#L79) | `impl` | [CHK04](test_check_accuracy.py#L119)、[CHK09](test_check_accuracy.py#L216)、[CHK10](test_check_accuracy.py#L259)、[CHK11](test_check_accuracy.py#L296) | 省略/None/显式覆盖、同入口、未知、不支持和执行异常 | 直接；无全名称笛卡尔积 |
| [accuracy.check_accuracy](../../../tests/python/utils/accuracy.py#L79) | `impl_ref` | [CHK04](test_check_accuracy.py#L119)、[CHK06](test_check_accuracy.py#L151)、[CHK09](test_check_accuracy.py#L216)、[CHK10](test_check_accuracy.py#L259)、[CHK11](test_check_accuracy.py#L296) | 省略/None/覆盖、参考有效性和错误优先级、跳过时不依赖参考 | 直接；无全名称笛卡尔积 |
| [accuracy.check_accuracy](../../../tests/python/utils/accuracy.py#L79) | `standard` | [CHK02](test_check_accuracy.py#L43)、[CHK03](test_check_accuracy.py#L59) | 零容差、自定义阈值使默认失败的结果通过，实际比较不替换 | 直接；配置格式由STD组专门验证 |
| [accuracy.check_accuracy](../../../tests/python/utils/accuracy.py#L79) | `skip_acc_check` | [CHK06](test_check_accuracy.py#L151)、[CHK07](test_check_accuracy.py#L183)、[CHK08](test_check_accuracy.py#L199)、[CHK10](test_check_accuracy.py#L259) | 默认/True/False、上下文None或bool覆盖、非bool拒绝、只查DUT | 直接；命令行parser不属于本组 |
| [common.get_test_impls](../../../tests/python/utils/common.py#L15) | `—` | [CHK04](test_check_accuracy.py#L119)、[CHK12](test_check_accuracy.py#L309) | 无形参；读取嵌套作用域的实现对并恢复 | 直接+间接；无 |
| [common.get_skip_acc_check](../../../tests/python/utils/common.py#L10) | `—` | [CHK07](test_check_accuracy.py#L183)、[CHK12](test_check_accuracy.py#L309) | 无形参；读取None/True/False及嵌套恢复 | 直接+间接；无 |
| [common.use_impls](../../../tests/python/utils/common.py#L21) | `impl` | [CHK04](test_check_accuracy.py#L119)、[CHK12](test_check_accuracy.py#L309) | 待测默认名及内外层独立作用域 | 直接；无 |
| [common.use_impls](../../../tests/python/utils/common.py#L21) | `impl_ref` | [CHK04](test_check_accuracy.py#L119)、[CHK12](test_check_accuracy.py#L309) | 参考默认名及内外层独立作用域 | 直接；无 |
| [common.use_impls](../../../tests/python/utils/common.py#L21) | `skip_acc_check` | [CHK07](test_check_accuracy.py#L183)、[CHK12](test_check_accuracy.py#L309) | None/True/False及正常/异常恢复 | 直接；上下文直接传非法skip值未单独测，入口有效值校验见CHK08 |
| [common.require_distinct_impls](../../../tests/python/utils/common.py#L32) | `operator` | [CHK01](test_check_accuracy.py#L20)、[CHK09](test_check_accuracy.py#L216)、[CHK10](test_check_accuracy.py#L259) | 注册算子查询真实registry，比较前校验 | 间接；未注册operator由REG04验证查询接口，未直接验证此helper |
| [common.require_distinct_impls](../../../tests/python/utils/common.py#L32) | `impl` | [CHK09](test_check_accuracy.py#L216)、[CHK10](test_check_accuracy.py#L259) | 相同名/同函数别名、未知、不支持架构 | 间接；仅选择有意义的错误组合 |
| [common.require_distinct_impls](../../../tests/python/utils/common.py#L32) | `impl_ref` | [CHK09](test_check_accuracy.py#L216)、[CHK10](test_check_accuracy.py#L259) | 同入口拒绝；缺失/不支持错误优先于DUT不支持skip | 间接；仅选择有意义的错误组合 |
| [common.require_test_impl](../../../tests/python/utils/common.py#L57) | `operator` | [CHK06](test_check_accuracy.py#L151)、[CHK10](test_check_accuracy.py#L259) | 跳过精度时真实查询注册算子 | 间接；未注册operator未直接验证此helper |
| [common.require_test_impl](../../../tests/python/utils/common.py#L57) | `impl` | [CHK06](test_check_accuracy.py#L151)、[CHK10](test_check_accuracy.py#L259) | 可用则返回executor、未知失败、不支持skip，不查询参考 | 间接；无 |
| [performance.performance_context](../../../tests/python/utils/performance.py#L20) | `test_case` | [PERF04](test_performance.py#L53)、[PERF07](test_performance.py#L159)、[INT01](test_performance_integration.py#L14) | 标签进入返回值和回调，嵌套恢复 | 直接；空标签默认路径无专门显示断言 |
| [performance.performance_context](../../../tests/python/utils/performance.py#L20) | `parameters` | [PERF04](test_performance.py#L53)、[PERF07](test_performance.py#L159)、[INT01](test_performance_integration.py#L14) | shape/dtype等元数据JSON化、内外层恢复 | 直接；部分；传入后再修改原dict的快照行为未专门测 |
| [performance.performance_context](../../../tests/python/utils/performance.py#L20) | `record_property` | [PERF04](test_performance.py#L53)、[PERF05](test_performance.py#L92)、[PERF06](test_performance.py#L123)、[PERF07](test_performance.py#L159)、[INT01](test_performance_integration.py#L14) | None、回调完整性/次数、失败不记录、回调作用域恢复 | 直接；使用普通列表回调；未验证pytest JUnit属性实际落盘 |
| [performance.performance_context](../../../tests/python/utils/performance.py#L20) | `enable_perf` | [PERF01](test_performance.py#L12)、[PERF02](test_performance.py#L27)、[PERF07](test_performance.py#L159) | None/True/False、覆盖代码开关、嵌套恢复 | 直接；CLI parser未测 |
| [performance.performance_context](../../../tests/python/utils/performance.py#L20) | `prof_dir` | [PERF05](test_performance.py#L92)、[PERF07](test_performance.py#L159) | None回退、上下文覆盖路径和恢复 | 直接；不穷举路径类型 |
| [performance.performance_context](../../../tests/python/utils/performance.py#L20) | `csv_path` | [PERF05](test_performance.py#L92)、[PERF07](test_performance.py#L159) | None回退、上下文覆盖及实际写入/恢复 | 直接；不穷举路径类型 |
| [performance._require_positive_integer](../../../tests/python/utils/performance.py#L40) | `value` | [DATA09](test_profile_data.py#L170)、[BENCH03](test_benchmark_npu.py#L114) | 正int通过，0/bool/小数/NaN/Inf拒绝 | 间接；负整数、所有int子类未分别展开；通过公开入口测试 |
| [performance._require_positive_integer](../../../tests/python/utils/performance.py#L40) | `name` | [DATA09](test_profile_data.py#L170)、[BENCH03](test_benchmark_npu.py#L114) | active/warmup出现在对应错误信息中 | 间接；不是开放给用户的任意字段名接口 |
| [performance.read_device_time_us](../../../tests/python/utils/performance.py#L47) | `directory` | [DATA01](test_profile_data.py#L26)、[DATA02](test_profile_data.py#L34)、[DATA03](test_profile_data.py#L41)、[DATA04](test_profile_data.py#L61)、[BENCH01](test_benchmark_npu.py#L12)、[INT01](test_performance_integration.py#L14) | 递归多文件、summary优先/statistic回退、缺文件/列/空/零/非法值 | 直接+间接；BOM等更多文件变体未专门测试；实际CANN格式需INT01验证 |
| [performance.read_device_time_us](../../../tests/python/utils/performance.py#L47) | `active` | [DATA01](test_profile_data.py#L26)、[DATA02](test_profile_data.py#L34)、[DATA09](test_profile_data.py#L170)、[BENCH01](test_benchmark_npu.py#L12) | 2/3等正整数作分母，0/bool/小数/NaN/Inf拒绝 | 直接；不重复warmup参数组合 |
| [performance.flatten_profile_directory](../../../tests/python/utils/performance.py#L76) | `directory` | [DATA07](test_profile_data.py#L114)、[BENCH01](test_benchmark_npu.py#L12) | 单host目录完整上移、冲突时内容和结构不变 | 直接+间接；部分；零host或多个host的no-op分支未单独覆盖 |
| [performance.profile_case_name](../../../tests/python/utils/performance.py#L95) | `parameters` | [DATA08](test_profile_data.py#L143)、[PERF04](test_performance.py#L53) | shape/dtype/特殊字符/长字符串；名称字符、长度和连续生成唯一性 | 直接+间接；部分；impl_pair排除、空参数默认显示、排序展示未专门断言 |
| [performance.profile_case_name](../../../tests/python/utils/performance.py#L95) | `impl` | [DATA08](test_profile_data.py#L143)、[PERF04](test_performance.py#L53) | 普通及含分隔符的长实现名参与安全唯一目录名 | 直接+间接；固定UUID/计数器值不写死 |
| [performance.profile_case_name.safe](../../../tests/python/utils/performance.py#L98) | `value` | [DATA08](test_profile_data.py#L143)、[PERF04](test_performance.py#L53) | tuple、dtype、字符串、特殊字符与长值 | 间接；部分；内部函数经外层间接覆盖；list变体未单独测试 |
| [performance.benchmark_npu](../../../tests/python/utils/performance.py#L117) | `executor` | [BENCH01](test_benchmark_npu.py#L12)、[BENCH02](test_benchmark_npu.py#L91)、[BENCH03](test_benchmark_npu.py#L114)、[INT01](test_performance_integration.py#L14) | 编排调用次序/次数、禁止副作用、真实NPU轻量运算 | 直接；采集中executor抛错的profiler退出清理未专门UT；NPU集成待实测 |
| [performance.benchmark_npu](../../../tests/python/utils/performance.py#L117) | `directory` | [BENCH01](test_benchmark_npu.py#L12)、[BENCH02](test_benchmark_npu.py#L91)、[BENCH03](test_benchmark_npu.py#L114)、[INT01](test_performance_integration.py#L14) | 创建、已有目录拒绝、参数错不创建、产物整理/解析 | 直接；权限/磁盘错误未注入 |
| [performance.benchmark_npu](../../../tests/python/utils/performance.py#L117) | `warmup` | [BENCH01](test_benchmark_npu.py#L12)、[BENCH03](test_benchmark_npu.py#L114)、[INT01](test_performance_integration.py#L14) | 2/1等正整数及0，正确传给schedule/影响循环次数 | 直接；非法类型分类由同一校验器DATA09覆盖，未对两个字段做全排列 |
| [performance.benchmark_npu](../../../tests/python/utils/performance.py#L117) | `active` | [BENCH01](test_benchmark_npu.py#L12)、[BENCH03](test_benchmark_npu.py#L114)、[INT01](test_performance_integration.py#L14) | 3/2、0、小数，影响schedule/循环/解析分母 | 直接+间接；bool/NaN/Inf由共享校验器DATA09覆盖；实际采样窗口待NPU验证 |
| [performance.write_result](../../../tests/python/utils/performance.py#L159) | `path` | [DATA05](test_profile_data.py#L68)、[DATA06](test_profile_data.py#L99)、[PERF05](test_performance.py#L92) | 创建父目录、新建/追加文件、不兼容schema时原文件不变 | 直接+间接；不验证所有文件系统故障 |
| [performance.write_result](../../../tests/python/utils/performance.py#L159) | `result` | [DATA05](test_profile_data.py#L68)、[DATA06](test_profile_data.py#L99)、[PERF05](test_performance.py#L92)、[INT01](test_performance_integration.py#L14) | 多字段、JSON/引号/逗号、追加、schema不兼容、耗时三位小数 | 直接+间接；部分；缺少average_device_task_us字段的写入分支未单测 |
| [performance.check_performance](../../../tests/python/utils/performance.py#L177) | `executor` | [PERF01](test_performance.py#L12)、[PERF04](test_performance.py#L53)、[PERF06](test_performance.py#L123)、[INT01](test_performance_integration.py#L14) | 绑定函数/实现元数据、精度状态True/False/缺失、采集失败、真实执行 | 直接；不完整executor的func/impl等缺失情况未测试 |
| [performance.check_performance](../../../tests/python/utils/performance.py#L177) | `csv_path` | [PERF01](test_performance.py#L12)、[PERF04](test_performance.py#L53)、[PERF05](test_performance.py#L92)、[PERF06](test_performance.py#L123) | 默认None不写、显式路径、上下文覆盖、失败不追加 | 直接；无 |
| [performance.check_performance](../../../tests/python/utils/performance.py#L177) | `enable_perf` | [PERF01](test_performance.py#L12)、[PERF02](test_performance.py#L27)、[PERF03](test_performance.py#L43)、[PERF07](test_performance.py#L159) | 默认None关闭、True/False、上下文覆盖、整数非法、恢复 | 直接；CLI字符串解析未测试 |
| [performance.check_performance](../../../tests/python/utils/performance.py#L177) | `prof_dir` | [PERF04](test_performance.py#L53)、[PERF05](test_performance.py#L92)、[PERF07](test_performance.py#L159) | 默认cwd/profilings、显式Path、上下文覆盖和恢复 | 直接；部分；波浪号expanduser及其他字符串路径未专门验证 |
| [performance.check_performance](../../../tests/python/utils/performance.py#L177) | `warmup` | [PERF04](test_performance.py#L53)、[BENCH03](test_benchmark_npu.py#L114)、[INT01](test_performance_integration.py#L14) | 自定义2透传有断言，真实集成使用1；错误校验由真实benchmark完成 | 直接+间接；部分；默认5在开启用例中执行过但未专门断言；PERF测量替身不验证正整数 |
| [performance.check_performance](../../../tests/python/utils/performance.py#L177) | `active` | [PERF04](test_performance.py#L53)、[BENCH03](test_benchmark_npu.py#L114)、[INT01](test_performance_integration.py#L14) | 自定义3透传有断言，真实集成使用2；错误校验由真实benchmark完成 | 直接+间接；部分；默认5在开启用例中执行过但未专门断言；不证明真实active窗口计量正确 |

### 尚无专门UT的接口（不能记作已覆盖）

| 接口 | 参数 | 当前状态 |
| --- | --- | --- |
| device.validate_npu_options | index、workers | 未有专门UT；npu_runtime自己的检查不等价于调用这个函数。 |
| device.use_npu_device | torch、index | 未有专门成功/异常设备与default device恢复UT。 |
| 上层conftest的CLI解析及pytest配置 | value、path、parser、metafunc等 | 实际收集会执行部分hook，但未有pytester/子进程级行为断言，不计作接口UT覆盖。 |

显式自定义assert_accuracy.path、更多目录cardinality分支、真实pytest报告落盘等缺口已在参数表逐项注明。
Triton/TileLang/AscendC生产kernel数值与性能、原地输入clone/reset、历史基线/回退阈值均不是这批框架UT已完成的能力。
