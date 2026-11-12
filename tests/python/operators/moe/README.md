# MegaMoE public-interface validation

## Scope and independent-reference exception

`auto` and `cann_ops_transformer` select one external implementation, not two independent
implementations. MegaMoE owns collective communication through a provider-specific
symmetric buffer. A replacement `torch_native` implementation cannot consume that
opaque buffer without defining another resource protocol. This contribution uses
the exception in the contribution guide: independent correctness references live
in the distributed tests instead of exposing a misleading interchangeable backend.
The buffer factory itself is a resource API, not a numerical operator.

- A3 BF16: an independent CPU FP32 dense expert FFN reference, with the exact same
  BF16 input/weights; no CANN or NPU computation is used in the reference. Check
  output shape/dtype, finiteness, pointwise error, relative RMS and exact local
  expert token counts. Cases cover one token/topk=1, masked tails, zero input,
  skewed routing with empty experts, and DeepSeek-V3 H=7168/I=2048 dimensions
  (8 experts/topk=4 to bound reference memory, not a full model configuration).
- A3 eight-rank BF16: run one HCCL group across eight NPUs with 16 experts and
  top-k 8. Every active token selects one expert on each rank. Validate the
  output against the CPU FP32 reference and check each rank's exact expert
  counts, including exclusion of the padded tokens.
- A3 W4A8/W8A8: compare the registered interface with direct CANN calls for
  SwiGLU and SiTU, with padded inputs and an active-token mask.
- A5: compare MXFP8 E4M3/E5M2, MXFP4 and MXFP8-W4 with direct CANN calls.
  Cover rank-varying token counts, per-expert and stacked weight layouts,
  supported storage formats, and BF16/FP32 routing weights.
- Both quantized paths check local expert token counts against an independent
  routing histogram. Output comparisons establish interface equivalence with
  CANN.
- A5 four-rank A8W4-FP accuracy uses a separate CPU reference. It decodes packed
  E2M1 weights and per-block E8M0 scales, simulates dispatch and post-activation
  E4M3 quantization, computes both expert projections and SiTU, and combines
  expert outputs with the routing weights. It calls the registered operator
  once using an eight-expert, top-k 4, all-rank routing case with rank-varying
  token counts. No direct MegaMoE call supplies the expected output.

The independent A5 check requires finite BF16 output, exact expert counts,
relative RMS error below 3.5%, and pointwise closeness with rtol 0.1 and
atol 5% of reference RMS. These initial thresholds allow FP8 rounding boundary
and activation approximation differences. A5 validation passed with the provider
used in PR #928; revalidate when changing the provider version.

All workers call the MegaMoE compute and buffer APIs directly. Both
`auto` and `cann_ops_transformer` are parameterized.

## Running

Install the current checkout's `sgl_kernel_npu` wheel, CANN 9.2.0,
matching torch/torch_npu, a compatible `cann_ops_transformer` MegaMoE provider,
pytest and ninja. Expose four NPUs for the four-rank cases and eight NPUs for
the eight-rank case. Run pytest serially (no xdist).
All cases are marked `extra_checks` and excluded by default, including when
selecting a node ID. Pass `-m extra_checks` explicitly to run them.
The parent checks architecture/device count; each case launches isolated workers.
All workers execute the same sequence of collectives. Worker timeouts kill the
entire process group. Missing NPUs on an otherwise matching architecture fail.

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python -m pytest -m extra_checks \
  tests/python/operators/moe/test_mega_moe.py -v -s -x --enable-perf=false
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m pytest -m extra_checks -v -s -x \
  tests/python/operators/moe/test_mega_moe.py::test_bf16_accuracy_eight_rank_group
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python -m pytest -m extra_checks -v -s -x -rs \
  tests/python/operators/moe/test_mega_moe.py::test_mxfp4_accuracy_four_rank
```

The generic `generate_data → check_accuracy → check_performance` executor assumes
one worker and interchangeable arguments. Here pytest manages configurations,
`generate_data` constructs BF16 inputs, and the worker performs reference checks
before timing. This custom distributed workflow is intentional; `--impl-ref` and
`--skip-acc-check` do not change this test (accuracy always runs). `--impl` does
not narrow the explicit auto/cann_ops_transformer matrix; use pytest `-k` if needed.

## Performance

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python -m pytest -m extra_checks \
  tests/python/operators/moe/test_mega_moe.py -k bf16 -v -s --enable-perf=true
```

Without an override, BF16 timing runs only for `perf_shape` (32/32/7168/2048).
BF16 timing excludes buffer construction and the reference: after correctness,
all ranks synchronize, run three warmups and ten measured calls. Report the
maximum of per-rank mean wall latency, including launch/synchronization cost,
as `PERF max_rank_mean_latency_us`; it is not pure kernel task time.
With `--enable-perf=true`, quantized cases use the same timing method on the
first layout/format after the output and count checks pass. Generic
`--csv-path`/`--prof-dir` do not capture child-process results; preserve stdout
and record device, CANN/provider/torch versions alongside measurements.

## Validation status

A5 accuracy and interface tests passed, as reported in [PR #928](https://github.com/sgl-project/sgl-kernel-npu/pull/928).
A3 correctness and hardware performance measurements remain pending. The
multi-NPU tests are opt-in through `-m extra_checks`; this contribution does
not add or modify GitHub workflows.

## 中文说明

`auto/cann_ops_transformer` 是同一实现的别名。由于 provider 的通信 buffer 是不透明资源，
这里按贡献指南的例外条款，使用测试内独立参考，而不注册无法复用相同 buffer
的 `torch_native` 后端；PR 应保留此例外说明。A3 BF16 使用独立 CPU FP32 FFN，
A3 INT 和 A5 MXFP 通过直接调用 CANN 验证注册接口等价性，并用路由直方图独立校验专家
计数。独立的数值精度校验覆盖 A3 BF16，以及新增的 A5 四卡 A8W4-FP 场景。
A5 参考路径在 CPU 上解码 MXFP4 权重和 E8M0 scale，模拟中间 MXFP8 量化，
计算 SiTU、两层 FFN 和路由加权合并。A5 精度和接口用例已通过，升级 provider 后应重新验证。

所有配置由 pytest 管理；为保证集合通信顺序，实际计算由独立的四进程或八进程完成，
不使用单 worker 通用精度/性能执行器。测试覆盖与限制、命令、依赖和性能口径
见上文；`--enable-perf` 控制精度通过后的性能采集，日志保留在标准输出。
默认仅 BF16 `perf_shape` 计时。启用性能采集后，量化用例在首个布局和格式下计时。

全部多卡用例标记为 `extra_checks`，需显式传入 `-m extra_checks` 执行；本次贡献
不增加或修改 GitHub workflows。A5 精度与接口测试已通过；A3 精度和硬件性能数据仍待验证。
