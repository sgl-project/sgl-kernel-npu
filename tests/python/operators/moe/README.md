# MegaMoE public-interface validation

## Scope and independent-reference exception

`auto` and `cann` select one external implementation, not two independent
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
  CANN. Independent numerical accuracy is covered by the A3 BF16 reference.

All workers call the MegaMoE compute and buffer APIs directly. Both
`auto` and `cann` are parameterized.

## Running

Install the current checkout's `sgl_kernel_npu` wheel, CANN 9.2.0,
matching torch/torch_npu, a compatible `cann_ops_transformer` MegaMoE provider,
pytest and ninja. Expose four NPUs for the four-rank cases and eight NPUs for
the eight-rank case. Run pytest serially (no xdist).
The parent checks architecture/device count; each case launches isolated workers.
All workers execute the same sequence of collectives. Worker timeouts kill the
entire process group. Missing NPUs on an otherwise matching architecture fail.

```bash
python -m pytest tests/python/sgl_kernel_npu/test_mega_moe_registration.py -q
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python -m pytest \
  tests/python/operators/moe/test_mega_moe.py -v -s -x --enable-perf=false
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m pytest -v -s -x \
  tests/python/operators/moe/test_mega_moe.py::test_bf16_accuracy_eight_rank_group
```

The generic `generate_data → check_accuracy → check_performance` executor assumes
one worker and interchangeable arguments. Here pytest manages configurations,
`generate_data` constructs BF16 inputs, and the worker performs reference checks
before timing. This custom distributed workflow is intentional; `--impl-ref` and
`--skip-acc-check` do not change this test (accuracy always runs). `--impl` does
not narrow the explicit auto/cann matrix; use pytest `-k` if needed.

## Performance

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python -m pytest \
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

## CI and validation status

Registration tests run in the CPU lint workflow. `pr-test-mega-moe.yml` builds
the checkout and runs four-rank A3/A5 tests and the eight-rank A3 BF16 test
with CANN 9.2.0. Configure repository
variables `MEGAMOE_PROVIDER_WHEEL_A3` and `MEGAMOE_PROVIDER_WHEEL_A5` with pinned,
architecture-matching provider wheel URLs. Missing configuration fails before
the build. No provider version or download URL is guessed by this contribution.

The authoring host has no Ascend NPU. Hardware correctness/performance results
are **pending**, not passing. Include actual CI logs and performance measurements
in the PR before claiming hardware validation.

Local validation: Python 3.12, CPU PyTorch 2.15.0; registration and reference/CLI
unit tests: **49 passed**. The distributed pytest file collects **23 cases**.
Workflow YAML, documentation links, Python example syntax, Black and isort
checks pass. Full `pre-commit run --all-files --show-diff-on-failure` also passes
in an isolated copy of the checkout. These checks do not execute the NPU kernel.

## 中文说明

`auto/cann` 是同一实现的别名。由于 provider 的通信 buffer 是不透明资源，
这里按贡献指南的例外条款，使用测试内独立参考，而不注册无法复用相同 buffer
的 `torch_native` 后端；PR 应保留此例外说明。A3 BF16 使用独立 CPU FP32 FFN，
A3 INT 和 A5 MXFP 通过直接调用 CANN 验证注册接口等价性，并用路由直方图独立校验专家
计数。独立的数值精度校验覆盖 A3 BF16。

所有配置由 pytest 管理；为保证集合通信顺序，实际计算由独立的四进程或八进程完成，
不使用单 worker 通用精度/性能执行器。测试覆盖与限制、命令、依赖和性能口径
见上文；`--enable-perf` 控制精度通过后的性能采集，日志保留在标准输出。
默认仅 BF16 `perf_shape` 计时。启用性能采集后，量化用例在首个布局和格式下计时。

CPU 注册测试已接入 lint；硬件 CI 需要配置上述两个 provider wheel 仓库变量。
本机没有 NPU，硬件精度与性能结果仍待 CI/目标机器运行，不应填写为通过。
