# A5 native INT8 FuseEP validation

This port enables `FuseMode.DISPATCH_FFN_COMBINE` (mode 2) on Ascend950. It is
the kernel dependency for the INT8 A5 regression reported while testing
[SGLang #40516](https://github.com/sgl-project/sglang/pull/40516).

Both grouped matrix multiplies retain INT8 inputs and weights with INT32
accumulation. Per-channel weight scales are passed to the A5 Fixpipe; the
vector epilogues apply per-token scales, SwiGLU, INT8 requantization, and
BF16 output conversion. Checkpoints are not converted to FP8 or BF16.

The A5 build includes the operator definition, tiling and ACLNN wrappers.
The shared pipeline selects CATLASS v1.6.1's `Ascend950` copies and Fixpipe,
uses CANN's A5 HCCL context accessors and communication engine, and obtains
the routing core count and UB size from the platform. A3 keeps its existing
CATLASS dependency, HCCL ABI and routing configuration.

GEMM2's per-token dequantization uses explicit element counts so its vector
operations do not depend on a previous operation's mask. Input dynamic
quantization synchronizes the vector reduction with the scalar scale read,
and waits for that read before reusing the scale buffer on Vector.

## Build on A5

Use the A5 CANN and torch-npu environment supported by this repository. The
build fetches CATLASS v1.6.1 and requires the system `moe_distribute_base.h`
already used by A5 `FusedDeepMoe`.

```bash
bash build.sh -a deepep Ascend950
python -m pip install --force-reinstall --no-deps output/deep_ep*.whl
```

Start fresh SGLang workers after installing the wheel. The runtime loads
`libcust_opapi.so` relative to the actual `deep_ep_cpp` extension before
consulting the process search path. An older external custom library must
not override the library in the wheel. The A5 build verifies both
`aclnnDispatchFFNCombine` exports and stops if either is absent.

The host-only loader regression can be run on Linux with
`python3 tests/python/deepep/test_op_api_loader.py -v`.

## Distributed INT8 kernel regression

The existing test compares against unfused INT8 dispatch, grouped matmul,
SwiGLU quantization and combine. Its reference explicitly selects INT8 on
A5 (where `use_fp8=True` selects FP8), asserts integer tensor dtypes, checks
output error and exact expert token counts, and can repeat on the same
buffers to exercise communication state reuse. The original mean absolute
error threshold of `1e-2` is unchanged.

From the repository root, run the Qwen3-30B-A3B expert shapes (hidden 2048,
gate/up width 1536, 128 experts, top-k 8) at decode, tail and prefill sizes:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_BUFFSIZE=200
export DEEPEP_HCCL_BUFFSIZE=200
for tokens in 1 17 128; do
  timeout 600s python tests/python/deepep/test_dispatch_ffn_combine.py \
    --num-processes 4 --num-tokens "$tokens" --hidden 2048 \
    --moe-intermediate-size 1536 --num-experts 128 --num-topk 8 \
    --repeat 3 --skip-benchmark || exit 1
done

# Concentrate traffic on one rank; other ranks receive no expert tokens.
timeout 600s python tests/python/deepep/test_dispatch_ffn_combine.py \
  --num-processes 4 --num-tokens 17 --hidden 2048 \
  --moe-intermediate-size 1536 --num-experts 128 --num-topk 8 \
  --active-ranks 0 --repeat 3 --skip-benchmark
```

Also rebuild for A3 (`bash build.sh -a deepep Ascend910_9382`), install that
wheel on A3 and run the same regression. Do not reuse an A5 wheel on A3.

The reported A5 17-token reproduction after the mask and synchronization
fixes passed all three repetitions on four ranks with mean errors of
0.001182556-0.001762390 and matching expert counts. This standalone result
does not establish full-model accuracy.

## SGLang regression

With the corresponding SGLang #40516 changes and the rebuilt A5 wheel,
run the same INT8 checkpoint from the SGLang repository:

```bash
PYTHONPATH="$PWD/python:${PYTHONPATH:-}" \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_TEST_MODEL_PATH=/path/to/Qwen-MoE-W8A8 \
SGLANG_TEST_TP_SIZE=4 \
SGLANG_TEST_FUSEEP_MODES=2 \
SGLANG_TEST_LOG_DIR=/tmp/fuseep-a5-int8 \
python -u test/registered/npu/basic_function/parameter/test_npu_fuseep_mode.py -v
```

The test exercises batched prefill/decode and concurrent arithmetic. A5
requires both the unfused baseline and each FuseEP mode to score at least
90% on 200 GSM8K questions, using five-shot completion and the last explicit
numeric `####` answer with last-number fallback. A3 retains its teacher-forced
logprob thresholds (mean absolute difference `< 0.1`, maximum `< 0.6`).
Server logs and ordinary GSM8K HTML reports are kept in the output directory.
Use the Qwen3.5 model override for the shared-expert regression.

For the user's saved A5 200-question responses, explicit-answer scoring gives
baseline 187/200 and mode 2 186/200. These are offline rescoring results, not
a fresh end-to-end pass. Full-model logprob equivalence and performance are
not established by these results.
