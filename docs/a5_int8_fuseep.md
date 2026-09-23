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

## Validation status

This is an unvalidated hardware port. The development host has neither CANN
nor an NPU: the A5/A3 compile, distributed kernel execution, numerical
comparison, and SGLang end-to-end regression below must be run before this
change is considered ready. Static checks do not establish NPU correctness.

## Build on A5

Use the A5 CANN and torch-npu environment supported by this repository. The
build fetches CATLASS v1.6.1 and requires the system `moe_distribute_base.h`
already used by A5 `FusedDeepMoe`.

```bash
bash build.sh -a deepep Ascend950
python -m pip install --force-reinstall --no-deps output/deep_ep*.whl
```

Verify the installed custom operator exports both symbols:

```bash
find output python/deep_ep -name 'libcust_opapi.so' -exec \
  nm -D --defined-only {} \; | grep -E 'aclnnDispatchFFNCombine(GetWorkspaceSize)?$'
```

Use a fresh process after installing the wheel. A stale DeepEP/custom OPP
installation still produces the missing-`aclnnDispatchFFNCombine` exception.

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

## SGLang regression

With the corresponding SGLang #40516 changes and the rebuilt A5 wheel,
run the existing INT8 checkpoint without skipping A5:

```bash
PYTHONPATH="$PWD/python:${PYTHONPATH:-}" \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_TEST_MODEL_PATH=/path/to/Qwen-MoE-W8A8 \
SGLANG_TEST_TP_SIZE=4 \
SGLANG_TEST_FUSEEP_MODES=2 \
SGLANG_TEST_LOG_DIR=/tmp/fuseep-a5-int8 \
python -u test/registered/npu/basic_function/parameter/test_npu_fuseep_mode.py -v
```

Run this command from the SGLang repository. It compares the same INT8
checkpoint with the unfused backend, including batched prefill/decode,
concurrent arithmetic requests and teacher-forced logprobs (mean absolute
difference `< 0.1`, maximum `< 0.6`). Use the Qwen3.5 model override for the
shared-expert regression. Performance and graph execution need separate
validation; these commands make no performance claim.
