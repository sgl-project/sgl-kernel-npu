# moe_dispatch_combine

# compile
bash build.sh

# install
./build_out/custom_opp_ubuntu_aarch64.run --install-path=/usr/local/Ascend/ascend-toolkit/latest/opp/

# require import env parameters before running
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/hwcomputing/bin/set_env.bash

## Experimental A2 DispatchFFNCombine (mode 2)

The opt-in build below produces **only** `DispatchFFNCombine` for `ascend910b`.
It does not replace the normal A2 `ops2` package or change the top-level wheel
build. Use a private installation and a separate process for validation.

A2 exposes `HcclCombineOpParam` with rank-indexed input windows; A3 exposes
`HcclOpResParamCustom` with remote-resource pointers. Interpreting A2's context
as A3's can dereference invalid peer addresses. `DEEPEP_FUSEEP_A2` selects the
A2 accessor at compile time; the existing A3 accessor remains the default.
The vector barrier before routing-result extraction is a separate correctness
fix: it orders merge/single-list copy output before extraction. It is not an
A2-only architectural requirement.

### Operator-only build

Requires CANN 9.0.1 (the tested version), ASC CMake support, and a separate
`catlass-v1-stable` checkout. The tested CATLASS commit is
`404b4309c4e6366f2019fdba8feac376e0021836`. Set `CATLASS_INCLUDE` to its
`include` directory and `ASCEND_CANN_PACKAGE_PATH` to the CANN installation.
From `csrc/deepep/ops`, use a fresh build directory:

```bash
export OPS_PROJECT_NAME=aclnnInner
cmake -S . -B build_a2 \
  -DCMAKE_PREFIX_PATH="$ASCEND_CANN_PACKAGE_PATH/lib64/cmake" \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_CANN_PACKAGE_PATH" \
  -DASCEND_COMPUTE_UNIT=ascend910b \
  -DASCEND_AUTOGEN_PATH="$PWD/build_a2/autogen" \
  -DASCEND_PYTHON_EXECUTABLE="$(command -v python3)" \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_BINARY_PACKAGE=True \
  -DDEEPEP_FUSEEP_A2=ON -DDEEPEP_CATLASS_INCLUDE_DIR="$CATLASS_INCLUDE"
cmake --build build_a2 --target binary -j4
cmake --build build_a2 --target package -j4
```

Install the generated `.run` into a new private directory, **not** the shared
CANN installation. Set `ASCEND_CUSTOM_OPP_PATH` to its `vendors/fuseep_a2`
directory and prepend its `op_api/lib` to `LD_LIBRARY_PATH`. When an installed
DeepEP package prepends its own vendor library, an explicit
`LD_PRELOAD=$ASCEND_CUSTOM_OPP_PATH/op_api/lib/libcust_opapi.so` selects the
experimental operator. This operator-only library does not contain the stock
dispatch/combine APIs, so generate the unfused reference in a separate process
without those overrides. Do not use this override as a general runtime install.

### Correctness regression

Reserve/check four idle devices first. From the repository root:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=127.0.0.1 MASTER_PORT=30371 WORLD_SIZE=1 RANK=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Stock A2 dispatch/combine environment, without the private library override:
python tests/python/deepep/test_dispatch_ffn_combine_a2.py \
  --reference-dir /tmp/fuseep-a2-reference --write-reference
# In the private A2 operator environment described above:
python tests/python/deepep/test_dispatch_ffn_combine_a2.py \
  --reference-dir /tmp/fuseep-a2-reference
python tests/python/deepep/test_dispatch_ffn_combine_a2.py \
  --reference-dir /tmp/fuseep-a2-reference --tokens 2048 --repeats 100 --sparse-case one
python tests/python/deepep/test_dispatch_ffn_combine_a2.py \
  --reference-dir /tmp/fuseep-a2-reference --tokens 2048 --repeats 100 --sparse-case empty
```

Random-routing tests check exact received counts, finite outputs, mean absolute
error below 0.01 against the unfused reference, bitwise repeatability, and
bitwise eager/graph agreement. Sparse cases check exact counts and repeatability
with a singleton or empty final expert; they do not compare to an unfused
reference. These are synthetic operator tests, not end-to-end benchmarks.

### Validation scope

The implementation was validated in an isolated operator build on four
Ascend910B2 devices with CANN 9.0.1, PyTorch 2.10.0 and torch_npu 2.10.0.post2:
BF16 activations, int8 weights, int32 expert IDs, hidden size 2048, gate/up width
1536, 128 experts, top-8, and 32 experts per rank. At 16 tokens/rank, ten calls
and 100 graph replays/rank passed; reference mean absolute error was
0.000149–0.000179. At 2048 tokens/rank, both sparse-tail cases passed 100 calls
with no count errors or changed outputs. The original sparse-tail validation
did not exercise graph replay; the regression above additionally requests it.

Full-model validation, A3 regression, multi-node/multiFlag transport validation,
rebuilding the upstream-integrated package and running the new regression harness,
other shapes/dtypes, and performance evaluation remain pending. No broad A2
compatibility or performance claim is made. The original test processes also
emitted Python forkserver signal-handler errors, despite passing rank assertions
and exiting zero; this environment issue remains to be investigated.
