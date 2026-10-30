# Ascend C Operator Sample: From `helloworld` to a Tested Operator

This tutorial is the second rung after
[`csrc/helloworld/README.md`](../../csrc/helloworld/README.md). `helloworld`
shows the smallest possible Ascend C kernel; this document walks through a
complete operator that is actually integrated into `sgl_kernel_npu`, tested
through the public Python API, and documented like any other repository
operator. The worked example is `elementwise_add` (`z = x + y`) in
[`csrc/elementwise_add`](../../csrc/elementwise_add).

Read [`docs/developer_guide/contribution_guide.md`](contribution_guide.md)
first: it is the normative contribution policy (API shape, `impl` dispatch,
tests, bilingual docs). This tutorial only adds the Ascend C specifics and the
verification gates that a kernel contributor must clear.

## 1. What the sample contains

| Layer | Path | What it demonstrates |
| --- | --- | --- |
| Device kernel | `csrc/elementwise_add/op_kernel/kernel_elementwise_add.cpp` | `CopyIn -> Compute -> CopyOut`, double-buffered queues, two dtype entries, a documented layout contract. |
| Host launcher | `csrc/elementwise_add/op_host/elementwise_add.cpp` | Input validation, UB-aware tiling, host-side alignment, `EXEC_KERNEL_CMD` launch, output shape restore. |
| Torch binding | `csrc/pytorch_extensions.cpp`, `include/sgl_kenel_npu_ops.h` | `npu::elementwise_add` schema plus the `PrivateUse1` implementation. |
| Build wiring | `cmake/legacy_kernels.cmake` | Host source in `OP_SRCS`, kernel source in `NO_WORKSPACE_KERNEL_SRCS`. |
| Public API | `python/sgl_kernel_npu/sgl_kernel_npu/operators/elementwise/elementwise_add.py` | `auto`/`ascendc`/`torch_native` registration and the keyword-only `impl` argument. |
| Reference | `python/sgl_kernel_npu/sgl_kernel_npu/implementation/torch_native/elementwise/elementwise_add.py` | Independent `x + y` used as the accuracy baseline. |
| Tests | `tests/python/operators/elementwise/test_elementwise_add.py` | Production and edge shapes, three data modes, negative cases, optional profiling. |
| API docs | `docs/api/{en,zh}/operators/elementwise/elementwise_add.md` | The bilingual documentation required by the contribution guide. |

## 2. Workflow

### Step 0 - Prepare the environment

```bash
git clone "https://github.com/<your_user_name>/sgl-kernel-npu.git"
cd sgl-kernel-npu
git switch -c feature/my-operator
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # CANN toolkit env
./build.sh -a kernels                                # builds sgl_kernel_npu
pip install -e python/sgl_kernel_npu                 # installs the wheel layout
```

Confirm the toolchain before writing code: `npu-smi info` must see a device and
`python -c "import torch, torch_npu"` must succeed in the same shell that runs
the tests. `./build.sh -a kernels` accepts an explicit `SOC_VERSION`
(`Ascend910B1` for A2, `Ascend910_9382` for A3); without it the script detects
the chip or falls back to `Ascend910_9382`.

### Step 1 - Fix the semantics and the layout contract

Write down the signature, dtypes, ranks, and the exact contract between host and
kernel *before* writing either side. For `elementwise_add` it is:

- kernel entries: `elementwise_add_fp16`, `elementwise_add_fp32`;
- host guarantees `totalLength % (GetBlockNum() * tileLength) == 0`;
- host guarantees `tileLength * sizeof(T) % 32 == 0`;
- the kernel never sees a tail and never calls `DataCopyPad`.

A contract you cannot state in three lines is usually a contract that will be
violated by a later change. Put it in a comment at the top of the kernel file.

### Step 2 - Write the device kernel

Start from `helloworld` and keep its skeleton: one class, `Init`, `Process`,
and `CopyIn`/`Compute`/`CopyOut`. Then make the three things explicit:

1. **dtype dispatch** - template the class on `T` and emit one
   `extern "C" __global__ __aicore__` entry per dtype. The host picks the entry,
   so the kernel file never guesses a dtype.
2. **tiling from the device** - `tileLength` is a kernel argument computed on
   the host from the UB size, not a constant. `tileCount` is derived from
   `totalLength / GetBlockNum() / tileLength`.
3. **buffering** - `BUFFER_NUM = 2` gives double buffering: the MTE2 copy of
   tile `i+1` overlaps the vector work on tile `i`. Every queue is allocated in
   `Init` with `tileLength * sizeof(T)`, and every `AllocTensor` is paired with
   `FreeTensor`.

Keep the loop structure `for (i...) { CopyIn(i); Compute(); CopyOut(i); }`; the
queue API is what serialises the stages safely.

### Step 3 - Write the host launcher

The host file owns everything the device must not do:

1. **validate** shape, rank, contiguity, dtype, and the `uint32` range of the
   element count, with `TORCH_CHECK` and actionable messages;
2. **tile** - derive `tileLength` from `PlatformAscendCManager` UB size
   (reserve some UB, keep the number of in-flight tiles in sync with the kernel),
   then pick `blockDim = min(coreNum, numTiles)`;
3. **align** - round the element count up to `blockDim * tileLength` and
   zero-pad on the host. Flatten the input first (`x.reshape({numel})`), because
   copying an N-D tensor into a 1-D buffer would broadcast;
4. **launch** - `EXEC_KERNEL_CMD(<entry>, blockDim, xFlat, yFlat, zFlat, totalLength, tileLength)`.
   The macro takes non-const lvalue references, so bind `view({-1})` results to
   named tensors;
5. **lifetime** - call `record_stream` on every host temporary that the
   asynchronous kernel can still read or write;
6. **restore** - `narrow` off the padding and `view` back to the caller's shape.

If you add a queue or scratch buffer, update the in-flight tile count in the
host file in the same commit; otherwise the UB allocation silently overflows
and the kernel fails with an opaque allocation error at run time.

### Step 4 - Wire the binding

Three small edits make the operator visible to Python:

- declare `at::Tensor elementwise_add(const at::Tensor &, const at::Tensor &)`
  in `include/sgl_kenel_npu_ops.h`;
- add `m.def("elementwise_add(Tensor x, Tensor y) -> Tensor");` to the
  `TORCH_LIBRARY_FRAGMENT(npu, m)` block in `csrc/pytorch_extensions.cpp`;
- add `m.impl("elementwise_add", TORCH_FN(sglang::npu_kernel::elementwise_add));`
  to the `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)` block;
- list the host source in `OP_SRCS` and the kernel source in
  `NO_WORKSPACE_KERNEL_SRCS` in `cmake/legacy_kernels.cmake`.

Rebuild, then check the schema before writing Python code:

```bash
python -c "import torch, torch_npu, sgl_kernel_npu; print(torch.ops.npu.elementwise_add)"
```

### Step 5 - Add the public API and an independent reference

Follow section 3 of the contribution guide: register the Ascend C entry as both
`auto` and `ascendc`, register `torch_native` as the reference, export the
function from the module `__init__.py`, and keep `impl` keyword-only. The
reference must not call the kernel; here it is plain `x + y`.

### Step 6 - Write the tests

Follow section 4 of the contribution guide. For this sample the tests:

- cover production shapes (`1x4096`, `16x7168`), a minimum shape (`1x1`), an
  unaligned numel (`7x1025`), 1-D and 3-D ranks, and an empty tensor;
- cover data modes (`normal`, `zeros`, `cancelling`) in a single
  `generate_data` helper;
- compare against `torch_native` with `check_accuracy` and only then collect
  performance for the shapes listed in `perf_shape`;
- assert that unsupported dtype and non-contiguous input are rejected
  (`pytest.raises`), so the validation code has coverage too.

### Step 7 - Document and submit

Add `docs/api/{en,zh}/operators/<module>/<op>.md` using the section order from
section 6 of the contribution guide, and add one row per language to
`docs/api/{en,zh}/API_list.md`. Then run the checks in section 3 and open the PR
against `main` with the correctness evidence attached.

## 3. Verification standard

A sample is "done" only when every gate below passes on real hardware. Record
the command, the chip, and the software versions in the PR; a gate without
evidence is not a gate.

| ID | Gate | Command | Pass condition |
| --- | --- | --- | --- |
| V1 | Extension builds | `./build.sh -a kernels` | Build exits 0 and `libsgl_kernel_npu.so` is refreshed in `python/sgl_kernel_npu/sgl_kernel_npu/lib/`. |
| V2 | Schema is registered | `python -c "import torch, torch_npu, sgl_kernel_npu; print(torch.ops.npu.elementwise_add)"` | Prints the operator handle without error. |
| V3 | Accuracy vs. reference | `python -m pytest tests/python/operators/elementwise/test_elementwise_add.py --skip-acc-check=false --enable-perf=false -v -s -x -rs` | All cases pass; `torch_native` is the reference. |
| V4 | Dtype coverage | same as V3 | Both `torch.float16` and `torch.float32` pass with the repository tolerances (`1e-3` fp16, `1e-5` fp32). |
| V5 | Edge shapes | same as V3 | `1x1`, `7x1025`, 1-D, 3-D, and empty inputs pass; no launch for empty input. |
| V6 | Input validation | same as V3 | bfloat16 and non-contiguous inputs raise `RuntimeError` with the documented messages. |
| V7 | Performance recorded (optional but expected for a real operator) | `python -m pytest tests/python/operators/elementwise/test_elementwise_add.py --skip-acc-check=false --prof-dir=results/profilings --csv-path=results/elementwise_add.csv -v -s` | CSV contains the `perf_shape` cases for fp16; report the numbers in the PR, not in the docs. |
| V8 | Style and docs | `pre-commit run --all-files --show-diff-on-failure` | No remaining formatting issues; both language docs and both index rows are updated. |

Cross-checks that catch the most common mistakes:

- a wrong kernel entry name fails at *build* time with a missing
  `aclrtlaunch_<name>.h`; the entry name in the kernel file, the host
  `#include`, and the `EXEC_KERNEL_CMD` call must match;
- a UB budget mistake fails at *first launch* with an allocation error; recount
  the in-flight tiles;
- an alignment mistake shows up as a mismatch only on unaligned numel, which is
  why `7x1025` is in the test matrix;
- a stale `impl` registration shows up as `ImplementationNotFoundError`; check
  that the module `__init__.py` exports the new function and that the package
  was reinstalled after the rebuild.

## 4. Checklist before opening the PR

- [ ] Kernel file states the host/kernel layout contract in a comment.
- [ ] Host file validates every assumption with `TORCH_CHECK` and a clear
      message; nothing is left to the kernel.
- [ ] All queues are allocated with matching sizes and freed on every path.
- [ ] Host temporaries are `record_stream`-ed; the output is trimmed and
      reshaped back.
- [ ] Binding added in the header, the schema fragment, the `PrivateUse1`
      fragment, and the CMake source lists.
- [ ] `auto` and one concrete implementation are registered; `torch_native`
      is an independent reference.
- [ ] Tests cover production shapes, edge shapes, dtypes, and negative cases.
- [ ] Accuracy passes before performance is collected.
- [ ] Both language API docs and both `API_list.md` indexes are updated.
- [ ] `pre-commit run --all-files` is clean.
- [ ] PR body lists the build command, chip, software versions, test command,
      test result, and (if collected) performance numbers.

## 5. References

- Contribution policy: [`docs/developer_guide/contribution_guide.md`](contribution_guide.md)
- Minimal kernel: [`csrc/helloworld/README.md`](../../csrc/helloworld/README.md)
- Test utilities: [`tests/python/README.md`](../../tests/python/README.md)
- Bilingual API example: [`docs/api/en/operators/norm/rmsnorm_bias.md`](../api/en/operators/norm/rmsnorm_bias.md)
- Ascend C operator development: <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html>
- Reference implementation of a full Ascend C operator PR:
  <https://github.com/sgl-project/sgl-kernel-npu/pull/788>
