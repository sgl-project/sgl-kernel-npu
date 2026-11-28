# AGENTS.md — working guide for agents in sgl-kernel-npu

This file is the entry point for AI agents working in this repository. Read it first, then open the relevant topic under `.agent/knowledge/`. Do not guess interfaces, build steps or hardware behavior from memory.

## 1. What this repository is

`sgl-kernel-npu` is the Ascend NPU kernel library used by SGLang. It contains three kinds of operators:

| Kind | Location | Notes |
| --- | --- | --- |
| CANN / AscendC operators | `csrc/<domain>/<op>/` | Built through the CANN op_host / op_kernel toolchain |
| Raw CCE operators | `csrc/cce/<op>/` | Device `.cce` compiled by bisheng, host side uses ACL Runtime, Torch binding in `torch_adapter.cpp` |
| HOST / AICPU operators | `csrc/<domain>/<op>/` | Pure host implementations or AICPU |

Public Python interfaces live in `python/sgl_kernel_npu/sgl_kernel_npu/operators/<domain>/`, with backends selected through `impl=` (`auto` / `ops-nn` / `ascendc` / `cce` / `torch_native` / `triton`).

## 2. Two build paths (do not mix them)

| Path | Command | Characteristics |
| --- | --- | --- |
| Refactor (used by current CI) | `python3 setup.py --arch-version {2201\|3510} --ops <ids\|all> [--artifact wheel]` | Per-operator build; operators register through `op_host` + `torch_adapter.cpp`; CCE operators are compiled by bisheng via `csrc/cmake/cce_kernels.cmake` |
| Legacy (full build) | `bash build.sh -a kernels [SOC_VERSION]` | Centralized build in `cmake/legacy_kernels.cmake`; AscendC kernels; registration in `csrc/pytorch_extensions.cpp`; does **not** compile raw `.cce` files under `csrc/cce/` |

Always confirm behavior against `setup.py`, `build.sh`, `cmake/`, `csrc/cmake/` and the CI workflows; documents such as `.opencode/ARCHITECTURE.md` may lag behind the code.

## 3. Directory overview

```text
csrc/                 C++ and device code
  cce/                raw CCE operators (op_host / op_kernel / torch_adapter)
  utils/              headers shared across operators (e.g. dtype.h)
  cmake/              unified build declarations and rules
python/sgl_kernel_npu/ Python package: public operators, registry, torch_native references
tests/python/         operator tests (operators/<domain>/test_<op>.py) and frameworks
docs/                 bilingual API docs and developer guides
.opencode/skills/     the repository's CCE development skill (read it, do not copy it)
.agent/knowledge/     the domain knowledge base that accompanies this file (section 4)
```

## 4. Knowledge base: `.agent/knowledge/`

Purpose: capture **verified, reusable** conclusions (interface contracts, hardware behavior, build details, pitfalls) so that agents do not repeat the same investigation.

It is a plain file convention: `AGENTS.md` plus Markdown under `.agent/knowledge/`. No knowledge-base plugin, vector index or external service is required. An agent reads this file, then opens the relevant topic from the index.

Rules:

1. The index and writing conventions live in `.agent/knowledge/README.md`; knowledge files are organized as `domain/topic.md`.
2. Read the index first and open only what the current task needs. When a knowledge file disagrees with the source, **the source wins**; fix the knowledge file in the same change.
3. Every entry states scope (branch / device / CANN version), conclusion, evidence (file paths, commands, measured results) and a change log.
4. Do not paste whole functions; cite `path:line`. Never include machine-specific paths, credentials or one-off logs.
5. When a conclusion expires (code change, toolchain upgrade), update or delete it instead of keeping a stale claim.

Current index:

| Domain | File | Read when |
| --- | --- | --- |
| utils | [utils/dtype.md](.agent/knowledge/utils/dtype.md) | Adding/changing a dtype, passing a dtype across layers, checking Ascend support for FP32/FP16/BF16 |
| utils | [utils/README.md](.agent/knowledge/utils/README.md) | Adding knowledge to the utils domain |

## 5. Hard engineering constraints

- **Source is the truth**: signatures, build options, supported ranges and hardware claims must be traceable to the current branch or to a measurement. When unsure, run a minimal verification.
- **CCE operators**: device code uses CCE intrinsics directly and must not pull in Ascend C APIs or `kernel_operator.h`; the host side submits to the caller's current stream via ACL Runtime. Read `.opencode/skills/cce-operator/SKILL.md` first.
- **Shared infrastructure** (`csrc/cmake/`, `csrc/CMakeLists.txt`, `python/.../utils/`, test frameworks): explain dependencies and impact, keep the change minimal, and flag it for maintainer review in the delivery notes.
- **Measurement rules**: accuracy uses the repository `check_accuracy` with an independent reference; performance uses the **device task time** from `check_performance` / `benchmark_npu`, never end-to-end wall clock.
- **Deliverables**: sources, formal tests and bilingual docs go into the repository; diagnostic scripts, profiling data and logs stay under ignored directories such as `build/`.

## 6. Common commands

```bash
# Build (refactor, single operator)
python3 setup.py --arch-version 2201 --ops <op_id> --without-third-party -j 16

# Install into an isolated directory and test (avoids polluting the environment)
pip install --no-deps --target <site> output/*.whl
PYTHONPATH=<site> python -m pytest tests/python/operators/<domain>/test_<op>.py \
  --impl=cce --impl-ref=torch_native -m "" -q --npu-device=<id>

# Style
pre-commit run --files <changed files>
```

Shared-device etiquette: check whether the target card is idle first; never preempt or reset devices used by someone else; state device, CANN version and measurement method with every performance claim.

## 7. Pre-delivery checklist

- [ ] The public interface is consistent end to end (Python `impl=`, `torch.ops` schema, host validation, device dispatch).
- [ ] Accuracy cases cover production shapes and edge cases; performance shapes are declared separately.
- [ ] The built artifact installs and passes tests; verification that was not run is stated explicitly.
- [ ] Docs (bilingual API docs plus the necessary README/knowledge updates) match the code.
- [ ] pre-commit passes (`no-commit-to-branch` failing on `main` is expected; do not commit to `main`).
