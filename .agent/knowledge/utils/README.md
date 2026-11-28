# utils domain

Covers the shared facilities under `csrc/utils/`: data types, common macros/constants, and host/device helper headers used by more than one operator. Changing these files affects multiple operators, so they fall under the "shared infrastructure" rule in `AGENTS.md`.

## Recorded topics

| Topic | File | Scope |
| --- | --- | --- |
| dtype | [dtype.md](dtype.md) | The `sgl::kernel_npu::utils::DTYPE` enum, cross-layer passing, extension flow, dav-2201 support matrix |

## Current directory responsibilities

| Path | Responsibility |
| --- | --- |
| `csrc/utils/dtype.h` | Cross-layer data-type enum; no ATen/ACL dependency |
| `csrc/utils/defines.h` | Visibility and other common macros |
| `csrc/utils/common.h`, `common_tiling.h` | Host-side common definitions |
| `csrc/utils/torch_helper.h` | Host-side Torch helpers |
| `csrc/utils/kernel/` | Device-side helpers (`math_utils` and friends) |

## Open topics (add when needed)

- Common macros and visibility conventions (`defines.h`).
- The intrinsic subset and accuracy available from device-side `kernel/math_utils.h`.
- Reuse boundaries of host-side tiling structures.

## Conventions for new knowledge in this domain

- Split by header or mechanism; do not merge unrelated facilities into one file.
- State the header location, how it is included (relative path or include directory), and which build paths compile it.
- When touching this domain, check **both** build paths: legacy (`cmake/legacy_kernels.cmake`) and refactor (`setup.py`).
