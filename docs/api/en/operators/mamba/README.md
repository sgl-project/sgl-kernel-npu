# mamba

Target-verification convolution for linear and tree speculative candidates.

Public import: `sgl_kernel_npu.operators.mamba`.

Select backends with `impl`; `auto` aliases the backend listed on each API page, with no runtime fallback. Registry import requires NPU architecture detection; implementations load lazily on first execution.

- [causal_conv1d_target_verify_npu](_causal_conv1d_target_verify_npu.md)

Linear mode overwrites `x`; tree mode allocates output. Both keep committed `conv_state` read only and write candidate scratch; see the API side-effect contract.
