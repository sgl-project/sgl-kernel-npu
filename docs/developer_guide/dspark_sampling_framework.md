# DSpark sampling operators

The public entry points are `row_argmax`, `markov_greedy`, `markov_sample`,
and `build_out_tokens` in `sgl_kernel_npu.operators.sample`. They take the
same arguments as their legacy `sgl_kernel_npu.sample` counterparts, plus the
keyword-only `impl` selector. `auto` and `triton` select the Triton backend.
The original imports are retained for existing callers.

The registry currently declares arch22 support only; arch35 is not yet
validated. `row_argmax`, `markov_greedy` and `build_out_tokens` also provide a
`torch_native` reference implementation. `markov_sample` has no registered
reference: its Philox stream, optional precomputed noise, distributed gather
and optional `logits_out` mutation require the dedicated tests in
`tests/python/sgl_kernel_npu/test_markov_sample.py`.

`row_argmax` accepts a two-dimensional FP32 tensor with unit column stride;
returns int64 indices, with the first maximum or first NaN selected. The
Markov greedy operator performs a linear projection and selects the first
maximum after adding it to the base logits. `build_out_tokens` returns an int64
buffer comprising the draft and bonus, including the uncommitted suffix; the
caller must use acceptance lengths to determine valid output tokens.

Run the new framework tests and the existing semantic and graph tests on an
Ascend NPU with the matching installed package:

```bash
python -m pytest tests/python/operators/sample/test_sampling.py -v -rs
python tests/python/sgl_kernel_npu/test_row_argmax.py
python tests/python/sgl_kernel_npu/test_markov_greedy.py
python tests/python/sgl_kernel_npu/test_build_out_tokens.py
python tests/python/sgl_kernel_npu/test_markov_sample.py
```

The framework tests use `auto` and `torch_native` by default for the three
operators with references. The sampling test compares the public operator with
an independent CPU reference using identical precomputed noise, then creates a
framework executor for optional profiling. The dedicated sampling tests also
cover Philox and mutable outputs. Performance collection is not enabled by
default; use the pytest `--enable-perf` option where appropriate.
