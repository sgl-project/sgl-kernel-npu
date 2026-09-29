# Sampling probability renormalization

Top-k and top-p use the original probability sort to select tokens. Top-p keeps
its inclusive cumulative sum minus the current probability comparison. The
filtered sorted probabilities are reduced in the original order and the row
sum is clamped to `1e-20` before normalization.

To avoid a full-vocabulary scatter, each sorted token ID is encoded as the FP32
integer `2 * token_id + nonzero_keep_bit`. Sorting these keys restores vocabulary
order, so the original probabilities can be read and written contiguously.
The full-sort indices must be a permutation; this is not a general scatter
replacement. Vocabulary sizes greater than `2**23` are rejected to preserve
exact FP32 integer encoding.

For nonempty, contiguous FP32 NPU inputs, Triton fuses filtering with key
encoding and restores the output in a second kernel. Native probability sort,
cumulative sum, filtered-value reduction and key sort are preserved. Other
inputs use the PyTorch packed-key implementation.

## Enable fused round-to-nearest division

Set these variables before starting every inference worker:

```bash
export SGL_KERNEL_NPU_SAMPLING_TRITON=1
export SGL_KERNEL_NPU_SAMPLING_TRITON_DIV=rn
```

The first variable defaults to `1`. The division mode defaults to `native`,
which performs the original PyTorch division after the restoration kernel.
`rn` fuses `tl.div_rn` into restoration; it does not replace division with a
reciprocal multiply. Setting `SGL_KERNEL_NPU_SAMPLING_TRITON=0` disables the
Triton path while retaining packed-key restoration.

No additional SGLang command-line option is required for this optimization.
Keep the existing model, MTP, temperature and top-k/top-p settings. Install the
updated kernel package and restart workers so they import the new Python files.
The calling SGLang version must already support the NPU non-greedy sampling
interfaces introduced by PR #32495.

## Precision and performance validation

On an NPU host with this checkout installed, run:

```bash
python -m pytest -q \
  tests/python/sgl_kernel_npu/test_speculative_probability.py \
  tests/python/sgl_kernel_npu/test_speculative_probability_triton.py
```

The fused tests cover both division modes, scalar and per-row thresholds,
top-k and top-p, ties, zero mass, one-hot inputs and vocabulary block tails.
A shared probability sort isolates restoration accuracy from independently
selected tie permutations. The public-interface test checks top-k followed by
top-p and verifies that both calls reach the requested Triton division mode.
NPU-only tests are skipped when no NPU is available; CPU results do not validate
Ascend compilation or division behavior.

For performance comparisons, use identical inputs, warm up compilation, and
synchronize device execution around measurements. Measure complete top-p and
end-to-end TPOT separately for disabled, native and rn modes. Keep the same
model, load, parallel configuration and sampling parameters, and record the
acceptance length alongside TPOT. Microbenchmark latency is not TPOT.
