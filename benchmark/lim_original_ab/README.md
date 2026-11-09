# Original vLLM-Ascend versus current LIM: stable-prefix A/B

This is an operator-only, BF16 A2/A3 experiment, not a reproduction of the
paper's C8 latency. Both implementations use B=16, Q=4, S=65536, K=2048,
C=12288, 32 heads, and a common page-aligned stable prefix of 65408. The
original operator does not support the current full-visible Q4 contract. Each
request starts with a full resident map and exactly 410 or 737 selected-union
misses. The map is restored outside each NPU event interval. The script uses
64 individual vendor LI calls as a correctness oracle; these are not a timed
LI baseline when `--only-lim` is used. No payload copies or attention run.

The original source is vllm-project/vllm-ascend commit
`3326f350c0de30a5170b86ecfce851322e3216e6`. The source archive obtained from
`https://codeload.github.com/vllm-project/vllm-ascend/tar.gz/3326f350c0de30a5170b86ecfce851322e3216e6`
had SHA-256 `c71229fe5a1b38ee7c902b3871aab5678f71b6ee4c8633e58d0bbcb39423ae9c`.
The current port was based on sgl-kernel-npu `28d243b43a669405e32ad73759f36139cbed82b9`;
the companion framework baseline was `f2f65896414d1645ea52d8872d07c25b115cfe31`
(the framework is not used by this operator-only A/B). The measured current
library SHA-256 was `61c08679342c8dd0b2a7acf33d89f5a1230701d9baffd4f5b764b28c198bbd85`.
Use independent directories for original source, original installed custom
operator, binding build, and results. The upstream package's dependencies and
CANN build tools must be available. The wrapper includes the pinned upstream
Torch adapter; it does not replace or patch upstream kernel or host code.

Example, on a provisioned Linux Ascend A3 system (set all paths yourself):

```bash
export VLLM_ASCEND_SOURCE=/absolute/path/to/pinned/vllm-ascend
export ORIGINAL_RUNTIME=/absolute/path/to/isolated/original-runtime
export ORIGINAL_BINDING_BUILD=/absolute/path/to/isolated/binding-build
export CURRENT_KERNEL_PYTHON_ROOT=/absolute/path/to/current/kernel-package-root
export RESULTS=/absolute/path/to/new/results
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p "$ORIGINAL_RUNTIME" "$ORIGINAL_BINDING_BUILD" "$RESULTS"
(cd "$VLLM_ASCEND_SOURCE/csrc" && bash build.sh --pkg --ops=fused_lightning_indexer_manage --soc=ascend910_93 -j8)
bash "$VLLM_ASCEND_SOURCE/csrc/build_out/cann-ops-transformer-custom_linux-aarch64.run" --install-path="$ORIGINAL_RUNTIME"
python3 benchmark/lim_original_ab/build_binding.py
export ORIGINAL_LIM_BINDING="$ORIGINAL_BINDING_BUILD/original_lim_binding.so"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$CURRENT_KERNEL_PYTHON_ROOT"
export LD_LIBRARY_PATH="$(python3 -c 'import pathlib, torch, torch_npu; print(str(pathlib.Path(torch_npu.__file__).parent / "lib") + ":" + str(pathlib.Path(torch.__file__).parent / "lib"))'):${LD_LIBRARY_PATH:-}"
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0  # Select an idle device after checking ownership.
```

Run each configuration in a fresh process. The original custom OPP environment
must be sourced only for original runs; do not contaminate the current run's
operator environment. The CLI refuses to overwrite results.

```bash
for misses in 410 737; do
  (source "$ORIGINAL_RUNTIME/vendors/custom_transformer/bin/set_env.bash"; \
   BENCH_IMPL=original python3 benchmark/lim_original_ab/bench_prefix.py --only-lim --misses "$misses" --warmup 20 --iterations 200 --rounds 3 --output "$RESULTS/original_$misses.json")
  BENCH_IMPL=current python3 benchmark/lim_original_ab/bench_prefix.py --only-lim --misses "$misses" --warmup 20 --iterations 200 --rounds 3 --output "$RESULTS/current_$misses.json"
done
```

Compare `input_sha256`, `reference_sorted_sha256`, and the resident-map hash
within each miss pair, as well as per-request plan counts and slot checks.
The example runs are sequential and are not paired or interleaved. Save logs,
exit codes, loaded-library hashes, device state, and runtime versions with any
published results. A common sorted reference hash does not imply equivalence
to the full-visible workload.

The executed experiment used a different, machine-specific script (SHA-256
`32015230fe0dc547c665a9d19c13149d1e1549f02aa2174111c0ef3cf0e71e3b`)
and wrapper source (SHA-256
`af98417b7141dcb76e15eb0d9b45a519ec00472efa1d4f844a332bb961f7d63e`).
The portable adaptation passed a fresh A3 smoke run with 410 misses: a new
binding build, both implementations, 1 warmup and 3 rounds of 2 samples
each. Input, reference, and resident-map hashes matched; all 16 request
plans and slots validated. This smoke run checks reproducibility and is not
a performance measurement. The formal results came from the executed script
identified above, not from this adapted script.
