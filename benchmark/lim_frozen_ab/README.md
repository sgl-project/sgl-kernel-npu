# Frozen-input LIM A/B and resident-layout ablation

These experiments compare the vLLM-Ascend LIM with the SGLang port in one
process, using identical pre-existing inputs and caller-owned output buffers.
The workload is A3 BF16, B=16, Q=4, 32 index heads, dimension 128, K=2048,
C=12288, actual KV length 65664, stable search prefix 65536, and state -1.
LI uses sparse mode 0; the port uses `full_visible=False`. No KV payload copy,
attention, model serving or C8 computation is measured. This is not an
end-to-end result or a reproduction of the paper's C8 numbers.

Use a trusted, provisioned Linux A3 environment with CANN 9.2.0, torch
2.10.0+cpu and torch_npu 2.10.0.post7. The original environment used image
`sha256:efe959d9f1b8b2a02c9d978f203ccdc391ff079ce3e923b24c5da2cf2a6bdac2`,
physical device 14, and vLLM-Ascend checkout
`bd42eea01d8312c4df469fabaed45f77a12df4d3` with its custom OPP. Its LIM
implementation matches the 20 operator source files pinned at
`3326f350c0de30a5170b86ecfce851322e3216e6` after LF normalization. The port
source was `ef24e7d5f8feba358aab61ba03d2ec75e1dc7e06`; the framework checkout
was `47b9d8c76226e3e73afdbe5d23bc0cfcb4e2f127`. The scripts check the loaded
`libsgl_kernel_npu.so` hash is exactly
`5ff20ae94d92779c87a9425097c88621063b7dc52387be7a61a449a77da0ab80`.
A separately rebuilt library will require verification and a deliberate
change to this guard followed by fresh testing. The original custom op is
registered before NPU initialization, and both namespaces must coexist.

Supply the trusted input directory separately. The files are not generated or
redistributed here, and are loaded on CPU with `weights_only=True`. Their
required SHA-256 values are enforced in the scripts:

- `shared.pt`: `8aecdbfc648236ef851dbe317811028ca5c652ab98e4f426b11525623f22abd8`
- `cache-hit95.pt`: `e8c5257b1c4889e84d4876766dc9a26cd3ced7fd171629bd047c24237a2e46d6`
- `cache-hit91.pt`: `ebb129b3607d41d104a675ec29d75efe283b7ec903f14d442b2cb994b5e02c7f`

The hit rate is per-query occurrence count; repeated sources selected by
multiple routes are counted in each route. The experiments check file and
input hashes, Top-K sets, per-route and union misses, slots and resident-map
bijections. Each timed LIM invocation restores the selected pool outside its
event interval. Run only on an idle, explicitly assigned device and configure
`PYTHONPATH`, `LD_LIBRARY_PATH` and the vLLM custom OPP for your environment.
The program writes new outputs only and records device, environment, tensor
addresses, loaded library paths/hashes and raw samples.

```bash
export INPUT_DIR=/absolute/path/to/frozen-inputs
export OUTPUT_DIR=/absolute/path/to/new-results
export ASCEND_RT_VISIBLE_DEVICES=<idle-physical-device>
# Source the matching Ascend environment and set the vLLM and SGLang package
# paths for the installed, matching custom OPP and kernel library.
mkdir -p "$OUTPUT_DIR"
for case in 95 91; do
  python3 benchmark/lim_frozen_ab/aligned_bench.py \
    --input-dir "$INPUT_DIR" --case "$case" \
    --output "$OUTPUT_DIR/aligned_${case}.json" \
    --profile-dir "$OUTPUT_DIR/aligned_${case}.profile" \
    --warmup 10 --iterations 50 --rounds 3 --timing both
done
python3 benchmark/lim_frozen_ab/pool_ablation.py \
  --input-dir "$INPUT_DIR" --case 95 --output "$OUTPUT_DIR/pool_95.json" \
  --profile-dir "$OUTPUT_DIR/pool_95.profile" \
  --warmup 10 --iterations 50 --rounds 3
```

The strict benchmark alternates original and ported operator order per sample;
Graph and eager each have 150 samples per backend. The layout experiment
captures one graph per backend and rotates four combinations: each backend with
frozen and low-position nonselected fillers. It preserves selected hit slots,
missing sets and cache capacity. Filler identities, their scores, and filler
slot assignments change together, so the result does not isolate any one of
those factors. Its Graph measurements have 150 samples per combination.
Profiling performs three separate calls per combination and is diagnostic,
not part of the event statistics. Graph capture excludes pool reset.

The exact scripts used for the measurements had SHA-256 values
`24c2bcfacf58e9f31ef3a11feae8f6abaa0e563abf3d5281000f682d20f6b63d`
(strict) and `b6f0a71049fca7ad68f999bd330942669803134146293ccac3e410c97e824e08`
(layout). The repository copies were formatted after measurement; AST equality
to those executed scripts was checked, but the formatted copies have not
been separately run on device.
