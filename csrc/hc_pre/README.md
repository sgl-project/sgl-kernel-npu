# HcPre migration

Imported from the clean local `vllm-ascend` checkout at commit
`9f9da5562089280a7f01c3d03e855b258e372807`:

- `csrc/moe/hc_pre/op_host/` and `op_kernel/`
- HcPre bindings in `csrc/torch_binding.cpp` and `csrc/torch_binding_meta.cpp`
- `tests/e2e/nightly/single_node/ops/singlecard_ops/test_npu_hc_pre.py`

The computation headers are copied unchanged, including license notices.
The binding retains its upstream Apache-2.0 notice; the Ascend C files retain
CANN Open Software License Agreement Version 2.0 notices.

## Engineering adaptations

- Build a dedicated `hc_pre_kernel` with `ascendc_library`, preserving manual
  synchronization flags and the mixed AIC/AIV task. The upstream A2/A3 and A5
  implementation branches are retained; no OPP installation is needed.
- Adapt the op definition and tiling context to the project's `ge_helper`.
  A local context preserves nullable optional shapes and launch metadata.
  Platform information comes from `PlatformAscendCManager`; tiling formulas
  and workspace size formulas are unchanged.
- Supply the OPP-generated device tiling structure explicitly: 53 fields,
  416 bytes, in upstream order. Pass the tiling key as a launch argument and
  copy tiling from GM into local memory. BF16 is the sole upstream input dtype.
- Register `torch.ops.npu.npu_hc_pre_v2` and `npu_hc_pre_v3`, including Meta
  implementations with upstream symbolic output shapes and defaults.
- Use the existing device-specific tiling cache for NPU graph capture. Run an
  eager warmup for each shape/attribute/premix configuration before capture.
- Preserve all five upstream precision cases, CPU reference, and tolerances;
  adapt them to `unittest`. Add Meta checks and graph replay checks for both
  input ranks, v2, and v3 with and without external `pre_mix`.

## NPU validation

On an Ascend machine with the project's CANN and torch-npu environment:

```bash
# A3
bash build.sh -a kernels Ascend910_9382
# A5 (run on the A5 machine, instead of the A3 command above)
bash build.sh -a kernels Ascend950PR_9599
pip install --force-reinstall --no-deps output/sgl_kernel_npu*.whl
python3 tests/python/sgl_kernel_npu/test_npu_hc_pre.py
```

Run separately on A3 and A5 to validate both implementation branches.
The kernels build rejects the DeepEP alias `Ascend950`; use the actual
AscendC-supported SoC name shown above for A5.
The installed wheel must be rebuilt and reinstalled before running these tests.
CPU/static checks do not establish kernel precision or graph replay correctness.
