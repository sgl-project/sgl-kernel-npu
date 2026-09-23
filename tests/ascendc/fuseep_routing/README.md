# FuseEP routing and INT8 input quantization diagnostic

This single-device test includes the production full-load routing/quantization
kernel and its host tiler. It needs CANN 9.1, but no DeepEP wheel or HCCL group.
Automatic synchronization is disabled, matching the operator build.

It poisons UB before running the kernel and verifies each routed INT8 element,
each per-token FP32 scale, expert counts and destination indices against an
independent CPU reference. Inputs use exactly representable BF16 values with
different scales per row. All comparisons are exact. Shapes cover 1/17/128
tokens, hidden size 2048, 128 experts and top-k 8, with three repeated calls.
This exercises the full-load path selected by these shapes, not the separate
gather implementation.

```bash
cmake -S tests/ascendc/fuseep_routing -B build/fuseep-routing \
  -DASC_DIR="$ASCEND_HOME_PATH/compiler/tikcpp/ascendc_kernel_cmake" \
  -DCATLASS_ARCH=3510
cmake --build build/fuseep-routing -j2
./build/fuseep-routing/test_routing_quant 0
```

Use `-DCATLASS_ARCH=2201` in a separate build directory on A3. The final
argument selects the device. `ROUTING_INCLUDE_DIR` can point to the routing
headers from an earlier commit for a before/after comparison.

Both the old and synchronized kernels passed on A3, and both compiled for
Ascend950. The A3 result alone does not establish that missing V-to-S
synchronization caused the reported A5 numerical error. An A5 failure here
isolates input routing/quantization from matrix multiplication, SwiGLU and
cross-rank transport; an A5 pass does not validate those later stages.
