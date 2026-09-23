# FuseEP per-token dequantization mask regression

This NPU test includes the production GEMM2 epilogue. It replaces only the
HCCL peer-address lookup with a local GM buffer, so it needs one device and
no distributed initialization. It does not load or install a DeepEP wheel.

The test fills UB with NaNs and runs the epilogue with both a full mask and
a mask enabling only eight FP32 lanes. Finite FP16 inputs and exact integer
per-token scales must produce the expected BF16 outputs in every lane.
The 18 cases cover 1/3/16 rows and 64/128/256 columns, including a partial
256-column UB tile. A non-finite value or any unequal element fails the test.

Build after the DeepEP build has fetched the platform's CATLASS dependency:

```bash
cmake -S tests/ascendc/fuseep_epilogue -B build/fuseep-epilogue \
  -DASC_DIR="$ASCEND_HOME_PATH/compiler/tikcpp/ascendc_kernel_cmake" \
  -DCATLASS_ARCH=3510
cmake --build build/fuseep-epilogue -j2
./build/fuseep-epilogue/test_epilogue 0
```

Use `-DCATLASS_ARCH=2201` in a separate build directory on A3. The last
argument selects the device. `CATLASS_INCLUDE_DIR` and `EPILOGUE_INCLUDE_DIR`
can point to other header checkouts for before/after comparisons.

On A3, the previous epilogue passes the nine full-mask cases but leaves
NaNs in all nine restricted-mask cases. The corrected epilogue passes all
18 cases. Both variants also compile for Ascend950 with CANN 9.1. These
results establish the mask-state defect; they do not validate the complete
A5 distributed operator or the SGLang regression.
