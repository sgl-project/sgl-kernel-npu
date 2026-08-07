# mHC AscendC custom operators

This directory contains the minimal CANN custom-operator project used to build
`HcPre` and `HcPost`. The operator sources were imported from the CANN
`cann-recipes-infer` repository and are referenced through `src/hc_pre` and
`src/hc_post`.

Build the A2 package from the repository root:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
scripts/build_mhc_custom_ops.sh ascend910b
```

The generated installer is written to:

```text
output/sgl_kernel_npu_mhc_ops-ascend910b-linux.aarch64.run
```

Install it and expose the custom OPP paths before starting SGLang:

```bash
output/sgl_kernel_npu_mhc_ops-ascend910b-linux.aarch64.run \
  --quiet \
  --install-path=/usr/local/Ascend/ascend-toolkit/latest/opp
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
```
