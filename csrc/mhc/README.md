# mHC AscendC custom operators

`hc_pre` and `hc_post` contain the operator sources imported from
`cann-recipes-infer`. The PyTorch-to-ACLNN bridge is under `bindings`.

The custom-OPP project is generated at build time with `msopgen`, following the
same approach as DeepEP. Only the MHC-specific host and kernel CMake files are
kept under `ops`; generated CMake infrastructure is not checked into the repo.

From the repository root, build the normal kernel wheel with:

```bash
bash build.sh -a kernels
```

The generated vendor OPP is bundled into the `sgl_kernel_npu` wheel, so no
separate system-wide custom-operator installation is needed.
