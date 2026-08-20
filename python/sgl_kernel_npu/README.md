<h2 align="left">
SGLang Kernels NPU
</h2>

## Introduction
SGLang Kernels for Ascend NPU

## Software and hardware
Supported Hardware Models: Ascend 910B, Ascend 910C, and Ascend 950 series products
(generic Ascend 950 builds use the 910C compatibility target; an explicit A5
compiler target enables the native Ascend 950 kernel build — see below)
Platform: aarch64/x86
Supporting Software
- Driver Ascend HDK 25.0.RC1.1, CANN 8.3.RC1 or later versions (refer to the "[CANN Software Installation Guide](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)" to install the CANN development kit package, as well as the supporting firmware and drivers)
- Before installing CANN software, you need to install the relevant [dependency list](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0045.html?Mode=PmIns&InstallType=netconda&OS=openEuler&Software=cannToolKit)
- Python >= 3.7
- pybind11 (install via `pip install pybind11`)

## Quick Start
### Compile and Run
1. Prepare the CANN environment variables (modify according to the installation path)
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. Build the project
Executing the engineering build script build.sh
```bash
# Detect the local NPU and build for it.
bash build.sh -a kernels

# Or name the SoC explicitly.
bash build.sh -a kernels 910
bash build.sh -a kernels 950
bash build.sh -a kernels Ascend950PR_9599
```

Omitting the SoC queries the local device with `npu-smi`. Hosts without an NPU —
build containers, for instance — fall back to `Ascend910_9382`, which is what
every target used to default to unconditionally, so container builds are
unaffected. Detection only picks a default: an explicit argument always wins.

`910B` (A2), `910`/`910C` (A3) and `950` (A5) select the SoC family. A generic or
auto-detected A5 target keeps the 910C compatibility compile path. Kernel-only
builds preserve explicit compiler targets such as `Ascend950PR_9599` and pass
them to AscendC. Both forms select the same Ascend 950 Gemma provider for the
wheel.

Every target exposes `sgl_kernel_npu.norm.gemma_rmsnorm`. The wheel build binds
that stable API to native `torch_npu` Gemma RMSNorm on Ascend 910 and to
standard ACLNN RMSNorm with `1 + weight` on Ascend 950. SGLang does not perform
runtime SoC detection for this operator.

Because the provider is chosen at build time, the source tree ships no
`norm/gemma_rmsnorm.py` — it only exists inside a built wheel. Running from a
source checkout (or `pip install -e .`) therefore raises `ImportError` for that
module rather than silently defaulting to the 910 operator.

### Installation
1. Pip install the `.whl` file into your Python environment
```bash
pip install output/sgl_kernel_npu*.whl

# (Optional) Confirm whether the import can be successfully
python -c "import sgl_kernel_npu; print(sgl_kernel_npu.__path__)"
```

2. Execute the environment variables for CANN (modify according to the installation path)
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
3. In the Python project, import `sgl_kernel_npu`.

### Test
Execute sgl_kernel_npu test scripts, for example
```bash
python3 tests/python/sgl_kernel_npu/test_hello_world.py
```
