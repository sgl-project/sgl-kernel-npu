# Contribution Guide

Welcome to **SGL-KERNEL-NPU**! We appreciate your interest in contributing. This guide provides a concise overview of how to set up your environment, run tests, build documentation, and open a Pull Request (PR). Whether you’re fixing a small bug or developing a major feature, we encourage following these steps for a smooth contribution process.

**SGL-KERNEL-NPU** is a kernel library for LLM inference engines, which provides optimized compute primitives (including Ascendc and Triton) especially for engines running on NPU. It has been used by [SGLang](https://github.com/sgl-project/sglang).

## Install SGL-KERNEL-NPU from Source

### Fork and clone the repository

**Note**: New contributors do **not** have the write permission to push to the official SGL-KERNEL-NPU repo. Please fork the repository under your GitHub account, then clone your fork locally.

```bash
git clone https://github.com/<your_user_name>/sgl-kernel-npu.git
```

### Build from source

Refer to [Install SGL-KERNEL-NPU from Source](../../python/sgl_kernel_npu/README.md).

## Ascend C Kernel Contribution

### Steps to add a new Ascend C kernel:

1. Implement the kernel in [csrc](https://github.com/sgl-project/sgl-kernel-npu/tree/main/csrc). You can start by reading the [helloworld](https://github.com/sgl-project/sgl-kernel-npu/tree/main/csrc/helloworld) kernel, a simple example of adding two bf16 vectors. If you are new to this, the [Ascend C Kernel Development Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html) could also be helpful.

2. Expose the interface in [include/sgl_kernel_npu_ops.h](https://github.com/sgl-project/sgl-kernel-npu/blob/main/include/sgl_kenel_npu_ops.h)

3. Create torch extension in [pytorch_extensions.cpp](https://github.com/sgl-project/sgl-kernel-npu/blob/main/csrc/pytorch_extensions.cpp)

4. Update [CMakeLists.txt](https://github.com/sgl-project/sgl-kernel-npu/blob/main/csrc/CMakeLists.txt) to include new kernel source

## Triton Kernel Contribution

Triton kernels are located at [sgl_kernel_npu](https://github.com/sgl-project/sgl-kernel-npu/tree/main/python/sgl_kernel_npu/sgl_kernel_npu)

## Development Tips

- How to write schema: [Schema reference](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func)

   ```cpp
   // We need def with schema here for torch.compile
   m.def("helloworld(Tensor x, Tensor y) -> Tensor");
   m.impl("helloworld", TORCH_FN(sglang::npu_kernel::helloworld));
   ```

## Run and add unit tests

If you add a new feature or fix a bug, please add corresponding unit tests [test](https://github.com/sgl-project/sgl-kernel-npu/tree/main/tests/python/sgl_kernel_npu) to ensure coverage and prevent regression.
SGL-KERNEL-NPU uses Python's built-in [unittest](https://docs.python.org/3/library/unittest.html) framework.

## Format code with pre-commit

We use [pre-commit](https://pre-commit.com/) to maintain consistent code style checks. Before pushing your changes, please run:

```bash
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
```

- **`pre-commit run --all-files`** manually runs all configured checks, applying fixes if possible. If it fails the first time, re-run it to ensure lint errors are fully resolved. Make sure your code passes all checks **before** creating a Pull Request.
- **Do not commit** directly to the `main` branch. Always create a new branch (e.g., `feature/my-new-feature`), push your changes, and open a PR from that branch.



Thank you for your interest in SGL-KERNEL-NPU. Happy coding!
