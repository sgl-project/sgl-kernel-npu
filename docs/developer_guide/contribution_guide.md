# Contribution Guide

Welcome to **SGL-KERNEL-NPU**! We appreciate your interest in contributing. This guide provides a concise overview of how to set up your environment, run tests, build documentation, and open a Pull Request (PR). Whether you’re fixing a small bug or developing a major feature, we encourage following these steps for a smooth contribution process.

## Install SGL-KERNEL-NPU from Source

### Fork and clone the repository

**Note**: New contributors do **not** have the write permission to push to the official SGL-KERNEL-NPU repo. Please fork the repository under your GitHub account, then clone your fork locally.

```bash
git clone https://github.com/<your_user_name>/sgl-kernel-npu.git
```

### Build from source

Refer to [Install SGL-KERNEL-NPU from Source](../../python/sgl_kernel_npu/README.md).

## Format code with pre-commit

We use [pre-commit](https://pre-commit.com/) to maintain consistent code style checks. Before pushing your changes, please run:

```bash
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
```

- **`pre-commit run --all-files`** manually runs all configured checks, applying fixes if possible. If it fails the first time, re-run it to ensure lint errors are fully resolved. Make sure your code passes all checks **before** creating a Pull Request.
- **Do not commit** directly to the `main` branch. Always create a new branch (e.g., `feature/my-new-feature`), push your changes, and open a PR from that branch.

## Run and add unit tests

If you add a new feature or fix a bug, please add corresponding unit tests to ensure coverage and prevent regression.
SGL-KERNEL-NPU uses Python's built-in [unittest](https://docs.python.org/3/library/unittest.html) framework

## Write documentations


## Test the accuracy


## Benchmark the speed


## Request a review


## General code style


## How to update sgl-kernel


## Tips for newcomers

Thank you for your interest in SGL-KERNEL-NPU. Happy coding!
