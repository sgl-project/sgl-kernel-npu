# SGLang-Kernel-NPU


## Introduction

**SGLang-Kernel-NPU** is the official kernel library of the [SGLang](https://github.com/sgl-project/sglang) framework for Ascend NPU. It delivers high-performance, production-ready compute primitives optimized for large language model (LLM) inference on Ascend hardware.

<div align="center">

[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/sgl-project/sgl-kernel-npu)

</div>

The library provides a comprehensive collection of optimized inference kernels for Ascend NPU, including attention mechanisms, normalization, activation functions, LoRA adapters, and more.

For contribution guidelines, please refer to the [Contribution Guide](docs/developer_guide/contribution_guide.md).


## Features

SGLang-Kernel-NPU provides a comprehensive set of optimized inference kernels:

**Attention:**
- Multi-Latent Attention (MLA) with Paged KV Cache support
- Grouped Query Attention (GQA)
- Decode Attention with optimized memory access patterns

**Flash Linear Attention (FLA):**
- Gated Delta Rule implementation
- Chunk-based operations for efficient memory usage

**Normalization:**
- RMSNorm
- Target-specific Gemma RMSNorm provider (native on 910, ACLNN on 950)
- Fused Add + RMSNorm + Bias
- Split QKV + RMSNorm + RoPE fusion

**Activation Functions:**
- SwiGLU
- Quantized SwiGLU (INT8)

**LoRA Adapters:**
- BGMV expand/shrink
- SGMV expand/shrink
- SGEMMV expand/shrink

**Speculative Decoding:**
- Efficient tree building
- Greedy tree verification

**MLA Preprocessing:**
- End-to-end fusion: RMSNorm → Dequant → MatMul → RoPE → ReshapeAndCache

**Mamba Support:**
- Causal Conv1D for state space models (SSM)

**KV Cache Management:**
- Paged Attention support
- Cache location assignment and update

**Other Utilities:**
- Lightning Indexer for sparse Top-K indexing
- Triangular matrix inverse
- Batch MatMul with transpose


## Accessing Ascend NPU
If you need to access Ascend NPU computing resources for development or testing, please visit the [HiDevLab - Online Development](https://hidevlab.huawei.com/online-develop-intro) page on the Huawei HiDevLab platform to apply for free access.


## Quick Start

Before building from source, initialize the submodule(s):
```bash
git submodule update --init --recursive
```

See [python/sgl_kernel_npu/README.md](python/sgl_kernel_npu/README.md) for the build and install instructions.
