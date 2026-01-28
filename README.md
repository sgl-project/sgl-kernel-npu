# SGLang-Kernel-NPU


## Introduction

**SGLang-Kernel-NPU** is the official kernel library of the [SGLang](https://github.com/sgl-project/sglang) framework for Ascend NPU. It delivers high-performance, production-ready compute primitives optimized for large language model (LLM) inference on Ascend hardware.

<div align="center">

[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/sgl-project/sgl-kernel-npu)

</div>

The library consists of two main components:
- **DeepEP-Ascend**: Ascend implementation of [DeepEP](https://github.com/deepseek-ai/DeepEP), providing highly optimized Expert Parallelism (EP) communication kernels for Mixture-of-Experts (MoE) models.
- **SGLang-Kernel-NPU**: A comprehensive collection of optimized inference kernels including attention mechanisms, normalization, activation functions, LoRA adapters, and more.

For contribution guidelines, please refer to the [Contribution Guide](docs/developer_guide/contribution_guide.md).


## Features

### DeepEP-Ascend

DeepEP-Ascend provides optimized all-to-all communication kernels for Expert Parallelism in MoE models.

**Communication Modes:**
- **Normal Mode**: High-throughput dispatch and combine operations for training and prefill phases (up to 65536 tokens/batch for A3 and 8192 tokens/batch for A2)
- **Low-Latency Mode**: Optimized for production inference with small batch sizes (128 tokens/batch), achieving sub-150us latency

**Key Capabilities:**
- Token dispatch and combine with automatic load balancing
- Fused MoE computation (`fused_deep_moe`)
- A3 full-mesh HCCS communication and A2 Intranode HCCS + internode RDMA communication
- INT8/FP8/BF16 quantization for reduced memory bandwidth

### SGLang-Kernel-NPU

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


## Quick Start

DeepEP-Ascend: Ascend Implementation of DeepEP. [README](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md)

SGLang-Kernel-NPU: Other SGLang Kernels for Ascend NPU. [README](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/README.md)


## DeepEP-Ascend Performance

### Normal Kernels with Pure HCCS

We test normal kernels on A3 384 SuperPOD, following the DeepSeek-V3/R1 pretraining setting (4096 tokens per batch, 7168 hidden size, top-8 experts, INT8 dispatching and BF16 combining).

| Type      | Dispatch #EP | Bottleneck bandwidth | Combine #EP | Bottleneck bandwidth |
| --------- | ------------ | -------------------- | ----------- | -------------------- |
| Intranode | 8            | 146 GB/s (HCCS)      | 8           | 125 GB/s (HCCS)      |
| Intranode | 16           | 107 GB/s (HCCS)      | 16          | 103 GB/s (HCCS)      |
| Intranode | 32           | 102 GB/s (HCCS)      | 32          | 95 GB/s (HCCS)       |
| Intranode | 64           | 81 GB/s (HCCS)       | 64          | 91 GB/s (HCCS)       |
| Intranode | 128          | 57 GB/s (HCCS)       | 128         | 81 GB/s (HCCS)       |

### Low-Latency Kernels with Pure HCCS

We test low-latency kernels on A3 384 SuperPOD, following a typical DeepSeek-V3/R1 production setting (128 tokens per batch, 7168 hidden size, top-8 experts, INT8 dispatching and BF16 combining).

| Dispatch #EP | Latency | Bandwidth      | Combine #EP | Latency | Bandwidth       |
| ------------ | ------- | -------------- | ----------- | ------- | --------------- |
| 8            | 132 us  | 58 GB/s (HCCS) | 8           | 126 us  | 116 GB/s (HCCS) |
| 16           | 139 us  | 55 GB/s (HCCS) | 16          | 135 us  | 109 GB/s (HCCS) |
| 32           | 153 us  | 49 GB/s (HCCS) | 32          | 151 us  | 97 GB/s (HCCS)  |
