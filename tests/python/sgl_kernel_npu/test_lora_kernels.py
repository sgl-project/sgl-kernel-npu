import random
import unittest

import sgl_kernel_npu
import torch
import torch_npu
from utils import reference_sgmv_expand, reference_sgmv_shrink

torch.set_printoptions(threshold=float("inf"))


class TestLoraKernels(unittest.TestCase):
    def test_sgemmv_shrink(self):
        batch_size = 2
        input_dim = 1024
        num_loras = 3
        dtype = torch.float16
        device_dtype = torch.float16

        possible_lora_ranks = [16, 32, 64]
        lora_ranks = random.sample(
            possible_lora_ranks,
            counts=[num_loras] * len(possible_lora_ranks),
            k=num_loras,
        )

        max_lora_rank = max(lora_ranks)

        possible_lora_scaling = [0.25, 0.5, 1.0, 2.0, 4.0]
        lora_scaling = random.sample(
            possible_lora_scaling,
            counts=[num_loras] * len(possible_lora_scaling),
            k=num_loras,
        )

        inputs = torch.randn(batch_size, input_dim, dtype=dtype)
        lora_a_weights = torch.randn(num_loras, max_lora_rank, input_dim, dtype=dtype)
        lora_indices_tensor = torch.randint(
            num_loras, (batch_size,), dtype=torch.int32, device="cpu"
        )
        seq_len_tensor = torch.ones(batch_size, dtype=torch.int32, device="cpu")
        lora_ranks_tensor = torch.tensor(lora_ranks, dtype=torch.int32, device="cpu")
        lora_scaling_tensor = torch.tensor(
            lora_scaling, dtype=torch.float32, device="cpu"
        )

        expect_output = reference_sgmv_shrink(
            inputs,
            lora_a_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            lora_scaling_tensor,
        )

        inputs_npu = inputs.to(dtype=device_dtype, device="npu")
        lora_a_weights_npu = lora_a_weights.to(dtype=device_dtype, device="npu")
        lora_indices_tensor_npu = lora_indices_tensor.to(device="npu")
        seq_len_tensor_npu = seq_len_tensor.to(device="npu")
        lora_ranks_tensor_npu = lora_ranks_tensor.to(device="npu")
        lora_scaling_tensor_npu = lora_scaling_tensor.to(device="npu")

        actual_output = torch.zeros(
            (batch_size, max_lora_rank), dtype=device_dtype, device=inputs_npu.device
        )

        torch.ops.npu.sgemmv_shrink(
            inputs_npu,
            lora_a_weights_npu,
            lora_indices_tensor_npu,
            seq_len_tensor_npu,
            lora_ranks_tensor_npu,
            lora_scaling_tensor_npu,
            actual_output,
        )

        actual_output_cpu = actual_output.to(dtype=dtype, device="cpu")

        self.assertTrue(
            torch.allclose(actual_output_cpu, expect_output, atol=1e-3, rtol=1e-3)
        )

    def test_sgemmv_expand(self):
        batch_size = 4
        output_dim = 1024
        num_loras = 8
        dtype = torch.float16
        device_dtype = torch.float16

        possible_lora_ranks = [16, 32, 64]
        lora_ranks = random.sample(
            possible_lora_ranks,
            counts=[num_loras] * len(possible_lora_ranks),
            k=num_loras,
        )

        max_lora_rank = max(lora_ranks)

        inputs = torch.randn(batch_size, max_lora_rank, dtype=dtype)
        lora_b_weights = torch.randn(num_loras, output_dim, max_lora_rank, dtype=dtype)
        lora_ranks_tensor = torch.tensor(lora_ranks, dtype=torch.int32, device="cpu")
        seq_len_tensor = torch.ones(batch_size, dtype=torch.int32, device="cpu")
        lora_indices_tensor = torch.randint(
            num_loras, (batch_size,), dtype=torch.int32, device="cpu"
        )
        slice_offsets = torch.tensor([0, output_dim], dtype=torch.int32, device="cpu")

        expect_output = reference_sgmv_expand(
            inputs,
            lora_b_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            slice_offsets,
        )

        inputs_npu = inputs.to(dtype=device_dtype, device="npu")
        lora_b_weights_npu = lora_b_weights.to(dtype=device_dtype, device="npu")
        lora_indices_tensor_npu = lora_indices_tensor.to(device="npu")
        seq_len_tensor_npu = seq_len_tensor.to(device="npu")
        lora_ranks_tensor_npu = lora_ranks_tensor.to(device="npu")
        slice_offsets_npu = slice_offsets.to(device="npu")

        actual_output = torch.zeros(
            (batch_size, output_dim), dtype=device_dtype, device=inputs_npu.device
        )

        torch.ops.npu.sgemmv_expand(
            inputs_npu,
            lora_b_weights_npu,
            lora_indices_tensor_npu,
            seq_len_tensor_npu,
            lora_ranks_tensor_npu,
            slice_offsets_npu,
            actual_output,
        )

        actual_output_cpu = actual_output.to(device="cpu")

        self.assertTrue(
            torch.allclose(actual_output_cpu, expect_output, atol=1e-3, rtol=1e-3)
        )


class TestSgmvKernels(unittest.TestCase):
    """sgmv_shrink / sgmv_expand tests.

    The kernels follow the same dtype protocol as ``AscendLoRABackend``
    (python/sglang/srt/lora/backend/ascend_backend.py):
    lora_indices/seq_len are **int32**, and x / weights / y all share the
    model dtype (fp16/bf16). ``test_sgmv_shrink_backend_dtype`` exercises the
    exact calling convention of the backend's former run_lora_a_sgemm flow
    (kernel scale=1.0, per-lora scaling applied in Python afterwards).
    """

    def test_sgmv_shrink(self):
        batch_size = 2
        input_dim = 1024
        num_loras = 3
        dtype = torch.float16
        device_dtype = torch.float16
        max_lora_rank = 64

        inputs = torch.randn(batch_size, input_dim, dtype=dtype)
        lora_a_weights = torch.randn(num_loras, max_lora_rank, input_dim, dtype=dtype)
        lora_indices_tensor = torch.randint(
            num_loras, (batch_size,), dtype=torch.int32, device="cpu"
        )
        seq_len_tensor = torch.ones(batch_size, dtype=torch.int32, device="cpu")
        lora_ranks_tensor = torch.full(
            (num_loras,), max_lora_rank, dtype=torch.int32, device="cpu"
        )
        lora_scaling_tensor = torch.ones(num_loras, dtype=torch.float32, device="cpu")

        expect_output = reference_sgmv_shrink(
            inputs,
            lora_a_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            lora_scaling_tensor,
        )

        inputs_npu = inputs.to(dtype=device_dtype, device="npu")
        lora_a_weights_npu = lora_a_weights.to(dtype=device_dtype, device="npu")
        lora_indices_tensor_npu = lora_indices_tensor.to(device="npu")
        seq_len_tensor_npu = seq_len_tensor.to(device="npu")

        actual_output = torch.zeros(
            (batch_size, max_lora_rank), dtype=device_dtype, device=inputs_npu.device
        )
        torch.ops.npu.sgmv_shrink(
            inputs_npu,
            lora_a_weights_npu,
            lora_indices_tensor_npu,
            seq_len_tensor_npu,
            actual_output,
            1.0,
        )

        actual_output_cpu = actual_output.to(device="cpu")

        self.assertTrue(
            torch.allclose(actual_output_cpu, expect_output, atol=1e-3, rtol=1e-3)
        )

    def test_sgmv_expand(self):
        batch_size = 4
        output_dim = 1024
        num_loras = 8
        dtype = torch.float16
        device_dtype = torch.float16
        max_lora_rank = 64

        inputs = torch.randn(batch_size, max_lora_rank, dtype=dtype)
        lora_b_weights = torch.randn(num_loras, output_dim, max_lora_rank, dtype=dtype)
        lora_indices_tensor = torch.randint(
            num_loras, (batch_size,), dtype=torch.int32, device="cpu"
        )
        seq_len_tensor = torch.ones(batch_size, dtype=torch.int32, device="cpu")
        lora_ranks_tensor = torch.full(
            (num_loras,), max_lora_rank, dtype=torch.int32, device="cpu"
        )
        slice_offsets = torch.tensor([0, output_dim], dtype=torch.int32, device="cpu")

        expect_output = reference_sgmv_expand(
            inputs,
            lora_b_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            slice_offsets,
        )

        inputs_npu = inputs.to(dtype=device_dtype, device="npu")
        lora_b_weights_npu = lora_b_weights.to(dtype=device_dtype, device="npu")
        lora_indices_tensor_npu = lora_indices_tensor.to(device="npu")
        seq_len_tensor_npu = seq_len_tensor.to(device="npu")

        actual_output = torch.zeros(
            (batch_size, output_dim), dtype=device_dtype, device=inputs_npu.device
        )
        torch.ops.npu.sgmv_expand(
            inputs_npu,
            lora_b_weights_npu,
            lora_indices_tensor_npu,
            seq_len_tensor_npu,
            actual_output,
            0,  # slice_offset
            output_dim,  # slice_size
        )

        actual_output_cpu = actual_output.to(device="cpu")

        self.assertTrue(
            torch.allclose(actual_output_cpu, expect_output, atol=1e-3, rtol=1e-3)
        )

    def test_sgmv_shrink_backend_dtype(self):
        # Contract test: sgmv_shrink now speaks the backend's convention
        # directly (int32 indices/seg_lens, y in the model dtype), so this
        # mirrors AscendLoRABackend's former run_lora_a_sgemm flow:
        #   - weight_indices / seg_lens are int32  (prepare_lora_batch)
        #   - the y buffer is dtype=x.dtype (fp16)
        #   - scale=1.0; per-lora scaling is applied in Python afterwards
        batch_size = 2
        input_dim = 1024
        num_loras = 3
        dtype = torch.float16
        device_dtype = torch.float16
        max_lora_rank = 64

        possible_lora_scaling = [0.25, 0.5, 1.0, 2.0, 4.0]
        lora_scaling = random.sample(
            possible_lora_scaling,
            counts=[num_loras] * len(possible_lora_scaling),
            k=num_loras,
        )

        inputs = torch.randn(batch_size, input_dim, dtype=dtype)
        lora_a_weights = torch.randn(num_loras, max_lora_rank, input_dim, dtype=dtype)
        # int32 -- exactly what AscendLoRABackend.prepare_lora_batch creates
        lora_indices_tensor = torch.tensor([0, 1], dtype=torch.int32, device="cpu")
        seq_len_tensor = torch.ones(batch_size, dtype=torch.int32, device="cpu")
        lora_ranks_tensor = torch.full(
            (num_loras,), max_lora_rank, dtype=torch.int32, device="cpu"
        )
        lora_scaling_tensor = torch.tensor(
            lora_scaling, dtype=torch.float32, device="cpu"
        )

        expect_output = reference_sgmv_shrink(
            inputs,
            lora_a_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            lora_scaling_tensor,
        )

        inputs_npu = inputs.to(dtype=device_dtype, device="npu")
        lora_a_weights_npu = lora_a_weights.to(dtype=device_dtype, device="npu")
        lora_indices_tensor_npu = lora_indices_tensor.to(device="npu")
        seq_len_tensor_npu = seq_len_tensor.to(device="npu")

        # fp16 y buffer -- exactly what the backend allocates (dtype=x.dtype)
        actual_output = torch.zeros(
            (batch_size, max_lora_rank), dtype=dtype, device=inputs_npu.device
        )
        torch.ops.npu.sgmv_shrink(
            inputs_npu,
            lora_a_weights_npu,
            lora_indices_tensor_npu,
            seq_len_tensor_npu,
            actual_output,
            1.0,
        )

        # per-lora scaling applied in Python, mirroring run_lora_a_sgemm
        scaling = (
            lora_scaling_tensor.to(device="npu")
            .gather(0, lora_indices_tensor_npu)
            .unsqueeze(-1)
        )
        actual_output *= scaling
        actual_output_cpu = actual_output.to(device="cpu")

        self.assertTrue(
            torch.allclose(actual_output_cpu, expect_output, atol=1e-3, rtol=1e-3),
            "Backend dtype convention (int32 indices + fp16 y buffer) "
            "mismatches the sgmv_shrink kernel protocol.",
        )


class TestSgemmcKernels(unittest.TestCase):
    """sgemmc_shrink / sgemmc_expand correctness tests.

    These are the cube-unit (``REGIST_MATMUL_OBJ``) matmul kernels meant to
    replace the naive vector-unit ``sgmv_*`` kernels in ``AscendLoRABackend``.

    Protocol (see op_kernel/sgemmc_*_kernel.cpp):
      - lora_indices / seq_len / lora_ranks / slice_offsets are **int32**;
      - x / weight / y are half/bf16 (Y_T = X_T = scalar_t, no fp32 buffer);
      - lora_scales is fp32 (the LoRABatchInfo convention) and is applied
        **inside** the shrink kernel (per-lora), so no Python-side scaling
        is needed;
      - shrink weight layouts: [num_loras, max_rank, hidden] or
        [num_loras, 1, max_rank, hidden]; with slice_count > 1 the per-lora
        region is [slices * max_rank, hidden] (natural layout works when all
        loras share the same rank, which is what the multi-slice test uses);
      - expand weight: [num_loras, output_dim, max_rank], slices addressed via
        slice_offsets; each slice block is [slice_size, max_rank].
    """

    def test_sgemmc_shrink(self):
        total_tokens = 5
        input_dim = 1024
        num_loras = 3
        dtype = torch.float16
        device_dtype = torch.float16
        max_lora_rank = 64

        possible_lora_ranks = [16, 32, 64]
        lora_ranks = random.sample(
            possible_lora_ranks,
            counts=[num_loras] * len(possible_lora_ranks),
            k=num_loras,
        )

        possible_lora_scaling = [0.25, 0.5, 1.0, 2.0, 4.0]
        lora_scaling = random.sample(
            possible_lora_scaling,
            counts=[num_loras] * len(possible_lora_scaling),
            k=num_loras,
        )

        seq_lens = [2, 3]
        total_tokens = sum(seq_lens)
        lora_indices = [0, 1]

        inputs = torch.randn(total_tokens, input_dim, dtype=dtype)
        # reference layout: [num_loras, max_rank, hidden], zero-padded beyond rank
        lora_a_weights = torch.zeros(num_loras, max_lora_rank, input_dim, dtype=dtype)
        for idx, rank in enumerate(lora_ranks):
            lora_a_weights[idx, :rank] = torch.randn(rank, input_dim, dtype=dtype)

        expect_output = reference_sgmv_shrink(
            inputs,
            lora_a_weights,
            torch.tensor(lora_indices, dtype=torch.int32, device="cpu"),
            torch.tensor(seq_lens, dtype=torch.int32, device="cpu"),
            torch.tensor(lora_ranks, dtype=torch.int32, device="cpu"),
            torch.tensor(lora_scaling, dtype=torch.float32, device="cpu"),
            num_slices=1,
        )

        # sgemmc accepts [num_loras, 1, max_rank, hidden] (host check passes
        # 3D/4D); with slices=1 the kernel reads [rank_i, hidden] from the
        # start of each lora's block, which is exactly the padded layout above.
        weight = lora_a_weights.unsqueeze(1).contiguous()

        inputs_npu = inputs.to(dtype=device_dtype, device="npu")
        weight_npu = weight.to(dtype=device_dtype, device="npu")
        lora_indices_npu = torch.tensor(lora_indices, dtype=torch.int32, device="npu")
        seq_len_npu = torch.tensor(seq_lens, dtype=torch.int32, device="npu")
        lora_ranks_npu = torch.tensor(lora_ranks, dtype=torch.int32, device="npu")
        lora_scaling_npu = torch.tensor(lora_scaling, dtype=torch.float32, device="npu")

        # y shares the model dtype (Y_T = scalar_t), per the unified protocol
        actual_output = torch.zeros(
            (total_tokens, max_lora_rank), dtype=device_dtype, device=inputs_npu.device
        )
        torch.ops.npu.sgemmc_shrink(
            inputs_npu,
            weight_npu,
            lora_indices_npu,
            seq_len_npu,
            lora_ranks_npu,
            lora_scaling_npu,
            actual_output,
            1,  # slice_count
        )

        self.assertTrue(
            torch.allclose(
                actual_output.to(device="cpu").float(),
                expect_output.float(),
                atol=1e-2,
                rtol=1e-2,
            )
        )

    def test_sgemmc_expand(self):
        total_tokens = 7
        output_dim = 1024
        num_loras = 4
        dtype = torch.float16
        device_dtype = torch.float16
        max_lora_rank = 64

        possible_lora_ranks = [16, 32, 64]
        lora_ranks = random.sample(
            possible_lora_ranks,
            counts=[num_loras] * len(possible_lora_ranks),
            k=num_loras,
        )

        seq_lens = [3, 4]
        total_tokens = sum(seq_lens)
        lora_indices = [0, 1]

        # x is the shrink output: [total_seq, max_rank]
        inputs = torch.randn(total_tokens, max_lora_rank, dtype=dtype)
        # expand weight layout: [num_loras, output_dim, max_rank],
        # only the first `rank` columns are used per lora
        lora_b_weights = torch.zeros(num_loras, output_dim, max_lora_rank, dtype=dtype)
        for idx, rank in enumerate(lora_ranks):
            lora_b_weights[idx, :, :rank] = torch.randn(output_dim, rank, dtype=dtype)

        slice_offsets = torch.tensor([0, output_dim], dtype=torch.int32, device="cpu")

        expect_output = reference_sgmv_expand(
            inputs,
            lora_b_weights,
            torch.tensor(lora_indices, dtype=torch.int32, device="cpu"),
            torch.tensor(seq_lens, dtype=torch.int32, device="cpu"),
            torch.tensor(lora_ranks, dtype=torch.int32, device="cpu"),
            slice_offsets,
        )

        inputs_npu = inputs.to(dtype=device_dtype, device="npu")
        weight_npu = lora_b_weights.to(dtype=device_dtype, device="npu")
        lora_indices_npu = torch.tensor(lora_indices, dtype=torch.int32, device="npu")
        seq_len_npu = torch.tensor(seq_lens, dtype=torch.int32, device="npu")
        lora_ranks_npu = torch.tensor(lora_ranks, dtype=torch.int32, device="npu")
        slice_offsets_npu = slice_offsets.to(device="npu")

        # base (yIn) is zeros -> the op returns just the LoRA delta, which is
        # what reference_sgmv_expand computes when base_output is None.
        actual_output = torch.zeros(
            (total_tokens, output_dim), dtype=device_dtype, device=inputs_npu.device
        )
        out = torch.ops.npu.sgemmc_expand(
            inputs_npu,
            weight_npu,
            lora_indices_npu,
            seq_len_npu,
            lora_ranks_npu,
            slice_offsets_npu,
            actual_output,
        )

        actual = out.to(device="cpu").float()
        expected = expect_output.float()
        self.assertTrue(
            torch.allclose(actual, expected, atol=1e-2, rtol=1e-2),
            f"sgemmc_expand mismatch: max_abs_diff="
            f"{(actual - expected).abs().max().item():.6f}, "
            f"mean_abs_diff={(actual - expected).abs().mean().item():.6f}, "
            f"max_abs_ref={(expected.abs().max().item()):.6f}",
        )

    def test_sgemmc_multi_slice_chain(self):
        # QKV-like pipeline: sgemmc_shrink with slice_count=3, then
        # sgemmc_expand with slice_offsets of 3 slices. All loras share the
        # same rank so the natural [num_loras, 1, slices*max_rank, hidden]
        # weight layout coincides with the kernel's packed-by-rank layout.
        input_dim = 1024
        max_lora_rank = 64
        num_loras = 2
        slices = 3
        output_dims = [512, 512, 512]
        total_output_dim = sum(output_dims)
        dtype = torch.float16
        device_dtype = torch.float16

        seq_lens = [2, 4]
        total_tokens = sum(seq_lens)
        lora_indices = [0, 1]
        lora_ranks = [max_lora_rank, max_lora_rank]  # uniform on purpose
        lora_scaling = [0.5, 2.0]

        inputs = torch.randn(total_tokens, input_dim, dtype=dtype)
        # shrink weights: [num_loras, slices*max_rank, hidden]
        lora_a_weights = torch.randn(
            num_loras, slices * max_lora_rank, input_dim, dtype=dtype
        )
        weight_a_4d = lora_a_weights.unsqueeze(1)  # [num_loras, 1, 3*rank, hidden]

        # expand weights: [num_loras, total_output_dim, max_rank]
        lora_b_weights = torch.randn(
            num_loras, total_output_dim, max_lora_rank, dtype=dtype
        )
        slice_offsets = [0]
        for d in output_dims:
            slice_offsets.append(slice_offsets[-1] + d)
        slice_offsets_t = torch.tensor(slice_offsets, dtype=torch.int32, device="cpu")

        # ---- reference (two-step) ----
        y_shrink_ref = reference_sgmv_shrink(
            inputs,
            lora_a_weights,
            torch.tensor(lora_indices, dtype=torch.int32, device="cpu"),
            torch.tensor(seq_lens, dtype=torch.int32, device="cpu"),
            torch.tensor(lora_ranks, dtype=torch.int32, device="cpu"),
            torch.tensor(lora_scaling, dtype=torch.float32, device="cpu"),
            num_slices=slices,
        )
        expect_final = reference_sgmv_expand(
            y_shrink_ref,
            lora_b_weights,
            torch.tensor(lora_indices, dtype=torch.int32, device="cpu"),
            torch.tensor(seq_lens, dtype=torch.int32, device="cpu"),
            torch.tensor(lora_ranks, dtype=torch.int32, device="cpu"),
            slice_offsets_t,
        )

        # ---- NPU: shrink ----
        inputs_npu = inputs.to(dtype=device_dtype, device="npu")
        weight_a_npu = weight_a_4d.to(dtype=device_dtype, device="npu")
        lora_indices_npu = torch.tensor(lora_indices, dtype=torch.int32, device="npu")
        seq_len_npu = torch.tensor(seq_lens, dtype=torch.int32, device="npu")
        lora_ranks_npu = torch.tensor(lora_ranks, dtype=torch.int32, device="npu")
        lora_scaling_npu = torch.tensor(lora_scaling, dtype=torch.float32, device="npu")

        y_shrink = torch.zeros(
            (total_tokens, slices * max_lora_rank),
            dtype=device_dtype,
            device=inputs_npu.device,
        )
        torch.ops.npu.sgemmc_shrink(
            inputs_npu,
            weight_a_npu,
            lora_indices_npu,
            seq_len_npu,
            lora_ranks_npu,
            lora_scaling_npu,
            y_shrink,
            slices,
        )

        shrink_actual = y_shrink.to(device="cpu").float()
        shrink_expected = y_shrink_ref.float()
        self.assertTrue(
            torch.allclose(shrink_actual, shrink_expected, atol=1e-2, rtol=1e-2),
            f"sgemmc multi-slice shrink mismatch: max_abs_diff="
            f"{(shrink_actual - shrink_expected).abs().max().item():.6f}, "
            f"mean_abs_diff={(shrink_actual - shrink_expected).abs().mean().item():.6f}, "
            f"max_abs_ref={(shrink_expected.abs().max().item()):.6f}",
        )

        # ---- NPU: expand (feeds the shrink output directly) ----
        weight_b_npu = lora_b_weights.to(dtype=device_dtype, device="npu")
        slice_offsets_npu = slice_offsets_t.to(device="npu")

        y_out = torch.zeros(
            (total_tokens, total_output_dim),
            dtype=device_dtype,
            device=inputs_npu.device,
        )
        out = torch.ops.npu.sgemmc_expand(
            y_shrink,
            weight_b_npu,
            lora_indices_npu,
            seq_len_npu,
            lora_ranks_npu,
            slice_offsets_npu,
            y_out,
        )

        final_actual = out.to(device="cpu").float()
        final_expected = expect_final.float()
        self.assertTrue(
            torch.allclose(final_actual, final_expected, atol=1e-2, rtol=1e-2),
            f"sgemmc multi-slice expand mismatch: max_abs_diff="
            f"{(final_actual - final_expected).abs().max().item():.6f}, "
            f"mean_abs_diff={(final_actual - final_expected).abs().mean().item():.6f}, "
            f"max_abs_ref={(final_expected.abs().max().item()):.6f}",
        )


class TestSgemmcExpandRace(unittest.TestCase):
    """Regression test for the sgemmc_expand per-block workspace race.

    The async ``GetTensorC`` staging buffer is padded to ``baseM`` rows (16)
    even though org M is 1 (``singleCoreM=1``). Sizing each block's workspace
    by ``singleCoreM * N`` left the staging 16x too small, so neighbouring
    blocks overwrote each other's C tiles -- visible only with enough blocks
    (>= 7) and nondeterministic across runs. This shape (seq_lens=[3, 4],
    N=1024, 7 blocks) reproduced it; run several times to smoke out races.
    """

    def test_multi_block_expand_race(self):
        output_dim = 1024
        num_loras = 4
        max_lora_rank = 64
        dtype = torch.float16

        g = torch.Generator().manual_seed(0)
        lora_ranks = [16, 32, 64, 16]
        seq_lens = [3, 4]
        total_tokens = sum(seq_lens)
        lora_indices = [0, 1]

        inputs = torch.randn(total_tokens, max_lora_rank, dtype=dtype, generator=g)
        lora_b_weights = torch.zeros(num_loras, output_dim, max_lora_rank, dtype=dtype)
        for idx, rank in enumerate(lora_ranks):
            lora_b_weights[idx, :, :rank] = torch.randn(
                output_dim, rank, dtype=dtype, generator=g
            )

        slice_offsets = torch.tensor([0, output_dim], dtype=torch.int32, device="cpu")
        expect = reference_sgmv_expand(
            inputs,
            lora_b_weights,
            torch.tensor(lora_indices, dtype=torch.int32),
            torch.tensor(seq_lens, dtype=torch.int32),
            torch.tensor(lora_ranks, dtype=torch.int32),
            slice_offsets,
        ).float()

        x_npu = inputs.to(device="npu")
        w_npu = lora_b_weights.to(device="npu")
        lora_indices_npu = torch.tensor(lora_indices, dtype=torch.int32, device="npu")
        seq_len_npu = torch.tensor(seq_lens, dtype=torch.int32, device="npu")
        lora_ranks_npu = torch.tensor(lora_ranks, dtype=torch.int32, device="npu")
        slice_offsets_npu = slice_offsets.to(device="npu")

        for run in range(4):
            y = torch.zeros((total_tokens, output_dim), dtype=dtype, device="npu")
            out = torch.ops.npu.sgemmc_expand(
                x_npu,
                w_npu,
                lora_indices_npu,
                seq_len_npu,
                lora_ranks_npu,
                slice_offsets_npu,
                y,
            )
            actual = out.to(device="cpu").float()
            self.assertTrue(
                torch.allclose(actual, expect, atol=1e-2, rtol=1e-2),
                f"run{run}: sgemmc_expand race regression, max_abs_diff="
                f"{(actual - expect).abs().max().item():.6f}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
