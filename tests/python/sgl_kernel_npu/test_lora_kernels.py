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
            lora_scaling, dtype=torch.float16, device="cpu"
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
        lora_scaling_tensor_npu = lora_scaling_tensor.to(
            dtype=torch.float16, device="npu"
        )

        actual_output = torch.zeros(
            (batch_size, max_lora_rank), dtype=torch.float, device=inputs_npu.device
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

        inputs_npu = inputs.to(dtype=torch.float, device="npu")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
