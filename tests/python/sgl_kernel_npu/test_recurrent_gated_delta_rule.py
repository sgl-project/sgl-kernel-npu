import logging
import os
import shutil
import time

import numpy as np
import sgl_kernel_npu
import torch
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger

EPSILON_FOR_DIVISION = 1e-9
# Global Configuration
_USE_GRAPH = False

if _USE_GRAPH:
    logger.setLevel(logging.ERROR)

    # Safely remove cache directories instead of using os.system
    for path in ["kernel_meta/", "/root/atc_data/kernel_cache/"]:
        if os.path.exists(path):
            shutil.rmtree(path)

    # "ENABLE_ACLNN" determines if it falls back to aclnn.
    # true: fallback to aclnn, false: online compilation
    os.environ["ENABLE_ACLNN"] = "false"
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)


def verify_result(output, golden):
    """Verifies if the output matches the golden expected values."""
    output = output.to(torch.float32).cpu().reshape(-1)
    golden = golden.to(torch.float32).cpu().reshape(-1)

    different_element_results = np.isclose(
        output, golden, rtol=2 ** (-8), atol=2 ** (-8), equal_nan=True
    )
    different_element_indexes = np.where(~different_element_results)[0]

    for index, real_index in enumerate(different_element_indexes):
        golden_data = golden[real_index]
        output_data = output[real_index]
        diff_ratio = abs(output_data - golden_data) / (
            golden_data + EPSILON_FOR_DIVISION
        )  # Added epsilon to prevent division by zero

        print(
            f"data index: {real_index:06d}, expected: {golden_data:-.9f}, "
            f"actual: {output_data:-.9f}, rdiff: {diff_ratio:-.6f}"
        )

        if index == 10:
            break


def visualize_error_flatline_ansi(output, golden, tolerance=2 ** (-8), width=100):
    """Prints an ANSI color-coded visual representation of the error map."""
    output = output.cpu()
    golden = golden.cpu()
    error = (
        torch.abs(output.to(torch.float) - golden.to(torch.float))
        .flatten()
        .cpu()
        .numpy()
    )

    total_count = len(error)
    fail_count = np.sum(error > tolerance)
    max_error = np.max(error)

    step = max(total_count // width, 1)
    compressed_error = [
        np.max(error[i : i + step]) for i in range(0, total_count, step)
    ]

    def ansi_block(color_code):
        return f"\033[{color_code}m  \033[0m"

    print(f"\n?? Flatline Color Error Map (normalized to {width} chars):\n")
    for e in compressed_error[:width]:
        if e > tolerance * 10:
            print(ansi_block("41"), end="")  # Red
        elif e > tolerance * 2:
            print(ansi_block("43"), end="")  # Yellow
        elif e > tolerance:
            print(ansi_block("42"), end="")  # Green
        else:
            print(ansi_block("47"), end="")  # Gray
    print()

    print("\n?? Summary:")
    print(f"  ?? Max Error: {max_error:.4e}")
    print(
        f"  ?? Failed elements (> {tolerance}): {fail_count}/{total_count} ({(fail_count / total_count) * 100:.2f}%)\n"
    )
    verify_result(output, golden)


class MyModel(torch.nn.Module):
    """Wrapper module for the recurrent_gated_delta_rule NPU operation."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        mix_qkv,
        recurrent_state,
        beta=None,
        scale=None,
        actual_seq_lengths=None,
        ssm_state_indices=None,
        nk=None,
        nv=None,
        intermediate_state=None,
        cache_indices=None,
        num_accepted_tokens=None,
        g=None,
        gk=None,
    ):

        return torch.ops.npu.recurrent_gated_delta_rule(
            mix_qkv,
            recurrent_state,
            beta=beta,
            scale=scale,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=ssm_state_indices,
            nk=nk,
            nv=nv,
            intermediate_state=intermediate_state,
            cache_indices=cache_indices,
            num_accepted_tokens=num_accepted_tokens,
            g=g,
            gk=gk,
        )


class TestCase:
    """Class to generate inputs, run golden PyTorch logic, run NPU logic, and compare results."""

    def __init__(
        self,
        b=64,
        mtp=2,
        nk=4,
        nv=8,
        dk=128,
        dv=128,
        is_continue=True,
        has_beta=True,
        has_scale=True,
        has_g=True,
        has_gk=False,
        has_num_accepted_tokens=True,
    ):
        self.b = b
        self.mtp = mtp
        self.nk = nk
        self.nv = nv
        self.dk = dk
        self.dv = dv
        self.is_continue = is_continue
        self.has_beta = has_beta
        self.has_scale = has_scale
        self.has_g = has_g
        self.has_gk = has_gk
        self.has_num_accepted_tokens = has_num_accepted_tokens

        self.max_slots = 65
        self.mix_qkv = None
        self.recurrent_state = None
        self.intermediate_state = None
        self.cache_indices = None
        self.be = None
        self.scale = None
        self.actual_seq_lengths = None
        self.ssm_state_indices = None
        self.num_accepted_tokens = None
        self.g = None
        self.gk = None
        self.out_npu = None
        self.state_npu = None
        self.out_golden = None
        self.state_golden = None

    def __repr__(self):
        return (
            f"B={self.b}, MTP={self.mtp}, Nk={self.nk}, Nv={self.nv}, Dk={self.dk}, Dv={self.dv}, "
            f"is_continue={self.is_continue}, has_beta={self.has_beta}, has_scale={self.has_scale}, "
            f"has_g={self.has_g}, has_gk={self.has_gk}, has_num_accepted_tokens={self.has_num_accepted_tokens}"
        )

    def generate_input(self):
        bs = self.b
        mtp = self.mtp
        nv = self.nv
        dv = self.dv
        dk = self.dk
        nk = self.nk
        S = mtp

        def make_mix_qkv(q_tensor, k_tensor, v_tensor):
            # The dimensions of q, k, v should be [T, nk, dk] and [T, nv, dv]
            # Flatten the feature dimension
            q_flat = q_tensor.flatten(start_dim=2)  # [T, nk*dk]
            k_flat = k_tensor.flatten(start_dim=2)  # [T, nk*dk]
            v_flat = v_tensor.flatten(start_dim=2)  # [T, nv*dv]
            # Concatenate
            return torch.cat([q_flat, k_flat, v_flat], dim=-1)

        if self.is_continue:
            self.actual_seq_lengths = (torch.ones(bs) * S).npu().to(torch.int32)
            if mtp > 1:
                self.intermediate_state = torch.rand(
                    (self.max_slots, S, nv, dv, dk), dtype=torch.bfloat16, device="npu"
                )

            self.recurrent_state = torch.rand(
                (self.max_slots, nv, dv, dk), dtype=torch.bfloat16, device="npu"
            )

            v_temp = torch.rand((bs, S, nv, dv), dtype=torch.bfloat16, device="npu")
            q_temp = torch.rand((bs, S, nk, dk), dtype=torch.bfloat16, device="npu")
            k_temp = torch.rand((bs, S, nk, dk), dtype=torch.bfloat16, device="npu")

            self.mix_qkv = make_mix_qkv(q_temp, k_temp, v_temp).contiguous()

            self.g = torch.rand((bs, S, nv), dtype=torch.float32, device="npu") * (-1.0)
            self.be = torch.rand((bs, S, nv), dtype=torch.bfloat16, device="npu")

            cache_indices = torch.randperm(self.max_slots, device="npu")[:bs].to(
                torch.int32
            )
            base_indices = (cache_indices * S).unsqueeze(1)  # shape: (bs, 1)
            offsets = torch.arange(S, device="npu", dtype=torch.int32)  # shape: (S,)

            self.ssm_state_indices = (base_indices + offsets).contiguous()
            if mtp > 1:
                self.cache_indices = cache_indices

            self.scale = dk**-0.5
            self.num_accepted_tokens = (torch.randint(1, mtp + 1, (bs,)).npu()).to(
                torch.int32
            )
        else:
            raise NotImplementedError(
                "Initialization for is_continue=False is not defined. "
                "The 'recurrent_gated_delta_rule' operator currently only supports continuous input tensors."
            )

        self.be = self.be if self.has_beta else None
        self.scale = self.scale if self.has_scale else None
        self.num_accepted_tokens = (
            self.num_accepted_tokens if self.has_num_accepted_tokens else None
        )
        self.g = self.g if self.has_g else None
        self.gk = self.gk if self.has_gk else None

    def run_golden(self):
        nk, dk, nv, dv = self.nk, self.dk, self.nv, self.dv
        dim_q = nk * dk
        dim_k = nk * dk
        dim_v = nv * dv
        B, Seq = self.mix_qkv.shape[0], self.mix_qkv.shape[1]
        T = B * Seq
        mix_qkv_fp32 = self.mix_qkv.to(torch.float32)
        mix_qkv_flat = mix_qkv_fp32.view(T, -1)

        q_flat, k_flat, v_flat = torch.split(
            mix_qkv_flat, [dim_q, dim_k, dim_v], dim=-1
        )

        # Reshape back to the original shapes required by the logic
        q = q_flat.view(T, nk, dk)  # [B, S, nk, dk]
        k = k_flat.view(T, nk, dk)  # [B, S, nk, dk]
        v = v_flat.view(T, nv, dv)  # [B, S, nv, dv]

        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        k = torch.nn.functional.normalize(k, p=2, dim=-1)

        scale = self.scale
        recurrent_state = self.recurrent_state.clone()

        if self.intermediate_state is not None:
            intermediate_state = self.intermediate_state.clone()
            cache_indices = self.cache_indices

            intermediate_state[cache_indices, 0] = recurrent_state[cache_indices]
            ssm_state = intermediate_state.view(-1, nv, dv, dk)
        else:
            ssm_state = recurrent_state

        initial_state = ssm_state.to(torch.float32)
        actual_seq_lengths = self.actual_seq_lengths
        num_accepted_tokens = self.num_accepted_tokens
        ssm_state_indices = self.ssm_state_indices.view(-1)

        T, n_heads_v, Dv = v.shape
        n_heads_qk = q.shape[-2]

        g = (
            torch.ones(T, n_heads_v).to(torch.float32)
            if self.g is None
            else self.g.to(torch.float32).reshape(T, n_heads_v).exp()
        )
        be = (
            torch.ones(T, n_heads_v).to(torch.float32)
            if self.be is None
            else self.be.to(torch.float32).view(T, n_heads_v)
        )
        beta = be.sigmoid()

        o = torch.empty_like(v).to(torch.float32)
        if scale is None:
            scale = k.shape[-1] ** -0.5
        q = q * scale

        seq_start = 0
        for i in range(len(actual_seq_lengths)):
            if num_accepted_tokens is None:
                init_state = initial_state[ssm_state_indices[seq_start]]
            else:
                init_state = initial_state[
                    ssm_state_indices[seq_start + num_accepted_tokens[i] - 1]
                ]

            for head_id in range(n_heads_v):
                S = init_state[head_id]  # [Dv, Dk]
                for slot_id in range(seq_start, seq_start + actual_seq_lengths[i]):
                    q_i = q[slot_id][head_id // (n_heads_v // n_heads_qk)]  # [Dk]
                    k_i = k[slot_id][head_id // (n_heads_v // n_heads_qk)]  # [Dk]
                    v_i = v[slot_id][head_id]  # [Dv]
                    alpha_i = g[slot_id][head_id]
                    beta_i = beta[slot_id][head_id]

                    S = S * alpha_i
                    x = (S * k_i.unsqueeze(-2)).sum(dim=-1)
                    y = (v_i - x) * beta_i  # [Dv]
                    S_ = y[:, None] * k_i[None, :]  # [Dv, Dk]
                    S = S + S_  # [Dv, Dk]

                    initial_state[ssm_state_indices[slot_id]][head_id] = S
                    o[slot_id][head_id] = (S * q_i.unsqueeze(-2)).sum(dim=-1)  # [Dv]

            seq_start += actual_seq_lengths[i]

        self.out_golden = o.view(B, Seq, nv, dv)
        self.state_golden = initial_state.view(self.max_slots * Seq, nv, dv, dk)

    def run_npu(self):
        model = MyModel().npu()
        if _USE_GRAPH:
            model = torch.compile(model, backend=npu_backend, dynamic=False)

        intermediate_state = None
        cache_indices = None

        if self.mtp > 1:
            intermediate_state = self.intermediate_state.clone()
            intermediate_state = intermediate_state.view(-1, self.nv, self.dv, self.dk)
            cache_indices = self.cache_indices

        initial_state = self.recurrent_state.clone()
        o = model(
            self.mix_qkv,
            initial_state,
            beta=self.be,
            scale=self.scale,
            actual_seq_lengths=self.actual_seq_lengths,
            ssm_state_indices=self.ssm_state_indices,
            nk=self.nk,
            nv=self.nv,
            intermediate_state=intermediate_state,
            cache_indices=cache_indices,
            num_accepted_tokens=self.num_accepted_tokens,
            g=self.g,
            gk=self.gk,
        )

        self.out_npu = o.to(torch.float32)
        if self.mtp > 1:
            self.state_npu = intermediate_state.to(torch.float32)
        else:
            self.state_npu = initial_state.to(torch.float32)

    def compare(self):
        eps = 2 ** (-8)
        is_close_o = torch.allclose(
            self.out_golden, self.out_npu, rtol=eps, atol=eps, equal_nan=False
        )
        is_close_state = torch.allclose(
            self.state_golden, self.state_npu, rtol=eps, atol=eps, equal_nan=False
        )

        if is_close_o and is_close_state:
            print(f"\t{eps=}: passed.")
            return True
        else:
            print(f"\t{eps=}: {is_close_o=}, {is_close_state=}")
            visualize_error_flatline_ansi(self.out_npu, self.out_golden)
            visualize_error_flatline_ansi(self.state_npu, self.state_golden)
            return False


def compatible_cases():
    """Defines the matrix of test cases to be executed."""
    res = []
    # format: {'shape': (b, mtp, dk, dv, nk, nv), 'is_cont': bool, 'option': (beta, scale, g, num_accepted_tokens, gk)}
    res.append(
        {
            "shape": (2, 1, 128, 128, 8, 16),
            "is_cont": True,
            "option": (True, True, True, True, False),
        }
    )
    res.append(
        {
            "shape": (64, 2, 128, 128, 8, 16),
            "is_cont": True,
            "option": (True, True, True, True, False),
        }
    )
    res.append(
        {
            "shape": (32, 8, 128, 128, 8, 16),
            "is_cont": True,
            "option": (True, True, True, True, False),
        }
    )
    res.append(
        {
            "shape": (32, 8, 128, 128, 16, 32),
            "is_cont": True,
            "option": (True, True, True, True, False),
        }
    )

    return res, "compatible"


def run_cases(cases):
    """Executes the provided test cases."""
    print("=" * 50)
    print(f"{cases[1]} cases, total: {len(cases[0])}, start time: {time.time()}")
    t0 = time.time()
    fault_count = 0

    for case in cases[0]:
        b, mtp, dk, dv, nk, nv = case["shape"]
        is_cont = case["is_cont"]
        beta, scale, g, n_acc_tokens, gk = case["option"]

        tc = TestCase(
            b=b,
            mtp=mtp,
            nk=nk,
            nv=nv,
            dk=dk,
            dv=dv,
            is_continue=is_cont,
            has_g=g,
            has_scale=scale,
            has_beta=beta,
            has_gk=gk,
            has_num_accepted_tokens=n_acc_tokens,
        )

        print(f"{tc}")
        tc.generate_input()
        tc.run_golden()
        tc.run_npu()

        if not tc.compare():
            fault_count += 1

    print(
        f"{cases[1]} cases time cost: {time.time() - t0:.4f}s, total: {len(cases[0])}, failed: {fault_count}"
    )


if __name__ == "__main__":
    # Specify NPU device; default to 0 for general environments, switch to 6 if strictly required
    target_device = "npu:0"
    torch.npu.set_device(target_device)

    run_cases(compatible_cases())
