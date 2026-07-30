import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


class TestFuseEPActivationAPI(unittest.TestCase):
    def test_buffer_uses_generic_activation_arguments(self):
        source = (REPO_ROOT / "python/deep_ep/deep_ep/buffer.py").read_text()
        tree = ast.parse(source)
        buffer_class = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "Buffer"
        )
        fused_deep_moe = next(
            node
            for node in buffer_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "fused_deep_moe"
        )
        argument_names = [argument.arg for argument in fused_deep_moe.args.args]

        self.assertEqual(
            argument_names[-6:],
            [
                "activation_type",
                "activation_alpha",
                "gate_clamp_max",
                "up_clamp_min",
                "up_clamp_max",
                "up_add",
            ],
        )
        self.assertNotIn("dispatch_ffn_combine_m3", source)

    def test_fused_deep_moe_kernel_dispatches_and_propagates_oai_parameters(self):
        kernel = (
            REPO_ROOT / "csrc/deepep/ops/op_kernel/fused_deep_moe.cpp"
        ).read_text()
        fused_header = (
            REPO_ROOT / "csrc/deepep/ops/op_kernel/fused_deep_moe.h"
        ).read_text()
        grouped_matmul = (
            REPO_ROOT
            / "csrc/deepep/ops/utils/op_kernel/operator/gemm/kernel/"
            "grouped_matmul_slice_m_per_token_dequant_swiglu_quant_multistage_workspace.h"
        ).read_text()
        epilogue = (
            REPO_ROOT
            / "csrc/deepep/ops/utils/op_kernel/operator/epilogue/block/"
            "block_epilogue_per_token_dequant_swiglu.h"
        ).read_text()

        self.assertIn("TILING_KEY_IS(2)", kernel)
        self.assertIn("TILING_KEY_IS(3)", kernel)
        for scalar in (
            "activationAlpha",
            "gateClampMax",
            "upClampMin",
            "upClampMax",
            "upAdd",
        ):
            self.assertIn(scalar, fused_header)
            self.assertIn(scalar, grouped_matmul)
            self.assertIn(scalar, epilogue)

        self.assertIn("if constexpr (DispatchPolicy::EXEC_FLAG & EXEC_FLAG_USE_SWIGLU_OAI)", epilogue)
        self.assertEqual(grouped_matmul.count("params.activationAlpha"), 2)


if __name__ == "__main__":
    unittest.main()
