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

    def test_dispatch_ffn_combine_kernel_dispatches_and_propagates_oai_parameters(self):
        kernel = (
            REPO_ROOT / "csrc/deepep/ops/op_kernel/dispatch_ffn_combine_swi_glu_oai.cpp"
        ).read_text()
        tiling = (
            REPO_ROOT / "csrc/deepep/ops/op_kernel/dispatch_ffn_combine_tiling.h"
        ).read_text()
        tiling_impl = (
            REPO_ROOT / "csrc/deepep/ops/op_host/dispatch_ffn_combine_tiling.cpp"
        ).read_text()
        op_def = (
            REPO_ROOT
            / "csrc/deepep/ops/op_host/dispatch_ffn_combine_swiglu_oai_def.cpp"
        ).read_text()
        epilogue = (
            REPO_ROOT / "csrc/deepep/ops/op_kernel/dispatch_ffn_combine_kernel/utils/"
            "block_epilogue_pertoken_swiglu.hpp"
        ).read_text()

        self.assertIn("#define SGLANG_SWIGLU_OAI", kernel)
        self.assertIn("TILING_KEY_IS(1000010)", kernel)
        self.assertIn("#if defined(SGLANG_SWIGLU_OAI)", epilogue)
        for scalar in (
            "activationAlpha",
            "gateClampMax",
            "upClampMin",
            "upClampMax",
            "upAdd",
        ):
            self.assertIn(scalar, tiling)
            self.assertIn(scalar, tiling_impl)
        for attr in (
            "activation_alpha",
            "gate_clamp_max",
            "up_clamp_min",
            "up_clamp_max",
            "up_add",
        ):
            self.assertIn(attr, op_def)


if __name__ == "__main__":
    unittest.main()
