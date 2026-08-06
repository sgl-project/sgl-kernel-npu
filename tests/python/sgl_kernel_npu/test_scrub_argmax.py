import unittest

import torch
from sgl_kernel_npu.sample.scrub_argmax import scrub_argmax_fused


class TestScrubArgmax(unittest.TestCase):
    def test_scrub_argmax(self):
        torch.manual_seed(1)
        delete_token_id = 17
        split_token_id = 29
        logits = torch.randn(32, 4097, dtype=torch.bfloat16, device="npu")
        logits[:, delete_token_id] = 100
        logits[::2, split_token_id] = 101

        token_ids = scrub_argmax_fused(
            logits,
            delete_token_id,
            split_token_id,
        )
        reference = logits.clone()
        reference[:, delete_token_id] = float("-inf")
        reference[:, split_token_id] = float("-inf")

        torch.testing.assert_close(
            token_ids.cpu(),
            reference.argmax(dim=-1).cpu(),
        )


if __name__ == "__main__":
    unittest.main()
