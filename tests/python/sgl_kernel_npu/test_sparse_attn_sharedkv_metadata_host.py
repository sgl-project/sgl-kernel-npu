"""Functional + validity tests for the host-side sparse_attn_sharedkv_metadata op.

Covers:
  * output contract (device int32[1024], enabled-core counts within hardware limits),
  * schedule validity (the FA cores form a contiguous partition that covers all batches),
  * input-contract rejection (invalid cmp_topk / cmp_ratio),
  * (optional) byte-identical to the live AICPU op on decode — only when the custom_ops
    vendor env is loaded; decode is version-insensitive so it matches regardless of which
    AICPU build is deployed.

Run:
    python tests/python/sgl_kernel_npu/test_sparse_attn_sharedkv_metadata_host.py
"""

import unittest

import sgl_kernel_npu  # noqa: F401  (loads libsgl_kernel_npu.so -> registers torch.ops.npu.*)
import torch
import torch_npu

HOST = torch.ops.npu.sparse_attn_sharedkv_metadata_host

_PROPS = torch.npu.get_device_properties(0)
AIC = int(_PROPS.cube_core_num)
AIV = int(_PROPS.vector_core_num)

# Buffer capacity (consumer strides). Runtime aic/aiv only bound enabled-entry counts.
FA_SLOTS = 36  # AIC_CORE_NUM
FD_SLOTS = 72  # AIV_CORE_NUM

# Optional live AICPU op (needs the custom_ops package + vendor aclnn env). Only used for
# the decode byte-identical cross-check.
try:
    import custom_ops  # noqa: F401

    _AICPU_OP = torch.ops.custom.npu_sparse_attn_sharedkv_metadata
except Exception:  # pragma: no cover - env dependent
    _AICPU_OP = None


def _topology():
    return AIC, AIV  # device-side expectations for assertions only


def host_call(cu, skv, cmp_ratio, has_cmp, cmp_topk=0):
    # aic/aiv/soc are queried inside the op (auto-detected from the device).
    return HOST(
        64,
        1,
        512,
        "TND",
        "PA_ND",
        cu,
        skv,
        int(skv.numel()),
        int(cmp_topk),
        int(cmp_ratio),
        4,
        3,
        127,
        0,
        True,
        bool(has_cmp),
    )


def aicpu_call(cu, skv, cmp_ratio, has_cmp, cmp_topk=0):
    return _AICPU_OP(
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=512,
        cu_seqlens_q=cu.npu(),
        seqused_kv=skv.npu(),
        batch_size=int(skv.numel()),
        cmp_topk=int(cmp_topk),
        cmp_ratio=int(cmp_ratio),
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=bool(has_cmp),
        device="npu",
    )


def _fa_fd(meta):
    m = meta.view(-1)
    return m[: FA_SLOTS * 8].view(FA_SLOTS, 8), m[
        FA_SLOTS * 8 : FA_SLOTS * 8 + FD_SLOTS * 8
    ].view(FD_SLOTS, 8)


def _enabled_fa(fa):
    return int((fa[:AIC, 0] == 1).sum())


_AICPU_OK_CACHE = None


def _aicpu_ok():
    """True iff the live AICPU op is actually runnable in this env (import + aclnn)."""
    global _AICPU_OK_CACHE
    if _AICPU_OK_CACHE is not None:
        return _AICPU_OK_CACHE
    if _AICPU_OP is None:
        _AICPU_OK_CACHE = False
        return False
    try:
        aicpu_call(
            torch.arange(2, dtype=torch.int32),
            torch.tensor([8], dtype=torch.int32),
            1,
            False,
        )
        _AICPU_OK_CACHE = True
    except Exception:
        _AICPU_OK_CACHE = False
    return _AICPU_OK_CACHE


class TestSparseAttnSharedkvMetadataHost(unittest.TestCase):
    def _assert_contract(self, o):
        self.assertEqual(tuple(o.shape), (1024,))
        self.assertEqual(o.dtype, torch.int32)
        self.assertEqual(o.device.type, "npu")

    def _assert_valid_schedule(self, o, batch):
        """Enabled FA cores must (1) be a contiguous prefix, (2) stay within the hardware
        cube-core count, (3) cover every batch (last enabled core's bn2_end == batch).
        """
        fa, fd = _fa_fd(o.cpu())
        enabled = _enabled_fa(fa)
        self.assertGreaterEqual(enabled, 1)
        self.assertLessEqual(enabled, AIC)
        # contiguous enabled prefix: core_enable==1 for [0, enabled), 0 after (within AIC)
        for i in range(enabled):
            self.assertEqual(int(fa[i, 0]), 1, f"core {i} should be enabled")
        if enabled < AIC:
            self.assertEqual(
                int(fa[enabled, 0]), 0, "core after last enabled must be disabled"
            )
        # full batch coverage: last enabled core closes the batch dimension
        self.assertEqual(
            int(fa[enabled - 1, 4]), batch, "last enabled core must finish all batches"
        )
        # FD is inert (supportFd disabled in the upstream build)
        self.assertEqual(int((fd[:AIV, 0] == 1).sum()), 0)

    # --- functional / validity --------------------------------------------------
    def test_decode_mixed(self):
        B = 8
        cu = torch.arange(B + 1, dtype=torch.int32)
        skv = torch.tensor(
            [128, 256, 512, 1024, 2048, 4096, 8192, 12345], dtype=torch.int32
        )
        o = host_call(cu, skv, 1, False)
        self._assert_contract(o)
        fa, _ = _fa_fd(o.cpu())
        # decode: one q-token/req -> one core per active request
        self.assertEqual(_enabled_fa(fa), B)
        self._assert_valid_schedule(o, B)

    def test_all_paths_uniform(self):
        B = 8
        cu = torch.arange(B + 1, dtype=torch.int32)
        skv = torch.full((B,), 4096, dtype=torch.int32)
        for cr, hc, tk in [(1, False, 0), (4, True, 512), (128, True, 0)]:
            o = host_call(cu, skv, cr, hc, tk)
            self._assert_contract(o)
            self._assert_valid_schedule(o, B)

    def test_prefill_uses_multiple_cores(self):
        T = 32768
        cu = torch.tensor([0, T], dtype=torch.int32)
        skv = torch.tensor([T], dtype=torch.int32)
        o = host_call(cu, skv, 1, False)
        self._assert_contract(o)
        fa, _ = _fa_fd(o.cpu())
        # large M -> work spread over several cube cores
        self.assertGreater(_enabled_fa(fa), 1)
        self._assert_valid_schedule(o, 1)

    def test_single_request_long_context(self):
        cu = torch.tensor([0, 1], dtype=torch.int32)  # decode, B=1
        skv = torch.tensor([65536], dtype=torch.int32)
        o = host_call(cu, skv, 1, False)
        self._assert_contract(o)
        self._assert_valid_schedule(o, 1)

    def test_determinism(self):
        cu = torch.arange(9, dtype=torch.int32)
        skv = torch.full((8,), 4096, dtype=torch.int32)
        a = host_call(cu, skv, 4, True, 512).cpu()
        b = host_call(cu, skv, 4, True, 512).cpu()
        self.assertTrue(torch.equal(a, b))

    def test_invalid_cmp_topk_rejected(self):
        cu = torch.arange(3, dtype=torch.int32)
        skv = torch.tensor([100, 100], dtype=torch.int32)
        # cmp_topk must be 0 / 512 / 1024; 64 is invalid -> the op must raise.
        with self.assertRaises(Exception):
            host_call(cu, skv, 4, True, cmp_topk=64)

    def test_invalid_cmp_ratio_rejected(self):
        cu = torch.arange(3, dtype=torch.int32)
        skv = torch.tensor([100, 100], dtype=torch.int32)
        # cmp_ratio must be 4 or 128 when has_cmp_kv; 7 is invalid -> must raise.
        with self.assertRaises(Exception):
            host_call(cu, skv, 7, True, cmp_topk=512)

    # --- optional cross-check vs the live AICPU op (decode only) ----------------
    @unittest.skipUnless(
        _aicpu_ok(), "live AICPU op (custom_ops + vendor aclnn) not runnable"
    )
    def test_byte_identical_decode_vs_aicpu(self):
        # Decode is M=1 -> version-insensitive, so host matches any deployed AICPU build.
        B = 8
        cu = torch.arange(B + 1, dtype=torch.int32)
        skv = torch.tensor(
            [128, 256, 512, 1024, 2048, 4096, 8192, 12345], dtype=torch.int32
        )
        for cr, hc, tk in [(1, False, 0), (4, True, 512), (128, True, 0)]:
            h = host_call(cu, skv, cr, hc, tk).cpu()
            a = aicpu_call(cu, skv, cr, hc, tk).cpu()
            fa_h, _ = _fa_fd(h)
            fa_a, _ = _fa_fd(a)
            for i in range(AIC):
                if int(fa_h[i, 0]) == 1:  # only enabled cores are read by the consumer
                    self.assertTrue(
                        torch.equal(fa_h[i], fa_a[i]),
                        f"decode core {i} mismatch (cmp_ratio={cr})",
                    )


if __name__ == "__main__":
    unittest.main()
