import unittest

import torch
from sgl_kernel_npu.kvcacheio import TransferDirection, transfer_mamba_state

# Typical Mamba state shapes (e.g. GLM-5.2-like config)
# conv_state: [num_layers, pool_size, conv_window, dim]
# ssm_state:  [num_layers, pool_size, nheads, head_dim]


def _make_device_buf(num_layers, device_size, state_shape, dtype, fill="ones"):
    shape = (num_layers, device_size, *state_shape)
    if fill == "ones":
        buf = torch.ones(shape, dtype=dtype, device="npu")
    elif fill == "zeros":
        buf = torch.zeros(shape, dtype=dtype, device="npu")
    elif fill == "arange":
        buf = torch.arange(
            torch.prod(torch.tensor(shape)).item(), dtype=dtype, device="npu"
        ).reshape(shape)
    elif fill == "randn":
        buf = torch.randn(shape, dtype=dtype, device="npu")
    else:
        buf = torch.zeros(shape, dtype=dtype, device="npu")
    return buf


def _make_host_buf(host_size, num_layers, state_shape, dtype, fill="zeros"):
    # Host layout: [host_size, num_layers, 1, *state_shape]
    shape = (host_size, num_layers, 1, *state_shape)
    if fill == "ones":
        buf = torch.ones(shape, dtype=dtype, device="cpu", pin_memory=True)
    elif fill == "zeros":
        buf = torch.zeros(shape, dtype=dtype, device="cpu", pin_memory=True)
    elif fill == "arange":
        buf = (
            torch.arange(torch.prod(torch.tensor(shape)).item(), dtype=dtype)
            .reshape(shape)
            .pin_memory()
        )
    elif fill == "randn":
        buf = torch.randn(shape, dtype=dtype, device="cpu").pin_memory()
    else:
        buf = torch.zeros(shape, dtype=dtype, device="cpu", pin_memory=True)
    return buf


class TestTransferMambaState(unittest.TestCase):
    """Tests for transfer_mamba_state D2H/H2D correctness."""

    def setUp(self):
        torch.npu.set_device(0)

    def _run_transfer(
        self, device_buf, host_buf, device_indices, host_indices, direction
    ):
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            transfer_mamba_state(
                device_buf=device_buf,
                host_buf=host_buf,
                device_indices=device_indices,
                host_indices=host_indices,
                direction=direction,
            )
        torch.npu.synchronize()

    # ------------------------------------------------------------------ #
    # D2H: device → host
    # ------------------------------------------------------------------ #
    def test_d2h_1d_state(self):
        """D2H with 1D state shape [num_layers, device_size, dim]."""
        num_layers, device_size, host_size, dim = 4, 8, 8, 128
        dtype = torch.bfloat16
        torch.manual_seed(42)

        device_buf = _make_device_buf(
            num_layers, device_size, (dim,), dtype, fill="randn"
        )
        host_buf = _make_host_buf(host_size, num_layers, (dim,), dtype, fill="zeros")

        device_indices = torch.arange(device_size, dtype=torch.int64)
        host_indices = torch.arange(host_size, dtype=torch.int64)

        self._run_transfer(
            device_buf, host_buf, device_indices, host_indices, TransferDirection.D2H
        )

        # Verify: host[slot, layer, 0, :] == device[layer, slot, :]
        for slot in range(device_size):
            for layer in range(num_layers):
                torch.testing.assert_close(
                    host_buf[slot, layer, 0, :].to(torch.float32),
                    device_buf[layer, slot, :].to(torch.float32).cpu(),
                    rtol=1e-3,
                    atol=1e-3,
                )

    def test_d2h_2d_state(self):
        """D2H with 2D state shape [num_layers, device_size, nheads, head_dim]."""
        num_layers, device_size, host_size = 3, 6, 6
        nheads, head_dim = 8, 64
        dtype = torch.float16
        torch.manual_seed(42)

        device_buf = _make_device_buf(
            num_layers, device_size, (nheads, head_dim), dtype, fill="randn"
        )
        host_buf = _make_host_buf(
            host_size, num_layers, (nheads, head_dim), dtype, fill="zeros"
        )

        device_indices = torch.arange(device_size, dtype=torch.int64)
        host_indices = torch.arange(host_size, dtype=torch.int64)

        self._run_transfer(
            device_buf, host_buf, device_indices, host_indices, TransferDirection.D2H
        )

        for slot in range(device_size):
            for layer in range(num_layers):
                torch.testing.assert_close(
                    host_buf[slot, layer, 0, :, :].to(torch.float32),
                    device_buf[layer, slot, :, :].to(torch.float32).cpu(),
                    rtol=1e-3,
                    atol=1e-3,
                )

    def test_d2h_subset_slots(self):
        """D2H transferring only a subset of slots with non-trivial index mapping."""
        num_layers, device_size, host_size, dim = 2, 10, 10, 32
        dtype = torch.float32

        device_buf = _make_device_buf(
            num_layers, device_size, (dim,), dtype, fill="arange"
        )
        host_buf = _make_host_buf(host_size, num_layers, (dim,), dtype, fill="zeros")

        # Transfer slots 1, 3, 5 from device to host slots 7, 2, 9
        device_indices = torch.tensor([1, 3, 5], dtype=torch.int64)
        host_indices = torch.tensor([7, 2, 9], dtype=torch.int64)

        self._run_transfer(
            device_buf, host_buf, device_indices, host_indices, TransferDirection.D2H
        )

        # Verify transferred slots
        for d_slot, h_slot in zip([1, 3, 5], [7, 2, 9]):
            for layer in range(num_layers):
                torch.testing.assert_close(
                    host_buf[h_slot, layer, 0, :],
                    device_buf[layer, d_slot, :].cpu(),
                    rtol=1e-6,
                    atol=1e-6,
                )

        # Verify non-transferred host slots remain zero
        non_transferred = set(range(host_size)) - {7, 2, 9}
        for h_slot in non_transferred:
            self.assertEqual(
                host_buf[h_slot].sum().item(),
                0,
                f"host slot {h_slot} should remain zero",
            )

    # ------------------------------------------------------------------ #
    # H2D: host → device
    # ------------------------------------------------------------------ #
    def test_h2d_1d_state(self):
        """H2D with 1D state shape."""
        num_layers, device_size, host_size, dim = 4, 8, 8, 128
        dtype = torch.bfloat16

        device_buf = _make_device_buf(
            num_layers, device_size, (dim,), dtype, fill="zeros"
        )
        host_buf = _make_host_buf(host_size, num_layers, (dim,), dtype, fill="arange")

        device_indices = torch.arange(device_size, dtype=torch.int64)
        host_indices = torch.arange(host_size, dtype=torch.int64)

        self._run_transfer(
            device_buf, host_buf, device_indices, host_indices, TransferDirection.H2D
        )

        for slot in range(device_size):
            for layer in range(num_layers):
                torch.testing.assert_close(
                    device_buf[layer, slot, :].to(torch.float32).cpu(),
                    host_buf[slot, layer, 0, :].to(torch.float32),
                    rtol=1e-3,
                    atol=1e-3,
                )

    def test_h2d_2d_state(self):
        """H2D with 2D state shape."""
        num_layers, device_size, host_size = 3, 6, 6
        nheads, head_dim = 8, 64
        dtype = torch.float16

        device_buf = _make_device_buf(
            num_layers, device_size, (nheads, head_dim), dtype, fill="zeros"
        )
        host_buf = _make_host_buf(
            host_size, num_layers, (nheads, head_dim), dtype, fill="arange"
        )

        device_indices = torch.arange(device_size, dtype=torch.int64)
        host_indices = torch.arange(host_size, dtype=torch.int64)

        self._run_transfer(
            device_buf, host_buf, device_indices, host_indices, TransferDirection.H2D
        )

        for slot in range(device_size):
            for layer in range(num_layers):
                torch.testing.assert_close(
                    device_buf[layer, slot, :, :].to(torch.float32).cpu(),
                    host_buf[slot, layer, 0, :, :].to(torch.float32),
                    rtol=1e-3,
                    atol=1e-3,
                )

    def test_h2d_subset_slots(self):
        """H2D transferring only a subset of slots."""
        num_layers, device_size, host_size, dim = 2, 10, 10, 32
        dtype = torch.float32

        device_buf = _make_device_buf(
            num_layers, device_size, (dim,), dtype, fill="zeros"
        )
        host_buf = _make_host_buf(host_size, num_layers, (dim,), dtype, fill="arange")

        # Transfer host slots 0, 4, 8 to device slots 3, 7, 1
        device_indices = torch.tensor([3, 7, 1], dtype=torch.int64)
        host_indices = torch.tensor([0, 4, 8], dtype=torch.int64)

        self._run_transfer(
            device_buf, host_buf, device_indices, host_indices, TransferDirection.H2D
        )

        for d_slot, h_slot in zip([3, 7, 1], [0, 4, 8]):
            for layer in range(num_layers):
                torch.testing.assert_close(
                    device_buf[layer, d_slot, :].cpu(),
                    host_buf[h_slot, layer, 0, :],
                    rtol=1e-6,
                    atol=1e-6,
                )

        # Verify non-transferred device slots remain zero
        non_transferred = set(range(device_size)) - {3, 7, 1}
        for d_slot in non_transferred:
            self.assertEqual(
                device_buf[:, d_slot, :].sum().cpu().item(),
                0,
                f"device slot {d_slot} should remain zero",
            )

    # ------------------------------------------------------------------ #
    # Round-trip: D2H then H2D should recover original data
    # ------------------------------------------------------------------ #
    def test_roundtrip_d2h_h2d(self):
        """D2H then H2D should recover the original device data."""
        num_layers, device_size, dim = 4, 8, 64
        dtype = torch.float16

        device_buf = _make_device_buf(
            num_layers, device_size, (dim,), dtype, fill="randn"
        )
        original = device_buf.clone()
        host_buf = _make_host_buf(device_size, num_layers, (dim,), dtype, fill="zeros")

        indices = torch.arange(device_size, dtype=torch.int64)

        # D2H: device → host
        self._run_transfer(
            device_buf, host_buf, indices, indices, TransferDirection.D2H
        )

        # Zero out device buffer
        device_buf.zero_()

        # H2D: host → device
        self._run_transfer(
            device_buf, host_buf, indices, indices, TransferDirection.H2D
        )

        torch.testing.assert_close(
            device_buf.to(torch.float32).cpu(),
            original.to(torch.float32).cpu(),
            rtol=1e-3,
            atol=1e-3,
        )

    # ------------------------------------------------------------------ #
    # float32 dtype
    # ------------------------------------------------------------------ #
    def test_d2h_float32(self):
        """D2H with float32 dtype."""
        num_layers, device_size, host_size, dim = 2, 4, 4, 32
        dtype = torch.float32
        torch.manual_seed(42)

        device_buf = _make_device_buf(
            num_layers, device_size, (dim,), dtype, fill="randn"
        )
        host_buf = _make_host_buf(host_size, num_layers, (dim,), dtype, fill="zeros")

        device_indices = torch.arange(device_size, dtype=torch.int64)
        host_indices = torch.arange(host_size, dtype=torch.int64)

        self._run_transfer(
            device_buf, host_buf, device_indices, host_indices, TransferDirection.D2H
        )

        # Sum-based check (like test_transfer_kv_dim_exchange)
        self.assertAlmostEqual(
            host_buf.sum().item(),
            device_buf.sum().cpu().item(),
            places=3,
            msg="host sum should equal device sum after D2H",
        )

    # ------------------------------------------------------------------ #
    # Single index
    # ------------------------------------------------------------------ #
    def test_single_index_transfer(self):
        """Transfer a single slot."""
        num_layers, device_size, host_size, dim = 3, 5, 5, 16
        dtype = torch.float32

        device_buf = _make_device_buf(
            num_layers, device_size, (dim,), dtype, fill="ones"
        )
        host_buf = _make_host_buf(host_size, num_layers, (dim,), dtype, fill="zeros")

        device_indices = torch.tensor([2], dtype=torch.int64)
        host_indices = torch.tensor([3], dtype=torch.int64)

        self._run_transfer(
            device_buf, host_buf, device_indices, host_indices, TransferDirection.D2H
        )

        # slot 2 of device → slot 3 of host
        for layer in range(num_layers):
            torch.testing.assert_close(
                host_buf[3, layer, 0, :],
                device_buf[layer, 2, :].cpu(),
                rtol=1e-6,
                atol=1e-6,
            )

        # Other host slots should remain zero
        for h_slot in [0, 1, 2, 4]:
            self.assertEqual(host_buf[h_slot].sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
