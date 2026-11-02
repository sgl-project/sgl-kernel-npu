import unittest

import torch
from sgl_kernel_npu.kvcacheio import (
    transfer_state_all_layer_direct_lf_pf,
    transfer_state_per_layer_direct_pf_lf,
)

NUM_LAYERS = 3
DEVICE_SLOTS = 8
HOST_SLOTS = 10


class TestTransferStateDirect(unittest.TestCase):
    def setUp(self):
        torch.npu.set_device(0)
        temporal = torch.arange(
            NUM_LAYERS * DEVICE_SLOTS * 2 * 4,
            dtype=torch.float32,
            device="npu",
        ).reshape(NUM_LAYERS, DEVICE_SLOTS, 2, 4)
        conv = (
            torch.arange(
                NUM_LAYERS * DEVICE_SLOTS * 3,
                dtype=torch.float16,
                device="npu",
            ).reshape(NUM_LAYERS, DEVICE_SLOTS, 3)
            + 7
        )
        self.device_states = [temporal, conv]
        self.host_states = [
            torch.zeros(
                (HOST_SLOTS, NUM_LAYERS, 1, 2, 4),
                dtype=torch.float32,
                pin_memory=True,
            ),
            torch.zeros(
                (HOST_SLOTS, NUM_LAYERS, 1, 3),
                dtype=torch.float16,
                pin_memory=True,
            ),
        ]

    def _submit_d2h(self, device_indices, host_indices):
        stream = torch.npu.Stream()
        event = torch.npu.Event()
        with torch.npu.stream(stream):
            transfer_state_all_layer_direct_lf_pf(
                device_states=self.device_states,
                host_states=self.host_states,
                device_indices=device_indices,
                host_indices=host_indices,
            )
            event.record(stream)
        event.synchronize()

    def _submit_h2d(self, device_indices, host_indices, layers):
        stream = torch.npu.Stream()
        event = torch.npu.Event()
        with torch.npu.stream(stream):
            for layer_id in layers:
                for device, host in zip(self.device_states, self.host_states):
                    transfer_state_per_layer_direct_pf_lf(
                        src=host,
                        dst=device[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        layer_id=layer_id,
                    )
            event.record(stream)
        event.synchronize()

    def test_d2h_all_layers_non_contiguous_indices(self):
        device_indices = torch.tensor([1, 4, 6], dtype=torch.int64)
        host_indices = torch.tensor([2, 5, 7], dtype=torch.int64)
        expected = [
            state[:, device_indices].transpose(0, 1).cpu()
            for state in self.device_states
        ]

        self._submit_d2h(device_indices, host_indices)

        for component, host in enumerate(self.host_states):
            torch.testing.assert_close(
                host[host_indices, :, 0],
                expected[component],
            )

    def test_h2d_single_layer_duplicate_host_source(self):
        layer = 1
        host_indices = torch.tensor([3, 3, 4], dtype=torch.int64)
        device_indices = torch.tensor([0, 2, 5], dtype=torch.int64)
        for component, host in enumerate(self.host_states):
            payload = torch.arange(
                host[3, layer, 0].numel(),
                dtype=host.dtype,
            ).reshape_as(host[3, layer, 0])
            host[3, layer, 0].copy_(payload + component * 10)
            host[4, layer, 0].copy_(payload + component * 20)
        for device in self.device_states:
            device.zero_()

        self._submit_h2d(device_indices, host_indices, [layer])

        for component, device in enumerate(self.device_states):
            torch.testing.assert_close(
                device[layer, 0].cpu(),
                self.host_states[component][3, layer, 0],
            )
            torch.testing.assert_close(
                device[layer, 2].cpu(),
                self.host_states[component][3, layer, 0],
            )
            torch.testing.assert_close(
                device[layer, 5].cpu(),
                self.host_states[component][4, layer, 0],
            )

    def test_h2d_preserves_mapping_order(self):
        layer = 2
        host_indices = torch.tensor([7, 2, 5], dtype=torch.int64)
        device_indices = torch.tensor([1, 6, 3], dtype=torch.int64)
        for component, host in enumerate(self.host_states):
            for offset, host_index in enumerate(host_indices.tolist()):
                host[host_index, layer, 0].fill_(component * 100 + offset + 1)
        for device in self.device_states:
            device.zero_()

        self._submit_h2d(device_indices, host_indices, [layer])

        for component, device in enumerate(self.device_states):
            for offset, device_index in enumerate(device_indices.tolist()):
                torch.testing.assert_close(
                    device[layer, device_index].cpu(),
                    torch.full_like(
                        device[layer, device_index].cpu(),
                        component * 100 + offset + 1,
                    ),
                )

    def test_empty_indices_are_a_noop(self):
        expected = [state.clone() for state in self.device_states]
        self._submit_h2d(
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
            [0],
        )
        for component, device in enumerate(self.device_states):
            torch.testing.assert_close(device, expected[component])

    def test_round_trip_contiguous_run(self):
        device_indices = torch.tensor([2, 3, 4], dtype=torch.int64)
        host_indices = torch.tensor([6, 7, 8], dtype=torch.int64)
        expected = [state.clone() for state in self.device_states]
        self._submit_d2h(device_indices, host_indices)
        for device in self.device_states:
            device[:, device_indices] = 0
        self._submit_h2d(device_indices, host_indices, range(NUM_LAYERS))
        for component, device in enumerate(self.device_states):
            torch.testing.assert_close(
                device[:, device_indices],
                expected[component][:, device_indices],
            )

    def test_round_trip_single_component_with_per_layer_h2d(self):
        device = self.device_states[0]
        host = self.host_states[0]
        device_indices = torch.tensor([1, 4, 6], dtype=torch.int64)
        host_indices = torch.tensor([2, 5, 7], dtype=torch.int64)
        expected = device[:, device_indices].clone()
        stream = torch.npu.Stream()
        event = torch.npu.Event()

        with torch.npu.stream(stream):
            transfer_state_all_layer_direct_lf_pf(
                device_states=[device],
                host_states=[host],
                device_indices=device_indices,
                host_indices=host_indices,
            )
            event.record(stream)
        event.synchronize()

        device[:, device_indices] = 0
        with torch.npu.stream(stream):
            for layer_id in range(NUM_LAYERS):
                transfer_state_per_layer_direct_pf_lf(
                    src=host,
                    dst=device[layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                )
            event.record(stream)
        event.synchronize()

        torch.testing.assert_close(device[:, device_indices], expected)

    def test_reject_pageable_host_memory(self):
        with self.assertRaisesRegex(RuntimeError, "pinned memory"):
            transfer_state_per_layer_direct_pf_lf(
                src=torch.empty_like(self.host_states[0], pin_memory=False),
                dst=self.device_states[0][0],
                src_indices=torch.tensor([0]),
                dst_indices=torch.tensor([0]),
                layer_id=0,
            )

    def test_round_trip_dense_permuted_payload(self):
        temporal_base = torch.arange(
            NUM_LAYERS * DEVICE_SLOTS * 2 * 4,
            dtype=torch.float32,
            device="npu",
        ).reshape(NUM_LAYERS, DEVICE_SLOTS, 2, 4)
        temporal = temporal_base.transpose(-1, -2)
        self.assertFalse(temporal.is_contiguous())
        host = torch.zeros(
            (HOST_SLOTS, NUM_LAYERS, 1, 4, 2),
            dtype=torch.float32,
            pin_memory=True,
        )
        device_indices = torch.tensor([1, 4, 6], dtype=torch.int64)
        host_indices = torch.tensor([2, 5, 7], dtype=torch.int64)
        expected = temporal[:, device_indices].clone()

        stream = torch.npu.Stream()
        event = torch.npu.Event()
        with torch.npu.stream(stream):
            transfer_state_all_layer_direct_lf_pf(
                device_states=[temporal],
                host_states=[host],
                device_indices=device_indices,
                host_indices=host_indices,
            )
            event.record(stream)
        event.synchronize()

        temporal[:, device_indices] = 0
        with torch.npu.stream(stream):
            for layer_id in range(NUM_LAYERS):
                transfer_state_per_layer_direct_pf_lf(
                    src=host,
                    dst=temporal[layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                )
            event.record(stream)
        event.synchronize()

        torch.testing.assert_close(temporal[:, device_indices], expected)

    def test_reject_non_dense_component(self):
        non_dense = self.device_states[0][..., ::2]
        host = torch.zeros(
            (HOST_SLOTS, NUM_LAYERS, 1, 2, 2),
            dtype=torch.float32,
            pin_memory=True,
        )
        with self.assertRaisesRegex(RuntimeError, "physically dense"):
            transfer_state_per_layer_direct_pf_lf(
                src=host,
                dst=non_dense[0],
                src_indices=torch.tensor([0]),
                dst_indices=torch.tensor([0]),
                layer_id=0,
            )

    def test_reject_out_of_range_index(self):
        with self.assertRaisesRegex(RuntimeError, "exceeds component slot count"):
            transfer_state_per_layer_direct_pf_lf(
                src=self.host_states[0],
                dst=self.device_states[0][0],
                src_indices=torch.tensor([0]),
                dst_indices=torch.tensor([DEVICE_SLOTS]),
                layer_id=0,
            )


if __name__ == "__main__":
    unittest.main()
