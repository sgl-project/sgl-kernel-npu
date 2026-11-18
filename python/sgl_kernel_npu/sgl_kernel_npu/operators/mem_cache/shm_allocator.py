"""Public helpers for NPU-registered host shared memory."""

from ...utils.shm_allocator import create_shm_tensor, free_shm

__all__ = ["create_shm_tensor", "free_shm"]
