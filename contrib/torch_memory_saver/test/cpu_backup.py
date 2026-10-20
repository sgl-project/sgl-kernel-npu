import logging
import sys

import torch
from torch_memory_saver import torch_memory_saver


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    print("Allocate tensor_with_backup")
    with torch_memory_saver.region(enable_cpu_backup=True):
        tensor_with_backup = torch.full(
            (20_000_000,), 10, dtype=torch.uint8, device="npu"
        )

    print("Allocate tensor_without_backup")
    with torch_memory_saver.region(enable_cpu_backup=False):
        tensor_without_backup = torch.full(
            (20_000_000,), 20, dtype=torch.uint8, device="npu"
        )

    print(f"{tensor_with_backup[:3]=} {tensor_without_backup[:3]=}")
    assert tensor_with_backup[:3].tolist() == [10, 10, 10]
    assert tensor_without_backup[:3].tolist() == [20, 20, 20]

    torch_memory_saver.pause()

    # occupy some space
    tensor_unrelated = torch.full((20_000_000,), 30, dtype=torch.uint8, device="npu")

    torch_memory_saver.resume()

    print(f"{tensor_with_backup[:3]=} {tensor_without_backup[:3]=}")
    assert tensor_with_backup[:3].tolist() == [10, 10, 10]
    assert tensor_without_backup[:3].tolist() != [20, 20, 20]


def run_full_content_check():
    """Verify the whole backup buffer (not just a prefix) survives pause->resume.

    Exercises the D2H -> aclrtMapMem -> H2D chain in core.cpp whose return
    values are now checked: a partial or failed copy would corrupt data
    somewhere in the buffer and fail these assertions.
    """
    n = 20_000_000
    with torch_memory_saver.region(enable_cpu_backup=True):
        tensor = torch.full((n,), 10, dtype=torch.uint8, device="npu")
    assert int(tensor.sum().item()) == 10 * n

    # two cycles, so the "cpu_backup buffer already allocated" branch in
    # pause() is covered as well
    for cycle in range(2):
        torch_memory_saver.pause()
        # dirty the physical memory so a missing H2D restore is visible
        torch.full((n,), 30, dtype=torch.uint8, device="npu")
        torch_memory_saver.resume()
        assert int(tensor.sum().item()) == 10 * n, f"cycle={cycle}"


def run_pause_sync_check():
    """pause() must back up the latest data even without a caller-side sync.

    Async writes are issued on a side stream right before pause(); pause()
    in core.cpp now synchronizes the device before the D2H backup, so after
    resume the tensor must hold the final written value rather than a torn
    mix captured while kernels were still in flight.
    """
    n = 20_000_000
    with torch_memory_saver.region(enable_cpu_backup=True):
        tensor = torch.zeros(n, dtype=torch.uint8, device="npu")

    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for v in range(1, 6):
            tensor.fill_(v)
    # deliberately no torch.npu.synchronize() here

    torch_memory_saver.pause()
    torch_memory_saver.resume()

    assert int(tensor[0].item()) == 5
    assert int(tensor.sum().item()) == 5 * n


def run_double_release_check():
    """Repeated pause()/resume() must be no-ops and must not clobber the backup.

    Before the is_paused guard in core.cpp, a second pause() would D2H-copy
    from already-unmapped virtual addresses, overwriting the good CPU backup
    with garbage; a second resume() would map a new physical handle onto a
    live address. After two pauses and resumes the data must stay intact.
    """
    n = 20_000_000
    with torch_memory_saver.region(enable_cpu_backup=True):
        tensor = torch.full((n,), 10, dtype=torch.uint8, device="npu")

    torch_memory_saver.pause()
    torch_memory_saver.pause()  # must be a no-op, not a backup-clobbering copy
    torch_memory_saver.resume()
    assert int(tensor.sum().item()) == 10 * n

    torch_memory_saver.resume()  # must be a no-op as well
    assert int(tensor.sum().item()) == 10 * n


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
    run_full_content_check()
    run_pause_sync_check()
    run_double_release_check()
