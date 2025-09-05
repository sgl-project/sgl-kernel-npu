from pathlib import Path


def get_binary_path_from_package(stem: str):
    dir_package = Path(__file__).parent
    candidates = [
        p for d in [dir_package, dir_package.parent] for p in d.glob(f"{stem}.*.so")
    ]
    assert (
        len(candidates) == 1
    ), f"Expected exactly one torch_memory_saver_cpp library, found: {candidates}"
    return candidates[0]
