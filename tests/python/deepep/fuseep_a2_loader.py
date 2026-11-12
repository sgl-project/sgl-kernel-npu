"""Select an isolated A2 op API without global LD_PRELOAD registration effects."""
import ctypes
import os
from pathlib import Path

_handles = []


def load_private_opapi(path):
    path = Path(path)
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError('Expected an absolute, existing A2 op API library')
    if os.environ.get('LD_PRELOAD'):
        raise RuntimeError('Run this isolated test without LD_PRELOAD')
    private = ctypes.CDLL(str(path), mode=os.RTLD_LOCAL | os.RTLD_NOW)
    named = ctypes.CDLL('libcust_opapi.so', mode=os.RTLD_LOCAL | os.RTLD_NOW)
    for symbol in ('aclnnDispatchFFNCombine', 'aclnnDispatchFFNCombineGetWorkspaceSize'):
        expected = ctypes.cast(getattr(private, symbol), ctypes.c_void_p).value
        actual = ctypes.cast(getattr(named, symbol), ctypes.c_void_p).value
        if actual != expected:
            raise RuntimeError(f'Custom-op library collision for {symbol}')
    _handles.extend((private, named))  # Keep both handles alive in each worker.
