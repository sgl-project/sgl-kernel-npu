import ctypes
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass


C = 128
D = 128
H = 16
HG = 16

RTOL = 1e-2
ATOL = 1e-5
MAX_RMSE_RATIO = 0.05
MIN_R2 = 0.99
HARD_FAIL_MAX = 1.0

REPO_ROOT = Path(__file__).resolve().parents[3]
CHUNK_O_CPP = REPO_ROOT / "csrc" / "mega_chunk_gdn" / "chunk_o.cpp"
KERNEL_INCLUDE = REPO_ROOT / "csrc" / "mega_chunk_gdn" / "include"
COMPILED_DIR = REPO_ROOT / "build" / "pto_jit"
DRIVER_INC = Path("/usr/local/Ascend/driver/kernel/inc")
JIT_FLAG_FLAVOR = os.environ.get("SGL_CHUNK_O_JIT_FLAG_FLAVOR", "cmake_debug")
LAUNCHER = os.environ.get("SGL_CHUNK_O_JIT_LAUNCHER", "generated_stub")
GENERATED_STUB_LIB = Path(
    os.environ.get("SGL_CHUNK_O_DEBUG_STUB_A", REPO_ROOT / "build" / "lib" / "libchunk_o_debug_kernel.a")
)
SHIM_SRC = Path(__file__).with_name("chunk_o_debug_stub_shim.cpp")


def _has_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(not _has_npu(), reason="NPU is required")


def _vp(tensor: torch.Tensor | None) -> ctypes.c_void_p:
    if tensor is None:
        return ctypes.c_void_p()
    return ctypes.c_void_p(tensor.data_ptr())


def _ascend_home() -> Path:
    home = os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get("ASCEND_HOME_PATH")
    if not home:
        pytest.skip("ASCEND_TOOLKIT_HOME or ASCEND_HOME_PATH is required for PTO JIT")
    return Path(home)


def _pto_root() -> Path:
    candidates = []
    if "PTO_LIB_PATH" in os.environ:
        candidates.append(Path(os.environ["PTO_LIB_PATH"]))
    candidates.extend(
        [
            REPO_ROOT / "thirdparty" / "pto-isa",
            REPO_ROOT / "megagdn-pto" / "third_party" / "pto-isa",
            Path("/sources/pto-isa"),
        ]
    )
    for candidate in candidates:
        if (candidate / "include").is_dir():
            return candidate
    pytest.skip("PTO include directory was not found")


def _block_dim(device: str) -> int:
    try:
        return int(getattr(torch.npu.get_device_properties(device), "cube_core_num", 20))
    except (RuntimeError, AssertionError):
        return 24


def _cmake_debug_flags(ascend_home: Path, pto_root: Path) -> list[str]:
    arch_home = ascend_home / "aarch64-linux"
    if not arch_home.exists():
        arch_home = ascend_home
    asc_devkit = ascend_home / "include" / "ascendc" / "asc_devkit_version.h"

    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "-DGDN_ENABLE_STANDALONE_COMPONENT_KERNELS",
        f"-DGDN_C={C}",
        f"-DGDN_D={D}",
        f"-DGDN_H={H}",
        f"-DGDN_HG={HG}",
        "-DTILING_KEY_VAR=0",
        "-O3",
        "-DNDEBUG",
        "--cce-disable-kernel-global-attr-check",
        "--cce-aicore-arch=dav-c220",
        "--cce-auto-sync",
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-record-overflow=true",
        "-mllvm",
        "-cce-aicore-addr-transform",
        "-mllvm",
        "-cce-aicore-dcci-insert-for-scalar=false",
        "-O3",
        "-std=c++17",
        "--cce-aicore-lang",
    ]
    if asc_devkit.exists():
        flags.extend(["-include", str(asc_devkit)])

    for include_dir in [
        arch_home / "asc" / "impl" / "adv_api",
        arch_home / "asc" / "impl" / "basic_api",
        arch_home / "asc" / "impl" / "utils",
        arch_home / "asc" / "include",
        arch_home / "asc" / "include" / "adv_api",
        arch_home / "asc" / "include" / "basic_api",
        arch_home / "asc" / "include" / "aicpu_api",
        arch_home / "asc" / "include" / "utils",
        arch_home / "tikcpp" / "tikcfw",
        arch_home / "tikcpp" / "tikcfw" / "interface",
        arch_home / "tikcpp" / "tikcfw" / "impl",
        REPO_ROOT / "csrc" / "mega_chunk_gdn",
        KERNEL_INCLUDE,
        pto_root / "include",
        arch_home / "include",
        arch_home / "include" / "experiment" / "runtime",
        arch_home / "include" / "experiment" / "msprof",
    ]:
        if include_dir.exists():
            flags.append(f"-I{include_dir}")
    return flags


def _megagdn_flags(ascend_home: Path, pto_root: Path) -> list[str]:
    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "-DMEMORY_BASE",
        "-DGDN_ENABLE_STANDALONE_COMPONENT_KERNELS",
        "-O2",
        "-std=gnu++17",
        "--cce-aicore-arch=dav-c220",
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-record-overflow=true",
        "-mllvm",
        "-cce-aicore-dcci-insert-for-scalar=false",
        "-Wno-macro-redefined",
        "-Wno-ignored-attributes",
        f"-I{KERNEL_INCLUDE}",
        f"-I{pto_root / 'include'}",
        f"-I{ascend_home / 'include'}",
        f"-I{ascend_home / 'pkg_inc'}",
        f"-I{ascend_home / 'pkg_inc' / 'runtime'}",
        f"-I{ascend_home / 'pkg_inc' / 'profiling'}",
        f"-DGDN_H={H}",
        f"-DGDN_HG={HG}",
        f"-DGDN_D={D}",
        f"-DGDN_C={C}",
    ]
    return flags


def _compile_flags(ascend_home: Path, pto_root: Path) -> tuple[str, list[str]]:
    if JIT_FLAG_FLAVOR == "megagdn":
        return "megagdn", _megagdn_flags(ascend_home, pto_root)
    if JIT_FLAG_FLAVOR != "cmake_debug":
        pytest.fail(
            "unknown SGL_CHUNK_O_JIT_FLAG_FLAVOR="
            f"{JIT_FLAG_FLAVOR!r}; expected 'cmake_debug' or 'megagdn'"
        )
    return "cmake_debug", _cmake_debug_flags(ascend_home, pto_root)


@lru_cache(maxsize=None)
def _compile_chunk_o() -> ctypes.CDLL:
    if shutil.which("bisheng") is None:
        pytest.skip("bisheng compiler is required for PTO JIT")

    ascend_home = _ascend_home()
    pto_root = _pto_root()
    COMPILED_DIR.mkdir(parents=True, exist_ok=True)
    flag_flavor, flags = _compile_flags(ascend_home, pto_root)
    if DRIVER_INC.is_dir():
        flags.append(f"-I{DRIVER_INC}")
    flags.extend(os.environ.get("PTO_DYNAMIC_EXTRA_FLAGS", "").split())

    lib_path = COMPILED_DIR / f"chunk_o_{flag_flavor}_H{H}_Hg{HG}_D{D}_C{C}.so"
    cmd = ["bisheng", *flags, str(CHUNK_O_CPP), "-o", str(lib_path)]
    print("chunk_o standalone compile cmd:", " ".join(cmd), flush=True)

    if not lib_path.exists() or lib_path.stat().st_mtime_ns < CHUNK_O_CPP.stat().st_mtime_ns:
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=300)
        if result.returncode != 0:
            pytest.fail(
                "failed to JIT compile chunk_o.cpp\n"
                f"stdout:\n{result.stdout[-4000:]}\n"
                f"stderr:\n{result.stderr[-4000:]}"
            )

    lib = ctypes.CDLL(str(lib_path.resolve()))
    lib.call_kernel.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p]
        + [ctypes.c_void_p] * 11
        + [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    )
    lib.call_kernel.restype = None
    return lib


@lru_cache(maxsize=None)
def _compile_generated_stub_shim() -> Path:
    if not GENERATED_STUB_LIB.exists():
        pytest.skip(f"generated chunk_o debug static library not found: {GENERATED_STUB_LIB}")

    ascend_home = _ascend_home()
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    torch_npu_lib = None
    try:
        import torch_npu as torch_npu_mod

        torch_npu_lib = Path(torch_npu_mod.__file__).resolve().parent / "lib"
    except ImportError:
        pass

    COMPILED_DIR.mkdir(parents=True, exist_ok=True)
    shim_so = COMPILED_DIR / "libchunk_o_debug_stub_shim.so"
    cmd = [
        os.environ.get("CXX", "/usr/bin/c++"),
        "-fPIC",
        "-shared",
        "-O2",
        "-o",
        str(shim_so),
        str(SHIM_SRC),
        "-Wl,--whole-archive",
        str(GENERATED_STUB_LIB),
        "-Wl,--no-whole-archive",
    ]

    for lib_dir in [
        torch_lib,
        torch_npu_lib,
        ascend_home / "lib64",
        ascend_home / "tools" / "simulator" / "Ascend910_9382" / "lib",
    ]:
        if lib_dir is not None and lib_dir.exists():
            cmd.append(f"-L{lib_dir}")
            cmd.append(f"-Wl,-rpath,{lib_dir}")

    ascendc_runtime = ascend_home / "lib64" / "libascendc_runtime.a"
    cmd.extend(
        [
            "-ltorch_npu",
            "-lascendcl",
            "-ltiling_api",
            "-lregister",
            "-lplatform",
            "-lascendalog",
            "-ldl",
        ]
    )
    if ascendc_runtime.exists():
        cmd.append(str(ascendc_runtime))
    cmd.extend(
        [
            "-lruntime",
            "-lerror_manager",
            "-lprofapi",
            "-lge_common_base",
            "-lmmpa",
            "-lascend_dump",
            "-lc_sec",
            "-ltiling_api",
            "-lregister",
            "-lplatform",
            "-lascendalog",
            "-ldl",
        ]
    )

    print("chunk_o generated-stub shim link cmd:", " ".join(cmd), flush=True)
    newest_input = max(GENERATED_STUB_LIB.stat().st_mtime_ns, SHIM_SRC.stat().st_mtime_ns)
    if not shim_so.exists() or shim_so.stat().st_mtime_ns < newest_input:
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=120)
        if result.returncode != 0:
            pytest.fail(
                "failed to link chunk_o generated-stub shim\n"
                f"stdout:\n{result.stdout[-4000:]}\n"
                f"stderr:\n{result.stderr[-4000:]}"
            )
    return shim_so


@lru_cache(maxsize=None)
def _load_generated_chunk_o_debug():
    shim_so = _compile_generated_stub_shim()
    print("chunk_o generated stub:", f"{shim_so}:call_chunk_o_debug_stub", flush=True)
    lib = ctypes.CDLL(str(shim_so.resolve()), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
    fn = lib.call_chunk_o_debug_stub

    fn.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p]
        + [ctypes.c_void_p] * 11
        + [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    )
    fn.restype = ctypes.c_uint32
    return fn


def _launch_chunk_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    s: torch.Tensor,
    g_t: torch.Tensor,
    mask: torch.Tensor,
    ws_qk: torch.Tensor,
    ws_qs: torch.Tensor,
    ws_gated: torch.Tensor,
    out: torch.Tensor,
    cu: torch.Tensor | None,
    block_dim: int,
    batch_size: int,
    total_tokens: int,
) -> None:
    stream = torch.npu.current_stream()._as_parameter_
    if LAUNCHER == "standalone":
        lib = _compile_chunk_o()
        lib.call_kernel(
            block_dim,
            stream,
            _vp(q),
            _vp(k),
            _vp(v_new),
            _vp(s),
            _vp(g_t),
            _vp(mask),
            _vp(ws_qk),
            _vp(ws_qs),
            _vp(ws_gated),
            _vp(out),
            _vp(cu),
            batch_size,
            total_tokens,
            total_tokens,
        )
        return

    if LAUNCHER != "generated_stub":
        pytest.fail(
            "unknown SGL_CHUNK_O_JIT_LAUNCHER="
            f"{LAUNCHER!r}; expected 'generated_stub' or 'standalone'"
        )

    fn = _load_generated_chunk_o_debug()
    print("chunk_o launcher: generated_stub", flush=True)
    ret = fn(
        block_dim,
        stream,
        _vp(q),
        _vp(k),
        _vp(v_new),
        _vp(s),
        _vp(g_t),
        _vp(mask),
        _vp(ws_qk),
        _vp(ws_qs),
        _vp(ws_gated),
        _vp(out),
        _vp(cu),
        batch_size,
        total_tokens,
        total_tokens,
    )
    assert ret == 0, f"aclrtlaunch_launch_chunk_o_debug returned {ret}"


def _ranges(total_tokens: int, cu_seqlens: torch.Tensor | None):
    if cu_seqlens is None:
        return [(0, total_tokens)]
    cu = cu_seqlens.cpu().tolist()
    return list(zip(cu, cu[1:]))


def _total_chunks(total_tokens: int, chunk_size: int, cu_seqlens: torch.Tensor | None) -> int:
    return sum((e - s + chunk_size - 1) // chunk_size for s, e in _ranges(total_tokens, cu_seqlens))


def _ref_cumsum(g: torch.Tensor, chunk_size: int, cu_seqlens: torch.Tensor | None) -> torch.Tensor:
    out = torch.zeros_like(g, dtype=torch.float32)
    for bos, eos in _ranges(g.shape[1], cu_seqlens):
        for offset in range(0, eos - bos, chunk_size):
            start = bos + offset
            end = min(start + chunk_size, eos)
            out[:, start:end, :] = g.float()[:, start:end, :].cumsum(dim=1)
    return out


def _transpose_gates(g_sum: torch.Tensor) -> torch.Tensor:
    return g_sum.squeeze(0).t().contiguous()


def _ref_chunk_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g_cumsum: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None,
):
    _, total_tokens, key_heads, hidden_size = k.shape
    heads = w.shape[2]
    group = heads // key_heads
    kf, wf, uf, gf = k.float(), w.float(), u.float(), g_cumsum.float()

    s_out = torch.zeros(_total_chunks(total_tokens, chunk_size, cu_seqlens), heads, hidden_size, hidden_size)
    v_new = torch.zeros_like(uf)
    final = torch.zeros(len(_ranges(total_tokens, cu_seqlens)), heads, hidden_size, hidden_size)

    chunk_base = 0
    for seq_idx, (bos, eos) in enumerate(_ranges(total_tokens, cu_seqlens)):
        num_chunks = (eos - bos + chunk_size - 1) // chunk_size
        for head in range(heads):
            key_head = head // group
            state = torch.zeros(hidden_size, hidden_size)
            for chunk_idx in range(num_chunks):
                start = bos + chunk_idx * chunk_size
                end = min(start + chunk_size, eos)
                gates = gf[0, start:end, head]
                last_gate = gates[end - start - 1]

                s_out[chunk_base + chunk_idx, head] = state
                vc = uf[0, start:end, head] - wf[0, start:end, head] @ state
                v_new[0, start:end, head] = vc

                decay = torch.exp(last_gate - gates)[:, None]
                kv = kf[0, start:end, key_head].T @ (vc * decay)
                state = torch.exp(last_gate) * state + kv
            final[seq_idx, head] = state
        chunk_base += num_chunks

    return s_out, v_new, final


def _ref_chunk_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    h_states: torch.Tensor,
    g_cumsum: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    _, total_tokens, key_heads, hidden_size = q.shape
    heads = v_new.shape[2]
    group = heads // key_heads
    qf, kf, vf, gf = q.float(), k.float(), v_new.float(), g_cumsum.float()
    out = torch.zeros_like(vf)

    chunk_base = 0
    for bos, eos in _ranges(total_tokens, cu_seqlens):
        num_chunks = (eos - bos + chunk_size - 1) // chunk_size
        for head in range(heads):
            key_head = head // group
            for chunk_idx in range(num_chunks):
                start = bos + chunk_idx * chunk_size
                end = min(start + chunk_size, eos)
                length = end - start
                qc = qf[0, start:end, key_head]
                kc = kf[0, start:end, key_head]
                vc = vf[0, start:end, head]
                gc = gf[0, start:end, head]
                state = h_states[chunk_base + chunk_idx, head]

                inter = (qc @ state) * torch.exp(gc)[:, None]
                qk = qc @ kc.T
                causal = torch.arange(length)[:, None] >= torch.arange(length)[None, :]
                gate = torch.exp(torch.minimum(gc[:, None] - gc[None, :], torch.zeros(length, length)))
                out[0, start:end, head] = inter + (qk * gate * causal.float()) @ vc
        chunk_base += num_chunks
    return out


def _r2(y_ref: torch.Tensor, y_pred: torch.Tensor) -> float:
    ref = y_ref.detach().cpu().numpy().ravel().astype(np.float64)
    pred = y_pred.detach().cpu().numpy().ravel().astype(np.float64)
    ss_res = np.sum((ref - pred) ** 2)
    ss_tot = np.sum((ref - np.mean(ref)) ** 2)
    return float("nan") if ss_tot < 1e-30 else 1.0 - ss_res / ss_tot


def _assert_stats_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    diff = (actual - expected).abs()
    max_abs = diff.max().item()
    if max_abs > HARD_FAIL_MAX:
        pytest.fail(f"{name} max_abs={max_abs:.6f} exceeds hard limit {HARD_FAIL_MAX}")
    if (diff <= ATOL + RTOL * expected.abs()).all():
        return

    mean_abs = float(expected.float().flatten().abs().mean())
    rmse = float(torch.sqrt((diff.float().flatten() ** 2).mean()))
    ratio = rmse / max(mean_abs, 1e-15)
    r2 = _r2(expected, actual)
    if mean_abs < 1e-9 and rmse < 5e-4:
        return
    assert ratio <= MAX_RMSE_RATIO and np.isfinite(r2) and r2 >= MIN_R2, (
        f"{name} max_abs={max_abs:.6f} rmse_ratio={ratio:.6f} r2={r2:.6f}"
    )


@pytest.mark.parametrize(
    ("total_tokens", "cu_list"),
    [
        pytest.param(128, None, id="fixed"),
        #pytest.param(256, [0, 64, 256], id="varlen"),
    ],
)
def test_chunk_o_cpp_can_run_when_jit_compiled(total_tokens: int, cu_list: list[int] | None) -> None:
    device_name = os.environ.get("GDN_NPU_DEVICE", "npu:0")
    torch.npu.set_device(device_name)
    device = torch.device(device_name)
    block_dim = _block_dim(device_name)
    cu_cpu = None if cu_list is None else torch.tensor(cu_list, dtype=torch.int32)
    cu = None if cu_cpu is None else cu_cpu.to(device).contiguous()
    batch_size = 1 if cu_cpu is None else cu_cpu.numel() - 1

    torch.manual_seed(42)
    k_cpu = F.normalize(torch.randn(1, total_tokens, HG, D, dtype=torch.float16), dim=-1, p=2)
    q_cpu = F.normalize(torch.randn(1, total_tokens, HG, D, dtype=torch.float16), dim=-1, p=2)
    w_cpu = torch.randn(1, total_tokens, H, D, dtype=torch.float16)
    u_cpu = torch.randn(1, total_tokens, H, D, dtype=torch.float16)
    g_in_cpu = F.logsigmoid(torch.randn(1, total_tokens, H, dtype=torch.float32))
    g_sum_cpu = _ref_cumsum(g_in_cpu, C, cu_cpu)

    num_chunks = _total_chunks(total_tokens, C, cu_cpu)
    s_ref, v_ref, _ = _ref_chunk_h(k_cpu, w_cpu, u_cpu, g_sum_cpu, C, cu_cpu)
    s_cpu = s_ref.reshape(num_chunks * H, D, D).to(torch.float16).contiguous()
    v_new_cpu = v_ref.to(torch.float16).contiguous()

    q = q_cpu.to(device).contiguous()
    k = k_cpu.to(device).contiguous()
    v_new = v_new_cpu.to(device).contiguous()
    s = s_cpu.to(device).contiguous()
    g_t = _transpose_gates(g_sum_cpu).to(device).contiguous()
    mask = torch.tril(torch.ones(C, C, device=device), diagonal=0).float().contiguous()
    ws_qk = torch.zeros(block_dim, C, C, device=device, dtype=torch.float16)
    ws_qs = torch.zeros(block_dim, C, D, device=device, dtype=torch.float16)
    ws_gated = torch.zeros_like(ws_qk)
    out = torch.empty(1, total_tokens, H, D, device=device, dtype=torch.float16)

    _launch_chunk_o(
        q,
        k,
        v_new,
        s,
        g_t,
        mask,
        ws_qk,
        ws_qs,
        ws_gated,
        out,
        cu,
        block_dim,
        batch_size,
        total_tokens,
    )
    torch.npu.synchronize()

    expected = _ref_chunk_o(q_cpu, k_cpu, v_new_cpu, s_ref, g_sum_cpu, C, cu_cpu)
    _assert_stats_close("chunk_o", out.float().cpu(), expected.float())
