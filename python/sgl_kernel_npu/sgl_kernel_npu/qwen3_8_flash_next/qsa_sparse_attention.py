"""NPU sparse GQA over physical slots with masked loads and FP32 reductions."""

import torch
import triton
import triton.language as tl

from .qsa_torch_attention import torch_attention

_MAX_GRID_PROGRAMS = 65535
# Offline experiments may override this through the fixed test/bench entrypoints.
_CONFIG_OVERRIDE = None
_FORCE_WIDE = False
_GROUP_MODEL = True


@triton.jit
def _sparse_partials(
    Q,
    K,
    V,
    Slots,
    Partial,
    Maxima,
    Sums,
    Output,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    Q_DIM: tl.constexpr,
    K_ROW: tl.constexpr,
    K_HEAD: tl.constexpr,
    K_DIM: tl.constexpr,
    V_ROW: tl.constexpr,
    V_HEAD: tl.constexpr,
    V_DIM: tl.constexpr,
    SLOT_ROW: tl.constexpr,
    SLOT_COL: tl.constexpr,
    HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    SPLITS: tl.constexpr,
    TILES: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
    START: tl.constexpr,
    WIDE_OFFSETS: tl.constexpr,
):
    local_item = tl.program_id(0).to(tl.int64)
    item = local_item + START
    row = item // HEADS
    head = item % HEADS
    split = tl.program_id(1)
    dims = tl.arange(0, DIM)
    if WIDE_OFFSETS:
        dims = dims.to(tl.int64)
    query = tl.load(Q + row * Q_ROW + head * Q_HEAD + dims * Q_DIM).to(tl.float32)
    maximum = tl.full((), -float("inf"), tl.float32)
    total = tl.full((), 0, tl.float32)
    numerator = tl.full((DIM,), 0, tl.float32)
    for tile in range(TILES):
        base = split * TILES + tile
        if WIDE_OFFSETS:
            base = base.to(tl.int64)
        cols = base * BLOCK + tl.arange(0, BLOCK)
        slots = tl.load(
            Slots + row * SLOT_ROW + cols * SLOT_COL, cols < WIDTH, other=-1
        )
        valid = (cols < WIDTH) & (slots >= 0)
        # Only negative padding is replaced; nonnegative slots are never clamped.
        if WIDE_OFFSETS:
            safe_slots = tl.maximum(slots, 0).to(tl.int64)
        else:
            safe_slots = tl.maximum(slots, 0).to(tl.int32)
        keys = tl.load(
            K
            + safe_slots[:, None] * K_ROW
            + (head // GROUP) * K_HEAD
            + dims[None, :] * K_DIM,
            valid[:, None],
            other=0,
        ).to(tl.float32)
        values = tl.load(
            V
            + safe_slots[:, None] * V_ROW
            + (head // GROUP) * V_HEAD
            + dims[None, :] * V_DIM,
            valid[:, None],
            other=0,
        ).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * SCALE
        # A selected -inf score is an anomaly, not a padding sentinel.
        scores = tl.where(scores == -float("inf"), float("nan"), scores)
        scores = tl.where(valid, scores, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(scores, axis=0))
        safe_max = tl.where(next_max == -float("inf"), 0, next_max)
        correction = tl.exp(maximum - safe_max)
        weights = tl.where(valid, tl.exp(scores - safe_max), 0)
        numerator = numerator * correction + tl.sum(weights[:, None] * values, axis=0)
        total = total * correction + tl.sum(weights, axis=0)
        maximum = next_max
    if SPLITS == 1:
        result = numerator / tl.where(total > 0, total, 1)
        tl.store(Output + item * DIM + dims, result)
    else:
        part = local_item * SPLITS + split
        tl.store(Partial + part * DIM + dims, numerator)
        tl.store(Maxima + part, maximum)
        tl.store(Sums + part, total)


@triton.jit
def _update_head(
    keys, values, query, valid, maximum, total, numerator, SCALE: tl.constexpr
):
    scores = tl.sum(keys * query[None, :], 1) * SCALE
    scores = tl.where(scores == -float("inf"), float("nan"), scores)
    scores = tl.where(valid, scores, -float("inf"))
    next_max = tl.maximum(maximum, tl.max(scores, 0))
    safe_max = tl.where(next_max == -float("inf"), 0, next_max)
    correction = tl.exp(maximum - safe_max)
    weights = tl.where(valid, tl.exp(scores - safe_max), 0)
    numerator = numerator * correction + tl.sum(weights[:, None] * values, 0)
    total = total * correction + tl.sum(weights, 0)
    return next_max, total, numerator


@triton.jit
def _store_head(
    Partial,
    Maxima,
    Sums,
    Output,
    local_item,
    item,
    split,
    maximum,
    total,
    numerator,
    DIM: tl.constexpr,
    SPLITS: tl.constexpr,
):
    dims = tl.arange(0, DIM)
    if SPLITS == 1:
        result = numerator / tl.where(total > 0, total, 1)
        tl.store(Output + item * DIM + dims, result)
    else:
        part = local_item * SPLITS + split
        tl.store(Partial + part * DIM + dims, numerator)
        tl.store(Maxima + part, maximum)
        tl.store(Sums + part, total)


@triton.jit
def _grouped_partials(
    Q,
    K,
    V,
    Slots,
    Partial,
    Maxima,
    Sums,
    Output,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    Q_DIM: tl.constexpr,
    K_ROW: tl.constexpr,
    V_ROW: tl.constexpr,
    SLOT_ROW: tl.constexpr,
    SLOT_COL: tl.constexpr,
    DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    SPLITS: tl.constexpr,
    TILES: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
    START_ROW: tl.constexpr,
):
    local_row = tl.program_id(0).to(tl.int64)
    row = local_row + START_ROW
    split = tl.program_id(1)
    dims = tl.arange(0, DIM)
    q0 = tl.load(Q + row * Q_ROW + dims * Q_DIM).to(tl.float32)
    q1 = tl.load(Q + row * Q_ROW + Q_HEAD + dims * Q_DIM).to(tl.float32)
    q2 = tl.load(Q + row * Q_ROW + 2 * Q_HEAD + dims * Q_DIM).to(tl.float32)
    m0 = tl.full((), -float("inf"), tl.float32)
    m1 = tl.full((), -float("inf"), tl.float32)
    m2 = tl.full((), -float("inf"), tl.float32)
    s0 = tl.full((), 0, tl.float32)
    s1 = tl.full((), 0, tl.float32)
    s2 = tl.full((), 0, tl.float32)
    a0 = tl.full((DIM,), 0, tl.float32)
    a1 = tl.full((DIM,), 0, tl.float32)
    a2 = tl.full((DIM,), 0, tl.float32)
    for tile in range(TILES):
        cols = (split * TILES + tile) * BLOCK + tl.arange(0, BLOCK)
        slots = tl.load(
            Slots + row * SLOT_ROW + cols * SLOT_COL, cols < WIDTH, other=-1
        )
        valid = (cols < WIDTH) & (slots >= 0)
        safe_slots = tl.maximum(slots, 0).to(tl.int32)
        keys = tl.load(
            K + safe_slots[:, None] * K_ROW + dims[None, :], valid[:, None], other=0
        ).to(tl.float32)
        values = tl.load(
            V + safe_slots[:, None] * V_ROW + dims[None, :], valid[:, None], other=0
        ).to(tl.float32)
        m0, s0, a0 = _update_head(keys, values, q0, valid, m0, s0, a0, SCALE)
        m1, s1, a1 = _update_head(keys, values, q1, valid, m1, s1, a1, SCALE)
        m2, s2, a2 = _update_head(keys, values, q2, valid, m2, s2, a2, SCALE)
    _store_head(
        Partial,
        Maxima,
        Sums,
        Output,
        local_row * 3,
        row * 3,
        split,
        m0,
        s0,
        a0,
        DIM,
        SPLITS,
    )
    _store_head(
        Partial,
        Maxima,
        Sums,
        Output,
        local_row * 3 + 1,
        row * 3 + 1,
        split,
        m1,
        s1,
        a1,
        DIM,
        SPLITS,
    )
    _store_head(
        Partial,
        Maxima,
        Sums,
        Output,
        local_row * 3 + 2,
        row * 3 + 2,
        split,
        m2,
        s2,
        a2,
        DIM,
        SPLITS,
    )


@triton.jit
def _merge_partials(
    Partial,
    Maxima,
    Sums,
    Output,
    DIM: tl.constexpr,
    SPLITS: tl.constexpr,
    START: tl.constexpr,
):
    item = tl.program_id(0).to(tl.int64)
    splits = tl.arange(0, SPLITS)
    dims = tl.arange(0, DIM)
    part = item * SPLITS + splits
    maxima = tl.load(Maxima + part)
    sums = tl.load(Sums + part)
    maximum = tl.max(maxima, axis=0)
    maximum = tl.where(maximum == -float("inf"), 0, maximum)
    factors = tl.exp(maxima - maximum)
    partials = tl.load(Partial + part[:, None] * DIM + dims[None, :])
    denominator = tl.sum(sums * factors, axis=0)
    numerator = tl.sum(partials * factors[:, None], axis=0)
    result = numerator / tl.where(denominator > 0, denominator, 1)
    tl.store(Output + (item + START) * DIM + dims, result)


# Full tensor contract and dispatch rationale: sparse_attention below.
_SUPPORTED_HEADS = {(24, 2), (12, 1), (6, 1), (3, 1)}


def _supported_width(width):
    if not 1025 <= width <= 2**31 - 1:
        return False
    for selected_blocks in (512, 2048):
        unit = selected_blocks + 1
        if (width + 1) % unit == 0 and (width + 1) // unit >= 2:
            return True
        # Exists c>=3 with 3<=width+1-unit*c<=c (shared MTP tail).
        low = max(3, (width + 1 + unit) // (unit + 1))
        high = (width - 2) // unit
        if low <= high:
            return True
    return False


def can_run_sparse_attention(q, k, v, slots) -> bool:
    """Check production metadata only; no device-value reads or fallback."""
    return (
        q.device.type == "npu"
        and q.ndim == k.ndim == v.ndim == 3
        and slots.ndim == 2
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.device == k.device == v.device == slots.device
        and k.shape == v.shape
        and q.shape[2] == k.shape[2] == 256
        and (q.shape[1], k.shape[1]) in _SUPPORTED_HEADS
        and q.shape[0] == slots.shape[0]
        and slots.dtype == torch.int32
        and k.shape[0] <= 2**31 - 1
        and _supported_width(slots.shape[1])
        and k.stride() == v.stride() == (k.shape[1] * 256, 256, 1)
        and q.stride(2) == 1
        and q.stride(1) >= 256
        and q.stride(0) >= (q.shape[1] - 1) * q.stride(1) + 256
        and slots.stride(1) == 1
        and slots.stride(0) >= slots.shape[1]
    )


def _configuration(q, k, v, slots):
    if _CONFIG_OVERRIDE is not None:
        return _CONFIG_OVERRIDE
    rows = q.shape[0]
    if rows != 1 and rows < 8:
        return 32, 8
    # Probe a conservative 32-split address bound before choosing a schedule.
    # The wrapper checks the selected launch configuration again.
    if _use_grouped(q, k, False) and not _wide_offsets(q, k, v, slots, 32, 32):
        if rows == 1:
            return 32, 16
        if rows >= 32:
            return 32, 1
        if rows >= 8:
            return 32, 4
    return 32, 8


def _wide_offsets(q, k, v, slots, block, splits):
    # Host metadata only: bound logical N, padded slot offsets, physical byte
    # spans. A true result disables three-head
    # reuse; the generic Triton path uses int64 offsets in that case.
    limit = 2**31 - 1
    return (
        _FORCE_WIDE
        or k.shape[0] > limit
        or slots.shape[1] + block * splits > limit
        or (slots.shape[1] + block * splits) * slots.stride(1) * slots.element_size()
        > limit
        or any(
            sum(max(0, n - 1) * s for n, s in zip(x.shape, x.stride()))
            * x.element_size()
            > limit
            for x in (q, k, v, slots)
        )
    )


def _use_torch(q, k, slots):
    # All accepted widths exceed one tile; only the local head pair matters.
    return _CONFIG_OVERRIDE is None and (q.shape[1], k.shape[1]) != (3, 1)


def dispatch_info(q, k, v, slots):
    """Read-only metadata for experiment reports; no device synchronization."""
    if not can_run_sparse_attention(q, k, v, slots):
        raise ValueError("Unsupported NPU sparse attention tensor configuration")
    if q.shape[0] == 0 or k.shape[0] == 0:
        return dict(path="empty", launches=0, scratch_bytes=0)
    if _use_torch(q, k, slots):
        kv_chunk = k.shape[1]
        count = max(
            1, min(32, (16 * 1024**2) // (slots.shape[1] * q.shape[-1] * 8 * kv_chunk))
        )
        return dict(
            path="torch_matrix",
            row_chunk=count,
            head_chunk=q.shape[1] // k.shape[1],
            kv_head_chunk=kv_chunk,
            gather_bytes=min(count, q.shape[0])
            * slots.shape[1]
            * q.shape[-1]
            * 8
            * kv_chunk,
            cache_copies=[],
            layout_copy_bytes=0,
        )
    block, splits = _configuration(q, k, v, slots)
    grouped = _use_grouped(q, k, _wide_offsets(q, k, v, slots, block, splits))
    items = q.shape[0] * q.shape[1]
    chunk = min(items, _MAX_GRID_PROGRAMS // splits)
    if grouped:
        chunk = chunk // 3 * 3
    path = (
        "torch_matrix"
        if _use_torch(q, k, slots)
        else ("direct" if splits == 1 else "split")
    )
    return dict(
        path=("grouped_" + path) if grouped else path,
        block=block,
        splits=splits,
        offset_bits=64 if _wide_offsets(q, k, v, slots, block, splits) else 32,
        cache_copies=[],
        layout_copy_bytes=0,
        launches=triton.cdiv(items, chunk) * (1 if splits == 1 else 2),
        scratch_bytes=0 if splits == 1 else chunk * splits * (q.shape[-1] + 2) * 4,
    )


def _use_grouped(q, k, wide_offsets):
    return (
        _GROUP_MODEL
        and not wide_offsets
        and q.dtype == torch.bfloat16
        and q.shape[1:] == (3, 256)
        and k.shape[1] == 1
    )


def sparse_attention(q, k, v, slots, softmax_scale=None):
    """Compute attention over selected physical KV rows for Qwen3.8-Flash-Next.

    Inputs and output
    -----------------
        Q       [R, Hq, 256]     BF16 query heads
        K/V     [N, Hkv, 256]    BF16 shared KV pool on this rank
        slots   [R, S]          int32 physical KV rows; negatives are padding
        output  [R, Hq, 256]     new contiguous BF16 tensor on the same NPU

    R is the total query-row count, including any caller-padded rows. It is
    not a fixed batch whitelist. N is pool capacity, not one request's context
    length. S includes padding columns and does not shrink with valid-key count.
    Slots are already physical rows, not compressed blocks or logical positions.
    Hq/Hkv are local head counts; the head dimension is fixed at 256.

    Upstream input contract
    -----------------------
    The current model has 24 total Q heads and 2 total KV heads, with DCP=1:

        attention TP       1      2      4      8
        local Hq:Hkv      24:2   12:1    6:1    3:1

    The full model's GDN shards conv_dim=10240 over the same attention TP.
    TP6/12/24 fail that divisibility constraint, even though attention alone
    could divide its heads. MTP shares the target's attention TP. Attention TP
    is not the service's total device count; NPU DCP>1 is excluded upstream.
    Both Q and the resolved KV pool must be BF16. Model dtype alone is not
    enough: a cache automatically resolved to FP8 is outside this version.

    Indexer configuration determines the selection width:
        S = (B+1)*c - 1 + t, with B in {512,2048} and compression ratio c>=2.
        Ordinary calls: t=0. Shared MTP tail: 3<=t<=c (draft steps + 1).
    For B=512,c=4 this gives S=2051, or S=2054/2055 with a shared MTP tail.
    Seventeen valid keys still occupy a full-width slots row with padding.
    The width check tests this formula, not the resource cost of huge configs.

    The wrapper checks host metadata:
    - Same NPU, the ranks/shapes/dtypes above, one of the four head pairs,
      0<=N<=INT32_MAX and 1025<=S<=INT32_MAX satisfying the width formula.
    - K/V strides exactly (Hkv*256,256,1): contiguous pool views, allowing
      storage offsets. No whole-pool copy or arbitrary-layout fallback.
    - Q strides: dim=1, head>=256, row>=(Hq-1)*head_stride+256.
      Slots strides: column=1, row>=S. Head/row gaps and storage offsets are
      allowed; all tensors need not pass is_contiguous().
    Unsupported tensor metadata raises ValueError; no reference fallback.
    softmax_scale accepts int/float (including bool via Python's int rule) or
    None; other types, including device tensors, raise TypeError. The legacy
    expression is scale or 1/16: None and numeric zero use the default 1/16.

    The caller guarantees contents; these are not scanned on the device or
    read on the host. Every nonnegative slot must be <N; there is no synchronous
    out-of-range error guarantee. Negative slots are masked before KV loads.
    Slot 0 is valid, and repeated slots retain repeated softmax contributions.
    Visibility and causality are already encoded in slots; no causal mask is
    rebuilt here. This physical-slot interpretation follows the upstream call
    chain, not every extra capability of the GPU kernels or Torch reference.

    R=0 or N=0 returns zeros without attention launches; N=0 requires all slots
    to be padding. S=0 is rejected. Nonempty all-padding inputs still enter a
    computation path and must produce exact zeros. Empty handling is not a
    fourth attention algorithm.

    Examples and computation paths
    ------------------------------
    Under default controls, metadata selects the path, not prefill/decode/verify.
    For the same tensors, changing the caller's phase label changes no math.

        Path | attention TP | Local Hq:Hkv      | Address check  | Implementation
        -----+--------------+-------------------+----------------+-------------------------
        1    | 8            | 3:1               | fast-path safe | Triton: three-head reuse
        2    | 1 / 2 / 4    | 24:2 / 12:1 / 6:1 | either range   | Torch + Triton hybrid
        3    | 8            | 3:1               | wide required  | Triton: int64 offsets

    Paths 1 and 3 are both online-softmax Triton. "Wide" means using int64
    offsets when the tensor spans exceed our int32 address-safety bounds.

    For TP8 BF16, each physical pool row uses 512 bytes for K and 512 for V.
    At N=4,194,304, K and V are each 2 GiB; N>=4,194,305 triggers the KV-span
    check and selects Path 3 (K+V exceed 4 GiB). This is ALLOCATED capacity
    for ONE layer on ONE rank, shared across requests, even if not full.
    It is not a single request's context length or memory summed over layers.
    Long contexts or many requests can require a larger pool; dispatch itself
    checks tensor metadata, not request lengths. Whether the current service
    actually allocates this capacity still needs runtime confirmation.

    The check also covers Q, slots and padded selection spans, which can
    independently select Path 3. The exact bounds are explained below.

    Example tensors (output always has the Q shape):

        Q                 K/V                    slots          path / splits
        [1,3,256]         [4096,1,256]            [1,2051]       1 / 16
        [8,3,256]         [4096,1,256]            [8,2055]       1 / 4
        [4096,3,256]      [4096,1,256]            [4096,2051]    1 / 1
        [32,24,256]       [4096,2,256]            [32,2051]      2
        [2,3,256]         [8388609,1,256]         [2,2051]       3 / 8

    Why these paths exist:
    1. Three-head Triton loads each [BLOCK,256] K/V tile once and reuses it
       across the three Q heads.
    2. The hybrid uses Triton masked gather,
       then Torch FP32 bmm/softmax/bmm. This production hybrid has measured
       graph benefits and bounded row gathers, with eager and memory tradeoffs.
       It is not mathematically required, nor an exception/reference fallback.
       The exact comparison baselines and costs are recorded in DELIVERY.md.
    3. Wide-address Triton uses one query head per task, with int64 offsets
       instead of three-head reuse.
       The example above allocated real contiguous K/V (8 GiB + 1024 bytes
       combined) in validation; it is not a zero-stride logical expansion.
       This path exists for address safety, not a benchmark-specific speedup.

    N fitting int32 does not imply that element offsets or byte spans fit.
    For BLOCK and L splits, _wide_offsets checks N, P=S+BLOCK*L, and
    P*slots.stride(1)*slots.element_size() against INT32_MAX. It also checks
    sum((size_i-1)*stride_i)*element_size for each of Q,K,V,slots (zero-sized
    axes contribute zero). This is the last-element byte displacement from
    the view's data pointer; storage offsets are already part of that pointer.
    P conservatively covers the padded selection range. The large-pool example
    is not a universal switch threshold. Row/output addressing stays int64.

    Tiles, splits and row chunks
    ---------------------------
    BLOCK=32 selection columns. Normally, safe three-head inputs use:
        R=1 / 2..7 / 8..31 / >=32  ->  L=16 / 8 / 4 / 1 splits.
    Few query rows need KV-axis parallelism; more rows need less scratch and
    merging. The scheduler first probes a conservative 32-split address bound
    (except R=2..7, which selects 8). If that probe fails it keeps 8 splits;
    the final address check decides reuse versus wide execution. Wide uses 8.
    These are tuning choices within one algorithm, not correctness boundaries
    or universal optima. Hybrid row chunks are described in torch_attention.py.

    Each split s covers T=ceil(S/(L*BLOCK)) consecutive tiles, starting at
    column s*T*BLOCK. Columns >=S and negative slots are masked. With L=1,
    write output directly; otherwise store FP32 maximum, normalizer and a
    256-element weighted numerator for each (query row, head, split).

    Flatten query rows and heads into R*Hq items. Each launch handles at most
    floor(65535/L) items, rounded down to a multiple of three for reuse.
    The first-stage grid is (items/3,L) for reuse or (items,L) for generic;
    merge has one task per item. Further launches advance the row/item offset
    and reuse scratch, so dynamic large R is not a fixed-grid limit. Sufficient
    device memory is still required; no arbitrary resource maximum is promised.

    How attention is computed
    -------------------------
    Let G=Hq/Hkv. Query head h reads KV head h//G. For every valid slot j:
        score_j = scale * dot(Q[r,h,:], K[slots[r,j], h//G, :])
        output[r,h,:] = sum_j softmax(score)_j * V[slots[r,j], h//G, :]
    Softmax runs over valid selection entries, including duplicates. Padding
    contributes nothing. Q/K/V are converted to FP32 for this computation.

    Online softmax keeps (m,l,a): score maximum, exponential sum, and weighted
    V sum. When a tile raises m to m', rescale the previous l and a by
    exp(m-m') before adding exp(score-m') and its weighted V contributions.
    For split states (m_s,l_s,a_s), merge using:
        M = max_s(m_s), f_s = exp(m_s-M)
        output = sum_s(f_s*a_s) / sum_s(f_s*l_s)
    This is a rescaled sum, not an average of split outputs. Safe maxima and
    denominator guards handle empty tiles/splits and exact all-padding zeros.
    The hybrid computes the same selected attention through FP32 matrices.

    Numerical policy and graph replay
    ---------------------------------
    INITIAL BF16 ACCEPTANCE: atol=rtol=0.02 against an independent CPU FP64
    oracle with direct BF16 quantization:
        abs(actual-reference) <= 0.02 + 0.02*abs(reference).
    This is not just 2% relative error and does not promise exact BF16 rounding.
    Six old 0.002 differences remain recorded; changing the acceptance policy
    was not an algorithm repair. See NUMERICAL_POLICY.md and the preserved
    regressions in tests/test_production.py; finite samples are not a proof
    for every legal input.

    Normal selected Q/K/V must be finite and the computation must not overflow.
    Computed -Inf scores become NaN before the padding mask is applied, so a
    valid -Inf cannot be hidden as zero weight beside finite scores. NaN/+Inf
    otherwise follow arithmetic. This explicit anomaly-exposure choice is not
    a uniform promise of all GPU paths. Nonfinite output is not a synchronous
    exception or automatic service termination, nor exhaustive detection of
    all numerical issues. Unused padding is isolated and all-padding rows
    remain exactly zero, even with unused nonfinite Q/K/V.

    Inputs are unchanged. Only inference forward is supported, not training
    backward. Warm up before capture; shapes, strides, addresses, scale and
    controls stay fixed. Q/K/V/slots can each be updated in place before replay;
    graph/output lifetimes and service anomaly diagnostics belong to the caller.
    Production keeps _CONFIG_OVERRIDE=None, _FORCE_WIDE=False, _GROUP_MODEL=True.
    Experimental overrides never bypass the metadata contract.
    """
    if softmax_scale is not None and not isinstance(softmax_scale, (int, float)):
        raise TypeError("softmax_scale must be a Python number or None")
    if not can_run_sparse_attention(q, k, v, slots):
        raise ValueError("Unsupported NPU sparse attention tensor configuration")
    rows, heads, dim = q.shape
    width = slots.shape[1]
    if rows == 0 or k.shape[0] == 0:
        return torch.zeros(q.shape, dtype=q.dtype, device=q.device)
    block, splits = _configuration(q, k, v, slots)
    wide_offsets = _wide_offsets(q, k, v, slots, block, splits)
    grouped = _use_grouped(q, k, wide_offsets)
    if _use_torch(q, k, slots):
        return torch_attention(q, k, v, slots, softmax_scale or dim**-0.5)
    items = rows * heads
    chunk_items = min(items, _MAX_GRID_PROGRAMS // splits)
    if grouped:
        chunk_items = chunk_items // 3 * 3
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    if splits == 1:
        partial = maxima = sums = output
    else:
        partial = torch.empty(
            (chunk_items, splits, dim), device=q.device, dtype=torch.float32
        )
        maxima = torch.empty(
            (chunk_items, splits), device=q.device, dtype=torch.float32
        )
        sums = torch.empty_like(maxima)
    for start in range(0, items, chunk_items):
        count = min(chunk_items, items - start)
        if grouped:
            _grouped_partials[(count // 3, splits)](
                q,
                k,
                v,
                slots,
                partial,
                maxima,
                sums,
                output,
                *q.stride(),
                k.stride(0),
                v.stride(0),
                *slots.stride(),
                dim,
                width,
                splits,
                triton.cdiv(width, splits * block),
                softmax_scale or dim**-0.5,
                block,
                start // 3,
                enable_fp_fusion=False,
            )
        else:
            _sparse_partials[(count, splits)](
                q,
                k,
                v,
                slots,
                partial,
                maxima,
                sums,
                output,
                *q.stride(),
                *k.stride(),
                *v.stride(),
                *slots.stride(),
                heads,
                heads // k.shape[1],
                dim,
                width,
                splits,
                triton.cdiv(width, splits * block),
                softmax_scale or dim**-0.5,
                block,
                start,
                wide_offsets,
                enable_fp_fusion=False,
            )
        if splits != 1:
            _merge_partials[(count,)](
                partial,
                maxima,
                sums,
                output,
                dim,
                splits,
                start,
                enable_fp_fusion=False,
            )
    return output
