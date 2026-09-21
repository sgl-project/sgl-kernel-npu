"""Exact NPU QSA Top-K for finite valid scores and contiguous row bounds.

Production artifact: this file only; no legacy kernel/experiments imports.
The finite-value premise is model-specific, not a restriction of GPU Top-K.
"""
from functools import lru_cache
import torch
import triton
import triton.language as tl
from triton.language.extra.cann import extension as ext

IMPLEMENTATION_VERSION = "qsa-model-v1"
MAX_WIDTH = 2**24
WORKSPACE_BYTES = 128 * 1024 * 1024
TILED_MAX_WIDTH = 262144


def select_implementation(rows, columns, topk):
    """Choose a schedule from score[R,M] and block_topk K, using host shapes.

    rows=R counts query rows; columns=M and topk=K count compressed blocks.
    Reject M>2**24 before choosing even the empty case. Return "shortcut" for
    R=0 or M<=K, but fast_topk handles these differently: R=0 only creates an
    empty [0,K] output; nonempty M<=K launches ordered-index/padding writes.
    Otherwise choose tiled if M<=262144 and R*M*4<128 MiB; choose hybrid for
    every other supported shape. R*M*4 is logical FP32 input size, excluding
    gaps in score.stride(0). No lengths, starts or scores are read on the host.

    These two thresholds preserve the measured coarse policy, not a semantic
    boundary or a proven optimal crossover. Individual L<=K rows still take
    short-row handling inside tiled/hybrid; their values do not change this
    host dispatch. See fast_topk for examples, chunking and exact computation.
    """
    if columns > MAX_WIDTH:
        raise ValueError(f"score width exceeds supported maximum {MAX_WIDTH}")
    if rows == 0 or columns <= topk:
        return "shortcut"
    if columns > TILED_MAX_WIDTH or rows * columns * 4 >= WORKSPACE_BYTES:
        return "hybrid"
    return "tiled"


@triton.jit
def _short(Lengths, Output, ROWS: tl.constexpr,
           K: tl.constexpr, BR: tl.constexpr, CORES: tl.constexpr):
    for block in range(tl.program_id(0), tl.cdiv(ROWS, BR), CORES):
        rows = block * BR + tl.arange(0, BR)
        length = tl.load(Lengths + rows.to(tl.int64), rows < ROWS, other=0)
        cols = tl.arange(0, K)
        result = tl.where(cols[None, :] < length[:, None], cols[None, :], -1)
        tl.store(Output + rows.to(tl.int64)[:, None]*K + cols[None, :], result, rows[:, None] < ROWS)


@lru_cache(maxsize=None)
def _vector_cores(device):
    return triton.runtime.driver.active.utils.get_device_properties(device)["num_vectorcore"]

_SORT_BLOCK = 4096
_MAX_PROGRAMS = 65535


@triton.jit
def _reduce_candidates(Scores, Lengths, Starts, Values, Threshold,
                       stride, WIDTH: tl.constexpr, OUT_WIDTH: tl.constexpr,
                       K: tl.constexpr, HAS_STARTS: tl.constexpr,
                       FIRST: tl.constexpr, FINAL: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    length = tl.load(Lengths + row)
    if length > K:
        start = 0
        if FIRST and HAS_STARTS:
            start = tl.load(Starts + row)
        cols = tile * BLOCK + tl.arange(0, BLOCK)
        valid = cols < WIDTH
        if FIRST:
            valid = valid & (cols < length)
        x = tl.load(Scores + row * stride + start + cols, valid, other=-float("inf"))
        ordered = ext.sort(x, descending=True)
        if FINAL:
            cutoff = ext.get_element(ordered, (K - 1,))
            tl.store(Threshold + row, cutoff)
        else:
            best = ext.extract_slice(ordered, (0,), (K,), (1,))
            tl.store(Values + row * OUT_WIDTH + tile * K + tl.arange(0, K), best)


@triton.jit
def _counts(Scores, Lengths, Starts, Threshold, Counts,
            stride: tl.constexpr,
            K: tl.constexpr, HAS_STARTS: tl.constexpr, TILES: tl.constexpr,
            BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    length = tl.load(Lengths + row)
    if length > K:
        start = 0
        if HAS_STARTS:
            start = tl.load(Starts + row)
        cols = tile * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(Scores + row * stride + start + cols, cols < length, other=-float("inf"))
        cutoff = tl.load(Threshold + row)
        greater = tl.sum(((cols < length) & (x > cutoff)).to(tl.int32), 0)
        equal = tl.sum(((cols < length) & (x == cutoff)).to(tl.int32), 0)
        tl.store(Counts + row * (2*TILES) + tile, greater)
        tl.store(Counts + row * (2*TILES) + TILES + tile, equal)


@triton.jit
def _emit_tiles(Scores, Lengths, Starts, Threshold, Counts, Output,
                stride: tl.constexpr,
                K: tl.constexpr, HAS_STARTS: tl.constexpr, TILES: tl.constexpr,
                BLOCK: tl.constexpr, CT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    length = tl.load(Lengths + row)
    if length <= K:
        if tile == 0:
            cols = tl.arange(0, K)
            tl.store(Output + row * K + cols, tl.where(cols < length, cols, -1))
    else:
        start = 0
        if HAS_STARTS:
            start = tl.load(Starts + row)
        t = tl.arange(0, CT)
        gs = tl.load(Counts + row*(2*TILES) + t, t<TILES, other=0)
        es = tl.load(Counts + row*(2*TILES) + TILES + t, t<TILES, other=0)
        total = tl.sum(gs, 0)
        gb = tl.sum(tl.where(t<tile, gs, 0), 0)
        eb = tl.sum(tl.where(t<tile, es, 0), 0)
        cols = tile * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(Scores + row*stride + start + cols, cols<length, other=-float("inf"))
        cutoff = tl.load(Threshold + row)
        # Exact index sorting produces contiguous writes for each category.
        greater = (cols < length) & (x > cutoff)
        equal = (cols < length) & (x == cutoff)
        local = tl.arange(0, BLOCK)
        gi = ext.sort(tl.where(greater, cols.to(tl.float32), float("inf")))
        ei = ext.sort(tl.where(equal, cols.to(tl.float32), float("inf")))
        gc = tl.sum(greater.to(tl.int32), 0)
        ec = tl.sum(equal.to(tl.int32), 0)
        tl.store(Output + row*K + gb + local, gi.to(tl.int32), local < gc)
        tl.store(Output + row*K + total + eb + local, ei.to(tl.int32),
                 (local < ec) & (total + eb + local < K))

@triton.jit
def _single_tile(Scores, Lengths, Starts, Output, stride,
                K: tl.constexpr, HAS_STARTS: tl.constexpr, BLOCK: tl.constexpr,
                ROWS: tl.constexpr, CORES: tl.constexpr):
    for row_id in range(tl.program_id(0), ROWS, CORES):
        row = row_id.to(tl.int64)
        length = tl.load(Lengths + row)
        out_cols = tl.arange(0, K)
        if length <= K:
            tl.store(Output + row * K + out_cols, tl.where(out_cols < length, out_cols, -1))
        else:
            start = 0
            if HAS_STARTS:
                start = tl.load(Starts + row)
            cols = tl.arange(0, BLOCK)
            x = tl.load(Scores + row * stride + start + cols, cols < length, other=-float("inf"))
            ordered = ext.sort(x, descending=True)
            cutoff = ext.get_element(ordered, (K - 1,))
            # FP32 index keys are exact: 0 <= key <= 2*BLOCK <= 8192.
            keys = tl.where((cols < length) & (x > cutoff), cols,
                            tl.where((cols < length) & (x == cutoff), BLOCK + cols, 2*BLOCK))
            ordered_ids = ext.sort(keys.to(tl.float32))
            chosen = ext.extract_slice(ordered_ids, (0,), (K,), (1,)).to(tl.int32)
            chosen = tl.where(chosen >= BLOCK, chosen - BLOCK, chosen)
            tl.store(Output + row*K + out_cols, chosen)


def tiled_topk(score, lengths, topk, row_starts=None):
    rows, columns = score.shape
    output = torch.empty((rows, topk), dtype=torch.int32, device=score.device)
    starts = lengths if row_starts is None else row_starts
    if columns <= _SORT_BLOCK:
        block = max(topk, triton.next_power_of_2(columns))
        layout = (score.stride(0), topk,
                  row_starts is not None, block)
        # One schedule for every R: one program per row up to the vector-core
        # count, then the same kernel loops over additional rows.
        cores = min(rows, _vector_cores(score.device.index))
        _single_tile[(cores,)](score, lengths, starts, output, *layout, rows, cores)
        return output
    # Each launch respects the product of grid axes, including all sort stages.
    tiles = triton.cdiv(columns, _SORT_BLOCK)
    chunk_rows = max(1, _MAX_PROGRAMS // tiles)
    for base in range(0, rows, chunk_rows):
        end = min(rows, base+chunk_rows)
        x, lens, out = score[base:end], lengths[base:end], output[base:end]
        starts = lens if row_starts is None else row_starts[base:end]
        threshold = torch.empty(end-base, dtype=torch.float32, device=score.device)
        source, width, first = x, columns, True
        while True:
            block = min(_SORT_BLOCK, max(topk, triton.next_power_of_2(width)))
            sort_tiles = triton.cdiv(width, block)
            final = sort_tiles <= 1
            # Keeping K from each tile is exact: every discarded value already
            # has at least K values no smaller than it in its own tile.
            values = threshold if final else torch.empty(
                (end-base, sort_tiles*topk), dtype=torch.float32, device=x.device)
            _reduce_candidates[(end-base, sort_tiles)](
                source, lens, starts, values, threshold, source.stride(0),
                width, sort_tiles*topk, topk,
                row_starts is not None, first, final, block)
            if final:
                break
            source, width, first = values, sort_tiles*topk, False
        counts = torch.empty((end-base, 2*tiles), dtype=torch.int32, device=x.device)
        args = (x, lens, starts, threshold, counts)
        layout = (x.stride(0), topk,
                  row_starts is not None, tiles, _SORT_BLOCK)
        _counts[(end-base, tiles)](*args, *layout)
        _emit_tiles[(end-base, tiles)](*args, out, *layout, triton.next_power_of_2(tiles))
    return output


@triton.jit
def _pack(X, Lengths, Starts, Packed, stride,
          M: tl.constexpr, K: tl.constexpr, HAS_STARTS: tl.constexpr,
          TILES: tl.constexpr, TASKS: tl.constexpr, CORES: tl.constexpr,
          BLOCK: tl.constexpr):
    for task in range(tl.program_id(0), TASKS, CORES):
        row = (task // TILES).to(tl.int64)
        cols = (task % TILES) * BLOCK + tl.arange(0, BLOCK)
        length = tl.load(Lengths + row)
        start = 0
        if HAS_STARTS:
            start = tl.load(Starts + row)
        # Only long-row valid scores are read. Invalid/short rows pack as -inf.
        values = tl.load(X + row * stride + start + cols,
                         (cols < length) & (length > K), other=-float("inf"))
        tl.store(Packed + row * M + cols, values, cols < M)


@triton.jit
def _finish(Lengths, Indices, Output, K: tl.constexpr,
            ROWS: tl.constexpr, CORES: tl.constexpr):
    for rid in range(tl.program_id(0), ROWS, CORES):
        row = rid.to(tl.int64)
        length = tl.load(Lengths + row)
        ranks = tl.arange(0, K)
        if length <= K:
            ids = tl.where(ranks < length, ranks, -1)
        else:
            # Finite valid scores strictly outrank every packed -inf.
            # No value reads, infinity repair, prefix scan or score rescan.
            ids = tl.load(Indices + row * K + ranks).to(tl.int32)
        tl.store(Output + row * K + ranks, ids)


def hybrid_topk(score, lengths, topk, row_starts=None):
    """One native selection per row chunk; 32-column alignment is retained.

    At M<=2**24 a single aligned FP32 row fits in 64 MiB. Chunking rows bounds
    the pack allocation to 128 MiB; output/native/graph memory is additional.
    Each launch is capped at vector cores; task loops cover all rows and tails.
    There is no column subdivision or candidate merge in this implementation.
    """
    rows, columns = score.shape
    output = torch.empty((rows, topk), dtype=torch.int32, device=score.device)
    cores = _vector_cores(score.device.index)
    packed_width = triton.cdiv(columns, 32) * 32
    chunk = max(1, min(65535, WORKSPACE_BYTES // (packed_width * 4)))
    for begin in range(0, rows, chunk):
        end = min(rows, begin + chunk)
        x, lens, out = score[begin:end], lengths[begin:end], output[begin:end]
        starts = lens if row_starts is None else row_starts[begin:end]
        packed = torch.empty((end-begin, packed_width), dtype=score.dtype, device=score.device)
        tiles = triton.cdiv(packed_width, 4096)
        tasks = (end-begin) * tiles
        grid = min(cores, tasks)
        _pack[(grid,)](x, lens, starts, packed, x.stride(0),
                      packed_width, topk, row_starts is not None,
                      tiles, tasks, grid, 4096)
        values, indices = torch.topk(packed, topk, dim=1, sorted=True)
        grid = min(cores, end-begin)
        _finish[(grid,)](lens, indices, out, topk, end-begin, grid)
    return output


def fast_topk(score, lengths, topk, row_starts=None):
    """Select exact sequence-local compressed QSA block indices.

    Inputs and output
    -----------------
        score       [R, M]   FP32 scores, with contiguous columns on an NPU
        lengths     [R]      contiguous int32 valid lengths on the same NPU
        row_starts  [R]      contiguous int32 starts, or None for all zeros
        output      [R, K]   new contiguous int32 tensor on the same NPU

        R           query rows in this call, including caller-padded rows
        M           physical columns in each score row, in compressed blocks
        K           block_topk output slots: Python int 512 or 2048
        S, L        a row's start and valid length; select score[r,S:S+L]

    M, K, S and L count compressed blocks, NOT original tokens. Current model:
    token budget=2048, compression ratio=4, hence block_topk=512. K=2048 is also
    supported by QSA configuration checks and the GPU interface. TP does not
    mechanically divide these dimensions by the TP degree. Output index b
    refers to score[r,S+b], not physical column b or an original token slot.

    Upstream construction and wrapper contract
    ------------------------------------------
    Prefill M is the total number of packed compressed keys across requests;
    R is the current query chunk, after the caller's prefill row chunking.
    Prefix sums and index_select provide each query's packed start; complete
    visible block counts determine its end. The framework passes ends-starts
    as lengths. Decode/verify M comes from compressed page-table capacity and
    the caller's MQA output width; each query has its own valid length, often
    with S=0. R is not fixed to the current number of requests, nor is M fixed
    to one request's valid length. Graph buffers use contiguous prefix slices.

    The wrapper checks host metadata and raises ValueError for violations:
    - The shapes/dtypes/devices above and strided tensor layout; score must
      have stride(1)=1 and nonnegative stride(0), which may exceed M. Bounds
      must be is_contiguous(), including framework-contiguous prefix slices.
    - K must be a Python int in {512,2048}; bool is not accepted. M<=2**24,
      even for R=0. Score byte span and output size must fit signed int64.
      Actual allocations still require enough device memory.
    GPU JIT permits score row stride and requires contiguous bounds through
    TensorMatcher. The width cap is this NPU wrapper's support boundary, not
    a GPU/model theoretical limit or a fixed maximum number of requests.

    The caller guarantees device contents; these are NOT scanned/validated:
    - 0<=S, 0<=L and S+L<=M. In-row endpoints beyond int32 are unsupported;
      normal cross-row row*stride addresses still use int64.
    - Every valid score is finite FP32. Finite negative values and ties,
      including many ReLU zeros, are supported. Valid NaN/+inf/-inf have no
      correctness guarantee: there is no runtime rejection, repair or rescan.
    QSA forms scores using sum(ReLU(QK))/positive_scale. ReLU is not sufficient
    to guarantee finiteness: invalid inputs or overflow can break the premise.
    The GPU general entrypoint does not explicitly enforce this model premise.
    Invalid columns may contain arbitrary data, including -inf/NaN; they are
    not read for selection. No full-score validation or host scalar sync is
    added. Unsupported metadata does not silently enter a reference fallback.

    Shape dispatch and examples
    ---------------------------
        score shape; K          schedule             output shape
        -----------------------+--------------------+--------------
        [0,4096]; 512           shortcut: empty      [0,512]
        [8,256]; 512            shortcut: M<=K       [8,512]
        [8,4096]; 512           tiled                [8,512]
        [2048,16384]; 512       hybrid (128 MiB)     [2048,512]
        [2,262145]; 2048        hybrid (width)       [2,2048]

    After metadata checks, R=0 returns an empty tensor with no kernel launch.
    For nonempty M<=K, the shape and caller bounds imply every L<=K: launch
    ordered relative-index/padding writes using lengths, without reading
    scores or starts. Otherwise tiled requires M<=262144 AND R*M*4<128 MiB;
    all other supported shapes use hybrid. The limits are coarse scheduling
    policies, not mathematical semantics or proven optimal performance points.
    A row with L<=K inside a wider matrix is also short, but cannot trigger
    the host shape shortcut: tiled/hybrid handle it internally on the device.

    Tiles and row chunks
    --------------------
    Tiled: M<=4096 uses one padded sort tile per long row, with programs up to
    the vector-core count looping over rows. Larger M uses 4096-element tiles,
    retains K values per tile, and recursively reduces these candidates until
    one tile gives the exact K-th-largest threshold. Discarding a tile's lower
    values is exact: each has at least K no-smaller values in its own tile.
    For T=ceil(M/4096), process at most max(1,65535//T) rows per chunk so the
    product of grid axes stays <=65535, including subsequent reduction stages.
    Tail masks exclude columns outside each row's valid interval.

    Hybrid: pack each row's [S,S+L) into relative columns starting at zero and
    pad to P=ceil(M/32)*32 columns. Use at most
    max(1,min(65535,128 MiB//(P*4))) rows per pack chunk. At M=2**24, each packed
    row uses 64 MiB, so three rows need chunks of two and one. Pack tasks loop
    over 4096-element tiles with a grid capped by the vector-core count.
    The 128 MiB budget limits one pack allocation, NOT total output/native
    workspace/graph-pool memory. Row chunks, alignment, masks and launch
    parameters are resource scheduling within these algorithms, not new paths.
    There is no extra wide-column subdivision or candidate-merge algorithm.

    How the paths compute the result
    --------------------------------
    Shortcut writes [0,...,L-1,-1,...,-1] for each row; it performs no score
    selection. Tiled uses the exact threshold, counts values strictly above
    it and values equal to it, then emits all greater values and enough ties.
    Per-tile counts give disjoint output ranges; relative indices are unique
    and legal. The single-tile variant applies the same threshold/tie rule.
    Short rows bypass score sorting and write ordered indices plus padding.

    Hybrid reads valid scores only for L>K; other pack entries are -inf.
    It invokes torch.topk once per row chunk. Finite valid scores strictly
    outrank padding, so selected native indices already have relative meaning.
    The finish kernel casts long-row indices to int32; for L<=K it ignores
    native indices and writes the ordered short-row result with -1 padding.
    No infinity repair, prefix scan or score-value rescan is needed.

    For L>K, exactly K indices select the largest-value multiset; ties may
    choose different indices across valid implementations and output need not
    be score-sorted. For L<=K, order is fixed. Valid block indices always form
    a prefix followed only by -1, as required by the downstream expansion.
    Top-K does not expand tokens, append MTP slots or map logical to physical
    addresses; those remain caller responsibilities.

    Inputs are unchanged. R/M may vary between calls. Warm up before graph
    capture; replay may update scores, lengths and starts independently in
    place while preserving the contract. Shapes, strides, storage, K and the
    presence/absence of starts remain fixed within that graph. Graph/output
    lifetimes and valid versus inert padded query rows remain with the caller.
    """
    if type(topk) is not int or topk not in (512, 2048):
        raise ValueError("QSA NPU top-k supports integer K=512 or 2048")
    if (not isinstance(score, torch.Tensor) or score.layout != torch.strided
            or score.device.type != "npu" or score.dtype != torch.float32
            or score.ndim != 2 or score.stride(1) != 1 or score.stride(0) < 0):
        raise ValueError("score must be NPU FP32 [R,M], contiguous in columns")
    rows, columns = score.shape
    implementation = select_implementation(rows, columns, topk)
    bounds = (lengths,) if row_starts is None else (lengths, row_starts)
    for tensor in bounds:
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tensor.shape != (rows,) or tensor.dtype != torch.int32
                or tensor.device != score.device or not tensor.is_contiguous()):
            raise ValueError("row bounds must be contiguous int32 [R] on the score device")
    limit = 2**63 - 1
    end = score.storage_offset() + max(rows-1, 0)*score.stride(0) + columns
    if end * score.element_size() > limit or rows * topk * 4 > limit:
        raise ValueError("score/output byte offsets must fit signed int64")
    if implementation == "shortcut":
        output = torch.empty((rows, topk), dtype=torch.int32, device=score.device)
        if rows:
            cores = min(_vector_cores(score.device.index), triton.cdiv(rows, 4))
            _short[(cores,)](lengths, output, rows, topk, 4, cores)
        return output
    if implementation == "hybrid":
        return hybrid_topk(score, lengths, topk, row_starts)
    return tiled_topk(score, lengths, topk, row_starts)
