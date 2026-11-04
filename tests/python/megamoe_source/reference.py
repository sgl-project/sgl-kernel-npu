"""Small, independent BF16 reference for the imported MegaMoe source.

Inputs represent all ranks on CPU. This is a correctness reference, not a
distributed implementation or a model for kernel performance. MXFP8 and
device execution are intentionally outside its scope.
"""

import torch
import torch.nn.functional as F


def _expert(x, w1, w2):
    projected = (w1.float() @ x.float()).to(torch.bfloat16).float()
    gate, up = projected.chunk(2)
    hidden = (F.silu(gate) * up).to(torch.bfloat16).float()
    return (w2.float() @ hidden).to(torch.bfloat16)


def reference_bf16(
    xs, ids, probabilities, first, second, shared_first=None, shared_second=None
):
    """Return (one BF16 output and local int32 expert counts per rank).

    Each rank owns ``first[r][e]`` and ``second[r][e]``. Global expert IDs
    use contiguous rank-major numbering. Token and probability tensors are
    local to each source rank. The reference accumulates routed contributions
    in owner-rank, then local-expert order with BF16 rounding at each stage.
    Shared experts are local to each source rank and added without a route
    probability. Each first projection uses the gate/up weight layout.
    """
    world = len(xs)
    if world < 2 or any(
        len(items) != world for items in (ids, probabilities, first, second)
    ):
        raise ValueError("Expected at least two equally populated rank lists")
    if (shared_first is None) != (shared_second is None):
        raise ValueError("Both shared weight lists must be supplied together")
    if shared_first is not None and (
        len(shared_first) != world or len(shared_second) != world
    ):
        raise ValueError("Shared weight lists must have one tensor per rank")
    experts = first[0].shape[0]
    hidden = xs[0].shape[1]
    if experts < 1 or experts * world > 1024 or hidden < 1:
        raise ValueError("Expert count and hidden size must be within supported bounds")
    for rank in range(world):
        if (
            xs[rank].ndim != 2
            or xs[rank].shape[1] != hidden
            or xs[rank].shape[0] < 1
            or xs[rank].dtype != torch.bfloat16
        ):
            raise ValueError("x must be BF16 [tokens, hidden]")
        if (
            ids[rank].ndim != 2
            or ids[rank].shape != probabilities[rank].shape
            or ids[rank].shape[0] != xs[rank].shape[0]
            or not 1 <= ids[rank].shape[1] <= experts * world
        ):
            raise ValueError("Each route tensor must have shape [tokens, topk]")
        if (
            ids[rank].dtype != torch.int32
            or probabilities[rank].dtype != torch.bfloat16
        ):
            raise ValueError("IDs must be int32 and routing weights BF16")
        if first[rank].ndim != 3 or second[rank].ndim != 3:
            raise ValueError("Expert weights must be rank-3 tensors")
        if first[rank].dtype != torch.bfloat16 or second[rank].dtype != torch.bfloat16:
            raise ValueError("Expert weights must be BF16")
        if (
            first[rank].shape[0] != experts
            or first[rank].shape[2] != hidden
            or first[rank].shape[1] < 2
            or first[rank].shape[1] % 2
        ):
            raise ValueError(
                "First weights must be [local experts, 2 * intermediate, hidden]"
            )
        if tuple(second[rank].shape) != (experts, hidden, first[rank].shape[1] // 2):
            raise ValueError(
                "Second weights must be [local experts, hidden, intermediate]"
            )
        if ids[rank].numel() and (
            ids[rank].min() < 0 or ids[rank].max() >= experts * world
        ):
            raise ValueError("Expert ID is out of range")
        if any(row.unique().numel() != row.numel() for row in ids[rank]):
            raise ValueError("A token must not select an expert more than once")
        if shared_first is not None:
            sw1, sw2 = shared_first[rank], shared_second[rank]
            if (
                sw1.dtype != torch.bfloat16
                or sw2.dtype != torch.bfloat16
                or sw1.ndim != 3
                or sw2.ndim != 3
            ):
                raise ValueError("Shared weights must be rank-3 BF16 tensors")
            if (
                sw1.shape[2] != hidden
                or sw1.shape[1] % 2
                or tuple(sw2.shape) != (sw1.shape[0], hidden, sw1.shape[1] // 2)
            ):
                raise ValueError("Shared expert shapes do not match")
    counts = [torch.zeros(experts, dtype=torch.int32) for _ in range(world)]
    outputs = []
    for rank, x in enumerate(xs):
        out = torch.zeros_like(x)
        for row in range(x.shape[0]):
            owner_parts = [
                torch.zeros(hidden, dtype=torch.bfloat16) for _ in range(world)
            ]
            slots = sorted(
                range(ids[rank].shape[1]), key=lambda slot: int(ids[rank][row, slot])
            )
            for slot in slots:
                owner, local = divmod(int(ids[rank][row, slot]), experts)
                counts[owner][local] += 1
                y = _expert(x[row], first[owner][local], second[owner][local])
                term = (y * probabilities[rank][row, slot]).to(torch.bfloat16)
                owner_parts[owner] = (owner_parts[owner] + term).to(torch.bfloat16)
            combined = torch.zeros(hidden, dtype=torch.bfloat16)
            for partial in owner_parts:
                combined = (combined + partial).to(torch.bfloat16)
            if shared_first is not None:
                for index in range(shared_first[rank].shape[0]):
                    combined = (
                        combined
                        + _expert(
                            x[row],
                            shared_first[rank][index],
                            shared_second[rank][index],
                        )
                    ).to(torch.bfloat16)
            out[row] = combined
        outputs.append(out)
    return outputs, counts
