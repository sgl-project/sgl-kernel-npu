"""Independent n-gram hashes using gathered history windows and reductions."""

import torch


def compute_n_gram_ids(
    oe_weights,
    oe_mods,
    exclusive_oe_embeder_size_sums,
    tokens,
    exclusive_req_len_sums,
    oe_token_table,
    row_indices,
    column_starts,
    batch_size,
    oe_n,
    oe_k,
    max_context_len,
):
    """Evaluate all n-gram orders per request without invoking the native op."""
    output = torch.empty(
        (tokens.shape[0], (oe_n - 1) * oe_k),
        dtype=torch.int32,
        device=tokens.device,
    )
    # Widen arithmetic so int32 token * weight products retain their value
    # before modular reduction; this does not adapt the native input contract.
    weights = oe_weights.to(torch.int64)
    mods = oe_mods.to(torch.int64)
    offsets = exclusive_oe_embeder_size_sums[:-1].reshape(oe_n - 1, oe_k)
    history_steps = torch.arange(oe_n, device=tokens.device)
    orders = torch.arange(2, oe_n + 1, device=tokens.device)
    order_mask = history_steps[None, :] < orders[:, None]
    table = oe_token_table.flatten()

    # Host iteration is confined to this correctness reference. Each request's
    # tokens/orders/embedding splits are evaluated with tensor operations.
    ends = exclusive_req_len_sums.cpu().tolist()
    rows = row_indices.cpu().tolist()
    columns = column_starts.cpu().tolist()
    start = 0
    for request in range(batch_size):
        end = ends[request]
        if end == start:
            continue
        positions = (
            columns[request]
            + torch.arange(end - start, device=tokens.device)[:, None]
            - history_steps[None, :]
        )
        history = table[rows[request] * max_context_len + positions.clamp_min(0)]
        invalid = (positions < 0) | (history < 0)
        # Once a negative sentinel or the row boundary is encountered, older
        # tokens contribute nothing, even if those older tokens are positive.
        valid_history = invalid.to(torch.int32).cumsum(dim=-1, dtype=torch.int32) == 0
        history = torch.where(valid_history, history, 0).to(torch.int64)
        terms = history[:, None, None, :] * weights[None, :, :, :]
        terms = torch.remainder(terms, mods[None, :, :, None])
        terms = torch.where(order_mask[None, :, None, :], terms, 0)
        ids = torch.remainder(terms.sum(dim=-1), mods) + offsets
        output[start:end] = ids.reshape(end - start, (oe_n - 1) * oe_k).to(torch.int32)
        start = end
    return output
