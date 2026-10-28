"""Torch implementation of DSpark Markov sampling, adapted from SGLang."""

import torch


def _mul32(a: torch.Tensor, b: int) -> tuple[torch.Tensor, torch.Tensor]:
    # (high, low) 32-bit words of a * b; b splits in halves so int64 never overflows.
    product_lo = a * (b & 0xFFFF)
    product_hi = a * (b >> 16)
    high = (product_hi + (product_lo >> 16)) >> 16
    low = (((product_hi & 0xFFFF) << 16) + product_lo) & 0xFFFFFFFF
    return high, low


def _philox(key: int, c0, c1, c2, c3) -> tuple[torch.Tensor, ...]:
    """Philox4x32-10, the same function as ``tl.philox``."""
    k0, k1 = key & 0xFFFFFFFF, (key >> 32) & 0xFFFFFFFF
    for _ in range(10):
        hi_b, lo_b = _mul32(c2, 0xCD9E8D57)
        hi_a, lo_a = _mul32(c0, 0xD2511F53)
        c0, c1, c2, c3 = hi_b ^ c1 ^ k0, lo_b, hi_a ^ c3 ^ k1, lo_a
        k0 = (k0 + 0x9E3779B9) & 0xFFFFFFFF
        k1 = (k1 + 0xBB67AE85) & 0xFFFFFFFF
    return c0, c1, c2, c3


def _check(
    base_logits,
    bias,
    temperatures,
    offset,
    step_dtype,
    vocab_start,
    gather,
    gather_logits,
    logits_out,
):
    assert base_logits.ndim == bias.ndim == 2
    assert bias.shape[0] == base_logits.shape[0]
    assert bias.shape[1] <= base_logits.shape[1]
    assert bias.stride(1) == base_logits.stride(1) == 1
    assert temperatures.shape == (base_logits.shape[0],)
    assert temperatures.dtype == torch.float32 and temperatures.stride(0) == 1
    assert offset.dtype == torch.int64 and offset.numel() == 1
    assert step_dtype in (torch.float32, torch.float16, torch.bfloat16), step_dtype
    # Philox words go to tokens in quads of global ids.
    assert vocab_start % 4 == 0, vocab_start
    if logits_out is not None:
        assert logits_out.dtype == step_dtype and logits_out.stride(1) == 1
        if gather is None:
            assert logits_out.shape == base_logits.shape
        else:
            assert gather_logits is not None
            assert logits_out.shape[0] == base_logits.shape[0]


def markov_sample_step_torch(
    *,
    base_logits,
    bias,
    temperatures,
    key,
    offset,
    step,
    step_dtype,
    vocab_start=0,
    gather=None,
    gather_logits=None,
    logits_out=None,
    has_sampled_rows=None,
    exp_noise=None,
):
    # None draws the noise for every row: reading the temperatures on the
    # host would break a graph capture.
    _check(
        base_logits,
        bias,
        temperatures,
        offset,
        step_dtype,
        vocab_start,
        gather,
        gather_logits,
        logits_out,
    )
    if exp_noise is not None:
        noise_shape = (
            logits_out.shape
            if gather is not None and logits_out is not None
            else base_logits.shape
        )
        assert exp_noise.shape == noise_shape
        assert exp_noise.device == base_logits.device
        assert exp_noise.dtype == torch.float32 and exp_noise.stride(1) == 1
    rows, width = bias.shape
    value = base_logits.to(torch.float32, copy=True)
    value[:, :width] += bias.float()
    value = value.to(step_dtype).float()
    if gather is not None and logits_out is not None:
        full = gather_logits(value.to(step_dtype))[:, : logits_out.shape[1]]
        return markov_sample_step_torch(
            base_logits=full,
            bias=torch.zeros_like(full),
            temperatures=temperatures,
            key=key,
            offset=offset,
            step=step,
            step_dtype=step_dtype,
            logits_out=logits_out,
            has_sampled_rows=has_sampled_rows,
            exp_noise=exp_noise,
        )
    padded = base_logits.shape[1]
    device = base_logits.device
    if logits_out is not None:
        logits_out.copy_(value)

    keys = value
    if has_sampled_rows is not False:
        if exp_noise is None:
            quads = vocab_start // 4 + torch.arange(
                (padded + 3) // 4, device=device, dtype=torch.int64
            )
            words = _philox(
                key,
                quads[None, :],
                torch.arange(rows, device=device, dtype=torch.int64)[:, None],
                torch.full((1, 1), step, device=device, dtype=torch.int64),
                offset.reshape(1, 1) & 0xFFFFFFFF,
            )
            words = torch.broadcast_tensors(*words)
            bits = torch.stack(words, dim=-1).reshape(rows, -1)[:, :padded]
            uniform = (bits.float() + 0.5) * 2.3283064365386963e-10
            exp_noise = -torch.log1p(-uniform)
        sampled = temperatures != 0
        safe_temperatures = torch.where(sampled, temperatures, 1.0)[:, None]
        noised = value / safe_temperatures - torch.log(exp_noise)
        keys = torch.where(sampled[:, None], noised, value)
    cols = torch.arange(padded, device=device)
    keys = torch.where(cols[None, :] < width, keys, -torch.inf)

    if width == 0:
        local_values = torch.full((rows,), -torch.inf, device=device)
        local_indices = torch.full(
            (rows,), 2147483647, dtype=torch.int32, device=device
        )
    else:
        local = keys.argmax(dim=1)
        local_values = keys.gather(1, local[:, None]).squeeze(1)
        local_indices = (local + vocab_start).to(torch.int32)
    partial = torch.stack(
        [local_values.float(), local_indices.view(torch.float32)], dim=1
    )
    gathered = partial if gather is None else gather(partial)
    gathered = gathered.view(-1, rows, 2)
    values = gathered[..., 0].transpose(0, 1)
    indices = gathered[..., 1].contiguous().view(torch.int32) & 2147483647
    indices = indices.transpose(0, 1).long()
    # Lowest index among the NaNs if any, else among the maxima.
    valid = indices != 2147483647
    nan = valid & values.isnan()
    has_nan = nan.any(dim=-1, keepdim=True)
    finite = torch.where(valid & ~nan, values, -torch.inf)
    maximum = finite.amax(dim=-1, keepdim=True)
    wins = torch.where(has_nan, nan, valid & (finite == maximum))
    return torch.where(wins, indices, 2147483647).amin(dim=-1)
