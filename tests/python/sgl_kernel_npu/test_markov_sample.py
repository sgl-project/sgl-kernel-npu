"""DSpark Markov sampling: CPU Philox reference, sharding and NPU graph replay."""

import unittest
from typing import Optional

import torch
from sgl_kernel_npu.sample import markov_sample_step

DEVICE = torch.device("npu")
_NO_INDEX = 2147483647
_UNIFORM_SCALE = 2.3283064365386963e-10
_PHILOX_ROUNDS = 10


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
    for _ in range(_PHILOX_ROUNDS):
        hi_b, lo_b = _mul32(c2, 0xCD9E8D57)
        hi_a, lo_a = _mul32(c0, 0xD2511F53)
        c0, c1, c2, c3 = hi_b ^ c1 ^ k0, lo_b, hi_a ^ c3 ^ k1, lo_a
        k0 = (k0 + 0x9E3779B9) & 0xFFFFFFFF
        k1 = (k1 + 0xBB67AE85) & 0xFFFFFFFF
    return c0, c1, c2, c3


def reference_markov_sample_step(
    *,
    base_logits: torch.Tensor,
    bias: torch.Tensor,
    temperatures: torch.Tensor,
    key: int,
    offset: torch.Tensor,
    step: int,
    step_dtype: torch.dtype,
    vocab_start: int = 0,
    logits_out: Optional[torch.Tensor] = None,
    has_sampled_rows: Optional[bool] = None,
) -> torch.Tensor:
    # None draws the noise for every row: reading the temperatures on the
    # host would break a graph capture.
    device = base_logits.device
    base_logits = base_logits.cpu()
    bias = bias.cpu()
    temperatures = temperatures.cpu()
    offset = offset.cpu()
    rows, width = bias.shape
    value = base_logits.to(torch.float32, copy=True)
    value[:, :width] += bias.float()
    value = value.to(step_dtype).float()
    padded = base_logits.shape[1]
    if logits_out is not None:
        logits_out.copy_(value)

    keys = value
    if has_sampled_rows is not False:
        quads = vocab_start // 4 + torch.arange(
            (padded + 3) // 4, device="cpu", dtype=torch.int64
        )
        words = _philox(
            key,
            quads[None, :],
            torch.arange(rows, device="cpu", dtype=torch.int64)[:, None],
            torch.full((1, 1), step, device="cpu", dtype=torch.int64),
            offset.reshape(1, 1) & 0xFFFFFFFF,
        )
        words = torch.broadcast_tensors(*words)
        bits = torch.stack(words, dim=-1).reshape(rows, -1)[:, :padded]
        uniform = (bits.float() + 0.5) * _UNIFORM_SCALE
        exp_noise = -torch.log1p(-uniform)
        sampled = temperatures != 0
        safe_temperatures = torch.where(sampled, temperatures, 1.0)[:, None]
        noised = value / safe_temperatures - torch.log(exp_noise)
        keys = torch.where(sampled[:, None], noised, value)
    cols = torch.arange(padded, device="cpu")
    keys = torch.where(cols[None, :] < width, keys, -torch.inf)

    if width == 0:
        return torch.full((rows,), _NO_INDEX, dtype=torch.int64, device=device)
    return (keys.argmax(dim=1) + vocab_start).to(device)


def _offset(value: int) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.int64, device=DEVICE)


def _call(impl, *, base, bias, temps, step_dtype, offset=7, step=2, **kwargs):
    return impl(
        base_logits=base,
        bias=bias,
        temperatures=temps,
        key=12345,
        offset=_offset(offset),
        step=step,
        step_dtype=step_dtype,
        **kwargs,
    )


class TestMarkovSampleStep(unittest.TestCase):
    def test_triton_matches_torch_reference(self):
        g = torch.Generator(device=DEVICE).manual_seed(0)
        rows, padded, width = 64, 9000, 8990
        for base_dtype, bias_dtype, step_dtype in [
            (torch.bfloat16, torch.bfloat16, torch.bfloat16),
            (torch.bfloat16, torch.bfloat16, torch.float32),
            (torch.float32, torch.float32, torch.float32),
            (torch.float16, torch.float16, torch.float16),
        ]:
            base = torch.randn(rows, padded, generator=g, device=DEVICE) * 4
            base = base.to(base_dtype)
            bias = torch.randn(rows, width, generator=g, device=DEVICE).to(bias_dtype)
            # Half greedy rows, half sampled at mixed temperatures.
            temps = torch.rand(rows, generator=g, device=DEVICE) * 1.5 + 0.1
            temps[::2] = 0.0
            out = {}
            for name, impl in [
                ("triton", markov_sample_step),
                ("torch", reference_markov_sample_step),
            ]:
                logits = torch.empty(rows, padded, dtype=step_dtype, device=DEVICE)
                tokens = _call(
                    impl,
                    base=base,
                    bias=bias,
                    temps=temps,
                    step_dtype=step_dtype,
                    vocab_start=100,
                    logits_out=logits,
                )
                out[name] = (tokens, logits)
            (t_tok, t_log), (r_tok, r_log) = out["triton"], out["torch"]
            self.assertTrue(torch.equal(t_log, r_log))
            expected_log = base.float()
            expected_log[:, :width] += bias.float()
            self.assertTrue(torch.equal(t_log, expected_log.to(step_dtype)))
            greedy = temps == 0
            expected_greedy = t_log.float()[:, :width].argmax(dim=1) + 100
            self.assertTrue(torch.equal(t_tok[greedy], expected_greedy[greedy]))
            # Same Philox bits on both sides; only a last-ulp log difference on a
            # near tie could split them.
            agree = (t_tok[~greedy] == r_tok[~greedy]).float().mean().item()
            self.assertGreaterEqual(agree, 0.95)
            self.assertTrue(((t_tok >= 100) & (t_tok < 100 + width)).all())

    def test_greedy_only_variant_matches_mixed_kernel(self):
        g = torch.Generator(device=DEVICE).manual_seed(3)
        base = torch.randn(16, 9000, generator=g, device=DEVICE).to(torch.bfloat16)
        bias = torch.randn(16, 8990, generator=g, device=DEVICE).to(torch.bfloat16)
        temps = torch.zeros(16, device=DEVICE)
        kwargs = dict(base=base, bias=bias, temps=temps, step_dtype=torch.bfloat16)
        mixed = _call(markov_sample_step, **kwargs)
        for impl in (markov_sample_step, reference_markov_sample_step):
            greedy_only = _call(impl, has_sampled_rows=False, **kwargs)
            self.assertTrue(torch.equal(greedy_only, mixed))

    def test_special_values_follow_argmax(self):
        base = torch.zeros(4, 16, device=DEVICE)
        base[0, 3] = float("nan")
        base[0, 9] = float("nan")
        base[1, :] = -float("inf")
        base[2, 5] = 2.0
        base[2, 11] = 2.0
        base[3, 15] = 100.0
        bias = torch.zeros(4, 15, device=DEVICE)
        temps = torch.zeros(4, device=DEVICE)
        for impl in (markov_sample_step, reference_markov_sample_step):
            tokens = _call(
                impl, base=base, bias=bias, temps=temps, step_dtype=torch.float32
            )
            # Lowest NaN, then lowest index of a tie; column 15 is padding.
            self.assertEqual(tokens.tolist(), [3, 0, 5, 0])

    def test_greedy_tie_across_blocks_takes_the_lower_index(self):
        base = torch.zeros(2, 9000, device=DEVICE)
        base[:, 1000] = base[:, 5000] = 5.0
        bias = torch.zeros(2, 9000, device=DEVICE)
        for impl in (markov_sample_step, reference_markov_sample_step):
            tokens = _call(
                impl,
                base=base,
                bias=bias,
                temps=torch.zeros(2, device=DEVICE),
                step_dtype=torch.float32,
            )
            self.assertEqual(tokens.tolist(), [1000, 1000])

    def test_strided_rows_match_contiguous(self):
        # The draft passes one step of a [bs, gamma, vocab] block, rows strided.
        g = torch.Generator(device=DEVICE).manual_seed(2)
        block = torch.randn(8, 5, 6000, generator=g, device=DEVICE)
        bias = torch.randn(8, 5990, generator=g, device=DEVICE)
        temps = torch.rand(8, generator=g, device=DEVICE)
        temps[:3] = 0.0
        logits_block = torch.empty(8, 5, 6000, device=DEVICE)
        strided = _call(
            markov_sample_step,
            base=block[:, 2],
            bias=bias,
            temps=temps,
            step_dtype=torch.float32,
            logits_out=logits_block[:, 2],
        )
        logits = torch.empty(8, 6000, device=DEVICE)
        contiguous = _call(
            markov_sample_step,
            base=block[:, 2].contiguous(),
            bias=bias,
            temps=temps,
            step_dtype=torch.float32,
            logits_out=logits,
        )
        self.assertTrue(torch.equal(strided, contiguous))
        self.assertTrue(torch.equal(logits_block[:, 2], logits))

    def test_sampling_follows_tempered_softmax(self):
        vocab, rows = 32, 1 << 16
        logits = torch.linspace(-3, 3, vocab, device=DEVICE)
        base = logits.expand(rows, vocab).contiguous()
        bias = torch.zeros(rows, vocab, device=DEVICE)
        for temperature in (0.5, 1.0, 2.0):
            temps = torch.full((rows,), temperature, device=DEVICE)
            tokens = _call(
                markov_sample_step,
                base=base,
                bias=bias,
                temps=temps,
                step_dtype=torch.float32,
            )
            freq = torch.bincount(tokens, minlength=vocab).float() / rows
            expected = torch.softmax(logits / temperature, dim=0)
            self.assertLess((freq - expected).abs().max().item(), 0.01)

    def test_noise_changes_with_offset_row_and_step(self):
        rows, vocab = 256, 4096
        base = torch.zeros(rows, vocab, device=DEVICE)
        bias = torch.zeros(rows, vocab, device=DEVICE)
        temps = torch.ones(rows, device=DEVICE)
        first = _call(
            markov_sample_step,
            base=base,
            bias=bias,
            temps=temps,
            step_dtype=torch.float32,
        )
        self.assertGreater(first.unique().numel(), rows // 2)
        for kwargs in ({"offset": 8}, {"step": 3}):
            other = _call(
                markov_sample_step,
                base=base,
                bias=bias,
                temps=temps,
                step_dtype=torch.float32,
                **kwargs,
            )
            self.assertLess((other == first).float().mean().item(), 0.01)

    def test_sharded_vocab_matches_full_vocab(self):
        # Every rank must pick the full vocab's token: the noise is keyed by
        # global token id, so the shards draw what the full vocab draws.
        g = torch.Generator(device=DEVICE).manual_seed(1)
        rows, shard = 32, 5000
        base = torch.randn(rows, 3 * shard, generator=g, device=DEVICE)
        bias = torch.randn(rows, 2 * shard - 7, generator=g, device=DEVICE)
        temps = torch.rand(rows, generator=g, device=DEVICE)
        temps[:8] = 0.0
        full = _call(
            markov_sample_step,
            base=base[:, : bias.shape[1]].contiguous(),
            bias=bias,
            temps=temps,
            step_dtype=torch.float32,
        )
        # The last shard is empty: all its columns are padding.
        shards = [
            (rank * shard, base[:, rank * shard : (rank + 1) * shard].contiguous())
            for rank in range(3)
        ]
        partials = {}

        def run(rank, gather):
            start, shard_base = shards[rank]
            return _call(
                markov_sample_step,
                base=shard_base,
                bias=bias[:, start : start + shard].contiguous(),
                temps=temps,
                step_dtype=torch.float32,
                vocab_start=start,
                gather=gather,
            )

        for rank in range(3):
            run(rank, lambda partial, rank=rank: partials.setdefault(rank, partial))
        stacked = torch.cat([partials[rank] for rank in range(3)])
        for rank in range(3):
            self.assertTrue(torch.equal(run(rank, lambda _: stacked), full))

    def test_kept_sharded_logits_match_full_vocab(self):
        # Kept logits are gathered whole and every rank picks from them: the
        # same tokens and the same logits as the full vocab.
        g = torch.Generator(device=DEVICE).manual_seed(2)
        rows, shard = 32, 5000
        vocab = 2 * shard - 7
        base = torch.randn(rows, 3 * shard, generator=g, device=DEVICE)
        bias = torch.randn(rows, vocab, generator=g, device=DEVICE)
        temps = torch.rand(rows, generator=g, device=DEVICE)
        temps[:8] = 0.0
        full_logits = torch.empty(rows, vocab, device=DEVICE)
        full = _call(
            markov_sample_step,
            base=base[:, :vocab].contiguous(),
            bias=bias,
            temps=temps,
            step_dtype=torch.float32,
            logits_out=full_logits,
        )
        locals_ = {}

        def run(rank, gather_logits):
            start = rank * shard
            logits_out = torch.empty(rows, vocab, device=DEVICE)
            tokens = _call(
                markov_sample_step,
                base=base[:, start : start + shard].contiguous(),
                bias=bias[:, start : start + shard].contiguous(),
                temps=temps,
                step_dtype=torch.float32,
                vocab_start=start,
                gather=lambda _: self.fail("candidates gathered"),
                gather_logits=gather_logits,
                logits_out=logits_out,
            )
            return tokens, logits_out

        for rank in range(3):
            run(
                rank,
                lambda local, rank=rank: locals_.setdefault(rank, local.clone()).repeat(
                    1, 3
                ),
            )
        stacked = torch.cat([locals_[rank] for rank in range(3)], dim=1)
        for rank in range(3):
            tokens, logits_out = run(rank, lambda _: stacked)
            self.assertTrue(torch.equal(tokens, full))
            self.assertTrue(torch.equal(logits_out, full_logits))

    def test_small_batches_and_partial_blocks(self):
        for rows, padded, width in [(1, 16, 15), (3, 97, 95)]:
            base = torch.zeros(rows, padded, device=DEVICE)
            bias = torch.zeros(rows, width, device=DEVICE)
            temps = torch.ones(rows, device=DEVICE)
            kwargs = dict(base=base, bias=bias, temps=temps, step_dtype=torch.float32)
            actual = _call(markov_sample_step, **kwargs)
            expected = _call(reference_markov_sample_step, **kwargs)
            self.assertTrue(torch.equal(actual, expected))

    def test_graph_replay_reads_updated_offset(self):
        base = torch.zeros(8, 2048, device=DEVICE)
        bias = torch.zeros(8, 2040, device=DEVICE)
        temps = torch.ones(8, device=DEVICE)
        offset = _offset(7)
        kwargs = dict(
            base_logits=base,
            bias=bias,
            temperatures=temps,
            key=12345,
            offset=offset,
            step=2,
            step_dtype=torch.float32,
        )
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                markov_sample_step(**kwargs)
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            actual = markov_sample_step(**kwargs)
        for value in (7, 8):
            offset.fill_(value)
            expected = reference_markov_sample_step(**kwargs)
            graph.replay()
            torch.npu.synchronize()
            self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
