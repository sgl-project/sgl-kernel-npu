"""Correctness test for npu_sparse_flash_attention_lse.

The reason this operator is vendored at all is that CANN's build of it refuses
``return_softmax_lse=True`` under ``layout_kv="PA_BSND"``, and ``PA_BSND`` is its
only paged layout -- so paged and LSE are mutually exclusive there. Decode
context parallelism needs a per-rank log-sum-exp to weight the cross-rank merge
with, which makes that refusal the blocker. See
``csrc/sparse_flash_attention/README.md``.

So the first test below is the whole point: if the paged+LSE call is rejected,
the gate was copied along with the kernel and nothing else here matters. The
rest establish that what comes back is right -- against CANN's own operator for
the output they share, and against a float64 reference for the LSE, which CANN
cannot produce here and so cannot be compared with.

The reference is built from the **actual** bf16 tensors upcast to float64, never
from the literals that generated them; otherwise it measures the rounding of the
input generation rather than the kernel.

Shapes are MLA decode's: a 512-wide nope part and a 64-wide rope part passed
separately, one latent KV head, paged KV. One NPU, no model, no collectives.
"""

import math
import unittest

import sgl_kernel_npu  # noqa: F401 - loads the torch custom-op library
import torch
import torch_npu

OP_NAME = "npu_sparse_flash_attention_lse"

TOKENS = 4
Q_HEADS = 16
KV_HEADS = 1  # MLA keeps one latent KV head
TOPK = 64
NUM_BLOCKS = 16
BLOCK_SIZE = 128
LORA_RANK = 512
ROPE_DIM = 64
SEED = 20260909

BF16_EPS = 3.906e-3


def _build_problem(device, empty_rows=0):
    """A DCP-shaped decode problem: op kwargs, plus what the reference needs.

    Every token is one decode step for its own sequence, so each batch entry has
    a query length of 1 and T == batch. The sparse indices are a left-aligned
    prefix of the positions this "rank" owns followed by a -1 tail, which is the
    contract the operator documents and the DCP remap emits.

    Two conventions here are the operator's, not free choices, and both were read
    out of the kernel rather than guessed:

    * ``sparse_indices`` is indexed by the **KV** head count, not the query head
      count (``SFATilingCheck::CheckTopkShape`` sets ``shapeParams.N = n2Size_``).
      With MLA's single latent KV head the selection is ``(T, 1, topk)`` and is
      shared by every query head of that token -- which is also what makes the
      empty-row case the real DCP one: a rank owning nothing for a token owns
      nothing for all of its heads.
    * ``actual_seq_lengths_query`` is **cumulative** under ``layout_query="TND"``:
      the kernel recovers a per-batch length as ``value[b] - value[b-1]`` and uses
      ``value[b-1]`` as the token base offset. Per-batch 1s would make every batch
      after the first look empty. ``actual_seq_lengths_kv`` under ``PA_BSND`` is
      the opposite -- per-batch, read as ``value[b]``.
    """
    generator = torch.Generator(device="cpu").manual_seed(SEED)

    def rnd(*shape):
        values = torch.randn(*shape, generator=generator, dtype=torch.float32)
        return (values * 0.5).to(torch.bfloat16)

    query = rnd(TOKENS, Q_HEADS, LORA_RANK)
    query_rope = rnd(TOKENS, Q_HEADS, ROPE_DIM)
    # PA_BSND is (num_blocks, block_size, kv_heads, dim).
    key = rnd(NUM_BLOCKS, BLOCK_SIZE, KV_HEADS, LORA_RANK)
    key_rope = rnd(NUM_BLOCKS, BLOCK_SIZE, KV_HEADS, ROPE_DIM)

    # One page-table row per token, each mapping to its own distinct blocks.
    blocks_per_seq = NUM_BLOCKS // TOKENS
    block_table = torch.arange(TOKENS * blocks_per_seq, dtype=torch.int32).reshape(
        TOKENS, blocks_per_seq
    )
    kv_len = blocks_per_seq * BLOCK_SIZE

    indices = torch.full((TOKENS, KV_HEADS, TOPK), -1, dtype=torch.int32)
    valid_counts = torch.zeros((TOKENS, KV_HEADS), dtype=torch.int64)
    for token in range(TOKENS):
        for head in range(KV_HEADS):
            if token < empty_rows:
                continue  # this "rank" owns nothing for this token
            n_valid = int(
                torch.randint(
                    1, min(TOPK, kv_len) + 1, (1,), generator=generator
                ).item()
            )
            selected = torch.randperm(kv_len, generator=generator)[:n_valid]
            indices[token, head, :n_valid] = selected.to(torch.int32)
            valid_counts[token, head] = n_valid

    kwargs = dict(
        query=query.to(device),
        key=key.to(device),
        value=key.to(device),  # MLA: value is the same latent tensor as key
        sparse_indices=indices.to(device),
        scale_value=1.0 / math.sqrt(LORA_RANK + ROPE_DIM),
        block_table=block_table.to(device),
        actual_seq_lengths_query=torch.arange(1, TOKENS + 1, dtype=torch.int32).to(
            device
        ),
        actual_seq_lengths_kv=torch.full((TOKENS,), kv_len, dtype=torch.int32).to(
            device
        ),
        query_rope=query_rope.to(device),
        key_rope=key_rope.to(device),
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        # 0, not 3: under DCP the indices are already rank-local, so local KV
        # length and global query length no longer share a coordinate system and
        # the operator must not apply its own causal crop.
        sparse_mode=0,
        attention_mode=2,
    )
    reference_inputs = dict(
        query=query,
        query_rope=query_rope,
        key=key,
        key_rope=key_rope,
        indices=indices,
        valid_counts=valid_counts,
        block_table=block_table,
        scale=kwargs["scale_value"],
    )
    return kwargs, reference_inputs


def _reference(ref):
    """Dense float64 attention over the selected positions, on the CPU.

    Returns ``(out [T, N1, D], lse [T, N1])`` with the LSE in natural log, which
    is what ``softmax_max + log(softmax_sum)`` produces.
    """
    query = ref["query"].to(torch.float64)
    query_rope = ref["query_rope"].to(torch.float64)
    key_paged = ref["key"].to(torch.float64)[:, :, 0, :]
    key_rope_paged = ref["key_rope"].to(torch.float64)[:, :, 0, :]
    scale = ref["scale"]

    out = torch.zeros((TOKENS, Q_HEADS, LORA_RANK), dtype=torch.float64)
    lse = torch.full((TOKENS, Q_HEADS), -math.inf, dtype=torch.float64)

    for token in range(TOKENS):
        blocks = ref["block_table"][token].tolist()
        # Flatten the paged KV back into linear position order, exactly as the
        # block table addresses it.
        keys = torch.cat([key_paged[b] for b in blocks], dim=0)
        keys_rope = torch.cat([key_rope_paged[b] for b in blocks], dim=0)
        n_valid = int(ref["valid_counts"][token, 0])
        if n_valid == 0:
            continue  # out stays 0, lse stays -inf
        selected = ref["indices"][token, 0, :n_valid].to(torch.int64)
        for head in range(Q_HEADS):
            logits = keys[selected] @ query[token, head]
            logits = logits + keys_rope[selected] @ query_rope[token, head]
            logits = logits * scale
            peak = logits.max()
            weights = torch.exp(logits - peak)
            total = weights.sum()
            out[token, head] = (weights @ keys[selected]) / total
            lse[token, head] = peak + torch.log(total)
    return out, lse


def _combine_lse(softmax_max, softmax_sum):
    """``softmax_max + log(softmax_sum)`` as NTG ``(N2, T, G)`` -> ``(T, N1)``.

    No epsilon on ``softmax_sum``: ``log(0)`` must be allowed to produce ``-inf``
    rather than be papered over, since the empty-row test below exists to pin what
    this actually returns.
    """
    combined = softmax_max.to(torch.float64) + torch.log(softmax_sum.to(torch.float64))
    n2, tokens, g = combined.shape
    # (N2, T, G) -> (T, N2*G), head index n2 * G + g, matching the query's N1.
    return combined.permute(1, 0, 2).reshape(tokens, n2 * g)


class TestSparseFlashAttentionLse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch_npu.npu.is_available():
            raise unittest.SkipTest("an Ascend NPU is required")
        if getattr(torch.ops.npu, OP_NAME, None) is None:
            # Built only for arch22 (A2/A3), so it is absent by design elsewhere.
            raise unittest.SkipTest(f"torch.ops.npu.{OP_NAME} is not built here")
        torch_npu.npu.set_device(0)
        cls.device = torch.device("npu:0")
        cls.op = getattr(torch.ops.npu, OP_NAME)

    def test_paged_kv_layout_accepts_return_softmax_lse(self):
        """The port's reason to exist. CANN's build fails this call at
        sparse_flash_attention_tiling.cpp:1994; if this one does too, the gate
        came across with the kernel."""
        kwargs, _ = _build_problem(self.device)

        _, softmax_max, softmax_sum = self.op(**kwargs, return_softmax_lse=True)
        torch_npu.npu.synchronize()

        # NTG: kv heads, tokens, query heads per kv head. Not a choice -- the
        # tiling derives the expected layout from layout_query and rejects a call
        # whose tensors disagree.
        want = (KV_HEADS, TOKENS, Q_HEADS // KV_HEADS)
        self.assertEqual(tuple(softmax_max.shape), want)
        self.assertEqual(tuple(softmax_sum.shape), want)
        self.assertEqual(softmax_max.dtype, torch.float32)
        self.assertEqual(softmax_sum.dtype, torch.float32)
        self.assertTrue(torch.isfinite(softmax_max.cpu()).all())
        self.assertTrue(torch.isfinite(softmax_sum.cpu()).all())
        self.assertTrue((softmax_sum.cpu() >= 0).all())

    def test_attention_out_matches_the_cann_operator(self):
        """Same kernel, two builds, so the output they share must agree. This is
        the no-op check that comes before the feature: a difference here means the
        port changed what the kernel computes."""
        kwargs, _ = _build_problem(self.device)

        try:
            theirs = torch_npu.npu_sparse_flash_attention(
                **kwargs, return_softmax_lse=False
            )[0]
        except Exception:
            # sparse_mode=0 is what DCP needs, but CANN's build may reject it for
            # reasons unrelated to the LSE gate. Fall back for BOTH operators so
            # the comparison stays apples-to-apples; the other tests keep using 0.
            kwargs["sparse_mode"] = 3
            theirs = torch_npu.npu_sparse_flash_attention(
                **kwargs, return_softmax_lse=False
            )[0]
        ours = self.op(**kwargs, return_softmax_lse=False)[0]
        torch_npu.npu.synchronize()

        # Measured bit-identical on A3 / CANN 9.1.0. The tolerance is one ulp of
        # the output dtype rather than zero, because a tiling difference could
        # legitimately reorder the accumulation.
        difference = (ours.float() - theirs.float()).abs().max().item()
        self.assertLessEqual(difference, BF16_EPS)

    def test_outputs_match_a_float64_reference(self):
        kwargs, reference_inputs = _build_problem(self.device)

        attention_out, softmax_max, softmax_sum = self.op(
            **kwargs, return_softmax_lse=True
        )
        torch_npu.npu.synchronize()

        expected_out, expected_lse = _reference(reference_inputs)
        actual_lse = _combine_lse(softmax_max.cpu(), softmax_sum.cpu())

        finite = torch.isfinite(expected_lse) & torch.isfinite(actual_lse)
        self.assertTrue(bool(finite.all()))
        lse_error = (actual_lse[finite] - expected_lse[finite]).abs().max().item()
        scale = max(expected_lse[finite].abs().max().item(), 1.0)
        self.assertLessEqual(lse_error / scale, 1e-2)

        out_error = (
            (attention_out.cpu().to(torch.float64) - expected_out).abs().max().item()
        )
        self.assertLessEqual(out_error, 5e-2)

    def test_a_row_owning_no_positions_contributes_nothing(self):
        """A DCP rank routinely owns none of the selected positions for a token.

        What the operator returns for that row is a kernel detail -- measured on
        A3 / CANN 9.1.0 it is ``softmax_max = -2e38`` (the SOFTMAX_MIN_NUM
        sentinel, not -inf) and ``softmax_sum = topk`` (the slot count, not 0,
        because with every index masked every slot equals the max and contributes
        exp(0) = 1). What the consumer depends on is not the value but the
        property: once centred on an ordinary global max, the row's merge weight
        must underflow to exactly zero, so no LSE combine needs a guard for it.
        """
        kwargs, _ = _build_problem(self.device, empty_rows=1)

        _, softmax_max, softmax_sum = self.op(**kwargs, return_softmax_lse=True)
        torch_npu.npu.synchronize()

        empty_max = softmax_max.cpu()[0, 0, 0].item()
        empty_sum = softmax_sum.cpu()[0, 0, 0].item()
        combined = _combine_lse(softmax_max.cpu(), softmax_sum.cpu())[0, 0].item()

        self.assertLess(combined, 0.0)
        # A populated row's LSE is O(1), so centring on 1.0 is representative:
        # what underflows against that underflows against a real global max too.
        self.assertEqual(math.exp(combined - 1.0), 0.0)
        # Never 0, so nothing downstream evaluates log(0) and an epsilon on
        # softmax_sum would be pointless rather than protective.
        self.assertNotEqual(empty_sum, 0.0)
        self.assertLess(empty_max, 0.0)

    def test_graph_capture_honours_replay_time_inputs(self):
        """DCP decode runs this operator inside a captured graph, and its tiling
        parses its inputs on the host. If that plan were baked in at record time,
        every replay would attend over the capture-time KV lengths while looking
        entirely healthy -- so both a data input and a length input are changed in
        place here and the replay is required to follow them."""
        kwargs, _ = _build_problem(self.device)
        query = kwargs["query"]
        kv_lengths = kwargs["actual_seq_lengths_kv"]
        kv_len = int(kv_lengths[0].item())

        # Every eager answer is taken BEFORE the capture, and the buffers are put
        # back as they were each time. Running the operator eagerly *between*
        # replays would leave the test unable to say whether a disagreement came
        # from the replay or from the eager call having moved the tiling slot the
        # recorded graph depends on. These calls double as the capture warmup: a
        # cache miss outside capture allocates and fills a stable slot.
        eager_long = self.op(**kwargs, return_softmax_lse=True)[0].clone()

        query.copy_(query.flip(0))
        eager_flipped = self.op(**kwargs, return_softmax_lse=True)[0].clone()
        query.copy_(query.flip(0))  # back to the original

        kv_lengths.fill_(kv_len // 2)
        eager_short = self.op(**kwargs, return_softmax_lse=True)[0].clone()
        kv_lengths.fill_(kv_len)
        torch_npu.npu.synchronize()

        # Halving the KV length masks out the selected positions beyond it, so
        # the three answers have to actually differ for the replays to prove
        # anything.
        self.assertGreater((eager_flipped - eager_long).abs().max().item(), 0.0)
        self.assertGreater((eager_short - eager_long).abs().max().item(), 0.0)

        graph = torch_npu.npu.NPUGraph()
        capture_stream = torch_npu.npu.Stream()
        with torch_npu.npu.graph(
            graph, stream=capture_stream, auto_dispatch_capture=True
        ):
            captured_out, _, _ = self.op(**kwargs, return_softmax_lse=True)
        torch_npu.npu.synchronize()

        graph.replay()
        torch_npu.npu.synchronize()
        torch.testing.assert_close(
            captured_out.cpu().float(), eager_long.cpu().float(), rtol=0, atol=0
        )

        # A changed data input: proves the operator is replayed at all, rather
        # than having run once during recording with its output left frozen.
        query.copy_(query.flip(0))
        graph.replay()
        torch_npu.npu.synchronize()
        torch.testing.assert_close(
            captured_out.cpu().float(), eager_flipped.cpu().float(), rtol=0, atol=0
        )

        # A changed length input: the one a baked-in tiling plan would ignore.
        query.copy_(query.flip(0))
        kv_lengths.fill_(kv_len // 2)
        graph.replay()
        torch_npu.npu.synchronize()
        torch.testing.assert_close(
            captured_out.cpu().float(), eager_short.cpu().float(), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
