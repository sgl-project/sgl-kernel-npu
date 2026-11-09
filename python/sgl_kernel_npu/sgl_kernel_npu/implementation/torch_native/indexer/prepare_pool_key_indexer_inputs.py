"""PyTorch reference for nonempty pooled-key indexer inputs."""

import torch


def prepare_pool_key_indexer_inputs(query, weights, ends, lengths, tails, table):
    # Kernel requires 1. non-empty pool_lens, 2. sentinel
    query_safe = torch.cat(
        (query, query.new_zeros((1, *query.shape[1:])))
    )  # append one all-0 query
    weights_safe = torch.cat(
        (weights, weights.new_zeros((1, weights.shape[1])))
    )  # append one all-0 weights
    query_ends_safe = torch.cat(
        (ends, ends.new_full((1,), query.shape[0] + 1))
    )  # append one query
    pool_lens_safe = torch.cat(
        (lengths.clamp_min(1), lengths.new_ones((1,)))
    )  # at least 1 key
    tail_lens_safe = torch.cat((tails, tails.new_zeros((1,))))  # append 0
    page_table_safe = torch.cat(
        (table, table.new_zeros((1, table.shape[1])))
    )  # append one all-0 page-table
    return (
        query_safe,
        weights_safe,
        query_ends_safe,
        pool_lens_safe,
        tail_lens_safe,
        page_table_safe,
    )
