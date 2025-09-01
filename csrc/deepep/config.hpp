#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <cstdlib>
#include <cctype>
#include <string>

namespace deep_ep {

struct Config {
    int num_sms;
    int num_max_nvl_chunked_send_tokens;
    int num_max_nvl_chunked_recv_tokens;
    int num_max_rdma_chunked_send_tokens;
    int num_max_rdma_chunked_recv_tokens;

    Config(
        int num_sms,
        int num_max_nvl_chunked_send_tokens,
        int num_max_nvl_chunked_recv_tokens,
        int num_max_rdma_chunked_send_tokens,
        int num_max_rdma_chunked_recv_tokens)
        : num_sms(num_sms),
          num_max_nvl_chunked_send_tokens(num_max_nvl_chunked_send_tokens),
          num_max_nvl_chunked_recv_tokens(num_max_nvl_chunked_recv_tokens),
          num_max_rdma_chunked_send_tokens(num_max_rdma_chunked_send_tokens),
          num_max_rdma_chunked_recv_tokens(num_max_rdma_chunked_recv_tokens)
    {}

    size_t get_nvl_buffer_size_hint(size_t hidden_bytes, int num_ranks) const
    {
        return hidden_bytes;
    }

    size_t get_rdma_buffer_size_hint(int64_t hidden_bytes, int num_ranks) const
    {
        return hidden_bytes;
    }
};

size_t get_low_latency_rdma_size_hint(int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts);

int get_value_from_env(const std::string &name, int defaultValue);
}  // namespace deep_ep
