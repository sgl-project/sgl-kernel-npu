#include "config.hpp"

namespace deep_ep {
size_t get_low_latency_rdma_size_hint(int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts) {
    return num_max_dispatch_tokens_per_rank;
}
}