#include "config.hpp"

namespace deep_ep {
size_t get_low_latency_rdma_size_hint(int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts)
{
    return num_max_dispatch_tokens_per_rank;
}

int get_value_from_env(const std::string &name, int defaultValue)
{
    int retValue = defaultValue;
    if (const char *rank_str = std::getenv(name.c_str())) {
        char *end;
        errno = 0;
        long val = std::strtol(rank_str, &end, 10);
        if (errno == ERANGE || *end != '\0' || !std::isdigit(*rank_str)) {
            return retValue;
        }
        retValue = static_cast<int>(val);
        return retValue;
    } else {
        return retValue;
    }
}
}  // namespace deep_ep
