#pragma once
#include <memory>

namespace deep_ep {

struct EventHandle {

    EventHandle() {
    }

    EventHandle(const EventHandle& other) = default;

    void current_stream_wait() const {
        return;
    }
};
} // namespace deep_ep
