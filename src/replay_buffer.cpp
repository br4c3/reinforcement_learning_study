#include "replay_buffer.h"

#include "random.h"

#include <stdexcept>

ReplayBuffer::ReplayBuffer(size_t capacity) : capacity_(capacity) {}

void ReplayBuffer::push(const Transition& transition) {
    if (buffer_.size() >= capacity_) {
        buffer_.pop_front();
    }

    buffer_.push_back(transition);
}

std::vector<Transition> ReplayBuffer::sample(size_t batch_size) {
    if (batch_size > buffer_.size()) {
        throw std::runtime_error("Not enough samples in replay buffer");
    }

    std::vector<Transition> batch;
    batch.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        int idx = global_rng().randint(0, static_cast<int>(buffer_.size() - 1));
        batch.push_back(buffer_[idx]);
    }

    return batch;
}

size_t ReplayBuffer::size() const {
    return buffer_.size();
}
