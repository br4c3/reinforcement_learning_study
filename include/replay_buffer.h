#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <deque>
#include <vector>

struct Transition {
    torch::Tensor state;
    int action;
    float reward;
    torch::Tensor next_state;
    bool done;
};

class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t capacity);

    void push(const Transition& transition);
    std::vector<Transition> sample(size_t batch_size);
    size_t size() const;

private:
    size_t capacity_;
    std::deque<Transition> buffer_;
};
