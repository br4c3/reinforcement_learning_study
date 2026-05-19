#pragma once

#include "q_network.h"
#include "replay_buffer.h"

#include <torch/torch.h>

#include <cstddef>

class DQNAgent {
public:
    DQNAgent(int state_dim, int action_dim, torch::Device device);

    int select_action(const torch::Tensor& state, float epsilon);
    void remember(const Transition& transition);
    float train_step(size_t batch_size, float gamma);
    void copy_weights_to_target();

    size_t replay_size() const;
    QNetwork& network();

private:
    int action_dim_;
    torch::Device device_;
    QNetwork q_net_;
    QNetwork target_net_;
    ReplayBuffer replay_buffer_;
    torch::optim::Adam optimizer_;
};
