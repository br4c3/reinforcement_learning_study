#include "dqn_agent.h"

#include "random.h"

#include <tuple>
#include <vector>

DQNAgent::DQNAgent(int state_dim, int action_dim, torch::Device device)
    : action_dim_(action_dim),
      device_(device),
      q_net_(state_dim, action_dim),
      target_net_(state_dim, action_dim),
      replay_buffer_(30000),
      optimizer_(q_net_->parameters(), torch::optim::AdamOptions(1e-3)) {
    q_net_->to(device_);
    target_net_->to(device_);
    copy_weights_to_target();
}

int DQNAgent::select_action(const torch::Tensor& state, float epsilon) {
    if (global_rng().uniform() < epsilon) {
        return global_rng().randint(0, action_dim_ - 1);
    }

    torch::NoGradGuard no_grad;

    torch::Tensor input = state.to(device_).unsqueeze(0);
    torch::Tensor q_values = q_net_->forward(input);

    return q_values.argmax(1).item<int>();
}

void DQNAgent::remember(const Transition& transition) {
    replay_buffer_.push(transition);
}

float DQNAgent::train_step(size_t batch_size, float gamma) {
    auto batch = replay_buffer_.sample(batch_size);

    std::vector<torch::Tensor> states;
    std::vector<torch::Tensor> next_states;
    std::vector<int64_t> actions;
    std::vector<float> rewards;
    std::vector<float> dones;

    states.reserve(batch_size);
    next_states.reserve(batch_size);
    actions.reserve(batch_size);
    rewards.reserve(batch_size);
    dones.reserve(batch_size);

    for (const auto& t : batch) {
        states.push_back(t.state);
        next_states.push_back(t.next_state);
        actions.push_back(static_cast<int64_t>(t.action));
        rewards.push_back(t.reward);
        dones.push_back(t.done ? 1.0f : 0.0f);
    }

    torch::Tensor state_batch = torch::stack(states).to(device_);
    torch::Tensor next_state_batch = torch::stack(next_states).to(device_);
    torch::Tensor action_batch = torch::tensor(actions, torch::kInt64).to(device_).unsqueeze(1);
    torch::Tensor reward_batch = torch::tensor(rewards, torch::kFloat32).to(device_);
    torch::Tensor done_batch = torch::tensor(dones, torch::kFloat32).to(device_);

    torch::Tensor q_values = q_net_->forward(state_batch);
    torch::Tensor current_q = q_values.gather(1, action_batch).squeeze(1);

    torch::Tensor target_q;
    {
        torch::NoGradGuard no_grad;

        torch::Tensor next_q_values = target_net_->forward(next_state_batch);
        torch::Tensor max_next_q = std::get<0>(next_q_values.max(1));

        target_q = reward_batch + gamma * max_next_q * (1.0f - done_batch);
    }

    torch::Tensor loss = torch::mse_loss(current_q, target_q);

    optimizer_.zero_grad();
    loss.backward();
    optimizer_.step();

    return loss.item<float>();
}

void DQNAgent::copy_weights_to_target() {
    torch::NoGradGuard no_grad;

    auto source_params = q_net_->named_parameters();
    auto target_params = target_net_->named_parameters();

    for (const auto& item : source_params) {
        target_params[item.key()].copy_(item.value());
    }
}

size_t DQNAgent::replay_size() const {
    return replay_buffer_.size();
}

QNetwork& DQNAgent::network() {
    return q_net_;
}
