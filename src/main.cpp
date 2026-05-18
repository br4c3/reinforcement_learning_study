// ================================================================
// Minimal C++ DQN Reinforcement Learning Project with LibTorch
// ================================================================
// Project structure:
//
// reinforcement_learning_study/
//   CMakeLists.txt
//   src/
//     main.cpp
//   include/
//   libtorch/
//
// This canvas contains BOTH files.
// Copy the CMakeLists.txt section into CMakeLists.txt.
// Copy the main.cpp section into src/main.cpp.
// ================================================================

/*
======================== CMakeLists.txt ========================

cmake_minimum_required(VERSION 3.18)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)

add_executable(main
    src/main.cpp
)

target_include_directories(main
    PRIVATE ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(main
    PRIVATE ${TORCH_LIBRARIES}
)

target_compile_options(main PRIVATE
    -Wall -Wextra -Wpedantic
)

set_property(TARGET main PROPERTY CXX_STANDARD 17)

=================================================================
*/

// ========================== src/main.cpp ==========================

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

// ---------------------------------------------------------------
// Utility random engine
// ---------------------------------------------------------------
class Random {
public:
    Random() : gen_(std::random_device{}()) {}

    int randint(int low, int high_inclusive) {
        std::uniform_int_distribution<int> dist(low, high_inclusive);
        return dist(gen_);
    }

    float uniform(float low = 0.0f, float high = 1.0f) {
        std::uniform_real_distribution<float> dist(low, high);
        return dist(gen_);
    }

private:
    std::mt19937 gen_;
};

Random RNG;

// ---------------------------------------------------------------
// Toy Environment
// ---------------------------------------------------------------
// State: [position_normalized]
// Actions:
//   0 -> move left
//   1 -> move right
// Goal: reach target position.
// Reward:
//   +10 when reaching target
//   otherwise negative distance penalty
// ---------------------------------------------------------------
struct StepResult {
    torch::Tensor next_state;
    float reward;
    bool done;
};

class LineWorldEnv {
public:
    LineWorldEnv(int min_pos = -10, int max_pos = 10, int target_pos = 7, int max_steps = 50)
        : min_pos_(min_pos),
          max_pos_(max_pos),
          target_pos_(target_pos),
          max_steps_(max_steps) {
        if (min_pos_ >= max_pos_) {
            throw std::invalid_argument("min_pos must be smaller than max_pos");
        }
        reset();
    }

    torch::Tensor reset() {
        position_ = RNG.randint(min_pos_, max_pos_);
        steps_ = 0;

        // Avoid starting exactly at target.
        if (position_ == target_pos_) {
            position_ = min_pos_;
        }

        return get_state();
    }

    StepResult step(int action) {
        if (action == 0) {
            position_ -= 1;
        } else if (action == 1) {
            position_ += 1;
        } else {
            throw std::invalid_argument("Invalid action");
        }

        position_ = std::max(min_pos_, std::min(max_pos_, position_));
        steps_++;

        const int distance = std::abs(target_pos_ - position_);
        bool reached_goal = (position_ == target_pos_);
        bool timeout = (steps_ >= max_steps_);
        bool done = reached_goal || timeout;

        float reward = reached_goal ? 10.0f : -static_cast<float>(distance) / 10.0f;

        return {get_state(), reward, done};
    }

    int action_dim() const {
        return 2;
    }

    int state_dim() const {
        return 1;
    }

private:
    torch::Tensor get_state() const {
        float normalized = static_cast<float>(position_) / static_cast<float>(max_pos_);
        return torch::tensor({normalized}, torch::kFloat32);
    }

private:
    int min_pos_;
    int max_pos_;
    int target_pos_;
    int max_steps_;
    int position_ = 0;
    int steps_ = 0;
};

// ---------------------------------------------------------------
// Replay Buffer
// ---------------------------------------------------------------
struct Transition {
    torch::Tensor state;
    int action;
    float reward;
    torch::Tensor next_state;
    bool done;
};

class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t capacity) : capacity_(capacity) {}

    void push(const Transition& transition) {
        if (buffer_.size() >= capacity_) {
            buffer_.pop_front();
        }
        buffer_.push_back(transition);
    }

    std::vector<Transition> sample(size_t batch_size) {
        if (batch_size > buffer_.size()) {
            throw std::runtime_error("Not enough samples in replay buffer");
        }

        std::vector<Transition> batch;
        batch.reserve(batch_size);

        for (size_t i = 0; i < batch_size; ++i) {
            int idx = RNG.randint(0, static_cast<int>(buffer_.size() - 1));
            batch.push_back(buffer_[idx]);
        }

        return batch;
    }

    size_t size() const {
        return buffer_.size();
    }

private:
    size_t capacity_;
    std::deque<Transition> buffer_;
};

// ---------------------------------------------------------------
// Q-Network
// ---------------------------------------------------------------
struct QNetworkImpl : torch::nn::Module {
    QNetworkImpl(int state_dim, int action_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(state_dim, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, action_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }

    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};
};

TORCH_MODULE(QNetwork);

// ---------------------------------------------------------------
// Copy network weights
// ---------------------------------------------------------------
void copy_weights(QNetwork& source, QNetwork& target) {
    torch::NoGradGuard no_grad;

    auto source_params = source->named_parameters();
    auto target_params = target->named_parameters();

    for (const auto& item : source_params) {
        const auto& name = item.key();
        target_params[name].copy_(item.value());
    }

    auto source_buffers = source->named_buffers();
    auto target_buffers = target->named_buffers();

    for (const auto& item : source_buffers) {
        const auto& name = item.key();
        target_buffers[name].copy_(item.value());
    }
}

// ---------------------------------------------------------------
// Epsilon-greedy action selection
// ---------------------------------------------------------------
int select_action(QNetwork& q_net, const torch::Tensor& state, float epsilon, int action_dim, torch::Device device) {
    if (RNG.uniform() < epsilon) {
        return RNG.randint(0, action_dim - 1);
    }

    torch::NoGradGuard no_grad;
    torch::Tensor input = state.to(device).unsqueeze(0);  // [1, state_dim]
    torch::Tensor q_values = q_net->forward(input);
    return q_values.argmax(1).item<int>();
}

// ---------------------------------------------------------------
// Single DQN training update
// ---------------------------------------------------------------
float train_step(
    QNetwork& q_net,
    QNetwork& target_net,
    ReplayBuffer& replay_buffer,
    torch::optim::Optimizer& optimizer,
    size_t batch_size,
    float gamma,
    torch::Device device
) {
    auto batch = replay_buffer.sample(batch_size);

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

    torch::Tensor state_batch = torch::stack(states).to(device);              // [B, state_dim]
    torch::Tensor next_state_batch = torch::stack(next_states).to(device);    // [B, state_dim]
    torch::Tensor action_batch = torch::tensor(actions, torch::kInt64).to(device).unsqueeze(1);
    torch::Tensor reward_batch = torch::tensor(rewards, torch::kFloat32).to(device);
    torch::Tensor done_batch = torch::tensor(dones, torch::kFloat32).to(device);

    torch::Tensor q_values = q_net->forward(state_batch);
    torch::Tensor current_q = q_values.gather(1, action_batch).squeeze(1);

    torch::Tensor target_q;
    {
        torch::NoGradGuard no_grad;
        torch::Tensor next_q_values = target_net->forward(next_state_batch);
        torch::Tensor max_next_q = std::get<0>(next_q_values.max(1));
        target_q = reward_batch + gamma * max_next_q * (1.0f - done_batch);
    }

    torch::Tensor loss = torch::mse_loss(current_q, target_q);

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    return loss.item<float>();
}

// ---------------------------------------------------------------
// Evaluate learned policy
// ---------------------------------------------------------------
float evaluate_policy(QNetwork& q_net, LineWorldEnv& env, int episodes, torch::Device device) {
    int success_count = 0;

    for (int ep = 0; ep < episodes; ++ep) {
        torch::Tensor state = env.reset();
        bool done = false;
        float total_reward = 0.0f;

        while (!done) {
            int action = select_action(q_net, state, 0.0f, env.action_dim(), device);
            StepResult result = env.step(action);
            state = result.next_state;
            total_reward += result.reward;
            done = result.done;
        }

        if (total_reward > 0.0f) {
            success_count++;
        }
    }

    return static_cast<float>(success_count) / static_cast<float>(episodes);
}

// ---------------------------------------------------------------
// Main
// ---------------------------------------------------------------
int main() {
    torch::manual_seed(42);

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "Using CUDA\n";
    } else {
        std::cout << "Using CPU\n";
    }

    LineWorldEnv env;

    const int state_dim = env.state_dim();
    const int action_dim = env.action_dim();

    QNetwork q_net(state_dim, action_dim);
    QNetwork target_net(state_dim, action_dim);

    q_net->to(device);
    target_net->to(device);

    // Copy initial weights to target network.
    copy_weights(q_net, target_net);

    torch::optim::Adam optimizer(q_net->parameters(), torch::optim::AdamOptions(1e-3));

    ReplayBuffer replay_buffer(10000);

    const int num_episodes = 600;
    const int max_steps_per_episode = 50;
    const size_t batch_size = 64;
    const float gamma = 0.99f;

    float epsilon = 1.0f;
    const float epsilon_min = 0.05f;
    const float epsilon_decay = 0.995f;

    const int warmup_steps = 500;
    const int target_update_interval = 20;

    int global_step = 0;

    for (int episode = 1; episode <= num_episodes; ++episode) {
        torch::Tensor state = env.reset();
        float episode_reward = 0.0f;
        float last_loss = 0.0f;

        for (int step = 0; step < max_steps_per_episode; ++step) {
            int action = select_action(q_net, state, epsilon, action_dim, device);
            StepResult result = env.step(action);

            replay_buffer.push({
                state.detach().clone(),
                action,
                result.reward,
                result.next_state.detach().clone(),
                result.done
            });

            state = result.next_state;
            episode_reward += result.reward;
            global_step++;

            if (replay_buffer.size() >= batch_size && global_step > warmup_steps) {
                last_loss = train_step(
                    q_net,
                    target_net,
                    replay_buffer,
                    optimizer,
                    batch_size,
                    gamma,
                    device
                );
            }

            if (result.done) {
                break;
            }
        }

        epsilon = std::max(epsilon_min, epsilon * epsilon_decay);

        if (episode % target_update_interval == 0) {
            copy_weights(q_net, target_net);
        }

        if (episode % 20 == 0) {
            float success_rate = evaluate_policy(q_net, env, 30, device);
            std::cout << "Episode " << episode
                      << " | Reward: " << episode_reward
                      << " | Loss: " << last_loss
                      << " | Epsilon: " << epsilon
                      << " | Success Rate: " << success_rate
                      << "\n";
        }
    }

    std::cout << "Training finished.\n";

    // Save model.
    torch::save(q_net, "dqn_lineworld.pt");
    std::cout << "Model saved to dqn_lineworld.pt\n";

    return 0;
}
