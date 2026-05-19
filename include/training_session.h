#pragma once

#include "dqn_agent.h"
#include "grid_world_env.h"

#include <cstddef>
#include <vector>

struct TrainingConfig {
    int num_episodes = 1200;
    int max_steps_per_episode = 80;
    size_t batch_size = 64;
    float gamma = 0.98f;
    float epsilon = 1.0f;
    float epsilon_min = 0.03f;
    float epsilon_decay = 0.995f;
    int warmup_steps = 500;
    int target_update_interval = 20;
    int plot_update_interval = 20;
};

struct TrainingMetrics {
    std::vector<double> episodes;
    std::vector<double> rewards;
    std::vector<double> losses;
    std::vector<double> success_rates;
    std::vector<double> epsilons;
};

class TrainingSession {
public:
    TrainingSession(GridWorldEnv& env, DQNAgent& agent, TrainingConfig config = {});

    TrainingMetrics run();
    float evaluate_policy(int episodes);

private:
    GridWorldEnv& env_;
    DQNAgent& agent_;
    TrainingConfig config_;
};
