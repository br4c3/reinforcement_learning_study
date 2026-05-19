#include "training_session.h"

#include <algorithm>
#include <iostream>

TrainingSession::TrainingSession(GridWorldEnv& env, DQNAgent& agent, TrainingConfig config)
    : env_(env), agent_(agent), config_(config) {}

TrainingMetrics TrainingSession::run() {
    int global_step = 0;
    float epsilon = config_.epsilon;
    TrainingMetrics metrics;

    for (int episode = 1; episode <= config_.num_episodes; ++episode) {
        torch::Tensor state = env_.reset();

        float episode_reward = 0.0f;
        float last_loss = 0.0f;

        for (int step = 0; step < config_.max_steps_per_episode; ++step) {
            int action = agent_.select_action(state, epsilon);
            StepResult result = env_.step(action);

            agent_.remember({
                state.detach().clone(),
                action,
                result.reward,
                result.next_state.detach().clone(),
                result.done
            });

            state = result.next_state;
            episode_reward += result.reward;
            global_step++;

            if (agent_.replay_size() >= config_.batch_size && global_step > config_.warmup_steps) {
                last_loss = agent_.train_step(config_.batch_size, config_.gamma);
            }

            if (result.done) {
                break;
            }
        }

        epsilon = std::max(config_.epsilon_min, epsilon * config_.epsilon_decay);

        if (episode % config_.target_update_interval == 0) {
            agent_.copy_weights_to_target();
        }

        if (episode % config_.plot_update_interval == 0) {
            float success_rate = evaluate_policy(50);

            metrics.episodes.push_back(static_cast<double>(episode));
            metrics.rewards.push_back(static_cast<double>(episode_reward));
            metrics.losses.push_back(static_cast<double>(last_loss));
            metrics.success_rates.push_back(static_cast<double>(success_rate));
            metrics.epsilons.push_back(static_cast<double>(epsilon));

            std::cout << "Episode " << episode
                      << " | Reward: " << episode_reward
                      << " | Loss: " << last_loss
                      << " | Epsilon: " << epsilon
                      << " | Success Rate: " << success_rate
                      << "\n";
        }
    }

    return metrics;
}

float TrainingSession::evaluate_policy(int episodes) {
    int success_count = 0;

    for (int ep = 0; ep < episodes; ++ep) {
        torch::Tensor state = env_.reset();
        bool done = false;
        int steps = 0;

        while (!done && steps < 100) {
            int action = agent_.select_action(state, 0.0f);
            StepResult result = env_.step(action);

            state = result.next_state;
            done = result.done;
            steps++;
        }

        if (env_.agent() == env_.goal()) {
            success_count++;
        }
    }

    return static_cast<float>(success_count) / static_cast<float>(episodes);
}
