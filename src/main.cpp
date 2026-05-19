#include "dqn_agent.h"
#include "grid_world_env.h"
#include "training_session.h"
#include "visualizer.h"

#include <torch/torch.h>

#include <iostream>

int main() {
    torch::manual_seed(42);

    torch::Device device(torch::kCPU);

    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "Using CUDA\n";
    } else {
        std::cout << "Using CPU\n";
    }

    GridWorldEnv env;
    DQNAgent agent(env.state_dim(), env.action_dim(), device);
    TrainingSession session(env, agent);

    TrainingMetrics metrics = session.run();

    std::cout << "\nTraining finished.\n";

    torch::save(agent.network(), "dqn_gridworld.pt");
    std::cout << "Model saved to dqn_gridworld.pt\n";

    std::cout << "\nSaving policy GIF...\n";
    Visualizer::save_policy_gif(agent, env);

    std::cout << "\nShowing final training plot.\n";
    Visualizer::plot_training_result(metrics);

    return 0;
}
