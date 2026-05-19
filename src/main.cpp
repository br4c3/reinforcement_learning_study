#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

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

struct StepResult {
    torch::Tensor next_state;
    float reward;
    bool done;
};

class GridWorldEnv {
public:
    GridWorldEnv()
        : width_(7),
          height_(7),
          start_({0, 0}),
          goal_({6, 6}),
          max_steps_(80) {
        obstacles_ = {
            {1, 1}, {2, 1}, {3, 1},
            {3, 2},
            {1, 3},
            {3, 4}, {4, 4}, {5, 4},
            {5, 5}
        };

        reset();
    }

    torch::Tensor reset() {
        agent_ = start_;
        steps_ = 0;
        return get_state();
    }

    StepResult step(int action) {
        auto [x, y] = agent_;

        int nx = x;
        int ny = y;

        if (action == 0) {
            ny -= 1;  // up
        } else if (action == 1) {
            ny += 1;  // down
        } else if (action == 2) {
            nx -= 1;  // left
        } else if (action == 3) {
            nx += 1;  // right
        } else {
            throw std::invalid_argument("Invalid action");
        }

        steps_++;

        bool hit_wall = nx < 0 || nx >= width_ || ny < 0 || ny >= height_;
        bool hit_obstacle = false;

        if (!hit_wall) {
            hit_obstacle = is_obstacle(nx, ny);
        }

        float reward = -0.05f;

        if (hit_wall || hit_obstacle) {
            reward = -1.0f;
        } else {
            agent_ = {nx, ny};

            const int old_dist = manhattan_distance(x, y, goal_.first, goal_.second);
            const int new_dist = manhattan_distance(nx, ny, goal_.first, goal_.second);

            if (new_dist < old_dist) {
                reward += 0.10f;
            } else {
                reward -= 0.10f;
            }
        }

        bool reached_goal = agent_ == goal_;
        bool timeout = steps_ >= max_steps_;
        bool done = reached_goal || timeout;

        if (reached_goal) {
            reward = 10.0f;
        }

        return {get_state(), reward, done};
    }

    int state_dim() const {
        return 2;
    }

    int action_dim() const {
        return 4;
    }

    int width() const {
        return width_;
    }

    int height() const {
        return height_;
    }

    std::pair<int, int> agent() const {
        return agent_;
    }

    std::pair<int, int> goal() const {
        return goal_;
    }

    bool is_obstacle(int x, int y) const {
        return std::find(
            obstacles_.begin(),
            obstacles_.end(),
            std::make_pair(x, y)
        ) != obstacles_.end();
    }

private:
    torch::Tensor get_state() const {
        float nx = static_cast<float>(agent_.first) / static_cast<float>(width_ - 1);
        float ny = static_cast<float>(agent_.second) / static_cast<float>(height_ - 1);

        return torch::tensor({nx, ny}, torch::kFloat32);
    }

    int manhattan_distance(int x1, int y1, int x2, int y2) const {
        return std::abs(x1 - x2) + std::abs(y1 - y2);
    }

private:
    int width_;
    int height_;
    std::pair<int, int> start_;
    std::pair<int, int> goal_;
    std::pair<int, int> agent_;
    std::vector<std::pair<int, int>> obstacles_;
    int max_steps_;
    int steps_ = 0;
};

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

struct QNetworkImpl : torch::nn::Module {
    QNetworkImpl(int state_dim, int action_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(state_dim, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 128));
        fc3 = register_module("fc3", torch::nn::Linear(128, action_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x);
    }

    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};
};

TORCH_MODULE(QNetwork);

void copy_weights(QNetwork& source, QNetwork& target) {
    torch::NoGradGuard no_grad;

    auto source_params = source->named_parameters();
    auto target_params = target->named_parameters();

    for (const auto& item : source_params) {
        target_params[item.key()].copy_(item.value());
    }
}

int select_action(
    QNetwork& q_net,
    const torch::Tensor& state,
    float epsilon,
    int action_dim,
    torch::Device device
) {
    if (RNG.uniform() < epsilon) {
        return RNG.randint(0, action_dim - 1);
    }

    torch::NoGradGuard no_grad;

    torch::Tensor input = state.to(device).unsqueeze(0);
    torch::Tensor q_values = q_net->forward(input);

    return q_values.argmax(1).item<int>();
}

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

    torch::Tensor state_batch = torch::stack(states).to(device);
    torch::Tensor next_state_batch = torch::stack(next_states).to(device);
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

float evaluate_policy(
    QNetwork& q_net,
    GridWorldEnv& env,
    int episodes,
    torch::Device device
) {
    int success_count = 0;

    for (int ep = 0; ep < episodes; ++ep) {
        torch::Tensor state = env.reset();
        bool done = false;
        int steps = 0;

        while (!done && steps < 100) {
            int action = select_action(q_net, state, 0.0f, env.action_dim(), device);
            StepResult result = env.step(action);

            state = result.next_state;
            done = result.done;
            steps++;
        }

        if (env.agent() == env.goal()) {
            success_count++;
        }
    }

    return static_cast<float>(success_count) / static_cast<float>(episodes);
}

void write_datablock(
    FILE* gp,
    const std::string& name,
    const std::vector<double>& x,
    const std::vector<double>& y
) {
    fprintf(gp, "$%s << EOD\n", name.c_str());

    for (size_t i = 0; i < x.size(); ++i) {
        fprintf(gp, "%f %f\n", x[i], y[i]);
    }

    fprintf(gp, "EOD\n");
}

void plot_training_result(
    const std::vector<double>& episodes,
    const std::vector<double>& rewards,
    const std::vector<double>& losses,
    const std::vector<double>& success_rates,
    const std::vector<double>& epsilons
) {
    FILE* gp = popen("gnuplot -persistent", "w");

    if (!gp) {
        std::cerr << "Failed to open gnuplot.\n";
        return;
    }

    write_datablock(gp, "reward", episodes, rewards);
    write_datablock(gp, "loss", episodes, losses);
    write_datablock(gp, "success", episodes, success_rates);
    write_datablock(gp, "epsilon", episodes, epsilons);

    fprintf(gp, "set terminal qt size 1200,800\n");
    fprintf(gp, "set grid\n");
    fprintf(gp, "set key outside\n");
    fprintf(gp, "set multiplot layout 2,2 title '2D GridWorld DQN Training Result'\n");

    fprintf(gp, "set title 'Episode Reward'\n");
    fprintf(gp, "set xlabel 'Episode'\n");
    fprintf(gp, "set ylabel 'Reward'\n");
    fprintf(gp, "unset yrange\n");
    fprintf(gp, "plot $reward using 1:2 with lines linewidth 2 title 'Reward'\n");

    fprintf(gp, "set title 'DQN Loss'\n");
    fprintf(gp, "set xlabel 'Episode'\n");
    fprintf(gp, "set ylabel 'Loss'\n");
    fprintf(gp, "unset yrange\n");
    fprintf(gp, "plot $loss using 1:2 with lines linewidth 2 title 'Loss'\n");

    fprintf(gp, "set title 'Success Rate'\n");
    fprintf(gp, "set xlabel 'Episode'\n");
    fprintf(gp, "set ylabel 'Success Rate'\n");
    fprintf(gp, "set yrange [0:1.05]\n");
    fprintf(gp, "plot $success using 1:2 with lines linewidth 2 title 'Success Rate'\n");

    fprintf(gp, "set title 'Epsilon Decay'\n");
    fprintf(gp, "set xlabel 'Episode'\n");
    fprintf(gp, "set ylabel 'Epsilon'\n");
    fprintf(gp, "set yrange [0:1.05]\n");
    fprintf(gp, "plot $epsilon using 1:2 with lines linewidth 2 title 'Epsilon'\n");

    fprintf(gp, "unset multiplot\n");
    fflush(gp);

    std::cout << "Close the plot window to exit.\n";
    std::cout << "Press ENTER to continue...\n";
    std::cin.get();

    pclose(gp);
}

void save_grid_frame(
    GridWorldEnv& env,
    int step,
    const std::string& output_dir
) {
    FILE* gp = popen("gnuplot", "w");

    if (!gp) {
        std::cerr << "Failed to open gnuplot.\n";
        return;
    }

    auto agent = env.agent();
    auto goal = env.goal();

    std::string filename =
        output_dir + "/frame_" + std::to_string(1000 + step) + ".png";

    fprintf(gp, "set terminal pngcairo size 700,700 enhanced font 'Arial,16'\n");
    fprintf(gp, "set output '%s'\n", filename.c_str());

    fprintf(gp, "unset key\n");
    fprintf(gp, "unset colorbox\n");
    fprintf(gp, "set size square\n");

    fprintf(gp, "set xrange [-0.5:%f]\n", env.width() - 0.5);
    fprintf(gp, "set yrange [%f:-0.5]\n", env.height() - 0.5);

    fprintf(gp, "set xtics 1\n");
    fprintf(gp, "set ytics 1\n");
    fprintf(gp, "set grid linewidth 1\n");

    fprintf(gp, "set title 'DQN GridWorld Policy | Step %d'\n", step);

    fprintf(gp, "$cells << EOD\n");

    for (int y = 0; y < env.height(); ++y) {
        for (int x = 0; x < env.width(); ++x) {
            int value = 0;

            if (env.is_obstacle(x, y)) {
                value = 1;
            }

            if (goal.first == x && goal.second == y) {
                value = 2;
            }

            if (agent.first == x && agent.second == y) {
                value = 3;
            }

            fprintf(gp, "%d %d %d\n", x, y, value);
        }
    }

    fprintf(gp, "EOD\n");

    fprintf(gp, "set palette defined (0 'white', 1 'black', 2 'green', 3 'red')\n");
    fprintf(gp, "plot $cells using 1:2:3 with points pt 5 ps 8 palette\n");

    fprintf(gp, "set output\n");
    fflush(gp);
    pclose(gp);
}

void save_policy_gif(
    QNetwork& q_net,
    GridWorldEnv& env,
    torch::Device device
) {
    const std::string frame_dir = "frames";

    std::filesystem::remove_all(frame_dir);
    std::filesystem::create_directory(frame_dir);

    torch::Tensor state = env.reset();
    bool done = false;
    int step = 0;

    save_grid_frame(env, step, frame_dir);

    while (!done && step < 100) {
        int action = select_action(q_net, state, 0.0f, env.action_dim(), device);
        StepResult result = env.step(action);

        state = result.next_state;
        done = result.done;
        step++;

        save_grid_frame(env, step, frame_dir);
    }

    std::cout << "Saved " << step + 1 << " frames to " << frame_dir << "/\n";

    int ret = std::system(
        "magick -delay 40 -loop 0 frames/frame_*.png gridworld_policy.gif"
    );

    if (ret != 0) {
        std::cout << "ImageMagick command failed.\n";
        std::cout << "If you are on Ubuntu, try:\n";
        std::cout << "  convert -delay 40 -loop 0 frames/frame_*.png gridworld_policy.gif\n";

        std::system(
            "convert -delay 40 -loop 0 frames/frame_*.png gridworld_policy.gif"
        );
    }

    std::cout << "GIF saved to gridworld_policy.gif\n";
}

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

    const int state_dim = env.state_dim();
    const int action_dim = env.action_dim();

    QNetwork q_net(state_dim, action_dim);
    QNetwork target_net(state_dim, action_dim);

    q_net->to(device);
    target_net->to(device);

    copy_weights(q_net, target_net);

    torch::optim::Adam optimizer(
        q_net->parameters(),
        torch::optim::AdamOptions(1e-3)
    );

    ReplayBuffer replay_buffer(30000);

    const int num_episodes = 1200;
    const int max_steps_per_episode = 80;
    const size_t batch_size = 64;
    const float gamma = 0.98f;

    float epsilon = 1.0f;
    const float epsilon_min = 0.03f;
    const float epsilon_decay = 0.995f;

    const int warmup_steps = 500;
    const int target_update_interval = 20;
    const int plot_update_interval = 20;

    int global_step = 0;

    std::vector<double> plot_episodes;
    std::vector<double> plot_rewards;
    std::vector<double> plot_losses;
    std::vector<double> plot_success_rates;
    std::vector<double> plot_epsilons;

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

        if (episode % plot_update_interval == 0) {
            float success_rate = evaluate_policy(q_net, env, 50, device);

            plot_episodes.push_back(static_cast<double>(episode));
            plot_rewards.push_back(static_cast<double>(episode_reward));
            plot_losses.push_back(static_cast<double>(last_loss));
            plot_success_rates.push_back(static_cast<double>(success_rate));
            plot_epsilons.push_back(static_cast<double>(epsilon));

            std::cout << "Episode " << episode
                      << " | Reward: " << episode_reward
                      << " | Loss: " << last_loss
                      << " | Epsilon: " << epsilon
                      << " | Success Rate: " << success_rate
                      << "\n";
        }
    }

    std::cout << "\nTraining finished.\n";

    torch::save(q_net, "dqn_gridworld.pt");
    std::cout << "Model saved to dqn_gridworld.pt\n";

    std::cout << "\nSaving policy GIF...\n";
    save_policy_gif(q_net, env, device);

    std::cout << "\nShowing final training plot.\n";
    plot_training_result(
        plot_episodes,
        plot_rewards,
        plot_losses,
        plot_success_rates,
        plot_epsilons
    );

    return 0;
}