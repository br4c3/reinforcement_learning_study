#include <torch/torch.h>
#include <raylib.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

constexpr int ACTIONS = 5;

enum Action {
    HOVER = 0,
    UP = 1,
    DOWN = 2,
    LEFT = 3,
    RIGHT = 4
};

struct Vec2 {
    double x;
    double y;
};

struct Obstacle {
    Vec2 pos;
    double radius;
};

struct State {
    Vec2 pos;
    Vec2 vel;
};

struct StepResult {
    State next_state;
    double reward;
    bool done;
    bool success;
    bool collision;
};

struct Transition {
    State state;
    int action;
    double reward;
    State next_state;
    bool done;
};

struct EvalResult {
    int success_count = 0;
    int collision_count = 0;
    double avg_reward = 0.0;
};

double dist(Vec2 a, Vec2 b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

class Drone2DEnv {
public:
    Drone2DEnv() {
        goal_ = {9.0, 9.0};
        obstacles_ = {
            {{4.0, 4.0}, 1.0},
            {{6.0, 7.0}, 1.0},
            {{7.5, 4.8}, 1.2}
        };
    }

    State reset() {
        state_ = {{0.5, 0.5}, {0.0, 0.0}};
        return state_;
    }

    StepResult step(int action) {
        constexpr double dt = 0.1;
        constexpr double acc = 1.2;
        constexpr double damping = 0.90;
        constexpr double max_speed = 1.8;

        State prev_state = state_;

        Vec2 a{0.0, 0.0};

        if (action == UP) a.y += acc;
        if (action == DOWN) a.y -= acc;
        if (action == LEFT) a.x -= acc;
        if (action == RIGHT) a.x += acc;

        state_.vel.x += a.x * dt;
        state_.vel.y += a.y * dt;

        state_.vel.x *= damping;
        state_.vel.y *= damping;

        state_.vel.x = std::clamp(state_.vel.x, -max_speed, max_speed);
        state_.vel.y = std::clamp(state_.vel.y, -max_speed, max_speed);

        state_.pos.x += state_.vel.x * dt;
        state_.pos.y += state_.vel.y * dt;

        bool out_of_bounds =
            state_.pos.x < 0.0 || state_.pos.x > 10.0 ||
            state_.pos.y < 0.0 || state_.pos.y > 10.0;

        bool collision = false;
        for (const auto& obs : obstacles_) {
            if (dist(state_.pos, obs.pos) <= obs.radius) {
                collision = true;
                break;
            }
        }

        double prev_goal_dist = dist(prev_state.pos, goal_);
        double goal_dist = dist(state_.pos, goal_);

        bool success = goal_dist < 0.8;

        double reward = -0.01;
        reward += 4.0 * (prev_goal_dist - goal_dist);
        reward += -0.002 * goal_dist;

        double nearest = nearestObstacleDistance(state_.pos);
        if (nearest < 1.5) {
            reward += -0.3 * (1.5 - nearest);
        }

        if (success) reward += 20.0;
        if (collision) reward -= 20.0;
        if (out_of_bounds) reward -= 20.0;

        bool done = success || collision || out_of_bounds;

        return {state_, reward, done, success, collision};
    }

    Vec2 goal() const {
        return goal_;
    }

    const std::vector<Obstacle>& obstacles() const {
        return obstacles_;
    }

    double nearestObstacleDistance(Vec2 p) const {
        double best = 1e9;

        for (const auto& obs : obstacles_) {
            best = std::min(best, dist(p, obs.pos) - obs.radius);
        }

        return best;
    }

    Vec2 nearestObstacleDelta(Vec2 p) const {
        double best = 1e9;
        Vec2 delta{0.0, 0.0};

        for (const auto& obs : obstacles_) {
            double d = dist(p, obs.pos) - obs.radius;
            if (d < best) {
                best = d;
                delta = {obs.pos.x - p.x, obs.pos.y - p.y};
            }
        }

        return delta;
    }

private:
    State state_;
    Vec2 goal_;
    std::vector<Obstacle> obstacles_;
};

void printProgress(int current, int total) {
    constexpr int width = 40;

    double ratio = static_cast<double>(current) / static_cast<double>(total);
    int filled = static_cast<int>(ratio * width);

    std::cout << "\r[";

    for (int i = 0; i < width; ++i) {
        std::cout << (i < filled ? "=" : " ");
    }

    std::cout << "] "
              << static_cast<int>(ratio * 100.0)
              << "% "
              << current << "/" << total
              << std::flush;
}

class TabularQLearning {
public:
    static constexpr int X_BINS = 15;
    static constexpr int Y_BINS = 15;
    static constexpr int VX_BINS = 7;
    static constexpr int VY_BINS = 7;

    TabularQLearning()
        : q_(X_BINS * Y_BINS * VX_BINS * VY_BINS * ACTIONS, 0.0) {}

    int selectAction(State s, double epsilon, std::mt19937& gen) {
        std::uniform_real_distribution<double> prob(0.0, 1.0);
        std::uniform_int_distribution<int> random_action(0, ACTIONS - 1);

        if (prob(gen) < epsilon) {
            return random_action(gen);
        }

        int best_action = 0;
        double best_value = qValue(s, 0);

        for (int a = 1; a < ACTIONS; ++a) {
            double v = qValue(s, a);
            if (v > best_value) {
                best_value = v;
                best_action = a;
            }
        }

        return best_action;
    }

    void update(
        State s,
        int action,
        double reward,
        State ns,
        bool done,
        double alpha,
        double gamma
    ) {
        double best_next = qValue(ns, 0);

        for (int a = 1; a < ACTIONS; ++a) {
            best_next = std::max(best_next, qValue(ns, a));
        }

        double target = reward;
        if (!done) {
            target += gamma * best_next;
        }

        double& q = qRef(s, action);
        q += alpha * (target - q);
    }

private:
    int bin(double v, double min_v, double max_v, int bins) const {
        v = std::clamp(v, min_v, max_v);
        double ratio = (v - min_v) / (max_v - min_v);
        int b = static_cast<int>(ratio * bins);

        if (b >= bins) b = bins - 1;
        if (b < 0) b = 0;

        return b;
    }

    int index(State s, int action) const {
        int xb = bin(s.pos.x, 0.0, 10.0, X_BINS);
        int yb = bin(s.pos.y, 0.0, 10.0, Y_BINS);
        int vxb = bin(s.vel.x, -2.0, 2.0, VX_BINS);
        int vyb = bin(s.vel.y, -2.0, 2.0, VY_BINS);

        return (((xb * Y_BINS + yb) * VX_BINS + vxb) * VY_BINS + vyb)
               * ACTIONS + action;
    }

    double qValue(State s, int action) const {
        return q_[index(s, action)];
    }

    double& qRef(State s, int action) {
        return q_[index(s, action)];
    }

    std::vector<double> q_;
};

void trainQLearning(TabularQLearning& agent) {
    std::random_device rd;
    std::mt19937 gen(rd());

    constexpr int episodes = 4000;
    constexpr int max_steps = 300;

    double alpha = 0.1;
    double gamma = 0.98;
    double epsilon = 1.0;
    double epsilon_min = 0.05;
    double epsilon_decay = 0.995;

    std::cout << "Training Q-learning...\n";

    for (int ep = 0; ep < episodes; ++ep) {
        printProgress(ep + 1, episodes);

        Drone2DEnv env;
        State s = env.reset();

        for (int t = 0; t < max_steps; ++t) {
            int action = agent.selectAction(s, epsilon, gen);
            auto result = env.step(action);

            agent.update(
                s,
                action,
                result.reward,
                result.next_state,
                result.done,
                alpha,
                gamma
            );

            s = result.next_state;

            if (result.done) {
                break;
            }
        }

        epsilon *= epsilon_decay;
        if (epsilon < epsilon_min) {
            epsilon = epsilon_min;
        }
    }

    std::cout << "\nQ-learning finished.\n";
}

torch::Tensor stateToTensor(const Drone2DEnv& env, State s) {
    Vec2 g = env.goal();
    Vec2 obs_delta = env.nearestObstacleDelta(s.pos);

    return torch::tensor(
        {
            static_cast<float>(s.pos.x / 10.0),
            static_cast<float>(s.pos.y / 10.0),
            static_cast<float>(s.vel.x / 2.0),
            static_cast<float>(s.vel.y / 2.0),
            static_cast<float>((g.x - s.pos.x) / 10.0),
            static_cast<float>((g.y - s.pos.y) / 10.0),
            static_cast<float>(obs_delta.x / 10.0),
            static_cast<float>(obs_delta.y / 10.0)
        },
        torch::kFloat32
    ).unsqueeze(0);
}

struct DQNImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};

    DQNImpl() {
        fc1 = register_module("fc1", torch::nn::Linear(8, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 128));
        fc3 = register_module("fc3", torch::nn::Linear(128, ACTIONS));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        return fc3(x);
    }
};

TORCH_MODULE(DQN);

class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t capacity)
        : capacity_(capacity), position_(0) {}

    void push(const Transition& t) {
        if (buffer_.size() < capacity_) {
            buffer_.push_back(t);
        } else {
            buffer_[position_] = t;
        }

        position_ = (position_ + 1) % capacity_;
    }

    std::vector<Transition> sample(size_t batch_size, std::mt19937& gen) {
        std::vector<Transition> batch;
        batch.reserve(batch_size);

        std::uniform_int_distribution<size_t> dist(0, buffer_.size() - 1);

        for (size_t i = 0; i < batch_size; ++i) {
            batch.push_back(buffer_[dist(gen)]);
        }

        return batch;
    }

    size_t size() const {
        return buffer_.size();
    }

private:
    size_t capacity_;
    size_t position_;
    std::vector<Transition> buffer_;
};

void copyWeights(DQN& target, DQN& policy) {
    torch::NoGradGuard no_grad;

    auto policy_params = policy->named_parameters();
    auto target_params = target->named_parameters();

    for (const auto& item : policy_params) {
        target_params[item.key()].copy_(item.value());
    }
}

int selectDQNAction(
    DQN& net,
    const Drone2DEnv& env,
    State s,
    double epsilon,
    std::mt19937& gen
) {
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_int_distribution<int> random_action(0, ACTIONS - 1);

    if (prob(gen) < epsilon) {
        return random_action(gen);
    }

    torch::NoGradGuard no_grad;

    auto q = net->forward(stateToTensor(env, s));
    return q.argmax(1).item<int>();
}

void trainDQNStep(
    DQN& policy,
    DQN& target,
    torch::optim::Adam& optimizer,
    ReplayBuffer& buffer,
    Drone2DEnv& env,
    size_t batch_size,
    double gamma,
    std::mt19937& gen
) {
    if (buffer.size() < batch_size) {
        return;
    }

    auto batch = buffer.sample(batch_size, gen);

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
        states.push_back(stateToTensor(env, t.state));
        next_states.push_back(stateToTensor(env, t.next_state));
        actions.push_back(t.action);
        rewards.push_back(static_cast<float>(t.reward));
        dones.push_back(t.done ? 1.0f : 0.0f);
    }

    auto state_batch = torch::cat(states, 0);
    auto next_state_batch = torch::cat(next_states, 0);

    auto action_batch = torch::tensor(actions, torch::kInt64).unsqueeze(1);
    auto reward_batch = torch::tensor(rewards, torch::kFloat32);
    auto done_batch = torch::tensor(dones, torch::kFloat32);

    auto q_values = policy->forward(state_batch);
    auto current_q = q_values.gather(1, action_batch).squeeze(1);

    torch::Tensor target_q;

    {
        torch::NoGradGuard no_grad;

        auto next_policy_q = policy->forward(next_state_batch);
        auto next_actions = std::get<1>(next_policy_q.max(1)).unsqueeze(1);

        auto next_target_q = target->forward(next_state_batch);
        auto selected_next_q = next_target_q.gather(1, next_actions).squeeze(1);

        target_q =
            reward_batch
            + static_cast<float>(gamma) * selected_next_q * (1.0f - done_batch);
    }

    auto loss = torch::smooth_l1_loss(current_q, target_q);

    optimizer.zero_grad();
    loss.backward();

    torch::nn::utils::clip_grad_norm_(policy->parameters(), 10.0);

    optimizer.step();
}

void trainDQN(DQN& policy) {
    std::random_device rd;
    std::mt19937 gen(rd());

    DQN target;
    copyWeights(target, policy);

    torch::optim::Adam optimizer(
        policy->parameters(),
        torch::optim::AdamOptions(3e-4)
    );

    ReplayBuffer buffer(50000);

    constexpr int episodes = 6000;
    constexpr int max_steps = 300;
    constexpr int target_update_interval = 100;
    constexpr size_t batch_size = 64;
    constexpr size_t warmup_size = 1000;

    double gamma = 0.98;
    double epsilon = 1.0;
    double epsilon_min = 0.05;
    double epsilon_decay = 0.996;

    std::cout << "\nTraining DQN...\n";

    for (int ep = 0; ep < episodes; ++ep) {
        printProgress(ep + 1, episodes);

        Drone2DEnv env;
        State s = env.reset();

        for (int t = 0; t < max_steps; ++t) {
            int action = selectDQNAction(policy, env, s, epsilon, gen);
            auto result = env.step(action);

            buffer.push({s, action, result.reward, result.next_state, result.done});

            if (buffer.size() >= warmup_size) {
                trainDQNStep(
                    policy,
                    target,
                    optimizer,
                    buffer,
                    env,
                    batch_size,
                    gamma,
                    gen
                );
            }

            s = result.next_state;

            if (result.done) {
                break;
            }
        }

        epsilon *= epsilon_decay;
        if (epsilon < epsilon_min) {
            epsilon = epsilon_min;
        }

        if ((ep + 1) % target_update_interval == 0) {
            copyWeights(target, policy);
        }
    }

    std::cout << "\nDQN finished.\n";
}

EvalResult evaluateQLearning(TabularQLearning& agent, int episodes) {
    EvalResult result;
    std::mt19937 gen(0);

    for (int ep = 0; ep < episodes; ++ep) {
        Drone2DEnv env;
        State s = env.reset();

        double total_reward = 0.0;

        for (int t = 0; t < 300; ++t) {
            int action = agent.selectAction(s, 0.0, gen);
            auto step = env.step(action);

            total_reward += step.reward;
            s = step.next_state;

            if (step.success) result.success_count++;
            if (step.collision) result.collision_count++;

            if (step.done) break;
        }

        result.avg_reward += total_reward;
    }

    result.avg_reward /= episodes;
    return result;
}

EvalResult evaluateDQN(DQN& policy, int episodes) {
    EvalResult result;
    std::mt19937 gen(0);

    for (int ep = 0; ep < episodes; ++ep) {
        Drone2DEnv env;
        State s = env.reset();

        double total_reward = 0.0;

        for (int t = 0; t < 300; ++t) {
            int action = selectDQNAction(policy, env, s, 0.0, gen);
            auto step = env.step(action);

            total_reward += step.reward;
            s = step.next_state;

            if (step.success) result.success_count++;
            if (step.collision) result.collision_count++;

            if (step.done) break;
        }

        result.avg_reward += total_reward;
    }

    result.avg_reward /= episodes;
    return result;
}

Vector2 worldToScreen(Vec2 p) {
    constexpr float scale = 70.0f;
    constexpr float margin = 50.0f;
    constexpr float screen_height = 800.0f;

    float sx = margin + static_cast<float>(p.x) * scale;
    float sy = screen_height - margin - static_cast<float>(p.y) * scale;

    return {sx, sy};
}

void renderEnv(
    const char* title,
    const Drone2DEnv& env,
    State s,
    const std::vector<Vector2>& path,
    const char* status
) {
    BeginDrawing();

    ClearBackground(RAYWHITE);

    DrawText(title, 20, 20, 24, BLACK);
    DrawText("Close window to continue", 20, 50, 18, DARKGRAY);
    DrawText(status, 570, 20, 24, MAROON);

    DrawRectangleLines(50, 50, 700, 700, BLACK);

    Vector2 goal_screen = worldToScreen(env.goal());
    DrawCircleV(goal_screen, 18, BLUE);
    DrawText("GOAL", goal_screen.x + 20, goal_screen.y - 10, 16, BLUE);

    for (const auto& obs : env.obstacles()) {
        Vector2 p = worldToScreen(obs.pos);
        DrawCircleV(p, static_cast<float>(obs.radius * 70.0), RED);
    }

    for (size_t i = 1; i < path.size(); ++i) {
        DrawLineV(path[i - 1], path[i], ORANGE);
    }

    Vector2 drone = worldToScreen(s.pos);
    DrawCircleV(drone, 10, GREEN);

    Vector2 vel_end = {
        drone.x + static_cast<float>(s.vel.x * 25.0),
        drone.y - static_cast<float>(s.vel.y * 25.0)
    };

    DrawLineV(drone, vel_end, DARKGREEN);

    DrawText(
        TextFormat(
            "pos=(%.2f, %.2f) vel=(%.2f, %.2f)",
            s.pos.x,
            s.pos.y,
            s.vel.x,
            s.vel.y
        ),
        20,
        760,
        18,
        BLACK
    );

    EndDrawing();
}

void visualizeQLearning(TabularQLearning& agent) {
    InitWindow(800, 800, "Q-learning 2D Drone");
    SetTargetFPS(30);

    std::mt19937 gen(0);

    Drone2DEnv env;
    State s = env.reset();

    std::vector<Vector2> path;
    path.push_back(worldToScreen(s.pos));

    bool done = false;
    bool success = false;
    bool collision = false;

    int step_count = 0;

    while (!WindowShouldClose()) {
        if (!done && step_count < 500) {
            int action = agent.selectAction(s, 0.0, gen);
            auto result = env.step(action);

            s = result.next_state;
            path.push_back(worldToScreen(s.pos));

            done = result.done;
            success = result.success;
            collision = result.collision;
            step_count++;
        }

        const char* status = "RUNNING";
        if (success) status = "SUCCESS";
        else if (collision) status = "COLLISION";
        else if (done) status = "DONE";

        renderEnv("Tabular Q-learning", env, s, path, status);
    }

    CloseWindow();
}

void visualizeDQN(DQN& policy) {
    InitWindow(800, 800, "DQN 2D Drone");
    SetTargetFPS(30);

    std::mt19937 gen(0);

    Drone2DEnv env;
    State s = env.reset();

    std::vector<Vector2> path;
    path.push_back(worldToScreen(s.pos));

    bool done = false;
    bool success = false;
    bool collision = false;

    int step_count = 0;

    while (!WindowShouldClose()) {
        if (!done && step_count < 500) {
            int action = selectDQNAction(policy, env, s, 0.0, gen);
            auto result = env.step(action);

            s = result.next_state;
            path.push_back(worldToScreen(s.pos));

            done = result.done;
            success = result.success;
            collision = result.collision;
            step_count++;
        }

        const char* status = "RUNNING";
        if (success) status = "SUCCESS";
        else if (collision) status = "COLLISION";
        else if (done) status = "DONE";

        renderEnv("DQN", env, s, path, status);
    }

    CloseWindow();
}

int main() {
    TabularQLearning q_agent;
    DQN dqn_policy;

    trainQLearning(q_agent);
    trainDQN(dqn_policy);

    constexpr int eval_episodes = 100;

    auto q_eval = evaluateQLearning(q_agent, eval_episodes);
    auto dqn_eval = evaluateDQN(dqn_policy, eval_episodes);

    std::cout << "\n========== Evaluation ==========\n";

    std::cout << "[Q-learning]\n";
    std::cout << "Success   : " << q_eval.success_count << "/" << eval_episodes << "\n";
    std::cout << "Collision : " << q_eval.collision_count << "/" << eval_episodes << "\n";
    std::cout << "AvgReward : " << q_eval.avg_reward << "\n";

    std::cout << "\n[DQN]\n";
    std::cout << "Success   : " << dqn_eval.success_count << "/" << eval_episodes << "\n";
    std::cout << "Collision : " << dqn_eval.collision_count << "/" << eval_episodes << "\n";
    std::cout << "AvgReward : " << dqn_eval.avg_reward << "\n";

    visualizeQLearning(q_agent);
    visualizeDQN(dqn_policy);

    return 0;
}