#include "grid_world_env.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

GridWorldEnv::GridWorldEnv()
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

torch::Tensor GridWorldEnv::reset() {
    agent_ = start_;
    steps_ = 0;
    return get_state();
}

StepResult GridWorldEnv::step(int action) {
    auto [x, y] = agent_;

    int nx = x;
    int ny = y;

    if (action == 0) {
        ny -= 1;
    } else if (action == 1) {
        ny += 1;
    } else if (action == 2) {
        nx -= 1;
    } else if (action == 3) {
        nx += 1;
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

int GridWorldEnv::state_dim() const {
    return 2;
}

int GridWorldEnv::action_dim() const {
    return 4;
}

int GridWorldEnv::width() const {
    return width_;
}

int GridWorldEnv::height() const {
    return height_;
}

std::pair<int, int> GridWorldEnv::agent() const {
    return agent_;
}

std::pair<int, int> GridWorldEnv::goal() const {
    return goal_;
}

bool GridWorldEnv::is_obstacle(int x, int y) const {
    return std::find(
        obstacles_.begin(),
        obstacles_.end(),
        std::make_pair(x, y)
    ) != obstacles_.end();
}

torch::Tensor GridWorldEnv::get_state() const {
    float nx = static_cast<float>(agent_.first) / static_cast<float>(width_ - 1);
    float ny = static_cast<float>(agent_.second) / static_cast<float>(height_ - 1);

    return torch::tensor({nx, ny}, torch::kFloat32);
}

int GridWorldEnv::manhattan_distance(int x1, int y1, int x2, int y2) const {
    return std::abs(x1 - x2) + std::abs(y1 - y2);
}
