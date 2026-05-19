#pragma once

#include <torch/torch.h>

#include <utility>
#include <vector>

struct StepResult {
    torch::Tensor next_state;
    float reward;
    bool done;
};

class GridWorldEnv {
public:
    GridWorldEnv();

    torch::Tensor reset();
    StepResult step(int action);

    int state_dim() const;
    int action_dim() const;
    int width() const;
    int height() const;
    std::pair<int, int> agent() const;
    std::pair<int, int> goal() const;
    bool is_obstacle(int x, int y) const;

private:
    torch::Tensor get_state() const;
    int manhattan_distance(int x1, int y1, int x2, int y2) const;

    int width_;
    int height_;
    std::pair<int, int> start_;
    std::pair<int, int> goal_;
    std::pair<int, int> agent_;
    std::vector<std::pair<int, int>> obstacles_;
    int max_steps_;
    int steps_ = 0;
};
