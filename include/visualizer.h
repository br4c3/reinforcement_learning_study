#pragma once

#include "dqn_agent.h"
#include "grid_world_env.h"
#include "training_session.h"

#include <cstdio>
#include <string>
#include <vector>

class Visualizer {
public:
    static void plot_training_result(const TrainingMetrics& metrics);
    static void save_policy_gif(DQNAgent& agent, GridWorldEnv& env);

private:
    static void write_datablock(
        FILE* gp,
        const std::string& name,
        const std::vector<double>& x,
        const std::vector<double>& y
    );

    static void save_grid_frame(GridWorldEnv& env, int step, const std::string& output_dir);
};
