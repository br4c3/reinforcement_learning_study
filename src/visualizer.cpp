#include "visualizer.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>

void Visualizer::write_datablock(
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

void Visualizer::plot_training_result(const TrainingMetrics& metrics) {
    FILE* gp = popen("gnuplot -persistent", "w");

    if (!gp) {
        std::cerr << "Failed to open gnuplot.\n";
        return;
    }

    write_datablock(gp, "reward", metrics.episodes, metrics.rewards);
    write_datablock(gp, "loss", metrics.episodes, metrics.losses);
    write_datablock(gp, "success", metrics.episodes, metrics.success_rates);
    write_datablock(gp, "epsilon", metrics.episodes, metrics.epsilons);

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

void Visualizer::save_grid_frame(GridWorldEnv& env, int step, const std::string& output_dir) {
    FILE* gp = popen("gnuplot", "w");

    if (!gp) {
        std::cerr << "Failed to open gnuplot.\n";
        return;
    }

    auto agent = env.agent();
    auto goal = env.goal();

    std::string filename = output_dir + "/frame_" + std::to_string(1000 + step) + ".png";

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

void Visualizer::save_policy_gif(DQNAgent& agent, GridWorldEnv& env) {
    const std::string frame_dir = "frames";

    std::filesystem::remove_all(frame_dir);
    std::filesystem::create_directory(frame_dir);

    torch::Tensor state = env.reset();
    bool done = false;
    int step = 0;

    save_grid_frame(env, step, frame_dir);

    while (!done && step < 100) {
        int action = agent.select_action(state, 0.0f);
        StepResult result = env.step(action);

        state = result.next_state;
        done = result.done;
        step++;

        save_grid_frame(env, step, frame_dir);
    }

    std::cout << "Saved " << step + 1 << " frames to " << frame_dir << "/\n";

    int ret = std::system("magick -delay 40 -loop 0 frames/frame_*.png gridworld_policy.gif");

    if (ret != 0) {
        std::cout << "ImageMagick command failed.\n";
        std::cout << "If you are on Ubuntu, try:\n";
        std::cout << "  convert -delay 40 -loop 0 frames/frame_*.png gridworld_policy.gif\n";

        std::system("convert -delay 40 -loop 0 frames/frame_*.png gridworld_policy.gif");
    }

    std::cout << "GIF saved to gridworld_policy.gif\n";
}
