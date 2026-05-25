#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <algorithm> 
#include <iomanip>
#include <unistd.h>


void printProgress(int current, int total)
{
    constexpr int barWidth = 40;

    double ratio = static_cast<double>(current) / total;
    int filled = static_cast<int>(ratio * barWidth);

    std::cout << "\r[";

    for (int i = 0; i < barWidth; ++i) {
        if (i < filled) {
            std::cout << "=";
        } else {
            std::cout << " ";
        }
    }

    std::cout << "] "
              << std::setw(3)
              << static_cast<int>(ratio * 100.0)
              << "% "
              << current << "/" << total
              << std::flush;
}

void printMap(int width, 
              int height, 
              std::vector<std::pair<int, int>> trajectories) 
{
    std::cout << "\r=====< MAP >=====\n";

    std::vector<std::vector<std::string>> map(
        height,
        std::vector<std::string>(width, "0")
    );

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (std::pair<int,int> yx: trajectories) {
                std::cout << yx.first << ", " << yx.second << "\n";
                if (yx.first == h && yx.second == w) {
                    map[h][w] = "1";
                    break;
                }
            }
        }
    }

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w)
            std::cout << std::setw(3) << map[h][w] << " ";      
        std::cout << '\n';
    }

    std::cout << std::flush;
    sleep(1);
}

constexpr int H = 5;
constexpr int W = 5;
constexpr int ACTIONS = 4;

enum Action {
    UP = 0,
    DOWN = 1,
    LEFT = 2,
    RIGHT = 3
};

struct State {
    int y;
    int x;
};

bool isWall(int y, int x) {
    return y == 1 && x == 1;
}

bool isGoal(int y, int x) {
    return y == 4 && x == 4;
}

bool inRange(int y, int x) {
    return y >= 0 && y < H && x >= 0 && x < W;
}

State step(State s, int action, double& reward, bool& done) {
    int ny = s.y;
    int nx = s.x;

    if (action == UP) ny--;
    if (action == DOWN) ny++;
    if (action == LEFT) nx--;
    if (action == RIGHT) nx++;

    if (!inRange(ny, nx) || isWall(ny, nx)) {
        ny = s.y;
        nx = s.x;
        reward = -5.0;
    } else {
        reward = -1.0;
    }

    done = false;

    if (isGoal(ny, nx)) {
        reward = 100.0;
        done = true;
    }

    return {ny, nx};
}

int argmaxAction(const std::array<double, ACTIONS>& q) {
    return std::max_element(q.begin(), q.end()) - q.begin();
}


int main() {
    std::vector<std::vector<std::array<double, ACTIONS>>> Q(
        H, std::vector<std::array<double, ACTIONS>>(W)
    );

    // Initialization
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            Q[y][x].fill(0.0);
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> prob(0.0,1.0);
    std::uniform_int_distribution<int> randomAction(0, ACTIONS-1);

    double alpha = 0.1;
    double gamma = 0.95;
    double epsilon = 0.2;

    int episodes = 5000;
    int maxSteps = 100;

    for (int ep = 0; ep < episodes; ++ep) {
        printProgress(ep+1, episodes);

        State s{0,0};
        for (int t = 0; t < maxSteps; ++t) {
            int action;

            if (prob(gen) < epsilon) {
                action = randomAction(gen);
            } else {
                action = argmaxAction(Q[s.y][s.x]);
            }

            double reward;
            bool done;

            State ns = step(s, action, reward, done);

            double bestNextQ = *std::max_element(
                Q[ns.y][ns.x].begin(),
                Q[ns.y][ns.x].end()
            );

            Q[s.y][s.x][action] = 
            Q[s.y][s.x][action] 
            + alpha * (reward + gamma * bestNextQ - Q[s.y][s.x][action]);

            s = ns;

            if (done) break;
        }

        epsilon *= 0.999;
        if (epsilon < 0.01) {
            epsilon = 0.01;
        }
    }
    std::cout << std::endl;

    std::cout << "Learned" << std::endl;
    
    State s{0,0};

    std::vector<std::pair<int,int>> traj;
    for (int t = 0; t < 30; ++t) {
        std::cout << "(" << s.y << ", " << s.x << ")" << std::endl;
        traj.push_back({s.y,s.x});
        printMap(H, W, traj);

        if (isGoal(s.y, s.x)) {
            std::cout << "GOAL" << std::endl;
            break;
        }

        int action = argmaxAction(Q[s.y][s.x]);
        double reward;
        bool done;
        s = step(s, action, reward, done);

        if (done) {
            std::cout << "(" << s.y << ", " << s.x << ") GOAL" << std::endl;
            break;
        }
    }

    return 0;
}