#pragma once

#include <random>

class Random {
public:
    Random();

    int randint(int low, int high_inclusive);
    float uniform(float low = 0.0f, float high = 1.0f);

private:
    std::mt19937 gen_;
};

Random& global_rng();
