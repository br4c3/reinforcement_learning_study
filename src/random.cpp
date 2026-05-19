#include "random.h"

Random::Random() : gen_(std::random_device{}()) {}

int Random::randint(int low, int high_inclusive) {
    std::uniform_int_distribution<int> dist(low, high_inclusive);
    return dist(gen_);
}

float Random::uniform(float low, float high) {
    std::uniform_real_distribution<float> dist(low, high);
    return dist(gen_);
}

Random& global_rng() {
    static Random rng;
    return rng;
}
