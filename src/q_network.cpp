#include "q_network.h"

QNetworkImpl::QNetworkImpl(int state_dim, int action_dim) {
    fc1 = register_module("fc1", torch::nn::Linear(state_dim, 128));
    fc2 = register_module("fc2", torch::nn::Linear(128, 128));
    fc3 = register_module("fc3", torch::nn::Linear(128, action_dim));
}

torch::Tensor QNetworkImpl::forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    return fc3->forward(x);
}
