#pragma once

#include <torch/torch.h>

struct QNetworkImpl : torch::nn::Module {
    QNetworkImpl(int state_dim, int action_dim);

    torch::Tensor forward(torch::Tensor x);

    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};
};

TORCH_MODULE(QNetwork);
