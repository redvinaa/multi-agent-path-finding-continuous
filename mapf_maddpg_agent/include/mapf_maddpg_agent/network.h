// Copyright 2021 Reda Vince

#ifndef MAPF_MADDPG_AGENT_NETWORK_H
#define MAPF_MADDPG_AGENT_NETWORK_H

#include <torch/torch.h>
#include <vector>

/*! \brief Network class inherited from torch
 */
struct Net : torch::nn::Module
{
    /*! \brief Create neural net with the given parameters
     */
    explicit Net(
        int _input_size,
        int _output_size = 2,
        int _hidden_layer_nodes = 10,
        int _n_hidden_layers = 1);

    /*! \brief Override forward() method
     *
     * \param x The input tensor
     * \return The output tensor
     */
    torch::Tensor forward(torch::Tensor x);

    /* \brief Stores the hidden layers
     */
    std::vector<torch::nn::Linear> layers;
    int input_size;
    int output_size;
    int hidden_layer_nodes;
    int n_hidden_layers;
};

#endif  // MAPF_MADDPG_AGENT_NETWORK_H
