// Copyright 2021 Reda Vince

#ifndef MAAC_AGENT_NETWORK_H
#define MAAC_AGENT_NETWORK_H

#include <torch/torch.h>
#include <vector>

/*! \brief Network class inherited from torch
 */
struct Net : torch::nn::Module
{
    /*! \brief Create neural net with the given parameters
     *
     * \param _batch_norm Perform batch normalization on input
     */
    explicit Net(
        int _input_size,
        int _output_size        = 2,
        bool _batch_norm        = true,
        int _hidden_layer_nodes = 10,
        int _n_hidden_layers    = 1);

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
    bool batch_norm;
    torch::nn::BatchNorm1d batch_norm_layer;
};

#endif  // MAAC_AGENT_NETWORK_H
