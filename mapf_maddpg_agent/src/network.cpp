// Copyright 2021 Reda Vince

#include <mapf_maddpg_agent/network.h>
#include <string>

Net::Net(int _input_size, int _output_size /* 2 */, int _hidden_layer_nodes /* 10 */, int _n_hidden_layers /* 1 */):
    input_size(_input_size),
    output_size(_output_size),
    hidden_layer_nodes(_hidden_layer_nodes),
    n_hidden_layers(_n_hidden_layers)
{
    assert(input_size > 0);
    assert(output_size > 0);
    assert(hidden_layer_nodes > 0);
    assert(n_hidden_layers > 0);

    for (int i=0; i < n_hidden_layers; i++)
    {
        torch::nn::Linear layer = nullptr;
        if (n_hidden_layers > 1)
        {
            if (i == 0)  // first layer
                layer = torch::nn::Linear(input_size, hidden_layer_nodes);
            else if (i == n_hidden_layers-1)  // last layer
                layer = torch::nn::Linear(hidden_layer_nodes, output_size);
            else  // middle layer
                layer = torch::nn::Linear(hidden_layer_nodes, hidden_layer_nodes);
        }
        else
            layer = torch::nn::Linear(input_size, output_size);
        layers.push_back(layer);

        std::string name;
        if (n_hidden_layers == 1)
            name = "Input_output_layer";
        else if (i == 0)
            name = "Input_layer";
        else if (i == n_hidden_layers-1)
            name = "Output_layer";
        else
            name = "Hidden_layer_" + std::to_string(i);
        register_module(name, layer);
    }
}

torch::Tensor Net::forward(torch::Tensor x)
{
    for (auto& layer : layers)
        x = layer->forward(x);

    return x;
}
