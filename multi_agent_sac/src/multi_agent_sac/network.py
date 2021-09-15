import torch
import torch.nn as nn
import numpy as np


## Generic ANN class
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size=10, n_hidden_layers=1, activation=nn.LeakyReLU):
        super(Network, self).__init__()

        self.input_size        = input_size
        self.output_size       = output_size
        self.hidden_layer_size = hidden_layer_size
        self.n_hidden_layers   = n_hidden_layers
        self.activation        = activation

        self.layers = nn.Sequential()

        if n_hidden_layers == 0:
            self.layers.add_module('single_layer', nn.Linear(input_size, output_size))
        else:
            for i in range(n_hidden_layers+1):
                if i == 0:
                    self.layers.add_module('input_layer', nn.Linear(input_size, hidden_layer_size))
                    self.layers.add_module('input_layer_activation', self.activation())
                elif i < n_hidden_layers:
                    self.layers.add_module(f'hidden_layer_{i}', nn.Linear(hidden_layer_size, hidden_layer_size))
                    self.layers.add_module(f'input_layer_{i}_activation', self.activation())
                else:
                    self.layers.add_module('output_layer', nn.Linear(hidden_layer_size, output_size))

    def forward(self, x):
        return self.layers(x)
