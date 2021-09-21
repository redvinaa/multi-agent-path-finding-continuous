import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


## Generic ANN class
class LinearNetwork(nn.Module):
    def __init__(self,
            input_size: int,
            output_size: int,
            hidden_layers: List[int],
            activation: nn.Module=nn.LeakyReLU):
        super(LinearNetwork, self).__init__()

        self.input_size      = input_size
        self.output_size     = output_size
        self.hidden_layers   = hidden_layers
        self.activation      = activation
        self.n_layers        = len(hidden_layers)

        self.layers = nn.Sequential()

        if self.n_layers == 0:
            self.layers.add_module('single_layer', nn.Linear(input_size, output_size))
        else:
            for i in range(self.n_layers+1):
                if i == 0:
                    self.layers.add_module('input_layer',
                        nn.Linear(input_size, hidden_layers[0]))
                    self.layers.add_module('input_layer_activation',
                        self.activation())
                elif i < self.n_layers:
                    self.layers.add_module(f'hidden_layer_{i}',
                        nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                    self.layers.add_module(f'input_layer_{i}_activation',
                        self.activation())
                else:
                    self.layers.add_module('output_layer',
                        nn.Linear(hidden_layers[i-1], output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


## Class for double Q-learning
#
#  https://github.com/ku2482/soft-actor-critic.pytorch.git
class DoubleQNetwork(nn.Module):
    def __init__(self,
            input_size: int,
            output_size: int,
            hidden_layers: List[int],
            activation: nn.Module=nn.LeakyReLU):
        super(DoubleQNetwork, self).__init__()

        self.Q1 = LinearNetwork(
            input_size, output_size, hidden_layers, activation)
        self.Q2 = LinearNetwork(
            input_size, output_size, hidden_layers, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Q1(x), self.Q2(x)


## Policy for use in the UnitActionsEnv
#
#  https://github.com/ku2482/soft-actor-critic.pytorch.git
#  https://stackoverflow.com/questions/54569726/how-does-rl-continuous-control-via-gaussian-policy-work
class TanhGaussianPolicy(nn.Module):
    LOG_STD_MIN: float=-20.
    LOG_STD_MAX: float=2.
    EPS: float=1e-6

    def __init__(self,
            input_size: int,
            output_size: int,
            hidden_layers: List[int],
            activation: nn.Module=nn.LeakyReLU):
        super(TanhGaussianPolicy, self).__init__()

        self.policy = LinearNetwork(
            input_size, output_size * 2, hidden_layers, activation)

    ## Returns the means and log_stds at the given state
    def forward(self, x: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor]:

        mean, log_std = torch.chunk(self.policy(x), 2, dim=-1)
        log_std = torch.clamp(
            log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return mean, log_std

    ## Calculate Tanh of Gaussian distribusion of mean and std
    #
    #  @return (actions (explore), entropies, actions (no explore),)
    def sample(self, obs: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        mean, log_std = self.forward(obs)
        std = log_std.exp()

        dist = torch.distributions.normal.Normal(mean, std)
        act_sampled = dist.rsample()
        act_sampled = torch.tanh(act_sampled)

        # entropies
        log_probs = dist.log_prob(act_sampled) \
            - torch.log(1 - act_sampled.square() + self.EPS)
        print(log_probs)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return act_sampled, entropies, torch.tanh(mean)
