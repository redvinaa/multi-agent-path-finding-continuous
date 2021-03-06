import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, TextIO


## Generic ANN class
class LinearNetwork(nn.Module):
    ## @param hidden_layers List of number of nodes in the hidden layers
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
        assert(len(x.shape) == 2)
        assert(x.shape[1]   == self.input_size)
        return self.layers(x)


## Class for double Q-learning
#
#  All observations and actions are flattened, and passed to the ANN, and the network
#  estimates the value from the perspective of each of the agents
class DoubleQNetwork(nn.Module):
    def __init__(self,
            n_agents: int,
            obs_size: np.ndarray,
            act_size: np.ndarray,
            hidden_layers: List[int],
            activation: nn.Module=nn.LeakyReLU):
        super(DoubleQNetwork, self).__init__()

        self.n_agents      = n_agents
        self.obs_size      = obs_size
        self.act_size      = act_size
        self.hidden_layers = hidden_layers
        self.activation    = activation

        self.input_size  = obs_size + n_agents * act_size
        self.output_size = n_agents

        self.Q1 = LinearNetwork(
            self.input_size, self.output_size, hidden_layers, activation)
        self.Q2 = LinearNetwork(
            self.input_size, self.output_size, hidden_layers, activation)

    ## Returns the estimated value of the state-action for each agent
    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor]:

        assert(len(obs.shape) == 2) # shape=(N, obs_size)
        assert(obs.shape[1]   == self.obs_size)
        N = obs.shape[0] # sample size

        assert(len(act.shape) == 3) # shape=(N, n_agents, act_size)
        assert(act.shape[1]   == self.n_agents)
        assert(act.shape[2]   == self.act_size)
        assert(N == act.shape[0])

        obs = torch.reshape(obs, (N, -1))
        act = torch.reshape(act, (N, -1))
        merged = torch.cat((obs, act), dim=1)

        vals_q1 = self.Q1(merged)
        vals_q2 = self.Q2(merged)

        return vals_q1, vals_q2


## Unit actions policy for the SAC algorithm
class TanhGaussianPolicy(nn.Module):
    LOG_STD_MIN: float=-20.
    LOG_STD_MAX: float=2.
    EPS:         float=1e-6

    def __init__(self,
            n_agents: int,
            obs_size: int,
            act_size: int,
            hidden_layers: List[int],
            activation: nn.Module=nn.LeakyReLU):
        super(TanhGaussianPolicy, self).__init__()

        self.n_agents      = n_agents
        self.obs_size      = obs_size
        self.act_size      = act_size
        self.hidden_layers = hidden_layers
        self.activation    = activation

        # input: state, output: means in each dim, then stds in each dim
        self.input_size  = obs_size
        self.output_size = 2 * act_size
        self.policy = LinearNetwork(
            self.input_size, self.output_size, hidden_layers, activation)

    ## Returns the means and log_stds at the given state, for one agent (Gaussian, no Tanh)
    def forward(self, obs: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor]:

        assert(len(obs.shape) == 2)  # shape=(N, obs_size)
        assert(obs.shape[1]   == self.obs_size)
        N = obs.shape[0] # sample size

        # divide last dimension
        mean, log_std = torch.chunk(self.policy(obs), 2, dim=-1)
        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return mean, log_std

    ## Calculate Tanh of Gaussian distribusion of mean and std at given state
    #
    #  @return (actions (explore), entropies, actions (no explore),)
    def sample(self, obs: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        assert(len(obs.shape) == 3)  # shape=(N, n_agents, obs_size)
        assert(obs.shape[1]   == self.n_agents)
        assert(obs.shape[2]   == self.obs_size)
        N = obs.shape[0] # sample size

        means = torch.empty((N, self.n_agents, self.act_size))
        stds  = torch.empty((N, self.n_agents, self.act_size))

        for i in range(self.n_agents):
            mean, log_std = self.forward(obs[:, i])
            std = log_std.exp()

            means[:, i] = mean
            stds[:, i]  = std

        dist             = torch.distributions.normal.Normal(means, stds)
        act_sampled      = dist.rsample()          # shape=(N, n_agents, act_size)
        act_sampled_tanh = torch.tanh(act_sampled) # shape=(N, n_agents, act_size)

        # entropies
        log_probs = dist.log_prob(act_sampled) \
            - torch.log(1 - act_sampled_tanh.square() + self.EPS)
        entropies = -log_probs.sum(dim=-1, keepdim=False) # sum in actions dim

        return act_sampled_tanh, entropies, torch.tanh(means)
