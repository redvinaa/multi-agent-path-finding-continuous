import numpy as np
import torch
from torch.distributions.normal import Normal


## Actor for the multi-agent SAC algorithm, with TanH normalized actions
#
class Actor:
    def __init__(self, obs_size, act_size, net, alpha, logger):
        self.obs_size = obs_size
        self.act_size = act_size
        self.net      = net
        self.alpha    = alpha
        self.logger   = logger
        self.L        = torch.nn.MSELoss()

        assert(net.output_size == 2*act_size)


    ## Return action for the given observations, with optional exploration
    def step(self, obs, explore):
        params = self.net(obs)  # shape: (N, 2*act_size,)
        means  = params[:, 0::2]
        stds   = params[:, 1::2]

        if explore:
            dist = Normal(means, stds)
            act = dist.sample()
        else:
            act = means

        act = torch.tanh(act)

        return act


    ## Return the negative log probability of selecting the action in the given state
    def prob(self, obs, act):
        params = self.net(obs)
        means  = params[:, 0::2]
        stds   = params[:, 1::2]

        act_gauss = torch.atanh(act)
        dist = Normal(means, stds)
        logits = -dist.log_prob(act_gauss)

        return logits


    ## Update weights based on the sampled minibatch and the values from the critic
    #
    #  @param sample Tuple of torch.Tensor-s: (obs, act, rew, next_obs, d)
    def update(self, sample, values):
        obs, act, rew, next_obs, d = sample
        act = self.step(obs, explore=True)
        loss = 
