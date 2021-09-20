import torch
import numpy as np
from multi_agent_sac.misc import soft_update
from copy import deepcopy


## Critic for the multi-agent SAC algorithm
class Critic:
    def __init__(self, n_agents, obs_size, act_size, net, alpha, gamma, tau, actor, logger):
        self.n_agents = n_agents
        self.obs_size = obs_size
        self.act_size = act_size
        self.net      = net
        self.tgt_net  = deepcopy(net)
        self.alpha    = alpha
        self.gamma    = gamma
        self.tau      = tau
        self.actor    = actor
        self.logger   = logger
        self.L        = torch.nn.MSELoss()


    ## Evaluate observations
    def eval(self, obs, act, numpy):
        obs     = torch.tensor(obs)           # shape: (N, n_agents, obs_len)
        act     = torch.tensor(act)           # shape: (N, n_agents, act_len)
        obs_act = torch.cat(obs, act, dim=2)  # shape: (N, n_agents, obs_len+act_len)
        vals    = self.net(obs_act)           # shape: (N, n_agents)

        if numpy:
            return vals.numpy()
        return vals


    ## Update critic net
    #
    #  @param sample Sampled minibatch
    #  @param step Index of training step
    def update(self, sample, step):
        obs, act, rew, next_obs, d = tuple([torch.tensor(arr) for arr in sample])

        # calculate update target
        next_act = self.actor.step(obs, explore=False)
        next_q_vals = self.eval(next_obs, next_act, numpy=False).detach()
        act_prob_logits = self.actor.prob(next_obs, next_act, logits=True)

        update_target = rew + self.gamma * (1 - d.to(torch.float32)) * \
            (next_q_vals - alpha * act_prob_logits)

        guess_q_vals = self.eval(obs, act, numpy=False)

        # step loss
        loss = self.L(guess_q_vals, update_target)

        if self.logger:
            self.logger.add_scalar('loss/critic', loss.item(), step)
