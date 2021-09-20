import os
import torch
import numpy as np
from multi_agent_sac.misc import soft_update
from copy import deepcopy
from tensorboardX import SummaryWriter
from typing import Optional


## Class for the multi-agents soft actor-critic algorithm
#
#  Based mainly on these:
#    - https://spinningup.openai.com/en/latest/algorithms/sac.html
#    - https://arxiv.org/abs/1810.02912
#
#  Actor net:
#    Takes an observation as input, and outputs [mean, std]*act_size (flattened)
#    for the Normal distribution, that then has to be fed into
#    a TanH for normalization
#
#  Critic net:
#    Takes [obs, act]*n_agents (flattened),
#    and outputs the approx. value of the given action-states for every agent
class MASAC:
    ## @param n_agents Number of agents
    #  @param obs_size Length of observation vector for one agent
    #  @param act_size Length of action vector for one agent
    #  @param gamma Discount factor
    #  @param tau Target networks update rate
    #  @param alpha Entropy coefficient
    #  @param actor_hidden_dim Number of neurons for single layer actor network
    #  @param critic_hidden_dim Number of neurons for single layer critic network
    #  @param logger Optional tensorboardX logger
    def __init__(self,
            n_agents: int,
            obs_size: int,
            act_size: int,
            gamma: float,
            tau: float,
            alpha: float,
            actor_hidden_dim: int,
            critic_hidden_dim: int,
            logger: Optional[SummaryWriter]=None):

        self.n_agents          = n_agents
        self.obs_size          = obs_size
        self.act_size          = act_size
        self.gamma             = gamma
        self.tau               = tau
        self.alpha             = alpha
        self.actor_hidden_dim  = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.logger            = logger

        #  actor net output size needs to be 2*act_size (mean and std in every dim)
        self.actor_net      = Network(obs_size, 2*act_size, actor_hidden_dim)
        self.actor_tgt_net  = deepcopy(self.actor_net)
        self.critic_net     = Network(n_agents * (obs_size + act_size),
            n_agents, critic_hidden_dim)
        self.critic_tgt_net = deepcopy(self.critic_net)

        self.L = torch.nn.MSELoss()
        self.actor_optim  = torch.optim.Adam(self.actor_net.parameters())
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters())


    ## Get actions for the given observations, with optional exploration
    #
    #  @param obs Observations, shape=(N, obs_size)
    #  @param explore Sample normal distribution (if True), or return mean
    #  @param use_tgt Whether to use target network
    #  @return Actions for the given observations, shape=(N, act_size)
    def __get_action(self,
            obs: torch.Tensor,
            *,
            explore: bool,
            use_tgt: bool) -> torch.Tensor:

        assert(len(obs.shape) == 2)

        if use_tgt:
            params = self.actor_tgt_net(obs)  # shape: (N, 2*act_size,)
        else:
            params = self.actor_net(obs)      # shape: (N, 2*act_size,)
        means  = params[:, 0::2]
        stds   = params[:, 1::2]

        if explore:
            dist = torch.distributions.normal.Normal(means, stds)
            act = dist.sample()
        else:
            act = means

        return torch.tanh(act)


    ## Get the negative log probabilities of selecting the actions in the given states
    #
    #  @param obs Observations, shape=(N, obs_size)
    #  @param act Actions for which to get the probabilities, shape=(N, act_size)
    #  @param use_tgt Whether to use target network
    #  @return Negative log probabilities for the selected actions
    def __get_log_prob(self,
            obs: torch.Tensor,
            act: torch.Tensor,
            *,
            use_tgt: bool) -> torch.Tensor:

        if use_tgt:
            params = self.actor_tgt_net(obs)
        else:
            params = self.actor_net(obs)
        means  = params[:, 0::2]
        stds   = params[:, 1::2]

        act_gauss = torch.atanh(act)
        dist      = torch.distributions.normal.Normal(means, stds)
        return -dist.log_prob(act_gauss)


    ## Calculate loss for actor update
    #
    #  @param sample tuple of torch.Tensor-s: (obs, act, rew, next_obs, d)
    #  @return Loss to perform SGD on
    def __get_actor_loss(self, sample: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.nn.MSELoss:

        obs, act, rew, next_obs, d = sample
        N = obs.shape[0]

        # sample actions using current policy, shape=(N, n_agents, act_size)
        sampled_act = self.__get_action(obs, explore=True, use_tgt=False).detach()

        q_vals = torch.empty((N, self.n_agents))
        for i in range(self.n_agents):
            act_cpy = act.copy()
            act_cpy[:, i, :] = sampled_act[:, i, :]

            val_i = self.__get_value(obs, act_cpy, use_tgt=False)[:, i]
            q_vals[:, i] = val_i

        logits = self.get_log_prob(obs, sampled_act, use_tgt=False)

        return self.L(q_vals, self.alpha*logits)


    ## Evaluate action-values, for all agents at once
    #
    #  @param obs Observations, shape=(N, n_agents, obs_size)
    #  @param act Actions for action-values, shape=(N, n_agents, act_size)
    #  @param use_tgt Whether to use target network
    #  @return Approximated values, shape=(N, n_agents)
    def __get_value(self,
            obs: torch.Tensor,
            act: torch.Tensor,
            *,
            use_tgt: bool) -> torch.Tensor:

        obs_act = torch.cat((obs, act), dim=2)  # shape: (N, n_agents, obs_len+act_len)

        if use_tgt:
            return self.critic_tgt_net(obs_act) # shape: (N, n_agents)
        return self.critic_net(obs_act)


    ## Calculate loss for critic update
    #
    #  @param sample tuple of torch.Tensor-s: (obs, act, rew, next_obs, d)
    #  @return Loss to perform SGD on
    def __get_critic_loss(self, sample: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.nn.MSELoss:

        obs, act, rew, next_obs, d = sample
        N = obs.shape[0]

        # calculate update target
        next_act        = self.__get_action(next_obs, explore=True, use_tgt=False)
        next_q_vals     = self.__get_value(next_obs, next_act, use_tgt=True).detach()
        act_prob_logits = self.__get_log_prob(next_obs, next_act, use_tgt=True)

        update_target = rew + self.gamma * (1 - d.int().float()) * \
            (next_q_vals - alpha * act_prob_logits)

        guess_q_vals = self.__get_value(obs, act, use_tgt=False)

        return self.L(guess_q_vals, update_target)


    ## Perform soft update on the networks, with parameter tau
    def __update_tgt_networks(self) -> type(None):
        soft_update(self.actor_tgt_net, self.actor_net, self.tau)
        soft_update(self.critic_tgt_net, self.critic_net, self.tau)


    ## Update critic and actor based on the sampled batch
    #
    #  @param sample tuple of np.ndarrays-s: (obs, act, rew, next_obs, d)
    #  @param step Index of training step (for logger)
    def update(self, sample: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            step: Optional[int]=None) -> type(None):

        obs, act, rew, next_obs, d = tuple([torch.Tensor(i) for i in sample])

        self.critic_optim.zero_grad()
        self.actor_optim.zero_grad()

        critic_loss = self.get_critic_loss(sample)
        actor_loss  = self.get_actor_loss(sample)

        if self.logger and step:
            self.logger.add_scalar('loss/critic', critic_loss.item(), step)
            self.logger.add_scalar('loss/actor',  actor_loss.item(), step)

        critic_loss.backward()
        actor_loss.backward()

        self.critic_optim.step()
        self.actor_optim.step()

        self.update_tgt_networks()


    ## Get actions for the given observations (no target parameters)
    #
    #  @param obs Observations, shape=(n_agents, obs_size)
    #  @param explore Sample normal distribution (if True), or return mean
    #  @return Actions for the given observations, shape=(n_agemts, act_size)
    def step(obs: np.ndarray, *, explore: bool) -> np.ndarray:

        assert(obs.shape == (self.n_agents, self.obs_size))

        obs = torch.Tensor(obs).unsqueeze(0)
        act = self.__get_action(obs, explore=True, use_tgt=False).squeeze(0)
        return act.numpy()


    ## Save agent and critic weights
    #
    #  @param directory Dir in which to save (two files, agent.p and critic.p)
    def save(self, directory: str) -> type(None):
        actor_model_file  = os.path.join(directory, 'agent.p')
        critic_model_file = os.path.join(directory, 'critic.p')

        torch.save(self.actor_net.state_dict(),  actor_model_file)
        torch.save(self.critic_net.state_dict(), critic_model_file)


    ## Load agent and critic weights
    #
    #  @param directory Dir from which to load (two files, agent.p and critic.p)
    def load(self, directory: str) -> type(None):
        actor_model_file = os.path.join(directory, 'agent.p')
        critic_model_file = os.path.join(directory, 'critic.p')

        self.critic_net.load_state_dict(torch.load(critic_model_file))
        self.actor_net.load_state_dict(torch.load(actor_model_file))
