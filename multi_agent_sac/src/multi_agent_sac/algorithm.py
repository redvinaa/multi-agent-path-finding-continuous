import os
import torch
import numpy as np
from multi_agent_sac.network import DoubleQNetwork, TanhGaussianPolicy
from multi_agent_sac.misc import soft_update, grad_false
from copy import deepcopy
from tensorboardX import SummaryWriter
from typing import List, Tuple, Optional


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
    #  @param actor_hidden Number of neurons for each layer of the actor network
    #  @param critic_hidden Number of neurons for each layer of the critic network
    #  @param logger Optional tensorboardX logger
    def __init__(self,
            n_agents: int,
            obs_size: int,
            act_size: int,
            gamma: float,
            tau: float,
            alpha: float,
            actor_hidden: List[int],
            critic_hidden: List[int],
            model_dir: str,
            logger: Optional[SummaryWriter]=None):

        self.n_agents      = n_agents
        self.obs_size      = obs_size
        self.act_size      = act_size
        self.gamma         = gamma
        self.tau           = tau
        self.alpha         = alpha
        self.actor_hidden  = actor_hidden
        self.critic_hidden = critic_hidden
        self.model_dir     = model_dir
        self.logger        = logger

        self.policy     = TanhGaussianPolicy(n_agents, obs_size, act_size, actor_hidden)
        self.tgt_policy = deepcopy(self.policy)
        grad_false(self.tgt_policy)
        self.critic     = DoubleQNetwork(n_agents, obs_size, act_size, critic_hidden)
        self.tgt_critic = deepcopy(self.critic)
        grad_false(self.tgt_critic)

        self.L = torch.nn.MSELoss()
        self.actor_optim  = torch.optim.Adam(self.policy.parameters())
        self.critic1_optim = torch.optim.Adam(self.critic.Q1.parameters())
        self.critic2_optim = torch.optim.Adam(self.critic.Q2.parameters())


    ## Calculate loss for actor update
    #
    #  @param sample tuple of torch.Tensor-s: (obs, act, rew, next_obs, d)
    #  @return Loss to perform SGD on
    def __get_actor_loss(self, sample: Tuple[torch.Tensor, torch.Tensor, \
            torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:

        obs, act, rew, next_obs, d = sample
        N = obs.shape[0]

        sampled_act, entropy, _ = self.policy.sample(obs)
        q1_val, q2_val = self.critic(obs, sampled_act)
        q = torch.min(q1_val, q2_val)

        actor_loss = torch.mean(-q - self.alpha * entropy)
        return actor_loss


    ## Calculate loss for critic update
    #
    #  @param sample tuple of torch.Tensor-s: (obs, act, rew, next_obs, d)
    #  @return Loss to perform SGD on
    def __get_critic_loss(self, sample: Tuple[torch.Tensor, torch.Tensor, \
            torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:

        obs, act, rew, next_obs, d = sample
        N = obs.shape[0]

        # calculate update target
        next_act, next_entropy, _ = self.policy.sample(next_obs)
        next_q1, next_q2 = self.tgt_critic(next_obs, next_act)
        next_q = torch.min(next_q1, next_q2) + self.alpha * next_entropy
        next_q = next_q.detach()
        target_q = rew + self.gamma * (1 - d) * next_q

        curr_q1, curr_q2 = self.critic(obs, act)

        return self.L(curr_q1, target_q), self.L(curr_q2, target_q)


    ## Update critic and actor based on the sampled batch
    #
    #  @param sample tuple of np.ndarrays-s: (obs, act, rew, next_obs, d)
    #  @param step Index of training step (for logger)
    def update(self, sample: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, \
            np.ndarray], step: Optional[int]=None) -> type(None):

        sample = tuple([torch.from_numpy(i.astype(np.float32)) for i in sample])
        obs, act, rew, next_obs, d = sample

        q1_loss, q2_loss = self.__get_critic_loss(sample)
        actor_loss  = self.__get_actor_loss(sample)

        if self.logger and step:
            self.logger.add_scalar('loss/critic1', q1_loss.item(), step)
            self.logger.add_scalar('loss/critic2', q2_loss.item(), step)
            self.logger.add_scalar('loss/actor',  actor_loss.item(), step)

        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        self.actor_optim.zero_grad()

        q1_loss.backward()
        q2_loss.backward()
        actor_loss.backward()

        self.critic1_optim.step()
        self.critic2_optim.step()
        self.actor_optim.step()

        # update target networks
        soft_update(self.tgt_policy, self.policy, self.tau)
        soft_update(self.tgt_critic, self.critic, self.tau)


    ## Get actions for the given observations (no target parameters)
    #
    #  @param obs Observations, shape=(n_agents, obs_size)
    #  @param explore Sample normal distribution (if True), or return mean
    #  @return Actions for the given observations, shape=(n_agemts, act_size)
    def step(self, obs: np.ndarray, *, explore: bool) -> np.ndarray:

        assert(obs.shape == (self.n_agents, self.obs_size))

        obs = torch.Tensor(obs).unsqueeze(0)
        act, _, _ = self.policy.sample(obs)
        return act.squeeze(0).detach().numpy()


    ## Save agent and critic weights
    def save(self) -> type(None):
        actor_model_file  = os.path.join(self.model_dir, 'agent.p')
        critic_model_file = os.path.join(self.model_dir, 'critic.p')

        torch.save(self.policy.state_dict(),  actor_model_file)
        torch.save(self.critic.state_dict(), critic_model_file)


    ## Load agent and critic weights
    def load(self) -> type(None):
        actor_model_file = os.path.join(self.model_dir, 'agent.p')
        critic_model_file = os.path.join(self.model_dir, 'critic.p')

        self.critic_net.load_state_dict(torch.load(critic_model_file))
        self.actor_net.load_state_dict(torch.load(actor_model_file))
