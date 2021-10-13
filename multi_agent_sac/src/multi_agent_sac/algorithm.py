import os
import torch
import numpy as np
from multi_agent_sac.network import DoubleQNetwork, TanhGaussianPolicy
from multi_agent_sac.misc import soft_update, hard_update, grad_false, update_params
from copy import deepcopy
from tensorboardX import SummaryWriter
from typing import List, Tuple, Optional


## Class for the multi-agents soft actor-critic algorithm
#
#  Based mainly on these:
#    - https://spinningup.openai.com/en/latest/algorithms/sac.html
#    - https://arxiv.org/abs/1810.02912
#    - https://github.com/ku2482/soft-actor-critic.pytorch.git
#
class MASAC:
    ## @param n_agents Number of agents
    #  @param obs_size Length of observation vector for one agent
    #  @param act_size Length of action vector for one agent
    #  @param gamma Discount factor
    #  @param tau Target networks update rate
    #  @param alpha Entropy coefficient
    #  @param auto_entropy Learn entropy coefficient
    #  @param actor_hidden Number of neurons for each layer of the policy network
    #  @param critic_hidden Number of neurons for each layer of the q network(s)
    #  @param logger Optional tensorboardX logger
    def __init__(self,
            n_agents: int,
            global_obs_size: int,
            obs_size: int,
            act_size: int,
            *,
            gamma: float,
            tau: float,
            actor_hidden: List[int],
            critic_hidden: List[int],
            model_dir: str,
            alpha: float=None,
            auto_entropy: bool=False,
            device: str):

        self.n_agents        = n_agents
        self.obs_size        = obs_size
        self.global_obs_size = global_obs_size
        self.act_size        = act_size
        self.gamma           = gamma
        self.tau             = tau
        self.auto_entropy    = auto_entropy
        self.actor_hidden    = actor_hidden
        self.critic_hidden   = critic_hidden
        self.model_dir       = model_dir
        self.device          = device

        self.policy        = TanhGaussianPolicy(n_agents, obs_size, \
            act_size, actor_hidden).to(self.device)

        self.critic        = DoubleQNetwork(n_agents, global_obs_size, \
            act_size, critic_hidden).to(self.device)
        self.tgt_critic    = DoubleQNetwork(n_agents, global_obs_size, \
            act_size, critic_hidden).to(self.device).eval()
        hard_update(self.tgt_critic, self.critic)
        grad_false(self.tgt_critic)

        self.policy_optim   = torch.optim.Adam(self.policy.parameters())
        self.critic1_optim = torch.optim.Adam(self.critic.Q1.parameters())
        self.critic2_optim = torch.optim.Adam(self.critic.Q2.parameters())

        if self.auto_entropy:
            self.target_entropy = \
                -torch.tensor([self.n_agents * self.act_size], device=self.device).item()
            self.log_alpha      = \
                torch.zeros((1,), requires_grad=True, device=self.device)
            self.alpha          = self.log_alpha.exp()
            self.entropy_optim  = torch.optim.Adam([self.log_alpha])
        else:
            self.alpha         = torch.Tensor(alpha, device=self.device)


    ## Calculate loss for policy update
    #
    #  @param sample tuple of torch.Tensor-s: (obs, act, rew, next_obs, d)
    #  @return Loss to perform SGD on
    def __get_policy_loss(self, sample: Tuple[torch.Tensor, torch.Tensor, \
            torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:

        obs, act, rew, next_obs, d = sample
        N = obs.shape[0] # sample size

        sampled_act, entropy, _ = self.policy.sample(obs[:, :-1])

        curr_q1, curr_q2 = self.critic(obs[:, -1], sampled_act)
        curr_q = torch.min(curr_q1, curr_q2)

        # we want to maximize this
        policy_loss = -torch.mean(curr_q + self.alpha * entropy)
        return policy_loss, entropy


    ## Calculate loss for critic update
    #
    #  @param sample tuple of torch.Tensor-s: (obs, act, rew, next_obs, d)
    #  @return Loss to perform SGD on, and entropies of the sampled actions
    def __get_critic_loss(self, sample: Tuple[torch.Tensor, torch.Tensor, \
            torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:

        obs, act, rew, next_obs, d = sample
        N = obs.shape[0] # sample size

        # get current eval
        curr_q1, curr_q2 = self.critic(obs[:, -1], act)

        # calculate update target
        with torch.no_grad():
            next_act, next_entropy, _ = self.policy.sample(next_obs[:, :-1])
            next_q1, next_q2 = self.tgt_critic(next_obs[:, -1], next_act)
            next_q = torch.min(next_q1, next_q2)

        target_q = rew + self.gamma * (1. - d) * next_q

        L = torch.nn.MSELoss()
        return L(curr_q1, target_q), L(curr_q2, target_q)


    ## Calculate loss for entropy coeffifient
    #
    #  @param entropy Entropies of the sampled actions from the last training session
    def __get_entropy_loss(self, entropy):
        return -torch.mean(self.log_alpha * (self.target_entropy - entropy).detach())


    ## Update critic and policy based on the sampled batch
    #
    #  @param sample tuple of np.ndarrays-s: (obs, act, rew, next_obs, d)
    def update(self, sample: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, \
            np.ndarray]) -> type(None):

        # convert to tensors
        sample = tuple([torch.from_numpy(i.astype(np.float32)).to(self.device) \
            for i in sample])
        obs, act, rew, next_obs, d = sample

        # get losses and update
        q1_loss, q2_loss = self.__get_critic_loss(sample)
        update_params(self.critic1_optim, q1_loss)
        update_params(self.critic2_optim, q2_loss)

        policy_loss, entropy = self.__get_policy_loss(sample)
        update_params(self.policy_optim, policy_loss)

        # update target networks
        soft_update(self.tgt_critic, self.critic, self.tau)

        # learn entropy
        if self.auto_entropy:
            entropy_loss = self.__get_entropy_loss(entropy)
            update_params(self.entropy_optim, entropy_loss)
            self.alpha = self.log_alpha.exp()

            return q1_loss.item(), q2_loss.item(), \
                policy_loss.item(), entropy_loss.item()

        else:
            return q1_loss.item(), q2_loss.item(), policy_loss.item()


    ## Get actions for the given observations (no target parameters)
    #
    #  @param obs Observations, shape=(n_agents, obs_size)
    #  @param explore Sample normal distribution (if True), or return mean
    #  @return Array of actions for the given observations, shape=(n_agemts, act_size)
    def step(self, obs: np.ndarray, *, explore: bool) -> np.ndarray:

        assert(obs.shape == (self.n_agents, self.obs_size))

        obs = torch.Tensor(obs).to(self.device).unsqueeze(0)
        act_explore, entropy, act_det = self.policy.sample(obs)
        if explore:
            act = act_explore
        else:
            act = act_det

        act_np = act.squeeze(0).detach().numpy()
        act_np = np.clip(act_np, -1, 1)
        return act_np


    ## Save agent and critic weights
    def save(self) -> type(None):
        policy_model_file = os.path.join(self.model_dir, 'agent.p')
        critic_model_file = os.path.join(self.model_dir, 'critic.p')

        torch.save(self.policy.state_dict(), policy_model_file)
        torch.save(self.critic.state_dict(), critic_model_file)


    ## Load agent and critic weights
    def load(self) -> type(None):
        policy_model_file  = os.path.join(self.model_dir, 'agent.p')
        critic_model_file  = os.path.join(self.model_dir, 'critic.p')

        self.critic.load_state_dict(torch.load(critic_model_file))
        self.policy.load_state_dict(torch.load(policy_model_file))
