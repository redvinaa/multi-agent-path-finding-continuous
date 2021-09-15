import numpy as np
import torch
from collections import deque


## Replay buffer to store transitions for multi-agent RL
class ReplayBuffer:
    def __init__(self, length, n_agents, obs_size, act_size):
        self.max_length = length
        self.n_agents   = n_agents
        self.obs_size   = obs_size
        self.act_size   = act_size

        self.buf = deque(maxlen=self.max_length)

    ## Push transition into buffer
    def push(self, obs, act, rew, next_obs, d):
        assert(obs.shape      == (self.n_agents, self.obs_size,))
        assert(act.shape      == (self.n_agents, self.act_size,))
        assert(rew.shape      == (self.n_agents,))
        assert(next_obs.shape == (self.n_agents, self.obs_size,))
        assert(d.shape        == (self.n_agents,))

        self.buf.appendleft((obs, act, rew, next_obs, d,))

    ## Return number of elements
    def __len__(self):
        return len(self.buf)

    ## Return the sum of rewards from the last n steps
    # @param n Length of the last episode to sum rewards for
    # @return List of rewards per agent
    def get_rewards(self, n):
        assert(n <= len(self.buf))

        vals = np.zeros((self.n_agents,))
        for i in range(n):
            rew = self.buf[i][2]
            vals += rew

        return vals

    ## Sample n elements from buffer
    #  @param n Sample size
    #  @return Tuple of ndarrays: (obs, act, rew, next_obs, done)
    def sample(self, n):
        indices = np.random.choice(len(self.buf), n, replace=False)
        obs, act, rew, next_obs, d = zip(*[self.buf[idx] for idx in indices])

        arr = np.array
        return (arr(obs), arr(act), arr(rew, dtype=np.float32), arr(next_obs), arr(d, dtype=np.bool),)
