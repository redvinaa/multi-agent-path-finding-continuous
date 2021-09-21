import numpy as np
import torch
from collections import deque
from typing import List, Tuple, Optional


## Replay buffer to store transitions for multi-agent RL
class ReplayBuffer:
    def __init__(self, length: int, n_agents: int, obs_size: int, act_size: int):
        self.max_length = length
        self.n_agents   = n_agents
        self.obs_size   = obs_size
        self.act_size   = act_size

        self.buf = deque(maxlen=self.max_length)

    ## Push transition into buffer
    def push(self,
            obs: np.ndarray,
            act: np.ndarray,
            rew: np.ndarray,
            next_obs: np.ndarray,
            d: np.ndarray) -> type(None):
        trans = (obs, act, rew, next_obs, d)
        trans = tuple([np.array(el) for el in trans])
        obs, act, rew, next_obs, d = trans

        assert(obs.shape      == (self.n_agents, self.obs_size,))
        assert(act.shape      == (self.n_agents, self.act_size,))
        assert(rew.shape      == (self.n_agents,))
        assert(next_obs.shape == (self.n_agents, self.obs_size,))
        assert(d.shape        == (self.n_agents,))

        self.buf.appendleft((obs, act, rew, next_obs, d,))

    ## Return number of elements
    def __len__(self) -> int:
        return len(self.buf)

    ## Return the sum of rewards from the last n steps
    #
    # @param n Length of the last episode to sum rewards for
    # @return List of rewards per agent
    def get_rewards(self, n: int=None) -> np.ndarray:
        if type(n) != type(None):
            assert(n > 0)
            assert(n <= len(self.buf))

        vals = np.zeros((self.n_agents,))
        if not type(n) == type(None):
            for i in range(n):
                rew = self.buf[i][2]
                vals += rew
        else:
            # search for last episode length based on buf['d']
            for n in range(self.n_agents):
                for i in range(len(self.buf)):
                    if self.buf[i][4][n] and not i == 0:
                        break

                    vals[n] += self.buf[i][2][n]

        return vals

    ## Sample n elements from buffer
    #
    #  @param n Sample size
    #  @return Tuple of ndarrays: (obs, act, rew, next_obs, done)
    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.choice(len(self.buf), n, replace=False)
        obs, act, rew, next_obs, d = zip(*[self.buf[idx] for idx in indices])

        a = np.array
        return a(obs), a(act), a(rew, dtype=np.float32), a(next_obs), a(d, dtype=np.bool)
