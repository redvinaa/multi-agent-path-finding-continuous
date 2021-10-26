import numpy as np
import torch
from collections import deque
from typing import List, Tuple, Optional


## Replay buffer to store transitions for multi-agent RL
class ReplayBuffer:
    def __init__(self, length: int, n_agents: int, obs_space: Tuple[int], act_space: Tuple[int]):
        self.max_length  = length
        self.n_agents    = n_agents
        self.obs_space   = obs_space
        self.act_space   = act_space

        self.buf = deque(maxlen=self.max_length)


    ## Push transition into buffer
    def push(self,
            obs: Tuple[np.ndarray, np.ndarray],
            act: np.ndarray,
            rew: np.ndarray,
            next_obs: Tuple[np.ndarray, np.ndarray],
            d: np.ndarray) -> type(None):

        trans = (obs, act, rew, next_obs, d)

        # check shapes
        assert(obs[0].shape      == (self.n_agents, self.obs_space[0],))
        assert(obs[1].shape      == (self.obs_space[-1],))
        assert(act.shape         == (self.n_agents, self.act_space[0],))
        assert(rew.shape         == (self.n_agents,))
        assert(next_obs[0].shape == (self.n_agents, self.obs_space[0],))
        assert(next_obs[1].shape == (self.obs_space[-1],))
        assert(d.shape           == (self.n_agents,))

        # newly added element is first (0 index)
        self.buf.appendleft(trans)


    ## Return number of elements
    def __len__(self) -> int:
        return len(self.buf)


    ## Return the sum of rewards from the last n steps
    #
    # @param n Length of the last episode to sum rewards for
    # @return Array of rewards per agent
    def get_rewards(self, n: int=None) -> np.ndarray:
        if type(n) != type(None):
            assert(n > 0)
            assert(n <= len(self.buf))

        vals = np.zeros((self.n_agents,))

        if type(n) != type(None):
            for i in range(n):
                rew = self.buf[i][2]
                vals += rew

        else: # search for last episode length based on dones
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
    def sample(self, n: int) -> List[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, \
            np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray]]:
        indices = np.random.choice(len(self.buf), n, replace=False)
        return [self.buf[idx] for idx in indices]
