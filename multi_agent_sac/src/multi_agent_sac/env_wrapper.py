from mapf_env import Environment
import numpy as np
from typing import Union, List, Tuple
import multiprocessing as mp
from multiprocessing import Process, Pipe
from copy import deepcopy


## Maps actions from (-1, 1) to env defined range
#
#  Input actions [[-1, 1], [-1, 1]]
#  are mapped to [[0, 1],  [-pi/2, pi/2]]
class UnitActionsEnv(Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, actions: Union[List[List[float]], np.ndarray], render: bool):
        # max linear speed is 1 m/s
        # max angular speed is pi/2 rad/s
        actions = np.array(actions)
        actions[:, 0] = (actions[:, 0] + 1) / 2
        actions[:, 1] *= np.pi/2

        return super().step(actions, render)


## Creates multiple environments parallelly
#
#  @param config Environment configuration
class ParallelEnv:
    def __piped_env(pipe: mp.connection.Connection,
            config: dict) -> type(None):

        env = UnitActionsEnv(
            config['map_image'],
            config['n_agents'])

        while True:
            cmd, data = pipe.recv()
            if cmd == 'step':
                obs, reward, done = env.step(data, False)
                pipe.send((obs, reward, done))
            elif cmd == 'reset':
                obs = env.reset()
                pipe.send(obs)
            elif cmd == 'close':
                pipe.close()
                break
            elif cmd == 'get_spaces':
                pipe.send((env.get_observation_space(), env.get_action_space()))
            else:
                raise NotImplementedError


    def __init__(self, config: dict, rng: np.random._generator.Generator):
        self.c         = config
        self.n_threads = config['n_threads']
        self.rng       = rng

        self.main_pipes, self.children_pipes = zip(*[Pipe() for _ in range(self.n_threads)])

        self.ps = []
        for child in self.children_pipes:
            config = self.c.copy()
            config.update({'seed': self.rng.integers(10000)})
            p = Process(target=ParallelEnv.__piped_env, args=(child, config))
            p.daemon = True
            self.ps.append(p)

        for p in self.ps:
            p.start()

        self.main_pipes[0].send(('get_spaces', None))
        spaces = self.main_pipes[0].recv()
        self.n_agents = len(spaces[0])
        self.obs_size = spaces[0][0]
        self.act_size = spaces[1][0]


    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert(actions.shape == (self.n_threads, self.n_agents, self.act_size))
        for pipe, act in zip(self.main_pipes, actions):
            pipe.send(('step', act))

        results = [pipe.recv() for pipe in self.main_pipes]
        obs, rew, d = zip(*results)
        return np.stack(obs), np.stack(rew), np.stack(d)


    def reset(self) -> type(None):
        for pipe in self.main_pipes:
            pipe.send(('reset', None))

        obs = [pipe.recv() for pipe in self.main_pipes]
        return np.stack(obs)


    def __del__(self) -> type(None):
        for pipe in self.main_pipes:
            pipe.send(('close', None))

        for p in self.ps:
            p.join()
