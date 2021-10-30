from mapf_env import Environment
import numpy as np
from typing import Union, List, Tuple
import multiprocessing as mp
from multiprocessing import Process, Pipe
from copy import deepcopy


## Creates multiple environments parallelly
#
#  @param config Environment configuration
class ParallelEnv:
    def __piped_env(pipe: mp.connection.Connection,
            config: dict) -> type(None):

        env = Environment(
            map_path         = config['map_image'],
            map_size         = tuple(config['map_size']),
            number_of_agents = config['n_agents'],
            seed             = config['seed'],
            robot_diam       = config['robot_diam'],
            max_steps        = config['episode_length'])

        while True:
            cmd, data = pipe.recv()
            if cmd == 'step':
                obs, reward, info, done = env.step(data, False)
                pipe.send((obs, reward, info, done))
            elif cmd == 'reset':
                obs = env.reset()
                pipe.send(obs)
            elif cmd == 'close':
                pipe.close()
                break
            elif cmd == 'get_spaces':
                pipe.send((env.get_observation_space(), env.get_action_space()))
            elif cmd == 'goal_reaching_reward':
                env.goal_reaching_reward = data
            elif cmd == 'goal_distance_reward_mult':
                env.goal_distance_reward_mult = data
            elif cmd == 'collision_reward':
                env.collision_reward = data
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
        self.obs_space, self.act_space = self.main_pipes[0].recv()
        self.n_agents = len(self.act_space)

    def get_observation_space(self):
        return self.obs_space
    def get_action_space(self):
        return self.act_space


    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert(actions.shape == (self.n_threads, self.n_agents, self.act_space[0]))
        for pipe, act in zip(self.main_pipes, actions):
            pipe.send(('step', act))

        results = [pipe.recv() for pipe in self.main_pipes]

        obs  = (np.empty((self.n_threads, self.n_agents, self.obs_space[0])),
            np.empty((self.n_threads, self.obs_space[-1])))
        rew  = np.empty((self.n_threads, self.n_agents))
        info = np.empty((self.n_threads, self.n_agents), dtype=dict)
        d    = np.empty((self.n_threads, self.n_agents))

        for i in range(self.n_threads):
            #  obs, rew, info, d = zip(*results)
            obs[0][i] = results[i][0][:-1]
            obs[1][i] = results[i][0][-1]
            rew[i]    = results[i][1]
            info[i]   = results[i][2]
            d[i]      = results[i][3]

        return (obs, rew, info, d)


    def reset(self) -> type(None):
        for pipe in self.main_pipes:
            pipe.send(('reset', None))

        results = [pipe.recv() for pipe in self.main_pipes]
        obs  = (np.empty((self.n_threads, self.n_agents, self.obs_space[0])),
            np.empty((self.n_threads, self.obs_space[-1])))

        for i in range(self.n_threads):
            #  obs, rew, info, d = zip(*results)
            obs[0][i] = results[i][:-1]
            obs[1][i] = results[i][-1]

        return obs


    def goal_reaching_reward(self, value):
        for pipe in self.main_pipes:
            pipe.send(('goal_reaching_reward', value))

    def goal_distance_reward_mult(self, value):
        for pipe in self.main_pipes:
            pipe.send(('goal_distance_reward_mult', value))

    def collision_reward(self, value):
        for pipe in self.main_pipes:
            pipe.send(('collision_reward', value))


    def __del__(self) -> type(None):
        for pipe in self.main_pipes:
            pipe.send(('close', None))

        for p in self.ps:
            p.join()
