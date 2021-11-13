#! /usr/bin/env python3.6

import argparse
import torch
import os
import numpy as np
from tensorboardX import SummaryWriter
from algorithm import MASAC
from buffer import ReplayBuffer
from env_wrapper import ParallelEnv
from mapf_env import Environment
from misc import from_unit_actions, to_unit_actions, LinearDecay
from rospkg import RosPack
import warnings
import json
import yaml
from typing import Tuple
import sys
from tqdm import tqdm
import shutil
import time


class TrainProcess:
    def __init__(self, env: ParallelEnv, model: MASAC, buffer: ReplayBuffer, \
            logger: SummaryWriter, log_dir: str, config: dict, \
            rng: np.random._generator.Generator) -> type(None):

        self.env     = env
        self.model   = model
        self.buffer  = buffer
        self.logger  = logger
        self.log_dir = log_dir
        self.c       = config
        self.rng     = rng

        index = int(self.log_dir.split('/')[-2][-1])
        self.coll_reward = LinearDecay(
            self.c['collision_reward_start'],
            self.c['collision_reward_end'],
            self.c['collision_reward_decay_ep_start'],
            self.c['collision_reward_decay_ep_end'])

        self.env.goal_reaching_reward(self.c['goal_reaching_reward'])
        self.env.goal_distance_reward_mult(self.c['goal_distance_reward_mult'])
        self.env.collision_reward(self.coll_reward.val)

        for self.ep in tqdm(range(self.c['n_episodes'])):
            t0 = time.time()
            self.__training_episode()
            t1 = time.time()

            if (self.ep % self.c['save_interval']) == 0 and self.ep != 0:
                self.model.save()

            if (self.ep % self.c['eval_interval']) == 0:
                self.__evaluation_episode()
                self.logger.add_scalars('evaluation/training_time', {'training_ep_time': t1 - t0}, self.ep)

            self.logger.add_scalars('evaluation/collision_reward', {'collision_reward': self.coll_reward.val}, self.ep)
            self.coll_reward.step()
            self.env.collision_reward(self.coll_reward.val)

        self.model.save()
        self.logger.export_scalars_to_json(os.path.join(self.log_dir, 'summary.json'))
        self.logger.close()


    def __evaluation_episode(self) -> type(None):
        log_actions      = np.empty((self.c['episode_length'], self.c['n_threads'], self.c['n_agents'], self.env.act_space[0]))
        log_rewards      = np.empty((self.c['episode_length'], self.c['n_threads'], self.c['n_agents']))
        log_reached_goal = np.empty((self.c['episode_length'], self.c['n_threads'], self.c['n_agents']))
        log_collision    = np.empty((self.c['episode_length'], self.c['n_threads'], self.c['n_agents']))

        obs_v = self.env.reset()

        for step in range(self.c['episode_length']):
            act_v = np.stack([self.model.step(obs_v[0][i], explore=False) for i in range(self.c['n_threads'])])

            next_obs_v, rewards_v, infos_v, dones_v = self.env.step(
                from_unit_actions(act_v, self.c['min_linear_speed'], self.c['max_linear_speed'], self.c['max_angular_speed']))

            obs_v = next_obs_v

            log_actions[step]      = act_v
            log_rewards[step]      = rewards_v
            log_reached_goal[step] = np.array([[i['reached_goal'] for i in info] for info in infos_v])
            log_collision[step]    = np.array([[i['collision']    for i in info] for info in infos_v])

        log_rewards      = np.sum(log_rewards,          axis=0)
        log_rewards      = np.average(log_rewards,      axis=0)
        log_rewards      = np.average(log_rewards,      axis=0)
        log_reached_goal = np.sum(log_reached_goal,     axis=0)
        log_reached_goal = np.average(log_reached_goal, axis=0)
        log_reached_goal = np.average(log_reached_goal, axis=0)
        log_collision    = np.sum(log_collision,        axis=0)
        log_collision    = np.average(log_collision,    axis=0)
        log_collision    = np.average(log_collision,    axis=0)
        log_actions_lin  = log_actions[:, :, :, 0].flatten()
        log_actions_ang  = log_actions[:, :, :, 1].flatten()

        self.logger.add_scalars('evaluation/reached_goal_average',   {'reached_goal_average':   log_reached_goal}, self.ep)
        self.logger.add_scalars('evaluation/collision_average',      {'collision_average':      log_collision},    self.ep)
        self.logger.add_scalars('evaluation/episode_reward_average', {'episode_reward_average': log_rewards},      self.ep)
        self.logger.add_histogram('evaluation/linear_actions',           log_actions_lin,   self.ep)
        self.logger.add_histogram('evaluation/angular_actions',          log_actions_ang,   self.ep)


    def __training_episode(self) -> type(None):
        obs_v = self.env.reset()

        n_updates = 0
        losses = {'critic1': 0., 'critic2': 0., 'entropy': 0., 'policy': 0.}
        for step in range(self.c['episode_length']):
            if len(self.buffer) < self.c['buffer_length']:
                # not learning yet, do fully random actions
                act_v = self.rng.random(self.c['n_threads'] * self.c['n_agents'] * self.env.get_action_space()[0])
                act_v = act_v.reshape((self.c['n_threads'], self.c['n_agents'], self.env.get_action_space()[0]))
                act_v = act_v * 2 - 1 # range=(-1, 1)
            else:
                act_v = np.stack([self.model.step(obs_v[0][i], explore=False) for i in range(self.c['n_threads'])])

            next_obs_v, rewards_v, infos_v, dones_v = self.env.step(
                from_unit_actions(act_v, self.c['min_linear_speed'], self.c['max_linear_speed'], self.c['max_angular_speed']))

            for i in range(self.c['n_threads']):
                obs      = (obs_v[0][i], obs_v[1][i])
                next_obs = (next_obs_v[0][i], next_obs_v[1][i])
                self.buffer.push(obs, act_v[i], rewards_v[i], next_obs, dones_v[i])
            obs_v = next_obs_v

            if (len(self.buffer) >= self.c['batch_size'] and
                (step % self.c['steps_per_update']) == 0):

                sample = self.buffer.sample(self.c['batch_size'])
                l_critic1, l_critic2, l_policy, l_entropy = \
                    self.model.update(sample)

                losses['critic1'] += l_critic1
                losses['critic2'] += l_critic2
                losses['entropy'] += l_entropy
                losses['policy']  += l_policy

                n_updates += 1

        if n_updates > 0:
            losses['critic1'] /= n_updates
            losses['critic2'] /= n_updates
            losses['entropy'] /= n_updates
            losses['policy']  /= n_updates

            self.logger.add_scalars('loss/critic', {
                'critic_1': losses['critic1'],
                'critic_2': losses['critic2']}, self.ep)
            self.logger.add_scalars('loss/policy', {'policy': losses['policy']}, self.ep)

            if self.c['auto_entropy']:
                self.logger.add_scalars('loss/entropy',  {'entropy': losses['entropy']}, self.ep)
                self.logger.add_scalars('evaluation/alpha', {'alpha': self.model.alpha}, self.ep)


## Runs multiple training sessions with different seeds
def train_MASAC(config: dict) -> type(None):
    config = config.copy()

    pkg_path = RosPack().get_path('multi_agent_sac')
    runs_dir = os.path.join(pkg_path, 'runs', config['run_name'])

    if config['override']:
        shutil.rmtree(runs_dir)

    os.makedirs(runs_dir, exist_ok=False)

    # save config
    with open(os.path.join(runs_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    torch.manual_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])

    device = torch.device('cuda' if config['device']=='cuda' and
        torch.cuda.is_available() else 'cpu')

    maps_dir_path = os.path.join(RosPack().get_path('mapf_environment'), 'maps')
    image = os.path.join(maps_dir_path, config['map_image'] + '.jpg')
    config.update({'map_image': image})

    print(f'Running {config["n_runs"]} sequential trainers')

    log_dir = os.path.join(runs_dir, 'run_*', 'logs')
    config.update({'log_dir': log_dir})
    longest_name = max([len(n) for n in config.keys()])
    for name, value in config.items():
        spaces = longest_name - len(name) + 4
        print(f'{spaces*" "}{name}: {value}')

    # run training n_runs times
    for i in range(config['n_runs']):
        print(f'\nTrainer {i}')
        run_dir = os.path.join(runs_dir, f'run_{i}')
        os.makedirs(run_dir, exist_ok=False)

        model_dir = os.path.join(run_dir, 'models')
        os.makedirs(model_dir, exist_ok=False)

        log_dir = os.path.join(run_dir, 'logs')
        os.makedirs(log_dir, exist_ok=False)

        logger = SummaryWriter(log_dir)
        env = ParallelEnv(config, rng)

        model = MASAC(
            n_agents          = config['n_agents'],
            obs_space         = env.get_observation_space(),
            act_space         = env.get_action_space(),
            gamma             = config['gamma'],
            tau               = config['tau'],
            auto_entropy      = config['auto_entropy'],
            actor_hidden      = [config['actor_hidden_dim']],
            critic_hidden     = [config['critic_hidden_dim']],
            model_dir         = model_dir,
            device            = device)

        buffer = ReplayBuffer(
            length=config['buffer_length'],
            n_agents=config['n_agents'],
            obs_space=env.get_observation_space(),
            act_space=env.get_action_space())

        TrainProcess(env, model, buffer, logger, log_dir, config, rng)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', default='test', nargs='?', type=str,
        help='Name of the configuration to be loaded')
    parser.add_argument('-o', '--override', action='store_true',
        help='Delete log and model files for current runs')

    config = parser.parse_args()
    config = vars(config)

    pkg_path    = RosPack().get_path('multi_agent_sac')
    params_file = os.path.join(pkg_path, 'params', config['run_name'] + '.yaml')
    with open(params_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        config.update(params)

    if config['device'] == 'cpu':
        # disable no NVIDIA driver warning
        warnings.filterwarnings("ignore", category=UserWarning)

    train_MASAC(config)
