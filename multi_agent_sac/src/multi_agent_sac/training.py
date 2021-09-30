#! /usr/bin/env python3.6

import argparse
import torch
import os
import numpy as np
from tensorboardX import SummaryWriter
from algorithm import MASAC
from buffer import ReplayBuffer
from env_wrapper import ParallelEnv
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

        for self.ep in tqdm(range(self.c['n_episodes'])):
            t0 = time.time()
            self.__training_episode()
            t1 = time.time()

            if (self.ep % self.c['save_interval']) == 0 and self.ep != 0:
                self.model.save()

            if (self.ep % self.c['eval_interval']) == 0:
                self.__evaluation_episode()
                self.logger.add_scalar('evaluation/training_ep_time', t1 - t0, self.ep)

        self.model.save()
        self.logger.export_scalars_to_json(os.path.join(self.log_dir, 'summary.json'))
        self.logger.close()


    def __evaluation_episode(self) -> type(None):
        log_actions = np.empty((self.c['n_threads'], self.c['episode_length'], \
            self.c['n_agents'], self.env.act_size))
        log_rewards = np.zeros((self.c['n_agents']))

        obs_v = self.env.reset()

        for step in range(self.c['episode_length']):
            act_v = np.stack([self.model.step(obs, explore=False) for obs in obs_v])

            next_obs_v, rewards_v, dones_v = self.env.step(act_v)

            obs_v = next_obs_v

            log_actions[:, step] = act_v
            log_rewards += np.average(rewards_v, axis=0) # average between parallel envs

        log_actions = np.average(log_actions, axis=0)
        data = {}
        for i, rew in enumerate(log_rewards):
            data.update({f'agent_{i}': rew})

        self.logger.add_scalars('evaluation/episode_reward',    data,                           self.ep)
        self.logger.add_histogram('evaluation/linear_actions', \
            log_actions[:,:,0].flatten(), self.ep)
        self.logger.add_histogram('evaluation/angular_actions', \
            log_actions[:,:,1].flatten(), self.ep)


    def __training_episode(self) -> type(None):
        obs_v = self.env.reset()

        n_updates = 0
        losses = {'critic1': 0., 'critic2': 0., 'entropy': 0., 'policy': 0.}
        for step in range(self.c['episode_length']):
            if len(self.buffer) < self.c['buffer_length']:
                # not learning yet, do fully random actions
                act_v = self.rng.random(self.c['n_threads'] * self.c['n_agents'] * self.env.act_size)
                act_v = act_v.reshape((self.c['n_threads'], self.c['n_agents'], self.env.act_size))
                act_v = act_v * 2 - 1 # range=(-1, 1)
            else:
                act_v = np.stack([self.model.step(obs, explore=True) for obs in obs_v])

            next_obs_v, rewards_v, dones_v = self.env.step(act_v)

            for obs, act, rewards, next_obs, dones in zip(obs_v, act_v, rewards_v, next_obs_v, dones_v):
                self.buffer.push(obs, act, rewards, next_obs, dones)
            obs_v = next_obs_v

            if (len(self.buffer) >= self.c['batch_size'] and
                (step % self.c['steps_per_update']) == 0):

                sample = self.buffer.sample(self.c['batch_size'])
                l_critic1, l_critic2, l_entropy, l_policy = \
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
            self.logger.add_scalar('loss/policy', losses['policy'], self.ep)

            if self.c['auto_entropy']:
                self.logger.add_scalar('loss/entropy', losses['entropy'], self.ep)
                self.logger.add_scalar('evaluation/alpha', self.model.alpha, self.ep)


## Runs multiple training sessions in parallel with different seeds
def parallel_train_MASAC(config: dict) -> type(None):
    #  mp.set_start_method('spawn')

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

    print(f'Running {config["n_runs"]} parallel trainers')

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
            obs_size          = env.obs_size,
            act_size          = env.act_size,
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
            obs_size=env.obs_size,
            act_size=env.act_size)

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

    parallel_train_MASAC(config)
