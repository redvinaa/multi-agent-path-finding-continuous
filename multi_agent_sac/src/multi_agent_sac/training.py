#! /usr/bin/env python3.6

import argparse
import torch
import os
import numpy as np
from tensorboardX import SummaryWriter
from algorithm import MASAC
from buffer import ReplayBuffer
from env_wrapper import UnitActionsEnv
from rospkg import RosPack
from tqdm import tqdm
import warnings
import json


class TrainMASAC:
    def __init__(self, config: dict):
        self.c = config

        pkg_path = RosPack().get_path('multi_agent_sac')

        run_idx = 0
        runs_dir = os.path.join(pkg_path, 'runs', self.c['run_name'])
        if os.path.isdir(runs_dir):
            runs = os.listdir(runs_dir)
            runs = [r for r in runs if 'run_' in r]
            if len(runs) == 0:
                run_idx = 0
            else:
                run_idx = max([int(r[4:]) for r in runs]) + 1

        self.model_dir = os.path.join(pkg_path, 'runs', \
            self.c['run_name'], f'run_{run_idx}', 'models')
        os.makedirs(self.model_dir, exist_ok=True)

        self.log_dir = os.path.join(pkg_path, 'runs', \
            self.c['run_name'], f'run_{run_idx}', 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = SummaryWriter(self.log_dir)

        torch.manual_seed(self.c['seed'])
        np.random.seed(self.c['seed'])

        maps_dir_path = os.path.join(RosPack().get_path('mapf_environment'), 'maps')
        image = os.path.join(maps_dir_path, self.c['map_image'] + '.jpg')
        self.env = UnitActionsEnv(image, self.c['n_agents'],
            max_steps=self.c['episode_length'], seed=self.c['seed'], collision_reward=-0.1)

        self.model = MASAC(
            n_agents          = self.c['n_agents'],
            obs_size          = self.env.get_observation_space()[0],
            act_size          = self.env.get_action_space()[0],
            gamma             = self.c['gamma'],
            tau               = self.c['tau'],
            alpha             = self.c['alpha'],
            auto_entropy      = self.c['auto_entropy'],
            actor_hidden      = [self.c['actor_hidden_dim']],
            critic_hidden     = [self.c['critic_hidden_dim']],
            model_dir         = self.model_dir,
            logger            = self.logger)

        self.buffer = ReplayBuffer(
            length=self.c['buffer_length'],
            n_agents=self.c['n_agents'],
            obs_size=self.env.get_observation_space()[0],
            act_size=self.env.get_action_space()[0])

        self.train_step_counter = 0
        self.train_ep_counter   = 0
        self.eval_step_counter  = 0
        self.eval_ep_counter    = 0

        # save config
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as f:
            json.dump(config, f)

        print(f'Initialized trainer: {self.c["run_name"]}, run_{run_idx}')

        longest_name = max([len(n) for n in self.c.keys()])
        for name, value in self.c.items():
            spaces = longest_name - len(name) + 4
            print(f'{spaces*" "}{name}: {value}')


    def evaluation_episode(self, render: bool) -> type(None):
        log_actions = np.empty((self.c['episode_length'], self.c['n_agents'], self.env.get_action_space()[0]))
        log_rewards = np.zeros((self.c['n_agents']))

        obs = self.env.reset()
        obs = np.array(obs)

        for step in range(self.c['episode_length']):
            act = self.model.step(obs, explore=False)

            next_obs, rewards, dones = self.env.step(act, render)

            next_obs = np.array(next_obs)
            obs = next_obs

            log_actions[step] = act
            log_rewards += rewards

            self.eval_step_counter += 1

        self.eval_ep_counter += 1

        data = {}
        for i, rew in enumerate(log_rewards):
            data.update({f'agent_{i}': rew})

        self.logger.add_scalars('evaluation/episode_reward',    data,                         self.train_ep_counter)
        self.logger.add_histogram('evaluation/linear_actions',  log_actions[:,:,0].flatten(), self.train_ep_counter)
        self.logger.add_histogram('evaluation/angular_actions', log_actions[:,:,1].flatten(), self.train_ep_counter)


    def training_episode(self) -> type(None):
        obs = self.env.reset()
        obs = np.array(obs)

        for step in range(self.c['episode_length']):
            if len(self.buffer) < self.c['buffer_length']:
                # not learning yet, do fully random actions
                act = np.random.random(self.c['n_agents']*self.env.get_action_space()[0]).reshape((self.c['n_agents'], -1)) * 2 - 1
            else:
                act = self.model.step(obs, explore=True)
            next_obs, rewards, dones = self.env.step(act, False)

            next_obs = np.array(next_obs)
            self.buffer.push(obs, act, rewards, next_obs, dones)
            obs = next_obs

            if (len(self.buffer) >= self.c['batch_size'] and
                (self.train_step_counter % self.c['steps_per_update']) == 0):

                sample = self.buffer.sample(self.c['batch_size'])
                self.model.update(sample, step=self.train_step_counter)

            self.train_step_counter += 1

        if (self.train_ep_counter % self.c['save_interval']) == 0 \
                and self.train_ep_counter != 0:
            self.model.save()

        self.train_ep_counter += 1


    ## Perform training with the provided config
    def run(self) -> type(None):

        for _ in tqdm(range(self.c['n_episodes'])):

            self.training_episode()

            if (self.train_ep_counter % self.c['eval_interval']) == 0:
                render = (self.train_ep_counter % 2000) == 0 and self.train_ep_counter != 0
                self.evaluation_episode(render)

        self.model.save()
        self.logger.export_scalars_to_json(os.path.join(self.log_dir, 'summary.json'))
        self.logger.close()


if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name',          default='test',      type=str,
        help='Name of the configuration')
    parser.add_argument('--map_image',         default='empty_4x4', type=str,
        help='Name of the map to load (from the mapf_environment package)')
    parser.add_argument('--buffer_length',     default=100000,      type=int,
        help='Lenght of the replay buffer')
    parser.add_argument('--n_episodes',        default=10000,       type=int,
        help='Number of episodes to train (evaluations are excluded)')
    parser.add_argument('--n_agents',          default=2,           type=int,
        help='Number of agents')
    parser.add_argument('--episode_length',    default=30,          type=int,
        help='1 second passes per step in simulation time')
    parser.add_argument('--steps_per_update',  default=10,          type=int,
        help='Gradient descent is performed every this many steps')
    parser.add_argument('--batch_size',        default=10000,       type=int,
        help='Sample this many transitions for the updates')
    parser.add_argument('--save_interval',     default=1000,        type=int,
        help='Save model weights after this many EPISODES')
    parser.add_argument('--eval_interval',     default=10,          type=int,
        help='Play an evaluation episode after this many training episodes')
    parser.add_argument('--actor_hidden_dim',  default=64,          type=int,
        help='Number of nodes in the (one) hidden layer of the policy network')
    parser.add_argument('--critic_hidden_dim', default=64,          type=int,
        help='Number of nodes in the (one) hidden layer of each of the critic networks')
    parser.add_argument('--seed',              default=5,           type=int,
        help='For reproducibility')
    parser.add_argument('--tau',               default=0.99,        type=float,
        help='Update the target critic network (1 -> hard update) at every update')
    parser.add_argument('--gamma',             default=0.90,        type=float,
        help='Discount factor')
    parser.add_argument('--alpha',             default=0.2,         type=float,
        help='Entropy coefficient')
    parser.add_argument('--auto_entropy',      default=True,        type=bool,
        help='Learn entropy coefficient')

    config = parser.parse_args()
    t = TrainMASAC(vars(config))
    t.run()
