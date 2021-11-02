#! /usr/bin/env python3.6

import argparse
import torch
import os
import numpy as np
from multi_agent_sac.algorithm import MASAC
from multi_agent_sac.misc import from_unit_actions
from multi_agent_sac.env_wrapper import ParallelEnv
from rospkg import RosPack
import json


class TestMASACRos:
    def __init__(self, config: dict):

        # load config
        pkg_path = RosPack().get_path('multi_agent_sac')
        self.run_dir   = os.path.join(pkg_path, 'runs', config['run_name'])
        self.model_dir = os.path.join(self.run_dir, f'run_{config["run_index"]}', 'models')
        self.log_dir   = os.path.join(self.run_dir, f'run_{config["run_index"]}', 'logs')

        with open(os.path.join(self.run_dir, 'config.json'), 'r') as f:
            self.c = json.load(f)

            # seed from config file
            torch.manual_seed(self.c['seed'])
            np.random.seed(self.c['seed'])

            self.c.update(config)

        # create env
        maps_dir_path = os.path.join(RosPack().get_path('mapf_environment'), 'maps')
        image = os.path.join(maps_dir_path, self.c['map_image'] + '.jpg')
        self.c.update({'n_threads': 1, 'map_image': image})

        self.env = ParallelEnv(self.c, np.random.default_rng())

        # create model, load weights
        device = torch.device('cuda' if self.c['device']=='cuda' and
            torch.cuda.is_available() else 'cpu')

        self.model = MASAC(  # most of these are not used
            n_agents          = self.c['n_agents'],
            obs_space         = self.env.get_observation_space(),
            act_space         = self.env.get_action_space(),
            gamma             = self.c['gamma'],
            tau               = self.c['tau'],
            auto_entropy      = self.c['auto_entropy'],
            actor_hidden      = [self.c['actor_hidden_dim']],
            critic_hidden     = [self.c['critic_hidden_dim']],
            model_dir         = self.model_dir,
            device            = device)

        self.model.load()

    def run(self) -> type(None):
        obs  = self.env.reset()

        while True:
            # get actions
            act = np.stack([self.model.step(o, explore=False) for o in obs[0]])

            # render
            self.env.render(500, False)

            # step
            obs, _, _, _ = \
                self.env.step(
                from_unit_actions(act, \
                self.c['min_linear_speed'],
                self.c['max_linear_speed'], \
                self.c['max_angular_speed']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name',  default='empty_4x4', nargs='?', type=str,
        help='Name of the run to load')
    parser.add_argument('run_index', default=0,      nargs='?', type=int,
        help='Index of the run to load')
    parser.add_argument('--seed',    default=0,                 type=int)

    config = parser.parse_args()
    test = TestMASACRos(vars(config))
    test.run()
