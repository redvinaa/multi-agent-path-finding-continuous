#!/usr/bin/env python3.6
# Copyright 2021 Reda Vince


import os
import sys
import unittest
import rospkg
from mapf_env import Environment
from multi_agent_sac.env_wrapper import UnitActionsEnv
from multi_agent_sac.algorithm import MASAC
from multi_agent_sac.buffer import ReplayBuffer
from multi_agent_sac.network import Network
import torch
import rosunit
import numpy as np


__author__ = "Reda Vince"
__copyright__ = "Copyright 2021 Reda Vince"

pkg = 'multi_agent_sac'
pkg_path = rospkg.RosPack().get_path('mapf_environment')


def random_action(env):
    return np.random.random(env.get_number_of_agents() * env.get_action_space()[0]).reshape((env.get_number_of_agents(), env.get_action_space()[0],))


class TestMASAC(unittest.TestCase):
    def test_unit_action_wrapper(self):
        image    = os.path.join(pkg_path, 'maps', 'test_4x4.jpg')
        env      =    Environment(image, 1, max_steps=10, seed=0)
        unit_env = UnitActionsEnv(image, 1, max_steps=10, seed=0)

        obs      = env.reset()
        unit_obs = unit_env.reset()
        self.assertEqual(obs, unit_obs)

        act      = np.array([[1., np.pi/2.]])
        unit_act = np.array([[1., 1.]])

        obs,      _, _ = env.step(act)
        unit_obs, _, _ = unit_env.step(unit_act)
        self.assertEqual(obs, unit_obs)

        act      = np.array([[0., -np.pi/2.]])
        unit_act = np.array([[-1., -1.]])

        obs,      _, _ = env.step(act)
        unit_obs, _, _ = unit_env.step(unit_act)
        self.assertEqual(obs, unit_obs)


    def test_buffer(self):
        image = os.path.join(pkg_path, 'maps', 'test_4x4.jpg')
        env = Environment(image, 2, max_steps=10)

        buf = ReplayBuffer(20, env.get_number_of_agents(), env.get_observation_space()[0], env.get_action_space()[0])

        obs = env.reset()
        done = False
        t = 0
        rew0 = 0.

        while not done:
            act = random_action(env)
            next_obs, rew, d = env.step(act)
            buf.push(obs, act, rew, next_obs, d)

            done = d[0]
            obs = next_obs
            rew0 += rew[0]

            t += 1
            if t == 5:
                self.assertEqual(len(buf), 5)

        self.assertEqual(buf.get_rewards(10)[0], rew0)

        sample = buf.sample(4)
        self.assertEqual(len(sample), 5)
        self.assertEqual(len(sample[0]), 4)
        self.assertEqual(sample[0][0].shape, (env.get_number_of_agents(), env.get_observation_space()[0]))


    def test_network(self):
        net = Network(2, 1)

        dummy_in  = np.random.random(10).reshape((-1, 2)).astype(np.float32)
        dummy_out = np.ones((5, 1)).astype(np.float32)

        dummy_in = torch.tensor(dummy_in)
        dummy_out = torch.tensor(dummy_out)

        opt = torch.optim.Adam(net.parameters())
        L   = torch.nn.MSELoss()
        losses = []
        for ep in range(10):
            opt.zero_grad()

            net_out = net(dummy_in)

            loss = L(net_out, dummy_out)
            loss_val = loss.item()
            losses.append(loss_val)
            self.assertNotEqual(loss_val, 0)

            loss.backward()
            opt.step()

        print(losses)
        self.assertGreater(losses[0], losses[-1])


if __name__ == '__main__':
    rosunit.unitrun(pkg, 'test_masac', TestMASAC)
