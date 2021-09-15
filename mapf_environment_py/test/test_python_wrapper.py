#!/usr/bin/env python3.6
# Copyright 2021 Reda Vince


import os
import sys
import unittest
import rospkg
from mapf_env import Environment
import rosunit
import numpy as np

__author__ = "Reda Vince"
__copyright__ = "Copyright 2021 Reda Vince"


class TestWrapper(unittest.TestCase):

    def test_env(self):
        # constructor
        pkg_path = rospkg.RosPack().get_path('mapf_environment')
        image = os.path.join(pkg_path, 'maps', 'test_4x4.jpg')
        env = Environment(image, 2)

        # reset
        env_obs = env.reset()
        self.assertFalse(env.is_done())

        # render && step
        done = False
        while not done:
            act = np.array([[1., -0.5], [0.1, 0.0]])
            obs, rew, d = env.step(act)
            done = d[0]
            env.render(10)

        self.assertEqual(env.get_observation_space()[0], 17)

if __name__ == '__main__':
    rosunit.unitrun('mapf_environment_py', 'test_wrapper', TestWrapper)
