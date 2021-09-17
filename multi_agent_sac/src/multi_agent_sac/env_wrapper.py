from mapf_env import Environment
import numpy as np


## Maps actions from (-1, 1) to env defined range
#
#  Input actions [[-1, 1], [-1, 1]]
#  are mapped to [[0, 1],  [-pi/2, pi/2]]
class UnitActionsEnv(Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, actions):
        # max linear speed is 1 m/s
        # max angular speed is pi/2 rad/s
        actions = np.array(actions)
        actions[:, 0] = (actions[:, 0] + 1) / 2
        actions[:, 1] *= np.pi/2

        return super().step(actions)
