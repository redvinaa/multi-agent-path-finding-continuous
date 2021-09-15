#! /usr/bin/env python3.6

import argparse
import torch
import os
import numpy as np
from tensorboardX import SummaryWriter
from agent import Agent
from rospkg import RosPack
import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist


class MASACRos:
    def __init__(self, config):
        self.config = config

        # load
        pkg_path = RosPack().get_path('multi_agent_sac')
        model_dir = os.path.join(pkg_path, 'runs', config.run_name, 'models')
        agent_weights_file = os.path.join(model_dir, 'agent.p')

        self.agent = Agent()
        self.agent.load(agent_weights_file)

        self.obs_subscribers = []
        self.act_publishers  = []

        for i in range(config.n_agents):
            agent_obs_cb = lambda obs: self.obs_callback(i, obs)
            sub = rospy.Subscriber(f'agent_{i}/observation', Float32MultiArray,
                agent_obs_cb, queue_size=1)
            self.obs_subscribers.append(sub)

            pub = rospy.Publisher(f'agent_{i}/cmd_vel', Twist, queue_size=1)
            self.act_publishers.append(pub)


    def obs_callback(self, agent_index, obs):
        act = self.agent.step(obs, expore=False)

        act_ros = Twist()
        act_ros.linear.x  = act[0]
        act_ros.angular.z = act[1]
        self.act_publishers[agent_index].publish(act_ros)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name',          default='test',      type=str)
    parser.add_argument('--map_image',         default='empty_4x4', type=str)
    parser.add_argument('--buffer_length',     default=int(1e6),    type=int)
    parser.add_argument('--n_episodes',        default=50000,       type=int)
    parser.add_argument('--n_agents',          default=2,           type=int)
    parser.add_argument('--episode_length',    default=25,          type=int)
    parser.add_argument('--steps_per_update',  default=100,         type=int)
    parser.add_argument('--num_updates',       default=4,           type=int)
    parser.add_argument('--batch_size',        default=1024,        type=int)
    parser.add_argument('--save_interval',     default=1000,        type=int)
    parser.add_argument('--actor_hidden_dim',  default=128,         type=int)
    parser.add_argument('--critic_hidden_dim', default=128,         type=int)
    parser.add_argument('--seed',              default=0,           type=int)
    parser.add_argument('--tau',               default=0.001,       type=float)
    parser.add_argument('--gamma',             default=0.99,        type=float)
    parser.add_argument('--alpha',             default=0.99,        type=float)

    config = parser.parse_args()
    rospy.init_node('multi_agent_sac')
    MASACRos(config)
    rospy.spin()
