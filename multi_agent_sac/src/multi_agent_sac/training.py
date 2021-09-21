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


## Perform training with the provided config
def run(config: object) -> type(None):
    pkg_path = RosPack().get_path('multi_agent_sac')
    model_dir = os.path.join(pkg_path, 'runs', config.run_name, 'models')
    os.makedirs(model_dir, exist_ok=True)

    log_dir = os.path.join(pkg_path, 'runs', config.run_name, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = SummaryWriter(log_dir)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    maps_dir_path = os.path.join(RosPack().get_path('mapf_environment'), 'maps')
    image = os.path.join(maps_dir_path, config.map_image + '.jpg')
    env = UnitActionsEnv(image, config.n_agents,
        max_steps=config.episode_length, seed=config.seed)
    #  env.render()

    model = MASAC(
        n_agents          = config.n_agents,
        obs_size          = env.get_observation_space()[0],
        act_size          = env.get_action_space()[0],
        gamma             = config.gamma,
        tau               = config.tau,
        alpha             = config.alpha,
        actor_hidden      = [config.actor_hidden_dim],
        critic_hidden     = [config.critic_hidden_dim],
        model_dir         = model_dir,
        logger            = logger)

    buffer = ReplayBuffer(
        length=config.buffer_length,
        n_agents=config.n_agents,
        obs_size=env.get_observation_space()[0],
        act_size=env.get_action_space()[0])

    t = 0
    for ep_i in range(config.n_episodes):

        obs = env.reset()
        obs = np.array(obs)

        if ((ep_i+1) % 10) == 0:
            print(f'Episode {ep_i+1} of {config.n_episodes}')

        for et_i in range(config.episode_length):
            act = model.step(obs, explore=True)

            next_obs, rewards, dones = env.step(act)
            next_obs = np.array(next_obs)
            buffer.push(obs, act, rewards, next_obs, dones)
            obs = next_obs
            t += 1

            if (len(buffer) >= config.batch_size and
                (t % config.steps_per_update) == 0):

                sample = buffer.sample(config.batch_size)
                model.update(sample, step=t)

        ep_rews = buffer.get_rewards(config.episode_length)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/episode_reward' % a_i,
                              a_ep_rew * config.episode_length, ep_i)

        if (ep_i % config.save_interval) == 0:
            inc_model_dir = os.path.join(model_dir, 'incremental', f'model_ep{ep_i+1}')
            os.makedirs(inc_model_dir, exist_ok=True)
            model.save()
            print('Saved model!')

    model.save()
    print('Saved model!')
    logger.export_scalars_to_json(str(os.path.join(log_dir, 'summary.json')))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name',          default='test',      type=str)
    parser.add_argument('--map_image',         default='empty_4x4', type=str)
    parser.add_argument('--buffer_length',     default=int(1e5),    type=int)
    parser.add_argument('--n_episodes',        default=5000,        type=int)
    parser.add_argument('--n_agents',          default=2,           type=int)
    parser.add_argument('--episode_length',    default=60,          type=int,
        help='0.5 seconds pass per step in simulation time')
    parser.add_argument('--steps_per_update',  default=10,          type=int)
    parser.add_argument('--batch_size',        default=800,         type=int)
    parser.add_argument('--save_interval',     default=1000,        type=int,
        help='This is per episode, not per step')
    parser.add_argument('--actor_hidden_dim',  default=128,         type=int)
    parser.add_argument('--critic_hidden_dim', default=128,         type=int)
    parser.add_argument('--seed',              default=0,           type=int)
    parser.add_argument('--tau',               default=0.999,       type=float)
    parser.add_argument('--gamma',             default=0.95,        type=float)
    parser.add_argument('--alpha',             default=0.99,        type=float)

    config = parser.parse_args()
    run(config)
