#! /usr/bin/env python3.6

import argparse
import json
from rospkg import RosPack
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List
import yaml


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', default='empty_4x4', nargs='?', type=str,
        help='Name of the configuration to be loaded')
    parser.add_argument('-s', '--show', action='store_true')
    parser.add_argument('-c', '--color', default='tab:blue', type=str,
        help='Color of the plots')

    args = parser.parse_args()

    pkg_path    = RosPack().get_path('multi_agent_sac')
    run_dir     = os.path.join(pkg_path, 'runs', args.run_name)

    # load config
    with open(os.path.join(pkg_path, 'params', args.run_name+'.yaml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # load training data
    summaries = []
    for run in glob(os.path.join(run_dir, 'run_*')):
        summary_f = os.path.join(run, 'logs', 'summary.json')
        if not os.path.isfile(summary_f):
            continue

        with open(summary_f) as f:
            data = json.load(f)
        summaries.append(data)

    if len(summaries) == 0:
        print('No summaries found')
        quit()

    print(f'Found {len(summaries)} runs')

    # strip path name from keys
    for summary in summaries:
        keys = list(summary.keys())
        for k in keys:
            new_k = k.split('/logs/')[1]
            summary[new_k] = summary.pop(k)

    plt.rcParams.update({'font.size': 22})

    ## Calculate, show and save plot for a given value
    #
    #  @param name Name or list of names for the plotted values
    #  @param save Save the figure to runs/<run_name>/figures/<name>.png
    #  @param log_scale Use log scale when plotting
    def draw_plot(name: Union[str, List[str]], *,
            log_scale: bool=False) -> type(None):
        data = []

        for summary in summaries:
            if type(name) == str:
                run_data = summary[name]
                data.append(run_data)
            elif type(name) == list:
                for n in name:
                    run_data = summary[n]
                    data.append(run_data)

        data = np.array(data)
        ep   = data[0, :, 1]*config['n_threads']/1000
        vals = data[:, :, 2]
        avg  = np.average(data[:, :, 2], axis=0)

        plt.grid()
        ax = plt.gca()
        ax.fill_between(ep, np.min(vals, axis=0), np.max(vals, axis=0), alpha=.2, color=args.color)
        if log_scale:
            ax.set_yscale('log')
        plt.plot(ep, avg, color=args.color)
        y_range = np.max(avg) - np.min(avg)
        y_top = np.max(avg) + y_range*0.2
        y_bot = np.min(avg) - y_range*0.2
        plt.ylim(bottom=y_bot, top=y_top)
        plt.xlabel('1000 episodes')
        plt.tight_layout()

        os.makedirs(os.path.join(run_dir, 'figures'), exist_ok=True)
        if type(name) == str:
            n = name
        else:
            n = name[0]
        plt.savefig(os.path.join(run_dir, 'figures', n.replace('/', '_')))

        if args.show:
            fig = plt.gcf()
            if type(name) == str:
                n = name
            else:
                n = name[0]
            fig.canvas.set_window_title(n)
            plt.show()

    draw_plot('evaluation/episode_reward_average/episode_reward_average')
    draw_plot('evaluation/collision_average/collision_average')
    draw_plot('evaluation/reached_goal_average/reached_goal_average')
    draw_plot(['loss/critic/critic_1', 'loss/critic/critic_2'])
    draw_plot('loss/entropy/entropy')
    draw_plot('loss/policy/policy')
    draw_plot('evaluation/collision_reward/collision_reward')
    draw_plot('evaluation/alpha/alpha', log_scale=True)
