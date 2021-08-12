#pragma once

#include "mapf_maddpg_agent/types.h"
#include "mapf_maddpg_agent/network.h"
#include "mapf_maddpg_agent/types.h"
#include <torch/torch.h>
#include <vector>

/*! \brief Multi-agent actor-critic algorithm, Critic class
 *
 * Based on: https://arxiv.org/abs/1706.02275
 */
class Critic
{
    public:
        int number_of_agents;
        Net* net;
        torch::optim::Optimizer* optim;

        Critic(int _number_of_agents, Net* _net, torch::optim::Optimizer* _optim);

        /*! \brief Calculate value of extended state
         * (that is, considering all agents)
         */
        Value get_value(CollectiveObservation state);

        /*! \brief Train network on batch of experiences
         *
         * This is basically a single-agent DQN algorithm,
         *
         * One experience consists of
         *   - CollectiveState x (= x1, ..., xn)
         *   - Reward r = avg(r1, ..., rn)
         *   - Actions a (= a1, ..., an)
         *   - CollectiveState x_ (the next state)
         *   - Done
         *
         *  The states include the rewards and also if done==true
         *
         * \sa Actor
         */
        std::vector<Value> train(std::vector<Experience> experiences);
};

