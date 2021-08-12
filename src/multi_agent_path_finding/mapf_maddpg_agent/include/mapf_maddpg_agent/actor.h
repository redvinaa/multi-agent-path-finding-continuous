#pragma once

#include "mapf_maddpg_agent/types.h"
#include "mapf_maddpg_agent/network.h"
#include <torch/torch.h>
#include <vector>
#include <gtest/gtest.h>

/*! \brief Multi-agent actor-critic algorithm, Actor class
 *
 * Based on: https://arxiv.org/abs/1706.02275
 */
class Actor
{
    private:
        int agent_index;
        Net* net;
        torch::optim::Optimizer* optim;

    public:
        Actor(int _agent_index, Net* _net, torch::optim::Optimizer* _optim);

        /*! \brief Calculate action based on (local) state
         */
        Action get_action(Observation obs) const;

        /*! \brief Train network on batch of experiences
         *
         * This is a policy-gradient algorithm,
         * the values calculated by the critic are used as baselines,
         *
         * \sa Critic
         */
        void train(std::vector<Experience> experiences, std::vector<Value> values);
};
