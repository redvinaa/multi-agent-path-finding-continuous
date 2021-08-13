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
    private:
        Net* net;
        torch::optim::Optimizer* optim;
        float gamma;

        /*! \brief Serialize, aka. create one long vector from the collective observations
         */
        std::vector<float> serialize_collective(CollectiveObservation coll_obs) const;

    public:
        Critic(Net* _net, torch::optim::Optimizer* _optim, float _gamma);

        /*! \brief Calculate value of extended state
         * (that is, considering all agents)
         */
        Value get_value(CollectiveObservation coll_obs) const;

        /*! \brief Vectorized overload
         */
        std::vector<Value> get_value(std::vector<CollectiveObservation> coll_obs) const;

        /*! \brief Train network on batch of experiences
         *
         * With respect to the CollectiveState, this is a single-agent 
         * Bellman update algorithm on the value function:
         * V(x) <- V(x) + step_size * [ r + V(x_) - V(x)]
         *
         * One experience consists of
         *   - CollectiveState x (= x1, ..., xn)
         *   - Reward r = avg(r1, ..., rn)
         *   - Actions a (= a1, ..., an)
         *   - CollectiveState x_ (the next state)
         *   - Done
         *
         *  \return Average loss
         *
         * \sa Actor
         */
        float train(std::vector<Experience> experiences);
};

