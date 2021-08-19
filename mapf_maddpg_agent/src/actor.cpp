#include "mapf_environment/types.h"
#include "mapf_maddpg_agent/types.h"
#include "mapf_maddpg_agent/actor.h"
#include "mapf_maddpg_agent/network.h"
#include <torch/torch.h>

Actor::Actor(Net* _net, torch::optim::Optimizer* _optim, float _entropy):
    net(_net), optim(_optim), entropy(_entropy) {}

Action Actor::get_action(Observation obs) const
{
    Action action;
    return action; // TODO
}

void Actor::train(std::vector<Experience> experiences, std::vector<Value> values)
{
}
