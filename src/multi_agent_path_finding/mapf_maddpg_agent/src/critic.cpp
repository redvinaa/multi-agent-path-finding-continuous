#include "mapf_maddpg_agent/critic.h"
#include "mapf_maddpg_agent/types.h"
#include "mapf_maddpg_agent/network.h"
#include <torch/torch.h>

Critic::Critic(int _number_of_agents, Net* _net, torch::optim::Optimizer* _optim):
    number_of_agents(_number_of_agents), net(_net) {}

Value Critic::get_value(CollectiveObservation obs)
{
    Value value;
    return value; // TODO
}

std::vector<Value> Critic::train(std::vector<Experience> experiences)
{
    std::vector<Value> values;
    return values; // TODO
}

