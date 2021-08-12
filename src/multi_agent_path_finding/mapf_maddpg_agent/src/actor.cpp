#include "mapf_maddpg_agent/actor.h"
#include "mapf_maddpg_agent/types.h"
#include "mapf_maddpg_agent/network.h"
#include <torch/torch.h>

Actor::Actor(int _agent_index, Net* _net, torch::optim::Optimizer* _optim):
    agent_index(_agent_index), net(_net), optim(_optim)
{

}

Action Actor::get_action(Observation obs) const
{
    Action action;
    return action; // TODO
}

void Actor::train(std::vector<Experience> experiences, std::vector<Value> values)
{
    
}
