#include <torch/torch.h>
#include "geometry_msgs/Twist.h"
#include <mapf_environment/Observation.h>
#include <mapf_maddpg_agent/Value.h>
#include <mapf_maddpg_agent/Experience.h>
#include <mapf_maddpg_agent/ExtendedState.h>
#include <vector>

using Value         = mapf_maddpg_agent::Value;
using Observation   = mapf_environment::Observation;
using Action        = geometry_msgs::Twist;
using ExtendedState = mapf_maddpg_agent::ExtendedState;
using Experience    = mapf_maddpg_agent::Experience;

class Critic {
	public:
		Critic(int _number_of_agents);
		Value get_state_value(ExtendedState state);
		std::vector<Value> train(std::vector<Experience> experiences);
};

