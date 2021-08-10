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

class Actor {
	public:
		Actor(int _agent_index);
		Action get_action(Observation obs);
		void train(std::vector<Experience> experiences, std::vector<Value> values);
};
