#include "mapf_maddpg_agent/actor.h"
#include <torch/torch.h>
#include <geometry_msgs/Twist.h>
#include <mapf_environment/Observation.h>

Actor::Actor(int agent_index) {
	
}

geometry_msgs::Twist Actor::step(mapf_environment::Observation obs, mapf_maddpg_agent::Value val) {
	return geometry_msgs::Twist();
}
geometry_msgs::Twist Actor::action_selector(mapf_environment::Observation obs) {
	return geometry_msgs::Twist();
}
