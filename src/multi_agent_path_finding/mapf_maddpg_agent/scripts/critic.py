#! /usr/bin/env python3

import rospy
from mapf_environment.msg import Action, Observation
from mapf_maddpg_agent.msg import Value

class Critic:
	def __init__(self):
		self.number_of_agents = rospy.get_param('~number_of_agents')

		self.observation_subscribers = []
		self.value_publishers        = []

		for i in range(self.number_of_agents):
			bound_callback = lambda obs: self.process_observation(i, obs)
			obs_sub = rospy.Subscriber(f'agent_{i}/observation', Observation, bound_callback, queue_size=1)
			self.observation_subscribers.append(obs_sub)

			val_pub = rospy.Publisher(f'agent_{i}/value', Value, queue_size=1)
			self.value_publishers.append(val_pub)

	def process_observation(self, agent_index, observation):
		value = Value()
		value.value = 1 # TODO
		value.message_id = observation.message_id
		self.value_publishers[agent_index].publish(value)

if __name__ == '__main__':
	rospy.init_node('critic')
	critic = Critic()
	try:
		rospy.spin()
	except rospy.ROSInterruptException:
		pass
