#! /usr/bin/env python3

import rospy
from mapf_environment.srv import AddAgent, AddAgentRequest
from mapf_environment.msg import Action, Observation
from mapf_maddpg_agent.msg import Value
from geometry_msgs.msg import Twist
import numpy as np

class Agent:
	def __init__(self):

		self.get_agent_index()

		self.observation_subscriber = rospy.Subscriber(f'agent_{self.agent_index}/observation', Observation, self.process_observation, queue_size=1)
		self.value_subscriber       = rospy.Subscriber(f'agent_{self.agent_index}/value', Value, self.process_value, queue_size=1)
		self.action_publisher       = rospy.Publisher(f'agent_{self.agent_index}/action', Action, queue_size=1)
		rospy.loginfo(f'Initialized agent {self.agent_index}')

	def get_agent_index(self):
		# call add_agent service of environment
		rospy.wait_for_service('add_agent')
		try:
			add_agent_client = rospy.ServiceProxy('add_agent', AddAgent)
			res = add_agent_client(AddAgentRequest())
			self.agent_index = res.agent_index
		except rospy.ServiceException as e:
			rospy.signal_shutdown("AddAgent service call failed: %s"%e)

	def process_observation(self, observation):
		self.last_observation = observation

	def process_value(self, value): # get value from global critic
		observation = self.last_observation
		action = Action()
		# TODO
		action.action.angular.z = 0.5
		action.message_id = observation.message_id
		self.action_publisher.publish(action)


if __name__ == '__main__':
	rospy.init_node('agent')
	agent = Agent()
	try:
		rospy.spin()
	except rospy.ROSInterruptException:
		pass
