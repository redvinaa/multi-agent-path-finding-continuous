#! /usr/bin/env python
#! /home/vince/study/multi-agent-path-finding/code/multi-agent-path-finding-continuous/.venv/bin/python

import rospy
from mapf_environment.srv import CallAgent, CallAgentResponse
from mapf_environment.srv import AddAgent, AddAgentRequest
from geometry_msgs.msg import Twist
import numpy as np
from agents import make_agent

class AgentRosWrapper:
	def __init__(self):
		self.agent_type = rospy.get_param('agent_type', 'user')
		self.agent = make_agent(self.agent_type)

		self.get_agent_index()
		self.call_agent_server = rospy.Service('agent_'+str(self.agent_index)+'/call_agent', CallAgent, self.call_agent)
		rospy.loginfo('Advertised service agent_'+str(self.agent_index)+'/call_agent')

	def get_agent_index(self):
		# call add_agent service of environment
		rospy.wait_for_service('add_agent')
		try:
			add_agent_client = rospy.ServiceProxy('add_agent', AddAgent)
			res = add_agent_client(AddAgentRequest())
			self.agent_index = res.agent_index
		except rospy.ServiceException as e:
			print("AddAgent service call failed: %s"%e)

	def call_agent(self, req):
		action = self.agent.step(req)
		response = CallAgentResponse()
		response.action = action
		return response


if __name__ == '__main__':
	rospy.init_node('agent')
	agent = AgentRosWrapper()
	try:
		rospy.spin()
	except rospy.ROSInterruptException:
		pass
