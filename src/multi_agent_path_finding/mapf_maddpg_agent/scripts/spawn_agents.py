#! /usr/bin/env python3

import roslaunch
import rospy

if __name__ == '__main__':
	rospy.init_node('spawn_agents')

	number_of_agents = rospy.get_param('number_of_agents', default=1)
	number_of_agents +=2
	rospy.loginfo(f'Running {number_of_agents} MADDPG agents')

	package = 'mapf_maddpg_agent'
	executable = 'agent.py'
	node = roslaunch.core.Node(package, executable)

	launch = roslaunch.scriptapi.ROSLaunch()
	launch.start()

	for i in range(number_of_agents):
		node = roslaunch.core.Node(package, executable, name=f'agent_{i}')
		process = launch.launch(node)

	#  process.stop()
