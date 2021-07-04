import rospy
from geometry_msgs.msg import Twist

class UserAgent:
	def __init__(self):
		self.cmd_vel = rospy.Subscriber('cmd_vel', Twist, self.cmd_vel_callback)
		self.twist = Twist()

	def cmd_vel_callback(self, twist):
		self.twist = twist

	def step(self, observation):
		return self.twist
