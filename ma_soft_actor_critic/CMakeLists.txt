cmake_minimum_required(VERSION 3.5)
project(ma_soft_actor_critic)

set(CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roslib
  roslint
)

catkin_package(
 CATKIN_DEPENDS rospy roslib roslint
)

catkin_python_setup()
catkin_package()

# testing
catkin_add_nosetests(test/test_masac.py)

# linting
roslint_python()
roslint_add_test()
