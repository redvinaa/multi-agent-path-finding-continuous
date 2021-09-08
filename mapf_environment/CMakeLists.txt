cmake_minimum_required(VERSION 3.5)
project(mapf_environment)

set(CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS
  roslib
  roslint
  roscpp
  sensor_msgs
  geometry_msgs
  message_generation
  image_transport
  cv_bridge
)

find_package(OpenCV 3.2 REQUIRED)
find_package(box2d REQUIRED)

add_message_files(
  FILES
  Observation.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${box2d_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)


catkin_package(
 INCLUDE_DIRS include
 LIBRARIES ${PROJECT_NAME}_lib
 CATKIN_DEPENDS roslib roslint roscpp sensor_msgs geometry_msgs image_transport cv_bridge
 DEPENDS box2d OpenCV
)


# add library
add_library(${PROJECT_NAME}_lib src/environment.cpp src/raycast_callback.cpp)
target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${catkin_LIBRARIES} box2d ${OpenCV_LIBRARIES})

add_executable(${PROJECT_NAME}_node src/ros_wrapper_node.cpp src/environment_wrapper.cpp)
target_link_libraries(${PROJECT_NAME}_node PUBLIC ${PROJECT_NAME}_lib)
set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME environment_node)


## Mark library for installation and executable
install(TARGETS ${PROJECT_NAME}_lib ${PROJECT_NAME}_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)


# testing
catkin_add_gtest(${PROJECT_NAME}-test test/test_environment.cpp)
target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME}_lib ${catkin_LIBRARIES} box2d ${OpenCV_LIBRARIES})

# linting
roslint_cpp()
roslint_python()
roslint_add_test()
