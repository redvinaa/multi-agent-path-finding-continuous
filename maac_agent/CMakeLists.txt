cmake_minimum_required(VERSION 3.5)
project(maac_agent)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../libtorch")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  mapf_environment
  sensor_msgs
  std_msgs
  tf
  message_generation
  roslint
)

find_package(Torch REQUIRED)

add_message_files(
  FILES
  Experience.msg
)

# add_service_files(
#   FILES
# )

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  mapf_environment
  maac_agent
)

catkin_package(
 INCLUDE_DIRS include ${TORCH_INCLUDE_DIRS}
 LIBRARIES maac_agent
 CATKIN_DEPENDS geometry_msgs roscpp mapf_environment sensor_msgs std_msgs tf message_runtime
 DEPENDS Torch
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/actor.cpp
  src/critic.cpp
  src/network.cpp
)

add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${TORCH_LIBRARIES}
)


## Mark executables for installation
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

catkin_add_gtest(${PROJECT_NAME}-test test/test_maac.cpp)
target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})

# linting
roslint_cpp()
