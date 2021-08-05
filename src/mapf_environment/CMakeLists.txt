cmake_minimum_required(VERSION 3.5)
project(mapf_environment)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  sensor_msgs
  std_msgs
  tf
  message_generation
  # cv_bridge
  image_transport
  ddynamic_reconfigure
  OpenCV
)

MESSAGE(STATUS "============================================${OpenCV_VERSION}")
find_package(box2d REQUIRED)

add_message_files(
  FILES
  Observation.msg
)

add_service_files(
  FILES
  AddAgent.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES mapf_environment
 CATKIN_DEPENDS geometry_msgs roscpp sensor_msgs std_msgs tf message_runtime
 DEPENDS box2d OpenCV
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${box2d_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)


# add library
add_library(${PROJECT_NAME}
  src/environment_core.cpp
  src/raycast_callback.cpp
)

add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES} box2d ${OpenCV_LIBRARIES}
)

# add executable
# add_executable(${PROJECT_NAME}_node
#     src/main.cpp
#     src/environment_wrapper.cpp
#     src/environment_core.cpp
#     src/raycast_callback.cpp
# )

# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME mapf_environment PREFIX "")

# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES} box2d ${OpenCV_LIBRARIES}
# )


# install
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)


# testing
catkin_add_gtest(${PROJECT_NAME}-test test/test_environment_core.cpp)
if(TARGET ${PROJECT_NAME}-test)
  target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
endif()
