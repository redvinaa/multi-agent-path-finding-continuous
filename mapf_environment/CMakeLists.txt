cmake_minimum_required(VERSION 3.5)
project(mapf_environment)

set(CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS
  pybind11_catkin
  roslib
  roslint
  roscpp
)

find_package(OpenCV 4 REQUIRED)
find_package(box2d REQUIRED)


catkin_package(
 INCLUDE_DIRS include
 LIBRARIES mapf_environment mapf_env
 CATKIN_DEPENDS pybind11_catkin roslib roslint roscpp
 DEPENDS box2d OpenCV
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${box2d_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)


# add library
add_library(${PROJECT_NAME}_lib
  src/environment.cpp
  src/raycast_callback.cpp
)

add_dependencies(${PROJECT_NAME}_lib ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_lib PRIVATE ${catkin_LIBRARIES} box2d ${OpenCV_LIBRARIES})
set_target_properties(${PROJECT_NAME}_lib PROPERTIES OUTPUT ${PROJECT_NAME})

## Mark library for installation
install(TARGETS ${PROJECT_NAME}_lib
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

pybind_add_module(mapf_env MODULE src/python_wrapper.cpp)
target_link_libraries(mapf_env PRIVATE ${PROJECT_NAME}_lib ${catkin_LIBRARIES} box2d ${OpenCV_LIBRARIES})
# catkin_install_python(PROGRAMS test/test_python_wrapper.py
#     DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION})

## Mark module for installation
install(TARGETS mapf_env
  DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)


# testing
catkin_add_gtest(${PROJECT_NAME}-test test/test_environment.cpp)
target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME}_lib ${catkin_LIBRARIES} box2d ${OpenCV_LIBRARIES})

catkin_add_nosetests(test/test_python_wrapper.py)

# linting
roslint_cpp()
roslint_python()
roslint_add_test()
