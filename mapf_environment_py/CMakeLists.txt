cmake_minimum_required(VERSION 3.5)
project(mapf_environment_py)

set(PYTHON_EXECUTABLE "/usr/bin/python3.6")

find_package(catkin REQUIRED COMPONENTS
    pybind11_catkin
    mapf_environment
)

find_package(box2d REQUIRED)
find_package(OpenCV 3.2 REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${box2d_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  /usr/local/include/eigen3
  /usr/local/include/eigen3/unsupported
)

catkin_package(
 CATKIN_DEPENDS pybind11_catkin mapf_environment
 DEPENDS box2d OpenCV
)

pybind_add_module(mapf_env MODULE src/python_wrapper.cpp)
target_link_libraries(mapf_env PUBLIC ${catkin_LIBRARIES} box2d ${OpenCV_LIBRARIES})


## Mark library for installation, executable and python module
install(TARGETS mapf_env
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)
