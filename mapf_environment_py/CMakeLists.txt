cmake_minimum_required(VERSION 3.5)
project(mapf_environment_py)

set(PYTHON_EXECUTABLE "/usr/bin/python3.6")

find_package(catkin REQUIRED COMPONENTS
    pybind11_catkin
    mapf_environment
)

include_directories(${catkin_INCLUDE_DIRS})

catkin_package(
 CATKIN_DEPENDS pybind11_catkin mapf_environment
)

pybind_add_module(mapf_env MODULE src/python_wrapper.cpp)
target_link_libraries(mapf_env PUBLIC ${catkin_LIBRARIES})


## Mark library for installation, executable and python module
install(TARGETS mapf_env
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)
