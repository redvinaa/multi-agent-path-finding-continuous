cmake_minimum_required(VERSION 3.0.2)
project(mapf_train_maac)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../libtorch")

find_package(catkin REQUIRED COMPONENTS
  maac_agent
  mapf_environment
  roscpp
  roslint
)

find_package(Torch REQUIRED)

catkin_package(
 INCLUDE_DIRS include ${TORCH_INCLUDE_DIRS}
 LIBRARIES mapf_train_maac
 CATKIN_DEPENDS maac_agent mapf_environment roscpp
 DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/training_loop.cpp
)

add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${TORCH_LIBRARIES}
)

add_executable(${PROJECT_NAME}_node src/run_training.cpp)
set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME train PREFIX "")
add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${TORCH_LIBRARIES}
)

## Mark executables for installation
install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

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

# Mark other files for installation (e.g. launch and bag files, etc.)
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

catkin_add_gtest(${PROJECT_NAME}-test test/test_train.cpp)
target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})

# linting
roslint_cpp()
