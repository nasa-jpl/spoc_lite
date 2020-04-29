cmake_minimum_required(VERSION 2.8.3)
project(ortho_projector)

## Find catkin macros and libraries
find_package(catkin REQUIRED
  COMPONENTS
  cv_bridge
  claraty_msgs
  image_transport
  nav_msgs
  nodelet
  roscpp
  sensor_msgs
  tf
)

## catkin specific configuration
catkin_package()

## Build
set(CMAKE_CXX_FLAGS "-std=c++11")

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(ortho_projector src/ortho_projector.cpp)
target_link_libraries(ortho_projector ${catkin_LIBRARIES})
add_dependencies(ortho_projector ${catkin_EXPORTED_TARGETS})

add_executable(ortho_projector_node src/ortho_projector_node.cpp)
target_link_libraries(ortho_projector_node ortho_projector ${catkin_LIBRARIES})

add_library(ortho_projector_nodelet src/ortho_projector_nodelet.cpp)
target_link_libraries(ortho_projector_nodelet ortho_projector ${catkin_LIBRARIES})

add_library(terrain_mapper src/terrain_mapper.cpp)
target_link_libraries(terrain_mapper ${catkin_LIBRARIES})
add_dependencies(terrain_mapper ${catkin_EXPORTED_TARGETS})

add_executable(terrain_mapper_node src/terrain_mapper_node.cpp)
target_link_libraries(terrain_mapper_node terrain_mapper ${catkin_LIBRARIES})

add_library(terrain_mapper_nodelet src/terrain_mapper_nodelet.cpp)
target_link_libraries(terrain_mapper_nodelet terrain_mapper ${catkin_LIBRARIES})
