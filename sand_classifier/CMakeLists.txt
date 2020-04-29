cmake_minimum_required(VERSION 2.8.3)
project(sand_classifier)

## Find catkin macros and libraries
find_package(catkin REQUIRED
  COMPONENTS
  cv_bridge
  image_transport
  message_filters
  nodelet
  roscpp
  sensor_msgs
)

find_package(OpenCV 2.4 REQUIRED)


## catkin specific configuration
catkin_package()


## Build settings
set(CMAKE_CXX_FLAGS "-std=c++11")


# Compile Yumi's terrain classification package
set(TCLIB "spoc_lite")

include_directories(
  ${TCLIB} ${TCLIB}/ibo ${TCLIB}/ibo/linearsvm
  ${OpenCV_INCLUDE_DIRS}
)

add_library(ibo_util ${TCLIB}/ibo/bco.cpp
                     ${TCLIB}/ibo/bco_mc.cpp
                     ${TCLIB}/ibo/bfo.cpp
                     ${TCLIB}/ibo/buo.cpp
                     ${TCLIB}/ibo/lbp.cpp)
target_link_libraries(ibo_util ${OpenCV_LIBS})

add_library(tc_svm ${TCLIB}/classification.cpp
                   ${TCLIB}/ibo/svm.cpp
                   ${TCLIB}/ibo/libsvm.cpp)
target_link_libraries(tc_svm ${OpenCV_LIBS} ibo_util)

add_library(tc_svm_ibo ${TCLIB}/classification_ibo.cpp
                       ${TCLIB}/ibo.cpp
                       ${TCLIB}/ibo/svm_linear.cpp
                       ${TCLIB}/ibo/linearsvm/linear.cpp
                       ${TCLIB}/ibo/linearsvm/tron.cpp)
target_link_libraries(tc_svm_ibo ${OpenCV_LIBS} ibo_util blas)


# Compile ROS wrappers
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(classifier src/classifier.cpp
                       src/classifier_v1.cpp
                       src/classifier_v2.cpp)
target_link_libraries(classifier tc_svm tc_svm_ibo ${catkin_LIBRARIES})


# Compile ROS nodes/nodelets
add_executable(sand_classifier_node src/sand_classifier_node.cpp)
target_link_libraries(sand_classifier_node classifier ${catkin_LIBRARIES})

add_library(sand_classifier_nodelet src/sand_classifier_nodelet.cpp)
target_link_libraries(sand_classifier_nodelet classifier ${catkin_LIBRARIES})


# Unit test
if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)

  add_rostest_gtest(test_ros_api tests/test_ros_api.test tests/test_ros_api.cpp)
  target_link_libraries(test_ros_api classifier ${catkin_LIBRARIES})

  add_rostest_gtest(test_classifier tests/test_classifier.test tests/test_classifier.cpp)
  target_link_libraries(test_classifier classifier ${catkin_LIBRARIES})
endif()
