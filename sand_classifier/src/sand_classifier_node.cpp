/*!
 *  @file   sand_classifier_node.cc
 *  @brief  ROS node for visually classify sand from any other else
 *  @author Kyon Otsu <otsu@jpl.nasa.gov>
 *  @date   2017-02-24
 *
 *  Set classifier_type param to specify a classifier.
 *    1 : IBO SVM
 *    2 : IBO Linear SVM (default)
 */

#include <ros/ros.h>
#include "sand_classifier/classifier.h"
#include "sand_classifier/classifier_v1.h"
#include "sand_classifier/classifier_v2.h"


int main(int ac, char **av)
{
  ros::init(ac, av, ros::this_node::getName());
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  std::shared_ptr<spoc::Classifier> classifier;

  // Choose classifier
  int version = -1;
  pnh.param<int>("classifier_version", version, 2);
  switch (version)
  {
    case 1:
      ROS_INFO("Initialize SandClassifierSVM (v1)");
      classifier.reset(new spoc::ClassifierSVM(nh, pnh));
      break;
    case 2: default:
      ROS_INFO("Initialize SandClassifierLinearSVM (v2)");
      classifier.reset(new spoc::ClassifierSVMIbo(nh, pnh));
      break;
  }

  ros::spin();
  return 0;
}
