/*!
 *  @file   sand_classifier_node.cc
 *  @brief  ROS nodelet for sand classifier
 *  @author Kyon Otsu <otsu@jpl.nasa.gov>
 *  @date   2017-02-24
 *
 *  This nodes use the classifier developed by Yumi Iwashita.
 */

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include "sand_classifier/classifier.h"
#include "sand_classifier/classifier_v1.h"
#include "sand_classifier/classifier_v2.h"

namespace spoc
{
  class SandClassifierNodelet : public nodelet::Nodelet
  {
    public:
      void onInit()
      {
        ros::NodeHandle &nh  = getNodeHandle();
        ros::NodeHandle &pnh = getPrivateNodeHandle();

        // Choose classifier
        int version = -1;
        pnh.param<int>("classifier_version", version, 2);
        switch (version)
        {
          case 1:
            classifier.reset(new spoc::ClassifierSVM(nh, pnh));
            break;
          case 2: default:
            classifier.reset(new spoc::ClassifierSVMIbo(nh, pnh));
            break;
        }
      }

    protected:
      std::shared_ptr<Classifier> classifier;
  };

}  // namespace spoc


#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(spoc::SandClassifierNodelet, nodelet::Nodelet)
