/*!
 *  @file   ortho_projector_nodelet.cpp
 *  @brief  project image onto 2D map 
 *  @author Kyon Otsu <otsu@jpl.nasa.gov>
 *  @date   2017-03-01
 */

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <boost/shared_ptr.hpp>
#include "ortho_projector/ortho_projector.h"

namespace ortho_projector
{
  class OrthoProjectorNodelet : public nodelet::Nodelet
  {
    public:
      void onInit()
      {
        ros::NodeHandle &nh  = getNodeHandle();
        ros::NodeHandle &pnh = getPrivateNodeHandle();
        classifier.reset(new OrthoProjector(nh, pnh));
      }

    protected:
      boost::shared_ptr<OrthoProjector> classifier;
  };

}  // namespace


#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(ortho_projector::OrthoProjectorNodelet, nodelet::Nodelet)
