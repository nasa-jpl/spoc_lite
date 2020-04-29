/*!
 *  @file   terrain_mapper_nodelet.cpp
 *  @brief  Map hazard probability
 *  @author Kyon Otsu <otsu@jpl.nasa.gov>
 *  @date   2017-03-01
 */

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <boost/shared_ptr.hpp>
#include "ortho_projector/terrain_mapper.h"

namespace ortho_projector
{
  class TerrainMapperNodelet: public nodelet::Nodelet
  {
    public:
      void onInit()
      {
        ros::NodeHandle &nh  = getNodeHandle();
        ros::NodeHandle &pnh = getPrivateNodeHandle();
        mapper.reset(new TerrainMapper(nh, pnh));
      }

    protected:
      boost::shared_ptr<TerrainMapper> mapper;
  };

}  // namespace


#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(ortho_projector::TerrainMapperNodelet, nodelet::Nodelet)
