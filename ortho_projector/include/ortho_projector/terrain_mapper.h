/*!
  @file    terrain_mapper.h
  @brief   Publish terrain class map containing hazard probability
  @author  Kyohei Otsu <otsu@jpl.nasa.gov>
  @date    2017-09-07
*/

#ifndef _TERRAIN_MAPPER_H_
#define _TERRAIN_MAPPER_H_

#include <ros/ros.h>
#include <claraty_msgs/FloatGrid.h>
#include <claraty_msgs/TerrainClassMap.h>

namespace ortho_projector
{
  class TerrainMapper
  {
    public:
      TerrainMapper(ros::NodeHandle &nh, ros::NodeHandle &pnh);
      ~TerrainMapper();

    protected:
      void prob_cb(const claraty_msgs::FloatGridConstPtr &msg);
      
      ros::NodeHandle nh_, pnh_;
      ros::Publisher  map_pub_;
      ros::Publisher  prob_pub_;
      ros::Subscriber prob_sub_;

      double prob_thresh_;
  };
}  // namespace ortho_projector

#endif  // _TERRAIN_MAPPER_H_
