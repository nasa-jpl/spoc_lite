/*!
 *  @file   terrain_mapper_node.cpp
 *  @brief  Publish terrain class map
 *  @author Kyon Otsu <otsu@jpl.nasa.gov>
 *  @date   2017-02-24
 */


#include "ortho_projector/terrain_mapper.h"

int main(int ac, char **av)
{
  ros::init(ac, av, ros::this_node::getName());
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  ortho_projector::TerrainMapper mapper(nh, pnh);
  ros::spin();
  return 0;
}

