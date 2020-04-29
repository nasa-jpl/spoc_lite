/*!
 *  @file   ortho_projector_node.cpp
 *  @brief  project image onto 2D map 
 *  @author Kyon Otsu <otsu@jpl.nasa.gov>
 *  @date   2017-02-24
 */


#include "ortho_projector/ortho_projector.h"

int main(int ac, char **av)
{
  ros::init(ac, av, ros::this_node::getName());
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  ortho_projector::OrthoProjector projector(nh, pnh);
  ros::spin();
  return 0;
}

