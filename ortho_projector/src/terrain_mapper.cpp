/*!
  @file    terrain_mapper.cpp
  @brief   Publish terrain class containing hazard probability
  @author  Kyohei Otsu <otsu@jpl.nasa.gov>
  @date    2017-09-07
*/

#include "ortho_projector/terrain_mapper.h"
#include <nav_msgs/OccupancyGrid.h>

namespace ortho_projector
{
  TerrainMapper::TerrainMapper(ros::NodeHandle &nh, ros::NodeHandle &pnh)
      : nh_(nh)
      , pnh_(pnh)
  {
    map_pub_  = nh_.advertise<claraty_msgs::TerrainClassMap>("map",          1, true);
    prob_pub_ = nh_.advertise<nav_msgs::OccupancyGrid      >("map_int", 1, true);
    prob_sub_ = nh_.subscribe("map_float", 1, &TerrainMapper::prob_cb, this);
    pnh_.param<double>("threshold", prob_thresh_, 0.4);
  }


  TerrainMapper::~TerrainMapper()
  {
  }


  void TerrainMapper::prob_cb(const claraty_msgs::FloatGridConstPtr &msg)
  {
    // Publish TerrainClassMap message
    if (map_pub_.getNumSubscribers() > 0)
    {
      claraty_msgs::TerrainClassMapPtr tc_map(new claraty_msgs::TerrainClassMap);
      tc_map->header = msg->header;
      tc_map->info = msg->info;
      tc_map->cells.resize(msg->data.size());
      auto cell = tc_map->cells.begin();
      auto prob = msg->data.begin();
      for (/* No initialization*/;
           cell != tc_map->cells.end(), prob != msg->data.end();
           ++cell, ++prob)
      {
        cell->label = (*prob > prob_thresh_)? (unsigned char)cell->SAND:
                                              (unsigned char)cell->SOIL;
        cell->probabilities.resize(2);
        cell->probabilities[cell->SOIL] = 1.0 - *prob;
        cell->probabilities[cell->SAND] = *prob;
        cell->probability = cell->probabilities[cell->SAND];
      }
      map_pub_.publish(tc_map);
    }

    // Publish OccupancyGrid message for debugging
    if (prob_pub_.getNumSubscribers() > 0)
    {
      nav_msgs::OccupancyGridPtr prob_map(new nav_msgs::OccupancyGrid);
      prob_map->header = msg->header;
      prob_map->info = msg->info;
      prob_map->data.resize(msg->data.size());
      auto dst = prob_map->data.begin();
      auto src = msg->data.begin();
      for (/* No initialization*/;
           dst != prob_map->data.end(), src != msg->data.end();
           ++dst, ++src)
      {
        *dst = 100 * *src;
      }
      prob_pub_.publish(prob_map);
    }
  }

}  // namespace ortho_projector
