/*!
 *  @file   ortho_projector.h
 *  @brief  project image onto 2D map 
 *  @author Kyon Otsu <otsu@jpl.nasa.gov>
 *  @date   2017-02-24
 */


#ifndef _ORTHO_PROJECTOR_H_
#define _ORTHO_PROJECTOR_H_

#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <image_transport/image_transport.h>
#include <tf/transform_listener.h>

#include <sensor_msgs/Image.h>
#include <std_msgs/Empty.h>

namespace ortho_projector
{

  class OrthoProjector
  {
    public:
      OrthoProjector(ros::NodeHandle &nh, ros::NodeHandle &pnh);
      ~OrthoProjector();

    protected:
      void image_cb(const sensor_msgs::ImageConstPtr &msg);
      void info_cb(const sensor_msgs::CameraInfoConstPtr &msg);
      void reset_cb(const std_msgs::EmptyConstPtr &msg);

      bool init_map_from_message(const sensor_msgs::ImageConstPtr &msg);
      void init_homography_from_camera_info(const sensor_msgs::CameraInfoConstPtr &msg);

      void project_to_plane(const cv::Mat &img, cv::Mat &ort, ros::Time t=ros::Time(0));
      void merge(const cv::Mat &src, const std::string &mode);

      cv::Mat resolve_tf(const std::string &target, 
                         const std::string &source, 
                         const ros::Time   t=ros::Time(0));
      void colorize(const cv::Mat &src, cv::Mat &dst,
                    const double min_v, const double max_v, const int colormap=2,
                    const double bgcolor=-1);

      // ROS node handle, publisher, subscribers 
      ros::NodeHandle nh_;
      ros::NodeHandle pnh_;
      image_transport::ImageTransport it_;

      image_transport::Publisher  ort_pub_;
      image_transport::Publisher  ort_col_pub_;
      image_transport::Subscriber img_sub_;
      ros::Publisher              fgrid_pub_;
      ros::Subscriber             info_sub_;
      ros::Subscriber             reset_sub_;
      tf::TransformListener       listener_;

      // tf frames
      std::string map_frame_;
      std::string base_frame_;
      std::string camera_frame_;

      // orthoprojected image
      cv::Mat ort_;
      int rows_, cols_, type_;
      std::string type_string_;
      cv::Point2f origin_;
      double length_, width_, cell_size_;

      bool initialized_;
      std::string merge_mode_;
      std::vector<int> margin_;

      // camera parameters
      cv::Mat H_b2i;  // base  to image homography
      cv::Mat H_m2b;  // map   to base  homography
      cv::Mat H_p2m;  // pixel to map   homography

  };

}  // namespace

#endif  // _ORTHO_PROJECTOR_H_
