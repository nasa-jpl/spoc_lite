/*!
 *  @file   ortho_projector.cpp
 *  @brief  project image onto 2D map 
 *  @author Kyon Otsu <otsu@jpl.nasa.gov>
 *  @date   2017-02-24
 */

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <claraty_msgs/FloatGrid.h>

#include "ortho_projector/ortho_projector.h"


namespace ortho_projector
{

  OrthoProjector::OrthoProjector(ros::NodeHandle &nh, ros::NodeHandle &pnh)
      : nh_(nh)
      , pnh_(pnh)
      , it_(nh_)
      , initialized_(false)
      , margin_(4, 0)  // top, right, bottom, left in pixels
  {
    //-- Setup OS publisher/subscriber
    img_sub_     = it_.subscribe("image",       1, &OrthoProjector::image_cb, this);
    info_sub_    = nh_.subscribe("camera_info", 1, &OrthoProjector::info_cb,  this);
    reset_sub_   = nh_.subscribe("reset",       1, &OrthoProjector::reset_cb, this);
    ort_pub_     = it_.advertise("map/image_raw", 1);
    ort_col_pub_ = it_.advertise("map/image_color", 1);
    fgrid_pub_   = nh_.advertise<claraty_msgs::FloatGrid>("map", 1, true);

    //-- Get parameters
    // map
    pnh_.param<double>("length"   , length_   , 50  );  // x length [m]
    pnh_.param<double>("width"    , width_    , 50  );  // y width  [m]
    pnh_.param<double>("cell_size", cell_size_, 0.25);  // edge length [m]
    rows_ = int(std::ceil(width_  / cell_size_));
    cols_ = int(std::ceil(length_ / cell_size_));
    origin_.x = 0;
    origin_.y = 0;

    // tf frames
    pnh_.param<std::string>("map_frame_id"   , map_frame_   , "MAP"  );
    pnh_.param<std::string>("base_frame_id"  , base_frame_  , "RNAV" );
    pnh_.param<std::string>("camera_frame_id", camera_frame_, "NCAML");

    // algorithm
    pnh_.param<std::string>("merge_mode", merge_mode_, "LOT");
    pnh_.param<int>("margin_top",    margin_[0], margin_[0]);
    pnh_.param<int>("margin_right",  margin_[1], margin_[1]);
    pnh_.param<int>("margin_bottom", margin_[2], margin_[2]);
    pnh_.param<int>("margin_left",   margin_[3], margin_[3]);
    for (size_t i = 0; i < margin_.size(); ++i)
      margin_[i] = std::max(margin_[i], 0);


    //-- Precompute map scaling homography
    H_p2m = cv::Mat::eye(3, 3, CV_32F);
    //H_p2m.at<float>(1, 2) = 0.5 * width_;
    H_p2m.at<float>(2, 2) = cell_size_;
  }


  OrthoProjector::~OrthoProjector()
  {
  }


  void OrthoProjector::info_cb(const sensor_msgs::CameraInfoConstPtr &msg)
  {
    init_homography_from_camera_info(msg);
  }


  void OrthoProjector::image_cb(const sensor_msgs::ImageConstPtr &msg)
  {
    //-- Check initialization
    if (!initialized_) init_map_from_message(msg);
    if (!H_b2i.data) return;

    //-- Convert image format
    cv::Mat img;
    try
    {
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, type_string_);
      img = cv_ptr->image;
    }
    catch (cv_bridge::Exception &e)
    {
      ROS_ERROR("Could not convert the image format");
      return;
    }

    //-- Ortho-projection
    cv::Mat ort_img;
    try
    {
      project_to_plane(img, ort_img, msg->header.stamp);
    }
    catch (...)
    {
      ROS_ERROR("Failed to add image");
      return;
    }
    merge(ort_img, merge_mode_);

    //-- Publish orthoprojected image
    if (ort_pub_.getNumSubscribers() > 0)
    {
        cv_bridge::CvImagePtr ort_bridge(new cv_bridge::CvImage);
        ort_bridge->header.stamp    = msg->header.stamp;
        ort_bridge->header.frame_id = map_frame_;
        ort_bridge->encoding        = type_string_;
        ort_bridge->image           = ort_;
        ort_pub_.publish(ort_bridge->toImageMsg());
    }

    //-- Colorlized orthoprojected image if it has single channels
    if (ort_col_pub_.getNumSubscribers() > 0)
    {
        if (type_ == CV_8UC1 || type_ == CV_32FC1)
        {
          cv::Mat ort_col;
          colorize(ort_, ort_col, 0.0, 1.0, cv::COLORMAP_JET, 0.5);

          cv_bridge::CvImagePtr ort_bridge(new cv_bridge::CvImage);
          ort_bridge->header.stamp    = msg->header.stamp;
          ort_bridge->header.frame_id = map_frame_;
          ort_bridge->encoding        = "bgr8";
          ort_bridge->image           = ort_col;
          ort_col_pub_.publish(ort_bridge->toImageMsg());
        }
        else
        {
          cv_bridge::CvImagePtr ort_bridge(new cv_bridge::CvImage);
          ort_bridge->header.stamp    = msg->header.stamp;
          ort_bridge->header.frame_id = map_frame_;
          ort_bridge->encoding        = type_string_;
          ort_bridge->image           = ort_;
          ort_pub_.publish(ort_bridge->toImageMsg());
          ort_col_pub_.publish(ort_bridge->toImageMsg());
        }
    }

    //-- Publish map
    if (fgrid_pub_.getNumSubscribers() > 0)
    {
      if (type_ == CV_32FC1)
      {
        claraty_msgs::FloatGrid::Ptr fgrid(new claraty_msgs::FloatGrid);
        fgrid->header.stamp       = msg->header.stamp;
        fgrid->header.frame_id    = map_frame_;
        fgrid->info.map_load_time = fgrid->header.stamp;
        fgrid->info.resolution    = cell_size_;
        fgrid->info.width         = cols_;
        fgrid->info.height        = rows_;
        fgrid->info.origin.position.x = origin_.x;
        fgrid->info.origin.position.y = origin_.y;
        fgrid->info.origin.position.z = 0;
        fgrid->info.origin.orientation.w = 1;
        fgrid->data.resize(cols_ * rows_);
        for (int i = 0; i < rows_; ++i)
        {
          for (int j = 0; j < cols_; ++j)
          {
            fgrid->data[i * rows_ + j] = ort_.at<float>(i, j);
          }
        }
        fgrid_pub_.publish(fgrid);
      }
    }
  } 


  void OrthoProjector::reset_cb(const std_msgs::EmptyConstPtr &msg)
  {
    ROS_INFO("Reinitializing the map");
    ort_ = cv::Mat::zeros(rows_, cols_, type_);
  }


  bool OrthoProjector::init_map_from_message(const sensor_msgs::ImageConstPtr &msg)
  {
    if (sensor_msgs::image_encodings::isMono(msg->encoding))
    {
      type_        = CV_8UC1;
      type_string_ = "mono8";
    }
    else if (sensor_msgs::image_encodings::isColor(msg->encoding))
    {
      type_        = CV_8UC3;
      type_string_ = "bgr8";
    }
    else if (msg->encoding == "TYPE_32FC1" || msg->encoding == "TYPE_64FC1" ||
             msg->encoding == "32FC1"      || msg->encoding == "64FC1"     )
    {
      type_        = CV_32FC1;
      type_string_ = "32FC1";
    }
    else
    {
      ROS_ERROR("Unsupported image format: %s", msg->encoding.c_str());
      return false;
    }
    ort_ = cv::Mat::zeros(rows_, cols_, type_);
    initialized_ = true;

    ROS_INFO("Map is initialized rows=%d cols=%d type=%s",
             rows_, cols_, type_string_.c_str());
    return true;
  }


  void OrthoProjector::init_homography_from_camera_info(
      const sensor_msgs::CameraInfoConstPtr &msg)
  {
    if (H_b2i.data) return;

    //-- Get base-to-camera transform from TF
    cv::Mat T_b2c;
    try
    {
      T_b2c = resolve_tf(base_frame_, camera_frame_);
    }
    catch (...)
    {
      return;
    }

    //-- Get projection matrix from image
    cv::Mat P_i2c(3, 4, CV_32F);
    for (int i = 0; i < P_i2c.rows; ++i)
    {
      for (int j = 0; j < P_i2c.cols; ++j)
      {
        P_i2c.at<float>(i, j) = msg->P[i * P_i2c.cols + j];
      }
    }
    cv::Mat P_i2b = P_i2c * T_b2c.inv();

    //-- Compute homography matrix
    cv::Mat H_i2b = (cv::Mat_<float>(3, 3) <<
        P_i2b.at<float>(0, 0), P_i2b.at<float>(0, 1), P_i2b.at<float>(0, 3), 
        P_i2b.at<float>(1, 0), P_i2b.at<float>(1, 1), P_i2b.at<float>(1, 3), 
        P_i2b.at<float>(2, 0), P_i2b.at<float>(2, 1), P_i2b.at<float>(2, 3));
    H_b2i = H_i2b.inv();

    // normalize
    for (int i = 0; i < H_b2i.rows; ++i)
    {
      for (int j = 0; j < H_b2i.cols; ++j)
      {
        H_b2i.at<float>(i, j) /= H_b2i.at<float>(2, 2);
      }
    }
  }


  void OrthoProjector::project_to_plane(const cv::Mat &img, cv::Mat &ort, ros::Time t)
  {
    //-- Get rover pose and convert to 2D transform
    cv::Mat T_m2b = resolve_tf(map_frame_, base_frame_, t);

    //-- Extract elements and compose homography
    cv::Mat H_m2b = (cv::Mat_<float>(3, 3) <<
        T_m2b.at<float>(0, 0), T_m2b.at<float>(0, 1), T_m2b.at<float>(0, 3), 
        T_m2b.at<float>(1, 0), T_m2b.at<float>(1, 1), T_m2b.at<float>(1, 3), 
        T_m2b.at<float>(3, 0), T_m2b.at<float>(3, 1), T_m2b.at<float>(3, 3));

    //-- Compute end-to-end homography
    cv::Mat H_p2i = H_p2m * H_m2b * H_b2i;

    //-- Transform
    cv::Mat mask(img.size(), CV_8U, cv::Scalar(0));
    cv::rectangle(mask, cv::Point(margin_[3], margin_[0]), 
                  cv::Point(img.cols - margin_[1] - 1, img.rows - margin_[2] - 1),
                  255, -1);
    cv::Mat img_masked(img.size(), img.type(), cv::Scalar(0));
    img.copyTo(img_masked, mask);
    cv::warpPerspective(img_masked, ort, H_p2i, ort_.size(), cv::INTER_LINEAR);
  }


  void OrthoProjector::merge(const cv::Mat &src, const std::string &mode)
  {
    if (mode == "FOT")
    {
      src.copyTo(ort_, ort_ == 0);
    }
    else if (mode == "LOT")
    {
      src.copyTo(ort_, src > 0);
    }
    else if (mode == "MAX")
    {
      src.copyTo(ort_, src > ort_);
    }
    else if (mode == "MIN")
    {
      src.copyTo(ort_, ort_ == 0);
      src.copyTo(ort_, src < ort_ & src > 0);
    }
    else
    {
      ROS_ERROR("Merge mode %s is not supported", mode.c_str());
    }
  }


  cv::Mat OrthoProjector::resolve_tf(const std::string &target, 
                                     const std::string &source, 
                                     const ros::Time   time)
  {
    tf::StampedTransform tfm;
    try
    {
      listener_.waitForTransform(target, source, time, ros::Duration(1.0));
      listener_.lookupTransform(target, source, time, tfm);
    }
    catch (tf::TransformException &e)
    {
      ROS_ERROR("Failed to get transformation from %s to %s", target.c_str(), source.c_str());
      throw e;
    }

    cv::Mat T = cv::Mat::eye(4, 4, CV_32F);
    T.at<float>(0, 3) = tfm.getOrigin().x();
    T.at<float>(1, 3) = tfm.getOrigin().y();
    T.at<float>(2, 3) = tfm.getOrigin().z();

    tf::Matrix3x3 m = tfm.getBasis();
    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        T.at<float>(i, j) = m[i][j];
      }
    }

    return T;
  }


  void OrthoProjector::colorize(const cv::Mat &src, cv::Mat &dst, 
      const double min_v, const double max_v, const int colormap, const double bgcolor)
  {
    cv::Mat src_scaled;
    double scale = 255.0 / (max_v - min_v);
    double offset = -255.0 * min_v / (max_v - min_v);
    src.convertTo(src_scaled, CV_8U, scale, offset);
    cv::applyColorMap(src_scaled, dst, colormap);
    dst.setTo(0, src == bgcolor);
  }


} // namespace

