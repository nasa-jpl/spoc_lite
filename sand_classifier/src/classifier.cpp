/*!
 *  @file   sand_classifier.cpp
 *  @brief  Base classifier class implementation
 *  @author Kyon Otsu <otsu@jpl.nasa.gov>
 *  @date   2017-02-24
 */

#include <iterator>

#include <ros/package.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

#include "sand_classifier/classifier.h"


namespace spoc
{
  Classifier::Classifier(ros::NodeHandle _nh, ros::NodeHandle _pnh)
      : nh(_nh)
      , pnh(_pnh)
      , it(nh)
      , img_sub_filter(it, "image",     1)
      , dsp_sub_filter(it, "disparity", 1)
      , sync(SyncPolicy(10), img_sub_filter, dsp_sub_filter)
  {
    //-- Load parameters
    pnh.param<double>("threshold",     threshold,     0.8);
    pnh.param<double>("min_disparity", min_disparity, -1 );
    min_disparity *= 256;  // stereo_pipline computes 256x disparity...

    pnh.param<int>("sand_id", id_sand_class, 1);

    //-- Setup publishers/subscribers
    img_sub  = it.subscribe("image", 1, &Classifier::image_cb, this);
    //sync.registerCallback(boost::bind(&Classifier::image_disparity_cb, this, _1, _2));

    ovly_pub = it.advertise("image_overlay", 1);
    labl_pub = it.advertise("image_label", 1);
    prob_pub = it.advertise("image_probability", 1);
  }


  Classifier::~Classifier()
  {
  }


  void Classifier::image_cb(const sensor_msgs::ImageConstPtr &msg)
  {
    cv::Mat bgr, label, probability;

    //-- Convert image format 
    if (!sensor_msgs::image_encodings::isColor(msg->encoding))
    {
      ROS_ERROR("Unsupported image encoding: %s", msg->encoding.c_str());
      return;
    }

    try
    {
      cv_bridge::CvImagePtr cv_ptr = 
        cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      bgr = cv_ptr->image;
    } 
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("Could not convert from '%s' to 'bgr8'.", 
                msg->encoding.c_str());
      return;
    }


    //-- Classify image
    if( !classify(bgr, label, probability) ) return;


    //-- Publish messages
    cv_bridge::CvImagePtr ovly_img(new cv_bridge::CvImage);  // Superimposed image
    cv_bridge::CvImagePtr labl_img(new cv_bridge::CvImage);  // Label ID image
    cv_bridge::CvImagePtr prob_img(new cv_bridge::CvImage);  // Probability image

    cv::Mat overlay;
    image_overlay(probability, overlay, bgr, label, cv::COLORMAP_JET);
    ovly_img->header   = msg->header;
    ovly_img->image    = overlay;
    ovly_img->encoding = sensor_msgs::image_encodings::BGR8;
    ovly_pub.publish(ovly_img->toImageMsg());

    labl_img->header   = msg->header;
    labl_img->image    = label;
    labl_img->encoding = sensor_msgs::image_encodings::MONO8;
    labl_pub.publish(labl_img->toImageMsg());

    prob_img->header   = msg->header;
    prob_img->image    = probability;
    prob_img->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    prob_pub.publish(prob_img->toImageMsg());
  }


  void Classifier::image_disparity_cb(const sensor_msgs::ImageConstPtr &img_msg,
                                      const sensor_msgs::ImageConstPtr &dsp_msg)
  {
    cv::Mat bgr, dsp, label, probability;

    fprintf( stderr, "cb debug 1\n" );

    //-- Convert image format 
    try
    {
      cv_bridge::CvImagePtr cv_ptr;
      cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
      bgr = cv_ptr->image;

      cv_ptr = cv_bridge::toCvCopy(dsp_msg, sensor_msgs::image_encodings::MONO16);
      dsp = cv_ptr->image;
    } 
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("Could not convert from '%s' to 'bgr8' or", img_msg->encoding.c_str());
      ROS_ERROR("Could not convert from '%s' to 'MONO16'.", dsp_msg->encoding.c_str());
    }

    //-- Classify image
    if( !classify(bgr, label, probability) ) return;

    if (min_disparity > 0)
    {
      label      .setTo(0, dsp <= min_disparity);
      probability.setTo(0, dsp <= min_disparity);
      label      .setTo(0, dsp >= 65500);
      probability.setTo(0, dsp >= 65500);
    }

    //-- Publish messages
    cv_bridge::CvImagePtr ovly_img(new cv_bridge::CvImage);  // Superimposed image
    cv_bridge::CvImagePtr labl_img(new cv_bridge::CvImage);  // Label ID image
    cv_bridge::CvImagePtr prob_img(new cv_bridge::CvImage);  // Probability image

    cv::Mat overlay;
    image_overlay(probability, overlay, bgr, label, cv::COLORMAP_JET);
    ovly_img->header   = img_msg->header;
    ovly_img->image    = overlay;
    ovly_img->encoding = sensor_msgs::image_encodings::BGR8;
    ovly_pub.publish(ovly_img->toImageMsg());

    labl_img->header   = img_msg->header;
    labl_img->image    = label;
    labl_img->encoding = sensor_msgs::image_encodings::MONO8;
    labl_pub.publish(labl_img->toImageMsg());

    prob_img->header   = img_msg->header;
    prob_img->image    = probability;
    prob_img->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    prob_pub.publish(prob_img->toImageMsg());
  }


  void Classifier::image_overlay(const cv::Mat &value,
                                       cv::Mat &dst,
                                 const cv::Mat &base,
                                 const cv::Mat &mask,
                                 const int      colormap)
  {
    //-- Change format
    cv::Mat value_8U;
    value.convertTo(value_8U, CV_8U, 255.0);

    //-- Apply color map
    cv::Mat value_colored;
    cv::applyColorMap(value_8U, value_colored, colormap);
    value_colored.setTo(0, ~mask);

    //-- Thresholding (tmp)
    value_colored.setTo(0, value < threshold);

    //-- Alpha blending
    cv::addWeighted(base, 0.8, value_colored, 0.5, 0.0, dst);
  }

}  // namespace spoc

