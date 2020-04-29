/*!
 *  @file   classifier.h
 *  @brief  Base classifier class implementation
 *  @author Kyon Otsu <otsu@jpl.nasa.gov>
 *  @date   2017-02-24
 */

#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <vector>

#include <opencv2/core/core.hpp>


namespace spoc
{

  /*! 
   * Base classifier class. 
   *
   * This class handles ROS message communication. The algorithms
   * must be implemented in classify() function in child classes.
   */
  class Classifier
  {
    public:
      /*! Constructor with ROS NodeHandle */
      Classifier(ros::NodeHandle nh, ros::NodeHandle pnh);

      /*! Destructor */
      ~Classifier();

    protected:
      /*! Subscribe color image, and publish results */
      virtual void image_cb(const sensor_msgs::ImageConstPtr &msg);

      /*! Subscribe color/disparity images. Results are filtered by distance */
      virtual void image_disparity_cb(const sensor_msgs::ImageConstPtr &img_msg,
                                      const sensor_msgs::ImageConstPtr &dst_msg);

      /*! 
       * Main classification implementation. Implement this in chid class. 
       * @param src_bgr  input color image
       * @param dst_labl output label image
       * @param dst_prob output probability image
       */
      virtual bool classify(const cv::Mat &src_bgr,
                                  cv::Mat &dst_labl,
                                  cv::Mat &dst_prob) = 0;

      /*! 
       * Superimpose color on input image based on value.
       * @param value     value in [0.0, 1.0]
       * @param dst       output image
       * @param base      base color image
       * @param mask      mask region to be superimposed
       * @param colormap  colormap specifier. See OpenCV's COLOR_* params
       */
      virtual void image_overlay(const cv::Mat &value,
                                       cv::Mat &dst,
                                 const cv::Mat &base,
                                 const cv::Mat &mask,
                                 int colormap=2);

      /*! @{ ROS node handles */
      ros::NodeHandle nh;
      ros::NodeHandle pnh;
      image_transport::ImageTransport it;
      /*! @} */

      /*! @{ ROS publishers/subscribers */
      image_transport::Subscriber img_sub;
      image_transport::SubscriberFilter img_sub_filter;
      image_transport::SubscriberFilter dsp_sub_filter;
      image_transport::Publisher  ovly_pub;
      image_transport::Publisher  labl_pub;
      image_transport::Publisher  prob_pub;
      /*! @} */

      /*! @{ Message synchronizers */
      typedef message_filters::sync_policies::ExactTime<
          sensor_msgs::Image,
          sensor_msgs::Image
          > SyncPolicy;
      message_filters::Synchronizer<SyncPolicy> sync;
      /*! @} */

      /*! Label ID of sand class */
      int id_sand_class;

      /*! Probability threshold */
      double threshold;

      /*! Disparity threshold for filtering */
      double min_disparity;
  };

}  // namespace spoc

#endif  // _CLASSIFIER_H_

