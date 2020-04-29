/*!
  @file    test_ros_api.cpp
  @brief   Unittest for ROS callbacks
  @author  Kyohei Otsu <otsu@jpl.nasa.gov>
  @date    2018-11-30
*/

#include <gtest/gtest.h>
#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>

// Hack: Workaround to test protected member methods
#define protected public
#include "sand_classifier/classifier_v2.h"
#undef protected


// -----------------------------------------------------------------------------
//   Fixture
// -----------------------------------------------------------------------------

class ROSAPITest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
      // ROS node handles
      ros::NodeHandle nh, pnh("~");

      // Instantiate classifier
      classifier_.reset(new spoc::ClassifierSVMIbo(nh, pnh));

      // Instantiate ROS subscribers
      auto ovly_f = boost::bind(&ROSAPITest::image_cb, this, _1, boost::ref(ovly_img_));
      ovly_sub_ = nh.subscribe<sensor_msgs::Image>("image_overlay",     1, ovly_f);
      auto labl_f = boost::bind(&ROSAPITest::image_cb, this, _1, boost::ref(labl_img_));
      labl_sub_ = nh.subscribe<sensor_msgs::Image>("image_label",       1, labl_f);
      auto prob_f = boost::bind(&ROSAPITest::image_cb, this, _1, boost::ref(prob_img_));
      prob_sub_ = nh.subscribe<sensor_msgs::Image>("image_probability", 1, prob_f);
    }

    void image_cb(const sensor_msgs::ImageConstPtr &msg, cv::Mat &img)
    {
      img = cv_bridge::toCvShare(msg)->image;
    }


    std::shared_ptr<spoc::Classifier> classifier_;

    ros::Subscriber ovly_sub_, labl_sub_, prob_sub_;
    cv::Mat         ovly_img_, labl_img_, prob_img_;
};


// -----------------------------------------------------------------------------
//   Test cases
// -----------------------------------------------------------------------------

TEST_F(ROSAPITest, callbackOnImage8UC3Succeeds)
{
  cv_bridge::CvImagePtr cv_img(new cv_bridge::CvImage);
  cv_img->image = cv::Mat::zeros(480, 640, CV_8UC3);
  cv_img->encoding = sensor_msgs::image_encodings::BGR8;
  cv_img->header.stamp = ros::Time::now();
  cv_img->header.frame_id = "camera_frame";
  classifier_->image_cb(cv_img->toImageMsg());

  // Spin once to invoke callback
  ros::spinOnce();

  EXPECT_EQ(480, ovly_img_.rows);
  EXPECT_EQ(640, ovly_img_.cols);
  EXPECT_EQ(CV_8UC3, ovly_img_.type());
  EXPECT_EQ(480, labl_img_.rows);
  EXPECT_EQ(640, labl_img_.cols);
  EXPECT_EQ(CV_8UC1, labl_img_.type());
  EXPECT_EQ(480, prob_img_.rows);
  EXPECT_EQ(640, prob_img_.cols);
  EXPECT_EQ(CV_32FC1, prob_img_.type());
}

TEST_F(ROSAPITest, callbackOnImage8UC1Fails)
{
  cv_bridge::CvImagePtr cv_img(new cv_bridge::CvImage);
  cv_img->image = cv::Mat::zeros(480, 640, CV_8UC1);
  cv_img->encoding = sensor_msgs::image_encodings::MONO8;
  cv_img->header.stamp = ros::Time::now();
  cv_img->header.frame_id = "camera_frame";
  classifier_->image_cb(cv_img->toImageMsg());

  // Spin once to invoke callback
  ros::spinOnce();

  EXPECT_EQ(nullptr, ovly_img_.data);
  EXPECT_EQ(nullptr, labl_img_.data);
  EXPECT_EQ(nullptr, prob_img_.data);
}

TEST_F(ROSAPITest, callbackOnImage32FC1Fails)
{
  cv_bridge::CvImagePtr cv_img(new cv_bridge::CvImage);
  cv_img->image = cv::Mat::zeros(480, 640, CV_32FC1);
  cv_img->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  cv_img->header.stamp = ros::Time::now();
  cv_img->header.frame_id = "camera_frame";
  classifier_->image_cb(cv_img->toImageMsg());

  // Spin once to invoke callback
  ros::spinOnce();

  EXPECT_EQ(nullptr, ovly_img_.data);
  EXPECT_EQ(nullptr, labl_img_.data);
  EXPECT_EQ(nullptr, prob_img_.data);
}

TEST_F(ROSAPITest, callbackOnEmptyImageFails)
{
  cv_bridge::CvImagePtr cv_img(new cv_bridge::CvImage);
  cv_img->image = cv::Mat();
  classifier_->image_cb(cv_img->toImageMsg());

  // Spin once to invoke callback
  ros::spinOnce();

  EXPECT_EQ(nullptr, ovly_img_.data);
  EXPECT_EQ(nullptr, labl_img_.data);
  EXPECT_EQ(nullptr, prob_img_.data);
}

// -----------------------------------------------------------------------------
//   Run all the tests
// -----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  ros::init(argc, argv, "test_ros_api");
  return RUN_ALL_TESTS();
}
