/*!
  @file    test_ros_api.cpp
  @brief   Unittest for ROS callbacks
  @author  Kyohei Otsu <otsu@jpl.nasa.gov>
  @date    2018-11-30
*/

#include <gtest/gtest.h>
#include <ros/ros.h>
#include <opencv2/core/core.hpp>

// Hack: Workaround to test protected member methods
#define protected public
#include "sand_classifier/classifier_v2.h"
#undef protected


// -----------------------------------------------------------------------------
//   Fixture
// -----------------------------------------------------------------------------

class ClassifierTest : public ::testing::Test
{
  protected:
    /** Execute before every test case */
    void SetUp() override
    {
      ros::NodeHandle nh, pnh("~");
      classifier_.reset(new spoc::ClassifierSVMIbo(nh, pnh));
    }

    /** Execute after every test case */
    void TearDown() override
    {
    }


    std::shared_ptr<spoc::Classifier> classifier_;
};



// -----------------------------------------------------------------------------
//   Test cases
// -----------------------------------------------------------------------------

TEST_F(ClassifierTest, classifyOnColorImageSucceeds)
{
  cv::Mat bgr = cv::Mat::zeros(480, 640, CV_8UC3);
  cv::Mat label, probability;

  bool classify_flg = classifier_->classify(bgr, label, probability);

  // Output image size should be identical to the input size
  EXPECT_EQ(true, classify_flg);
  EXPECT_EQ(bgr.rows, label.rows);
  EXPECT_EQ(bgr.cols, label.cols);
  EXPECT_EQ(bgr.rows, probability.rows);
  EXPECT_EQ(bgr.cols, probability.cols);

  // Type check
  EXPECT_EQ(CV_8UC1,  label.type());
  EXPECT_EQ(CV_32FC1, probability.type());

  // Value check
  double min, max;
  cv::minMaxLoc(probability, &min, &max);
  EXPECT_GE(min, 0.0);
  EXPECT_LE(max, 1.0);
}

TEST_F(ClassifierTest, classifyOnLargeImageSucceeds)
{
  cv::Mat bgr = cv::Mat::zeros(768, 1024, CV_8UC3);
  cv::Mat label, probability;

  bool classify_flg = classifier_->classify(bgr, label, probability);

  EXPECT_EQ(true, classify_flg);
  EXPECT_EQ(bgr.rows, label.rows);
  EXPECT_EQ(bgr.cols, label.cols);
  EXPECT_EQ(bgr.rows, probability.rows);
  EXPECT_EQ(bgr.cols, probability.cols);
}

TEST_F(ClassifierTest, classifyOnGrayscaleImageFails)
{
  cv::Mat bgr = cv::Mat::zeros(768, 1024, CV_8UC1);
  cv::Mat label, probability;

  bool classify_flg = classifier_->classify(bgr, label, probability);

  EXPECT_EQ(false, classify_flg);
}

TEST_F(ClassifierTest, classifyOnEmptyImageFails)
{
  cv::Mat bgr;
  cv::Mat label, probability;

  bool classify_flg = classifier_->classify(bgr, label, probability);

  EXPECT_EQ(false, classify_flg);
}

TEST_F(ClassifierTest, classificationAccuracyAboveThreshold)
{
}

// -----------------------------------------------------------------------------
//   Run all the tests
// -----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  ros::init(argc, argv, "test_classifier");
  return RUN_ALL_TESTS();
}
