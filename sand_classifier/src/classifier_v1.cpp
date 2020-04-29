/*!
 *  @file    sand_classifier_v1.cpp
 *  @brief   SVM-based sand classifiers 
 *  @author  Yumi Iwashita <iwashita@jpl.nasa.gov>
 *  @author  Kyohei Otsu <otsu@jpl.nasa.gov>
 *  @date    2017-07-14
 *
 *  This classes use the SVM-based classifiers developed by Yumi Iwashita.
 */

#include <ros/ros.h>
#include <ros/package.h>

#include "sand_classifier/classifier_v1.h"

#include "ibo/bco.h"
#include "ibo/bco_mc.h"
#include "ibo/buo.h"
#include "ibo/buo.h"
#include "ibo/lbp.h"
#include "ibo/bfo.h"
#include "svm.h"
#include "libsvm.h"

#include "classification_ibo.h"


// constants  TODO: Remove this 
#define CLASS_NUM 5


using namespace std;
using namespace cv;
using namespace ibo;


namespace spoc
{

  ClassifierSVM::ClassifierSVM(ros::NodeHandle nh, ros::NodeHandle pnh)
      : Classifier(nh, pnh)
      , svm(0)
  {
    //-- Load parameters
    pnh.param<bool>("use_ave",       feature_options.use_ave,       true);
    pnh.param<bool>("use_lbp",       feature_options.use_lbp,       false);
    pnh.param<bool>("use_dct",       feature_options.use_dct,       false);
    pnh.param<bool>("use_intensity", feature_options.use_intensity, false);

    //-- Load trained model
    std::string model_name;
    pnh.param<std::string>("model_name", model_name, "");

    int kernel_type = LINEAR;
    svm.SetParameters(C_SVC, kernel_type, 1.0, (double)10, (double)-40);
    svm.DummyGetMemory();
    if (model_name.find("/") == 0)
    {
      svm.svmLoadModel(model_name.c_str());
    }
    else
    {
      std::string path = ros::package::getPath("sand_classifier")
                         + "/" + model_name;
      svm.svmLoadModel(path.c_str());
    }

    //-- Set LBP parameters
    lbp_parameter.SDIM = ibo::makeELBP(8, lbp_parameter.Flg_LBP);
    lbp_parameter.db = 5;
  }


  ClassifierSVM::~ClassifierSVM()
  {
  }


  bool ClassifierSVM::classify(const cv::Mat &src_bgr, 
			       cv::Mat &dst_labl, 
			       cv::Mat &dst_prob)
  {
    /* Adaptation from img_sub_classificatoin.cpp:image_cb() */

    if(src_bgr.channels() == 1 || src_bgr.empty() == true ){
      return false;
    }


//#define USE_GREYWORLD
#ifdef USE_GREYWORLD
    std::vector<cv::Mat> bgr_planes(3);
    cv::Scalar avg = cv::mean(src_bgr);
    cv::split(src_bgr, bgr_planes);
    bgr_planes[0] *= 128.0 / avg[0];
    bgr_planes[1] *= 128.0 / avg[1];
    bgr_planes[2] *= 128.0 / avg[2];
    cv::merge(bgr_planes, src_bgr);
#endif

    cv::Mat src_mat = src_bgr.clone();

    //-- ADAPTATION FROM HERE
    ibo::mc_image mc_src;
    mc_src.rows = src_mat.rows;
    mc_src.cols = src_mat.cols;
    mc_src.chans = src_mat.channels();
    mc_src.data.resize( mc_src.rows*mc_src.cols*mc_src.chans );
    ibo::mc_copy_Mat_2_image( src_mat, mc_src );  

    //-- extract features
    std::vector<ibo::struct_bfo> test;
    std::vector<ibo::struct_feature_dimension> feature_dim, feature_dim_4_train;
    int feature_window_size = 25;
    double feature_window_overlap_ratio = 0.5;
    tc::setFeature( feature_options, src_mat, mc_src, test, feature_dim, feature_window_size, feature_window_overlap_ratio, lbp_parameter );

    int w_size = floor(feature_window_size / 2);
    if(w_size < 0) w_size = 0;

    std::vector<cv::Mat> prob_mat;// = cv::Mat::zeros( src_mat.rows, src_mat.cols, CV_64FC1 );
    for(int k=0; k<CLASS_NUM; k++){
      cv::Mat tmp_prob_mat;
      tmp_prob_mat = cv::Mat::zeros( src_mat.rows, src_mat.cols, CV_64FC1 );
      prob_mat.push_back( tmp_prob_mat );
    }
    cv::Mat count_mat = cv::Mat::zeros( src_mat.rows, src_mat.cols, CV_64FC1 );

    double probability[CLASS_NUM] = { 0 };
    double *probability_frame = new double [CLASS_NUM * test.size()];
    int estimated_class;
    for(int j=0; j<test.size(); j++){
      svm.svmPredictProbability(test[j], probability);

      int r = test[j].y;
      int c = test[j].x;
      int left   = test[j].left;
      int right  = test[j].right;
      int up     = test[j].up;
      int bottom = test[j].bottom;

      for(int k=0; k<CLASS_NUM; k++){
        for(int sr=up; sr<=bottom; sr++){
          double* _prob_mat = prob_mat[k].ptr<double>(sr);
          double* _count_mat = count_mat.ptr<double>(sr);
          for(int sc=left; sc<=right; sc++){
            _prob_mat[sc] += probability[k];
            if( k == 0 )
              _count_mat[sc] ++;
          }
        }
      }

    }
    if( probability_frame ) delete [] probability_frame;

    // average and estimated class
    cv::Mat estimated_mat = cv::Mat::zeros( src_mat.rows, src_mat.cols, CV_8UC1 );
    for(int r=w_size; r<(src_mat.rows-w_size); r++){
      uchar* _estimated_mat = estimated_mat.ptr<uchar>(r);
      double* _count_mat = count_mat.ptr<double>(r);

      for(int c=w_size; c<(src_mat.cols-w_size); c++){

        double tmp_prob = 0.0;
        for(int k=0; k<CLASS_NUM; k++){
          double* _prob_mat = prob_mat[k].ptr<double>(r);
          _prob_mat[c] /= _count_mat[c];
          if( _prob_mat[c] > tmp_prob ){
            tmp_prob = _prob_mat[c];
            _estimated_mat[c] = k;
          }
        }

      }
    }

    for(int k=0; k<CLASS_NUM; k++){
      cv::Mat tmp_pubImage_prob = cv::Mat::zeros( src_mat.rows, src_mat.cols, CV_32FC1 );
      cv::Mat recog_mat = cv::Mat::zeros( src_mat.rows, src_mat.cols, CV_8UC3 );

      for(int r=w_size; r<(src_mat.rows-w_size); r++){
        uchar* _estimated_mat = estimated_mat.ptr<uchar>(r);
        double* _prob_mat = prob_mat[k].ptr<double>(r);
        cv::Vec3b* _recog_mat = recog_mat.ptr<cv::Vec3b>(r);
        float* _tmp_pubImage_prob = tmp_pubImage_prob.ptr<float>(r);

        for(int c=w_size; c<(src_mat.cols-w_size); c++){
          int estimated_class = _estimated_mat[c];
          double value = _prob_mat[c];

          if( k == id_sand_class ){
            _tmp_pubImage_prob[c] = (float)value * 255.0;
            // _tmp_pubImage_prob[c].val[0] = class_color[k][0];
            // _tmp_pubImage_prob[c].val[1] = class_color[k][1];
            // _tmp_pubImage_prob[c].val[2] = class_color[k][2];
          }

          if( estimated_class == k && value > threshold ){
            int intensity = std::min( (value * 255.0 * 2.0), 255.0 );
            _recog_mat[c].val[0] = intensity;
            _recog_mat[c].val[1] = intensity;
            _recog_mat[c].val[2] = intensity;
          }
        }
      }
    }
    //-- ADAPTATION ENDS HERE

    //-- Convert to ouput type
    cv::Mat mask(src_mat.size(), CV_8U);
    cv::rectangle(mask, 
                  cv::Point(w_size, w_size), 
                  cv::Point(src_mat.cols - w_size - 1, src_mat.rows - w_size - 1),
                  255,
                  -1);

    // probability (0.0 to 1.0)
    cv::Mat prob_sand;
    prob_mat[id_sand_class].convertTo(prob_sand, CV_32F);
    prob_sand.copyTo(dst_prob, mask);

    // label (set to 255)
    cv::Mat labl_sand;
    cv::compare(prob_sand, threshold, labl_sand, cv::CMP_GT);
    labl_sand.copyTo(dst_labl, mask);
    return true;
  }

}  // namespace spoc

