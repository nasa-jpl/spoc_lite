/*!
 *  @file    sand_classifier_ibo.cpp
 *  @brief   SVM-based sand classifiers 
 *  @author  Yumi Iwashita <iwashita@jpl.nasa.gov>
 *  @author  Kyohei Otsu <otsu@jpl.nasa.gov>
 *  @date    2017-07-14
 *
 *  This classes use the SVM-based classifiers developed by Yumi Iwashita.
 */

#include <ros/ros.h>
#include <ros/package.h>

#include "sand_classifier/classifier_v2.h"

#include "ibo/bco.h"
#include "ibo/bco_mc.h"
#include "ibo/buo.h"
#include "ibo/buo.h"
#include "ibo/lbp.h"
#include "ibo/bfo.h"
#include "svm.h"
#include "libsvm.h"

#include "classification_ibo.h"


using namespace std;
using namespace cv;
using namespace ibo;


namespace spoc
{

  ClassifierSVMIbo::ClassifierSVMIbo(ros::NodeHandle nh, ros::NodeHandle pnh) 
      : Classifier(nh, pnh)
      , svm(0)
      , linear_svm(0)
  {
    //-- Load trained model
    std::string model_name;
    pnh.param<std::string>("model_name", model_name, "");

    //-- Load setting_file
    std::string setting_file_name;
    pnh.param<std::string>("setting_file", setting_file_name, "");
    
    //-- Load setting file
    struct_file file_info; // this is not used in this ClassifireSVM_ibo_SVM but used in other
    load_settings(setting_file_name, file_info, this->other_info);
    
    if (this->other_info.load_train_model)
      svm_setup(model_name, this->other_info.c_start, this->other_info.g_start);
    
    this->lbp_parameter.SDIM = ibo::makeELBP(8, this->lbp_parameter.Flg_LBP);
    this->lbp_parameter.db = this->other_info.lbp_threshold;
    
  }


  ClassifierSVMIbo::~ClassifierSVMIbo()
  {
  }


  void ClassifierSVMIbo::svm_setup(std::string model_name, int c, int g)
  {
    // to load pre-trained SVM model
    if(other_info.classifier_type == "SVM") {
      svm.SetParameters(other_info.kernel_type, (double)c, (double)g);
      svm.DummyGetMemory();
      svm.svmLoadModel(model_name.c_str());
    }
    else if (other_info.classifier_type == "LINEARSVM"){
      linear_svm.SetParameters(other_info.solver_type, 1);  // 1: automatic parameter search
      linear_svm.DummyGetMemory();
      linear_svm.svmLoadModel( model_name.c_str() );
    }
    cout << "svm model loaded" << endl;
  }


  bool ClassifierSVMIbo::classify(const cv::Mat &mat_src,
				  cv::Mat &mat_label_dst, 
				  cv::Mat &mat_prob_dst)
  {
    if(mat_src.channels() == 1 || mat_src.empty() == true)
      return false;

    cv::Mat mat_src_copy = mat_src.clone();

    vector<struct_bfo> test;
    vector<struct_feature_dimension> feature_dim;


    //SetFeature(test, feature_dim, lbp_parameter, mat_src_copy, mat_label, other_info);
    SetFeature(test, feature_dim, lbp_parameter, mat_src_copy, Mat(), other_info);
		
    int sample_num = test.size();

    // normalization -------------------------------------------------
    if (other_info.feature_norm_type == "L1") {
      cout << "normalization L1" << endl;
      for (int j = 0; j < sample_num; j++) 
        bfo_normalize_l1(test[j].feature, feature_dim,
                         other_info.lower, other_info.upper);
    }
    else if (other_info.feature_norm_type == "L2") {
      cout << "normalization L2" << endl;
      for (int j = 0; j < sample_num; j++)
	      bfo_normalize_l2(test[j].feature, feature_dim);
    }

    int actual_num_labels = other_info.bco_label_list.n_label;
    double *probability = new double [actual_num_labels];
    double *probability_frame = new double [sample_num * actual_num_labels];
    memset(probability_frame, 0, sizeof(double)*sample_num*actual_num_labels);
		
    // prediction each frame
    for (int j = 0; j < sample_num; j++) {
      if (other_info.classifier_type == "SVM")
        svm.svmPredictProbability(test[j], probability);
      else if(other_info.classifier_type == "LINEARSVM")
        linear_svm.svmPredictProbability(test[j], probability);

      memcpy(probability_frame+j*actual_num_labels, probability,
             sizeof(double)*actual_num_labels);
		  
      // if( other_info.classifier_type == "SVM" )
      //   estimated_class = svm.svmGetClass();
      // else if( other_info.classifier_type == "LINEARSVM" )
      //   estimated_class = linear_svm.svmGetClass();
    }
		
    // calc probability at each pixel and display
    vector<Mat> prob_mat_vec;
    Mat estimated_class_mat;
    Size mat_size(mat_src.cols, mat_src.rows);
    calc_prob_and_get_id_each_pixel(prob_mat_vec, estimated_class_mat,
                                    mat_size, probability_frame, test,
                                    other_info);

    make_label_prob_target_id(prob_mat_vec, estimated_class_mat, 
                              mat_label_dst, mat_prob_dst, other_info);
  
    if (probability) delete [] probability;
    if (probability_frame) delete [] probability_frame;

    return true;
  }

}  // namespace spoc

