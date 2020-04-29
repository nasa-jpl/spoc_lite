#ifndef __IBO_H__
#define __IBO_H__

#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>

#include "ibo/bco.h"
#include "ibo/buo.h"
#include "ibo/buo.h"
#include "ibo/lbp.h"
#include "ibo/bfo.h"
//#include "ibo/pca.h"
#include "ibo/svm.h"
#include "ibo/libsvm.h"
#include "ibo/svm_linear.h"

struct struct_list{
  std::string file_name;
  unsigned char flg;// used or not
};


struct struct_roc_threshold{
  double start;
  double end;
  double gap;
  int num;
  
  struct_roc_threshold(){
    start = 0.0;
    end = 1.0;
    gap = 0.01;
    num = (end - start) / gap + 1;
  };
};

struct struct_info
{
  int evaluate_class_id;
  int save_predicted_image;
  std::string color_space_type;// OPPONENT, GRAY, RGB, HSV, CIELab, XYZ
  std::vector<std::string> feature_type; // AVERAGE, LBP, DCT
  int lbp_threshold;
  std::string feature_norm_type; // NONE, L1, L2
  std::string classifier_type; // SVM, LINEARSVM, RF
  bool load_train_model;
  double window_overlap;
  int window_size;
  int w_size;
  int window_skip;
  struct_roc_threshold roc_threshold;

  int n_of_n_folds;

  ibo::struct_bco_label bco_label_list;

  // for svm
  int kernel_type;
  int svm_type;
  int svm_probability;

  // for svm
  int solver_type;

  // for svm & linearsvm
  int c_start;
  int c_end;
  int c_gap;
  int g_start;
  int g_end;
  int g_gap;

  // for rf
  int maxdepth_start;
  int maxdepth_end;
  int maxdepth_gap;
  int maxtrees_start;
  int maxtrees_end;
  int maxtrees_gap;

  // for normalization (L1_NORMALIZATION)
  double lower;
  double upper;
  
  struct_info(){
    save_predicted_image = 0;
    n_of_n_folds = 2;
    evaluate_class_id = -1;
    color_space_type = "OPPONENT";
    lbp_threshold = 5;
    feature_norm_type = "NONE";
    classifier_type = "SVM";
    load_train_model = false;
    window_overlap = 0.5;
    window_size = 25;
    w_size = 0;
    window_skip = 1;
    kernel_type = LINEAR;
    svm_type = C_SVC;
    svm_probability = 1;
    solver_type = L2R_LR_DUAL;
    c_start = 10;
    c_end = 20;
    c_gap = 2;
    g_start = -40;
    g_end = -20;
    g_gap = 2;
    maxdepth_start = 25;
    maxdepth_end = 25;
    maxdepth_gap = 20;
    maxtrees_start = 25;
    maxtrees_end = 25;
    maxtrees_gap = 20;

    lower = 0.0;
    upper = 1.0;
  };
};


struct struct_file{
  std::string train_src_fd;
  std::string test_src_fd;
  std::string train_label_fd;
  std::string test_label_fd;
  std::string src_extension;
  std::string label_extension;

  std::string result_fd;
  std::string result_file;

  std::string svm_model_name;
  std::string rf_model_name;
  std::string svm_eval_file;
  std::string rf_eval_file;
  std::string predicted_image_file;

  std::string train_file;
  std::string test_file;

  std::vector<struct_list> train_file_lists;
  std::vector<struct_list> test_file_lists;

  void reset_flg(){
    int total_files = train_file_lists.size();
    for(int i=0; i<total_files; i++)
      train_file_lists[i].flg = 0;
    
    total_files = test_file_lists.size();
    for(int i=0; i<total_files; i++)
      test_file_lists[i].flg = 0;
  };
};

int load_settings( std::string setting_file_name, struct_file &file_info, struct_info &other_info );



#endif // __IBO_H__
