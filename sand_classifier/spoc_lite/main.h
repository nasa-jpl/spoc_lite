#ifndef __MAIN_H__
#define __MAIN_H__

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

#include "ibo.h"

enum{ SAMPLE_BALANCED, SAMPLE_ALL };


struct struct_rgb{
  int rgb[3];
};

struct struct_eval{
  int tp;
  int tn;
  int fn;
  int fp;
  int tp_ident;
  int fn_ident;
  double recall;
  double precision;
  double FOR;
  double F;
  double CCR;
};



struct struct_xy{
  int x;
  int y;
};






void LoadFileName( struct_file &file_info );

void SetMatFeature( std::vector<ibo::struct_bfo> data, ibo::struct_feature_dimension feature_dim_4_train, int class_id, cv::Mat &mat_data, cv::Mat &mat_data_class );



void MakeRandomNumber( int num_list, std::vector<int> &rand_id );
void initialize_eval( struct_eval &eval );
void read_chars_with_space( FILE *fp, std::string &str );
void evaluate_target_id( std::vector<cv::Mat> prob_mat_vec, cv::Mat &estimated_class_mat, cv::Mat mat_label, struct_eval *eval, struct_info other_info );
void visualize_predicted_target_id( std::string fullpath, cv::Mat estimated_class_mat, std::vector<cv::Mat> prob_mat_vec, cv::Mat &overlay_mat, struct_info other_info );



void LoadRandomNumber( int loop, int num_list, char rname_rand[], std::vector<int> &rand_id );

#define conf_all(e_class,gt_class,c,g) conf_all[(e_class)*nclass*num_c*num_g + (gt_class)*num_c*num_g + (c)*num_g + (g)]
#define gt_num_all(gt_class,c,g)       gt_num_all[(gt_class)*num_c*num_g + (c)*num_g + (g)]

#endif // __MAIN_H__
