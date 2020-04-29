#ifndef __CLASSIFY_H__
#define __CLASSIFY_H__

#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>

#include "ibo.h"

#include "ibo/bco.h"
#include "ibo/buo.h"
#include "ibo/buo.h"
#include "ibo/lbp.h"
#include "ibo/bfo.h"
//#include "ibo/pca.h"
#include "ibo/svm.h"
#include "ibo/libsvm.h"
#include "ibo/svm_linear.h"


#if 0
// for stand alone
class ClassifierSVMIbo
{
 public:
  ClassifierSVMIbo();// this is for ROS
  ClassifierSVMIbo( std::string mode_name, struct_info other_info, int c, int g );// for Yumi
  ~ClassifierSVMIbo();

  ibo::struct_lbp_parameter lbp_parameter;
  struct_info other_info;

  ibo::LIBSVM svm;
  LIBSVM_LINEAR linear_svm;
  
  void svm_setup( std::string model_name, int c, int g );  
  bool classify( const cv::Mat &mat_src, cv::Mat &mat_label_dst, cv::Mat &mat_prob_dst );

 protected:
  
};
#endif


struct struct_extractFeature{
  //ibo::mc_image mc_src;
  cv::Mat src_mat;
  cv::Mat lbp_mat;
  std::vector<ibo::struct_bfo> feature;
  std::vector<ibo::struct_feature_dimension> feature_dim;
};

void SetFeature( std::vector<ibo::struct_bfo> &feature, std::vector<ibo::struct_feature_dimension> &feature_dim, ibo::struct_lbp_parameter lbp_parameter, cv::Mat mat_src, cv::Mat mat_label, struct_info other_info );
void SetFeature_BalancedLoad( std::vector<ibo::struct_bfo> &feature, std::vector<ibo::struct_feature_dimension> &feature_dim, ibo::struct_lbp_parameter lbp_parameter, cv::Mat mat_src, cv::Mat mat_label, int maxnum_feature_one_load, int label_next, struct_info other_info );

void calc_prob_and_get_id_each_pixel( std::vector<cv::Mat> &prob_mat_vec, cv::Mat &estimated_class_mat, cv::Size mat_size, double *probability_frame, std::vector<ibo::struct_bfo> test, struct_info other_info );
void make_label_prob_target_id( std::vector<cv::Mat> prob_mat_vec, cv::Mat estimated_class_mat, cv::Mat &mat_label_4_evaluate_class, cv::Mat &mat_prob_4_evaluate_class, struct_info other_info);

inline 
void extractFeature( int r, int c, int w_size, int label, struct_extractFeature *v_extractFeature, float* f, struct_info other_info )
{
 std::vector<ibo::struct_feature_dimension> tmp_feature_dim;
  int feature_counter = 0;
   
  int left, right, up, bottom;
  int roi_width, roi_height;
  int rr, cc, b;

  int src_rows = v_extractFeature->src_mat.rows;
  int src_cols = v_extractFeature->src_mat.cols;
  int src_chans = v_extractFeature->src_mat.channels();

  left = std::max( 0, (c-w_size) ); right = std::min( (src_cols-1), (c+w_size) );
  up = std::max( 0, (r-w_size) );   bottom = std::min( (src_rows-1), (r+w_size) );
   
  // prepare feature      
  ibo::struct_bfo tmp_feature;
  tmp_feature.y = r;
  tmp_feature.x = c;
  tmp_feature.left  = left;
  tmp_feature.right = right;
  tmp_feature.up = up;
  tmp_feature.bottom = bottom;
  tmp_feature.chans = src_chans;
  tmp_feature.cols = (2*w_size + 1);
  tmp_feature.rows = (2*w_size + 1);
  tmp_feature.label = label;
  tmp_feature.status = 1;//1-flg_ignore; // 1: include for evaluation, 0: ignore
	
  double time_wall;
  //printf("feature calc ");

  roi_width = right - left;
  roi_height = bottom - up;
  cv::Rect roi(left, up, roi_width, roi_height);
	
  for(int i=0; i<other_info.feature_type.size(); i++){
    // LBP
    if( other_info.feature_type[i] == "LBP" ){
      ibo::struct_feature_dimension tmp_feature_dim_lbp;
      tmp_feature_dim_lbp.start = feature_counter;
      cv::Mat lbp_mat_roi = v_extractFeature->lbp_mat(roi); // this does not take memory
      double area = (double)roi_width/100.0;// * roi_height;
      
      //int dim = calc_lbp_hist(lbp_mat_roi, (float*)f, lbp_parameter.Flg_LBP);// rotation invariant LBP. a bit slow
      int dim = ibo::calc_lbp_hist(lbp_mat_roi, (float*)f, NULL);// normal LBP
      for(int i=0; i<dim; i++){
	tmp_feature.feature.push_back( (double)f[i]/area );
	//tmp_feature.feature.push_back( 0.0 );//debug
	feature_counter ++;
      }
      tmp_feature_dim_lbp.type = ibo::TYPE_LBP;
      tmp_feature_dim_lbp.end = feature_counter;
      tmp_feature_dim_lbp.num = tmp_feature_dim_lbp.end - tmp_feature_dim_lbp.start;
      tmp_feature_dim.push_back( tmp_feature_dim_lbp );
    }
    // AVERAGE
    else if( other_info.feature_type[i] == "AVERAGE" ){
      ibo::struct_feature_dimension tmp_feature_dim_sd;
      tmp_feature_dim_sd.start = feature_counter;
	    
      time_wall = ibo::buo_get_wall_time();
      if( src_chans > 3 ){
	printf("FEATURE_AVE can be calculated with 1 or 3 channels only\n");
	getchar();
      }
      double average[3]={0.0}, stddev[3]={0.0}, asum[3]={0.0};
      int count = 0;
#if 1
      for(rr=up; rr<=bottom; rr++){
	if( src_chans == 1 ){
	  uchar* _src_mat = v_extractFeature->src_mat.ptr<uchar>(rr);
	  for(cc=left; cc<=right; cc++){
	    *(average) += (double)_src_mat[cc];
	    count ++;
	  }
	}
	else{
	  cv::Vec3b* _src_mat = v_extractFeature->src_mat.ptr<cv::Vec3b>(rr);
	  for(cc=left; cc<=right; cc++){
	    for(b=0; b<src_chans; b++){
	      //printf("%d ", _src_mat[cc].val[b]);
	      *(average+b) += (double)_src_mat[cc].val[src_chans-b-1];
	    }
	    count ++;
	  }
	}
      }
#else
      if( src_chans == 1 ){
	uchar* _src_mat = v_extractFeature->src_mat.ptr<uchar>(r);
	*(average) += (double)_src_mat[c];
      }
      else{
	cv::Vec3b* _src_mat = v_extractFeature->src_mat.ptr<cv::Vec3b>(r);
	for(b=0; b<src_chans; b++)
	  *(average+b) += (double)_src_mat[c].val[src_chans-b-1];
      }
      count ++;
#endif
      if(count == 0){
	std::cout << "weird" << std::endl;
	getchar();
	return;//break;
      }
      for (int b=0; b < (src_chans); b++)
	average[b] /= (double)count;

      int b_num = src_chans;
      if( other_info.color_space_type == "CIELab" )
	b_num = src_chans - 1;
      for (int b=0; b < b_num; b++){
	tmp_feature.feature.push_back( average[b] );
	feature_counter ++;
      }
   
      tmp_feature_dim_sd.type = ibo::TYPE_AVE_SD;
      tmp_feature_dim_sd.end = feature_counter;
      tmp_feature_dim_sd.num = tmp_feature_dim_sd.end - tmp_feature_dim_sd.start;
      tmp_feature_dim.push_back( tmp_feature_dim_sd );
    }
    // DCT
    else if( other_info.feature_type[i] == "DCT" ){
      ibo::struct_feature_dimension tmp_feature_dim_dct;
      tmp_feature_dim_dct.start = feature_counter;
      int roi_width2 = roi_width;
      int roi_height2 = roi_height;
      if( (roi_width2%2) == 1 )  roi_width2 -= 1;
      if( (roi_height2%2) == 1 ) roi_height2 -= 1;
      if( roi_width2<=0 || roi_height2<=0 ){
	std::cout << "window size is too small" << std::endl;
	getchar();
	return;
      }
      cv::Rect roi(left, up, roi_width2, roi_height2);
      cv::Mat src_mat_roi = v_extractFeature->src_mat(roi); // this does not take memory
	      
      cv::Mat dct_mat;// = Mat::zeros( src_mat_roi.rows, src_mat_roi.cols, CV_64FC1 );
      cvtColor( src_mat_roi, dct_mat, CV_BGR2GRAY );
      dct_mat.convertTo( dct_mat, CV_64FC1 );
      dct( dct_mat, dct_mat );
	  
      // normalization 0-255
      dct_mat.convertTo( dct_mat, CV_8UC1 );
	  
      int dct_num = w_size;// this is the minimum size
      for(int i=0; i<dct_num; i++){
	uchar* _dct_mat = dct_mat.ptr<uchar>(i);
	for(int j=0; j<dct_num; j++){
	  //tmp_feature.feature.push_back( (double)dct_mat.at<uchar>(i,j) );
	  tmp_feature.feature.push_back( (double)_dct_mat[j] );
	  feature_counter ++;
	}
      }
	    
      tmp_feature_dim_dct.type = ibo::TYPE_DCT;
      tmp_feature_dim_dct.end = feature_counter;
      tmp_feature_dim_dct.num = tmp_feature_dim_dct.end - tmp_feature_dim_dct.start;
      tmp_feature_dim.push_back( tmp_feature_dim_dct );
    }
  }

  //cout << tmp_feature.feature.size() << " " << tmp_feature.feature[0] << endl;
  v_extractFeature->feature.push_back( tmp_feature );
  
  v_extractFeature->feature_dim.clear();
  copy(tmp_feature_dim.begin(), tmp_feature_dim.end(), back_inserter(v_extractFeature->feature_dim));
}

#endif //__CLASSIFY_H__
