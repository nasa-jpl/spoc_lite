/*
 author: Yumi Iwashita
*/
#include <stdio.h>
#include <string>
#include <vector>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>

#include "ibo/bco.h"
#include "ibo/buo.h"
#include "ibo/buo.h"
#include "ibo/lbp.h"
#include "ibo/bfo.h"
//#include "ibo/pca.h"
#include "ibo/svm.h"
#include "ibo/libsvm.h"
#include "ibo/svm_linear.h"

#include "classification_ibo.h"
#include "main.h"
#include "ibo.h"

using namespace std;
using namespace cv;
using namespace ibo;

  
#if 1
  // for ngan ROS
#include "sand_classifier/classifier_v2.h"
namespace spoc
{
  ClassifierSVMIbo::ClassifierSVMIbo(ros::NodeHandle nh, ros::NodeHandle pnh) : Classifier(nh, pnh), svm(0), linear_svm(0)
  {
    //-- Load trained model
    std::string model_name;
    pnh.param<std::string>("model_name", model_name, "" );
    //std::string model_name = "result_ave_lbp/svm_type_0_c_10_g_-20_w_25.model"; // replace later

    //-- Load setting_file
    std::string setting_file_name;// = "setting_ave_lbp.txt"; // replace later
    pnh.param<std::string>("setting_file", setting_file_name, "" );
    
    //-- Load setting file
    struct_file file_info; // this is not used in this ClassifireSVM_ibo_SVM but used in other
    load_settings( setting_file_name, file_info, this->other_info );
    
    if( this->other_info.load_train_model )
      svm_setup( model_name, this->other_info.c_start, this->other_info.g_start );
    this->lbp_parameter.SDIM = makeELBP(8, this->lbp_parameter.Flg_LBP);
    this->lbp_parameter.db = this->other_info.lbp_threshold;
    
  }
  ClassifierSVMIbo::~ClassifierSVMIbo()
  {
  }

  void ClassifierSVMIbo::svm_setup( string model_name,  int c, int g )
  {
    // to load pre-trained SVM model
    if( other_info.classifier_type == "SVM" ){
      svm.SetParameters(other_info.kernel_type, (double)c, (double)g);
      svm.DummyGetMemory();
      svm.svmLoadModel( model_name.c_str() );
    }
    else if( other_info.classifier_type == "LINEARSVM" ){
      linear_svm.SetParameters(other_info.solver_type, 1);// 1: automatic parameter search
      linear_svm.DummyGetMemory();
      linear_svm.svmLoadModel( model_name.c_str() );
    }
    cout << "load svm model" << endl;
  }

  bool ClassifierSVMIbo::classify( const cv::Mat &mat_src, cv::Mat &mat_label_dst, cv::Mat &mat_prob_dst )
  {
    if(mat_src.channels() == 1 || mat_src.empty())
      return false;

    cv::Mat mat_src_copy = mat_src.clone();

    vector<struct_bfo> test;
    vector<struct_feature_dimension> feature_dim;


    //SetFeature( test, feature_dim, lbp_parameter, mat_src_copy, mat_label, other_info );
    SetFeature( test, feature_dim, lbp_parameter, mat_src_copy, Mat(), other_info );
		
    int sample_num = test.size();

    // normalization -------------------------------------------------
    if( other_info.feature_norm_type == "L1" ){
      cout << "normalization L1" << endl;
      for(int j=0; j<sample_num; j++)
	bfo_normalize_l1( test[j].feature, feature_dim, other_info.lower, other_info.upper );
    }
    if( other_info.feature_norm_type == "L2" ){
      cout << "normalization L2" << endl;
      for(int j=0; j<sample_num; j++)
	bfo_normalize_l2( test[j].feature, feature_dim );
    }

    int actual_num_labels = other_info.bco_label_list.n_label;
    double *probability = new double [actual_num_labels];
    double *probability_frame = new double [sample_num * actual_num_labels];
    memset( probability_frame, 0, sizeof(double)*sample_num*actual_num_labels );
		
    // prediction each frame
    for(int j=0; j<sample_num; j++){

      if( other_info.classifier_type == "SVM" )
	svm.svmPredictProbability(test[j], probability);
      else if( other_info.classifier_type == "LINEARSVM" )
	linear_svm.svmPredictProbability(test[j], probability);

      memcpy( probability_frame+j*actual_num_labels, probability, sizeof(double)*actual_num_labels );
		  
      // if( other_info.classifier_type == "SVM" )
      //   estimated_class = svm.svmGetClass();
      // else if( other_info.classifier_type == "LINEARSVM" )
      //   estimated_class = linear_svm.svmGetClass();
    }
		
    // calc probability at each pixel and display
    vector<Mat> prob_mat_vec;
    Mat estimated_class_mat;
    Size mat_size( mat_src.cols, mat_src.rows );
    calc_prob_and_get_id_each_pixel( prob_mat_vec, estimated_class_mat, mat_size, probability_frame, test, other_info );

    make_label_prob_target_id( prob_mat_vec, estimated_class_mat, mat_label_dst, mat_prob_dst, other_info);
  
    if( probability ) delete [] probability;
    if( probability_frame ) delete [] probability_frame;

    return true;
  }

}  // namespace spoc

#else
//for stand alone 
ClassifierSVMIbo::ClassifierSVMIbo() : svm(0), linear_svm(0)
{
  //-- Load trained model
  std::string model_name = "result_ave_lbp/svm_type_0_c_10_g_-20_w_25.model"; // replace later
  //pnh.param<std::string>( "model_name", model_name, "" );
  
  //-- Load setting file
  struct_file file_info; // this is not used in this ClassifireSVM_ibo_SVM but used in other
  std::string setting_file_name = "setting_ave_lbp.txt"; // replace later
  load_settings( setting_file_name, file_info, this->other_info );
  
  if( this->other_info.load_train_model )
    svm_setup( model_name, this->other_info.c_start, this->other_info.g_start );
  
  this->lbp_parameter.SDIM = makeELBP(8, this->lbp_parameter.Flg_LBP);
  this->lbp_parameter.db = this->other_info.lbp_threshold;
  
}

ClassifierSVMIbo::ClassifierSVMIbo( string model_name, struct_info other_info, int c, int g ) : svm(0), linear_svm(0)
{
  if( other_info.load_train_model )
    svm_setup( model_name, c, g );
  
  this->lbp_parameter.SDIM = makeELBP(8, this->lbp_parameter.Flg_LBP);
  this->lbp_parameter.db = other_info.lbp_threshold;  
  
  // copyt other_info
  this->other_info = other_info;
}

ClassifierSVMIbo::~ClassifierSVMIbo()
{
}


void ClassifierSVMIbo::svm_setup( string model_name,  int c, int g )
{
  // to load pre-trained SVM model
  if( other_info.classifier_type == "SVM" ){
    svm.SetParameters(other_info.kernel_type, (double)c, (double)g);
    svm.DummyGetMemory();
    svm.svmLoadModel( model_name.c_str() );
  }
  else if( other_info.classifier_type == "LINEARSVM" ){
    linear_svm.SetParameters(other_info.solver_type, 1);// 1: automatic parameter search
    linear_svm.DummyGetMemory();
    linear_svm.svmLoadModel( model_name.c_str() );
  }
  cout << "load svm model" << endl;
}

bool ClassifierSVMIbo::classify( const cv::Mat &mat_src, cv::Mat &mat_label_dst, cv::Mat &mat_prob_dst )
//			     struct_info other_info, struct_lbp_parameter lbp_parameter )
{
  if(mat_src.channels() == 1 || mat_src.empty())
    return false;

  vector<struct_bfo> test;
  vector<struct_feature_dimension> feature_dim;

  //SetFeature( test, feature_dim, lbp_parameter, mat_src, mat_label, other_info );
  SetFeature( test, feature_dim, lbp_parameter, mat_src, Mat(), other_info );
		
  int sample_num = test.size();

  // normalization -------------------------------------------------
  if( other_info.feature_norm_type == "L1" ){
    cout << "normalization L1" << endl;
    for(int j=0; j<sample_num; j++)
      bfo_normalize_l1( test[j].feature, feature_dim, other_info.lower, other_info.upper );
  }
  if( other_info.feature_norm_type == "L2" ){
    cout << "normalization L2" << endl;
    for(int j=0; j<sample_num; j++)
      bfo_normalize_l2( test[j].feature, feature_dim );
  }

  int actual_num_labels = other_info.bco_label_list.n_label;
  double *probability = new double [actual_num_labels];
  double *probability_frame = new double [sample_num * actual_num_labels];
  memset( probability_frame, 0, sizeof(double)*sample_num*actual_num_labels );
		
  // prediction each frame
  for(int j=0; j<sample_num; j++){

    if( other_info.classifier_type == "SVM" )
      svm.svmPredictProbability(test[j], probability);
    else if( other_info.classifier_type == "LINEARSVM" )
      linear_svm.svmPredictProbability(test[j], probability);

    memcpy( probability_frame+j*actual_num_labels, probability, sizeof(double)*actual_num_labels );
		  
    // if( other_info.classifier_type == "SVM" )
    //   estimated_class = svm.svmGetClass();
    // else if( other_info.classifier_type == "LINEARSVM" )
    //   estimated_class = linear_svm.svmGetClass();
  }
		
  // calc probability at each pixel and display
  vector<Mat> prob_mat_vec;
  Mat estimated_class_mat;
  Size mat_size( mat_src.cols, mat_src.rows );
  calc_prob_and_get_id_each_pixel( prob_mat_vec, estimated_class_mat, mat_size, probability_frame, test, other_info );

  make_label_prob_target_id( prob_mat_vec, estimated_class_mat, mat_label_dst, mat_prob_dst, other_info);
  
  if( probability ) delete [] probability;
  if( probability_frame ) delete [] probability_frame;

  return true;
}
#endif


// test
void SetFeature( vector<struct_bfo> &feature, vector<struct_feature_dimension> &feature_dim, struct_lbp_parameter lbp_parameter, 
		 Mat src_mat, Mat mat_label, struct_info other_info )
{
  char rname[512];
  int src_rows = src_mat.rows;
  int src_cols = src_mat.cols;
  int src_chans = src_mat.channels();
  double time_wall;
  
  double time_wall_start = ibo::buo_get_wall_time();
  
  int rr, cc, r, c;
  int window_size = other_info.window_size;
  double feature_window_overlap_ratio = other_info.window_overlap;

  if( mat_label.empty() ){
    mat_label = Mat::zeros(src_rows, src_cols, CV_32FC1 );
    for (r=0; r < src_rows; r++){
      float* _mat_label = mat_label.ptr<float>(r);
      for (c=0; c < src_cols; c++)
	_mat_label[c] = -1;
    }
  }

  if( (src_rows != mat_label.rows) || (src_cols != mat_label.cols) ){
    cout << "image size is different\n" << endl;
    return;
  }
  
  // int w_size = floor(window_size / 2);
  // if(w_size < 0) w_size = 0;
  int w_size = other_info.w_size;

  // int window_skip = window_size * (1.0-feature_window_overlap_ratio);
  // if( window_skip == 0 ) window_skip = 1;// at least move the window 1 pixel
  // cout << "window_skip " << window_skip << endl;
  int window_skip = other_info.window_skip;

  // //------------------------------------------
  // // put offset
  // int rgb_offset[3] = { -10, 0, -10 };
  // for (int r=0; r < src_rows; r++){
  //   Vec3b* _src_mat = src_mat.ptr<Vec3b>(r);
  //   for (int c=0; c < src_cols; c++)
  //     for(int b=0; b < src_chans; b++)
  // 	if( _src_mat[c].val[b] > 230 )
  // 	  _src_mat[c].val[b] += rgb_offset[b];
  // }
  // //------------------------------------------


  
  if( other_info.color_space_type == "OPPONENT" ){
    time_wall = buo_get_wall_time();
    bco_convert_to_opposite_color( src_mat );
    time_wall = buo_get_wall_time() - time_wall;
    printf("bco_convert_to_opposite_color %lf\n", time_wall*1000.0 );
  }

  // imshow( "src_mat", src_mat );
  // waitKey( 0 );

  if( other_info.color_space_type == "GRAY" ){
    cvtColor(src_mat, src_mat, CV_RGB2GRAY);
    src_chans = src_mat.channels();
  }
  else if( other_info.color_space_type == "HSV" )
    if( src_chans == 3 )
      cvtColor( src_mat, src_mat, CV_BGR2HSV );
  else if( other_info.color_space_type == "CIELab" )
    if( src_chans == 3 )
      cvtColor( src_mat, src_mat, CV_BGR2Lab );
  else if( other_info.color_space_type == "XYZ" )
    if( src_chans == 3 )
      cvtColor( src_mat, src_mat, CV_BGR2XYZ );

  time_wall = ibo::buo_get_wall_time() - time_wall_start;

  Mat lbp_mat;
  for(int i=0; i<other_info.feature_type.size(); i++){
    if( other_info.feature_type[i] == "LBP" ){
      time_wall = buo_get_wall_time();
      calc_lbp( src_mat, lbp_mat, lbp_parameter.db );// ch3 converted into ch1
      time_wall = buo_get_wall_time() - time_wall;
      printf("calc_lbp %lf\n", time_wall*1000.0 );
      break;
    }
  }

  time_wall = ibo::buo_get_wall_time() - time_wall_start;

  float *lbp_feature = new float[8 * lbp_parameter.SDIM];
  float* f = lbp_feature;

  struct_extractFeature v_extractFeature;
  v_extractFeature.src_mat = src_mat.clone();
  v_extractFeature.lbp_mat = lbp_mat.clone();
  
  
  time_wall = buo_get_wall_time();
  int loop_count = 0;
  int label_current;
  for (r=w_size; r < (src_rows-w_size); r+=window_skip){
    float* _mat_label = mat_label.ptr<float>(r);
    for (c=w_size; c < (src_cols-w_size); c+=window_skip){
      
      //label_current = mc_label_src.data[r*mc_label_src.cols*mc_label_src.chans + c*mc_label_src.chans + 0];
      label_current = (int)_mat_label[c];

      // feature extraction
      extractFeature( r, c, w_size, label_current, &v_extractFeature, f, other_info );	

      loop_count ++;
    }
  }
  time_wall = buo_get_wall_time() - time_wall;
  printf("feature calc total %lf (loop %d)\n", time_wall*1000.0, loop_count);

  feature.clear();
  feature_dim.clear();
  copy(v_extractFeature.feature.begin(), v_extractFeature.feature.end(), back_inserter(feature));
  copy(v_extractFeature.feature_dim.begin(), v_extractFeature.feature_dim.end(), back_inserter(feature_dim));

  if( lbp_feature ) delete [] lbp_feature;
}

void SetFeature_BalancedLoad( vector<struct_bfo> &feature, vector<struct_feature_dimension> &feature_dim, struct_lbp_parameter lbp_parameter, Mat src_mat, Mat mat_label, int maxnum_feature_one_load, int label_next, struct_info other_info )
{
  int src_rows = src_mat.rows;
  int src_cols = src_mat.cols;
  int src_chans = src_mat.channels();
  double time_wall;

  int window_size = other_info.window_size;
  
  if( (src_rows != mat_label.rows) || (src_cols != mat_label.cols) ){
    cout << "image size is different\n" << endl;
    return;
  }
  
  // int w_size = floor(window_size / 2);
  // if(w_size < 0) w_size = 0;
  int w_size = other_info.w_size;

  if( other_info.color_space_type == "OPPONENT" ){
    time_wall = buo_get_wall_time();
    //mc_convert_to_opposite_color( mc_src );
    bco_convert_to_opposite_color( src_mat );
    time_wall = buo_get_wall_time() - time_wall;
    printf("bco_convert_to_opposite_color %lf\n", time_wall*1000.0 );
  }

  if( other_info.color_space_type == "GRAY" ){
    cvtColor(src_mat, src_mat, CV_RGB2GRAY);
    src_chans = src_mat.channels();
  }
  else if( other_info.color_space_type == "HSV" )
    if( src_chans == 3 )
      cvtColor( src_mat, src_mat, CV_BGR2HSV );
  else if( other_info.color_space_type == "CIELab" )
    if( src_chans == 3 )
      cvtColor( src_mat, src_mat, CV_BGR2Lab );
  else if( other_info.color_space_type == "XYZ" )
    if( src_chans == 3 )
      cvtColor( src_mat, src_mat, CV_BGR2XYZ );

  Mat lbp_mat;
  for(int i=0; i<other_info.feature_type.size(); i++){
    if( other_info.feature_type[i] == "LBP" ){
      time_wall = buo_get_wall_time();
      calc_lbp( src_mat, lbp_mat, lbp_parameter.db );// ch3 converted into ch1
      time_wall = buo_get_wall_time() - time_wall;
      printf("calc_lbp %lf\n", time_wall*1000.0 );
      break;
    }
  }

  float *lbp_feature = new float[8 * lbp_parameter.SDIM];
  float* f = lbp_feature;

  Mat label_mat;
  {
    int r, c, b;
    int label_chans = 3;
    label_mat = Mat::zeros( src_rows, src_cols, CV_8UC3 );

    for (r=0; r < src_rows; r++){
      float* _mat_label = mat_label.ptr<float>(r);
      Vec3b* _label_mat = label_mat.ptr<Vec3b>(r);
      for (c=0; c < src_cols; c++){
	for (b=0; b < label_chans; b++)
	  _label_mat[c].val[b] = (int)_mat_label[c] + 1;
      }
    }
  }
  // imshow( "src_mat", src_mat );
  // imshow( "label_mat", label_mat );
  // waitKey( 0 );

  vector< vector<Point> > label_contours;
  bco_Extract( label_mat, label_contours );

  int rr, cc;
  uchar *flg = new uchar [src_rows * src_cols];
  memset( flg, 0, sizeof(uchar)*src_rows*src_cols );

  struct_extractFeature v_extractFeature;
  v_extractFeature.src_mat = src_mat.clone();
  v_extractFeature.lbp_mat = lbp_mat.clone();

  int count_label_next = 0;
  for(int i=0; i<label_contours.size(); i++){
    Rect rect = boundingRect( label_contours[i] );

    // rectangle( label_mat, rect, Scalar(255,255,255) );
    // imshow( "label", label_mat );
    // waitKey( 0 );

    int count_label = 0;
    int max_count_label = rect.height * rect.width;
    do{
      if( count_label++ >= max_count_label ) break;
      //cout << "count_label " << count_label << " " <<  max_count_label << endl;

      int r = rand() % rect.height;
      int c = rand() % rect.width;
      r += rect.y;
      c += rect.x;
      if( r < w_size || r >= (src_rows-w_size) || c < w_size || c >= (src_cols-w_size) ) continue;
      if( flg[r*src_cols + c] == 1 ) continue;

      flg[r*src_cols + c] = 1;

      // check the class in this rect. If this is not the target one, break
      float* _mat_label = mat_label.ptr<float>(r);
      int label_current = (int)_mat_label[c];
      if( label_current != label_next )
	continue;
      //break;
      
#if 1
      // -----------------------------------
      // check labels (if label_current is not majority in this window, skip) 
      {
	struct_bfo tmp_feature;
	tmp_feature.left  = max( 0, (c-w_size) ); 
	tmp_feature.right = min( (src_cols-1), (c+w_size) );
	tmp_feature.up = max( 0, (r-w_size) );
	tmp_feature.bottom = min( (src_rows-1), (r+w_size) );
      
	double ratio_label_current = 0.9;
	int label;
	int label_current_counter = 0;
	int label_others_counter = 0;
	for(int rr=tmp_feature.up; rr<tmp_feature.bottom; rr++){
	  float* __mat_label = mat_label.ptr<float>(rr);
	  for(int cc=tmp_feature.left; cc<tmp_feature.right; cc++){
	    label = (int)__mat_label[cc];//mc_label_src.data[rr*mc_label_src.cols*mc_label_src.chans + cc*mc_label_src.chans + 0];
	    if( label == label_next ) label_current_counter ++;
	    else label_others_counter ++;
	  }
	}
	if( (double)label_current_counter/((double)label_current_counter+(double)label_others_counter) < ratio_label_current ) continue;
      }
      // -----------------------------------
#endif

#if 1 //--------------------------
      if( label_current == label_next ){
	extractFeature( r, c, w_size, label_current, &v_extractFeature, f, other_info );	
	
	if( count_label_next++ >= maxnum_feature_one_load ) break;
      }
#endif //--------------------------
      
    }while(1);
    
  }

  //feature.clear();
  feature_dim.clear();
  copy(v_extractFeature.feature.begin(), v_extractFeature.feature.end(), back_inserter(feature));
  copy(v_extractFeature.feature_dim.begin(), v_extractFeature.feature_dim.end(), back_inserter(feature_dim));

  if(flg) delete [] flg;
      

#if 1
  int num_label2 = other_info.bco_label_list.n_label;
  int *num_feature2 = new int [num_label2];
  memset( num_feature2, 0, sizeof(int)*num_label2 );
  for(int i=0; i<feature.size(); i++){
    for(int j=0; j<num_label2; j++){
      if( feature[i].label == other_info.bco_label_list.label_id[j] )
	num_feature2[j] ++;
    }
  }
  printf("in train dataset .......................\n");
  for(int j=0; j<num_label2; j++)
    printf("label %d : %d sets\n", other_info.bco_label_list.label_id[j], num_feature2[j]);
  if( num_feature2 ) delete [] num_feature2;
#endif
  if( lbp_feature ) delete [] lbp_feature;
}

void calc_prob_and_get_id_each_pixel( vector<Mat> &prob_mat_vec, Mat &estimated_class_mat, Size mat_size, double *probability_frame, vector<struct_bfo> test, struct_info other_info )
{
  int src_rows = mat_size.height;
  int src_cols = mat_size.width;
  int actual_num_labels = other_info.bco_label_list.n_label;

  estimated_class_mat = Mat::zeros( src_rows, src_cols, CV_8UC1 );
  Mat tmp_prob_mat_vec = Mat::zeros( src_rows, src_cols, CV_64FC1 );

  for(int j=0; j<other_info.bco_label_list.n_label; j++){
    Mat tmp_prob_mat_vec2 = Mat::zeros( src_rows, src_cols, CV_64FC1 );
    int r, c;
    int left, right, up, bottom;
    
    //int id_in_prob = id_map_from_label_to_prob[j];
    int id_category = other_info.bco_label_list.label_id[j];
    //if( id_in_prob < 0 ) continue;
    int sample_num = test.size();

    // pixel
    for(int jj=0; jj<sample_num; jj++){
      double prob = probability_frame[jj*actual_num_labels + id_category];
      r = test[jj].y;
      c = test[jj].x;
      left   = test[jj].left;
      right  = test[jj].right;
      up     = test[jj].up;
      bottom = test[jj].bottom;
      
      for(int sr=up; sr<=bottom; sr++){
	double* _tmp_prob_mat_vec = tmp_prob_mat_vec.ptr<double>(sr);
	double* _tmp_prob_mat_vec2 = tmp_prob_mat_vec2.ptr<double>(sr);
	uchar* _estimated_class_mat = estimated_class_mat.ptr<uchar>(sr);
	for(int sc=left; sc<=right; sc++){
	  if( _tmp_prob_mat_vec[sc] < prob ){
	    _tmp_prob_mat_vec[sc] = prob;
	    _estimated_class_mat[sc] = id_category;
	  }
	  if( _tmp_prob_mat_vec2[sc] < prob )
	    _tmp_prob_mat_vec2[sc] = prob ;
	}
      }		 
    }// pixel (jj)
    //prob_mat_vec.push_back( tmp_prob_mat_vec );
    prob_mat_vec.push_back( tmp_prob_mat_vec2 );
  }
}
  
 void make_label_prob_target_id( vector<Mat> prob_mat_vec, Mat estimated_class_mat, Mat &mat_label_4_evaluate_class, Mat &mat_prob_4_evaluate_class, struct_info other_info)
 {
   int rows = estimated_class_mat.rows;
   int cols = estimated_class_mat.cols;

   mat_label_4_evaluate_class = Mat::zeros( rows, cols, CV_8UC1 );
   mat_prob_4_evaluate_class  = Mat::zeros( rows, cols, CV_32FC1 );

   int r, c;
   if( other_info.evaluate_class_id >= 0 ){
     //mat_prob_4_evaluate_class = prob_mat_vec[other_info.evaluate_class_id].clone();
     for(r=0; r<rows; r++){
       double* _prob_mat_vec = prob_mat_vec[other_info.evaluate_class_id].ptr<double>(r);
       float* _prob_mat = mat_prob_4_evaluate_class.ptr<float>(r);
       uchar* _estimated_class_mat = estimated_class_mat.ptr<uchar>(r);
       uchar* _mat_label_4_evaluate_class = mat_label_4_evaluate_class.ptr<uchar>(r);
       for(c=0; c<cols; c++){
	 if( _estimated_class_mat[c] == other_info.evaluate_class_id ){
	   _prob_mat[c] = _prob_mat_vec[c];
	   _mat_label_4_evaluate_class[c] = 255;
	 }
       }
     }
   }
   
 }

