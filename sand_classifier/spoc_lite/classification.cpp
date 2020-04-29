#include <stdio.h>
#include <string>
#include <vector>

#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/nonfree/nonfree.hpp>

#include "ibo/bco.h"
#include "ibo/bco_mc.h"
#include "ibo/buo.h"
#include "ibo/buo.h"
#include "ibo/lbp.h"
#include "ibo/bfo.h"
#include "classification.h"

using namespace std;
using namespace cv;
using namespace ibo;

namespace tc
{
  void setFeature( struct_feature_options feature_options, cv::Mat src_mat, mc_image mc_src, std::vector<struct_bfo> &feature, std::vector<struct_feature_dimension> &feature_dim, int feature_window_size, double feature_window_overlap_ratio, struct_lbp_parameter lbp_parameter )
  {
    Mat lbp_mat;
    int mc_src_rows = mc_src.rows;
    int mc_src_cols = mc_src.cols;
    int mc_src_chans = mc_src.chans;

    int w_size = floor(feature_window_size / 2);
    if(w_size < 0) w_size = 0;
    
    int window_skip = feature_window_size * feature_window_overlap_ratio;
    if( window_skip == 0 ) window_skip = 1;// at least move the window 1 pixel

    
#ifdef USE_OPPOSITE_COLOR
    mc_convert_to_opposite_color( mc_src, feature_options.use_intensity );
    mc_copy_image_2_Mat( mc_src, src_mat );
    // {
    //   Mat src_mat;
    //   mc_copy_image_2_Mat( mc_src, src_mat );
    //   imshow( "s", src_mat );
    //   waitKey( 0 );
    // }
#endif

    //#if defined (FEATURE_LBP)
    calc_lbp( src_mat, lbp_mat, lbp_parameter.db );// ch3 converted into ch1

    float *lbp_feature = new float[8 * lbp_parameter.SDIM];
    float* f = lbp_feature;
    //#endif


#if defined (USE_HSV) || defined (USE_CIELab) || defined (USE_XYZ)
    mc_image mc_src2;
    {
      Mat src_mat;
      mc_copy_image_2_Mat( mc_src, src_mat );
      if( mc_src.chans == 3 ){
#ifdef USE_HSV
	cvtColor( src_mat, src_mat, CV_BGR2HSV );
#endif
#ifdef USE_CIELab
	cvtColor( src_mat, src_mat, CV_BGR2Lab );
#endif
#ifdef USE_XYZ
	cvtColor( src_mat, src_mat, CV_BGR2XYZ );
#endif
	mc_copy_Mat_2_image( src_mat, mc_src2 );
      }
      else
	mc_copy_image_2_image( mc_src, mc_src2 );
    }
#endif
  
    int layer = 1;
    
    int left, right, up, bottom;
    int roi_width, roi_height;

    int rr, cc;
    //for (int r=0; r < mc_src_rows; r+=window_skip){
    //for (int c=0; c < mc_src_cols; c+=window_skip){
    for (int r=w_size; r < (mc_src_rows-w_size); r+=window_skip){
      for (int c=w_size; c < (mc_src_cols-w_size); c+=window_skip){
	vector<struct_feature_dimension> tmp_feature_dim;
	int feature_counter = 0;
	
	// prepare feature      
	struct_bfo tmp_feature;
	tmp_feature.y = r;
	tmp_feature.x = c;
	tmp_feature.left  = max( 0, (c-w_size) ); 
	tmp_feature.right = min( (mc_src_cols-1), (c+w_size) );
	tmp_feature.up = max( 0, (r-w_size) );
	tmp_feature.bottom = min( (mc_src_rows-1), (r+w_size) );
	tmp_feature.chans = mc_src_chans;
	tmp_feature.cols = (2*w_size + 1);
	tmp_feature.rows = (2*w_size + 1);
	tmp_feature.label = 1;//label_current;//mc_label_image.data[r*mc_label_image.cols*mc_label_image.chans + c*mc_label_image.chans + 0];
	tmp_feature.status = 1; // 1: include for evaluation, 0: ignore
	
#if 1
	//#ifdef FEATURE_LBP
	if( feature_options.use_lbp ){
	  struct_feature_dimension tmp_feature_dim_lbp;
	  tmp_feature_dim_lbp.start = feature_counter;
	  //for(int layer=1; layer<=pyramid_layer; layer++){
	  // Rect(x,y,width,height)
	  left = max( 0, (c-w_size*layer) ); right = min( (mc_src_cols-1), (c+w_size*layer) );
	  up = max( 0, (r-w_size*layer) );   bottom = min( (mc_src_rows-1), (r+w_size*layer) );
	  roi_width = right - left;
	  roi_height = bottom - up;
	  Rect roi(left, up, roi_width, roi_height);
	  Mat lbp_mat_roi = lbp_mat(roi); // this does not take memory
	  double area = (double)roi_width/100.0;// * roi_height;
	
	  int dim = calc_lbp_hist(lbp_mat_roi, (float*)f, NULL);// normal LBP
	  for(int i=0; i<dim; i++){
	    tmp_feature.feature.push_back( (double)lbp_feature[i]/area );
	    feature_counter ++;
	  }
	  //}
	  //tmp_feature_dim_lbp.type = TYPE_LBP;
	  tmp_feature_dim_lbp.end = feature_counter;
	  tmp_feature_dim_lbp.num = tmp_feature_dim_lbp.end - tmp_feature_dim_lbp.start;
	  tmp_feature_dim.push_back( tmp_feature_dim_lbp );
	}
	//#endif
#endif
	//#ifdef FEATURE_AVE_SD
	if( feature_options.use_ave) {
	  struct_feature_dimension tmp_feature_dim_sd;
	  tmp_feature_dim_sd.start = feature_counter;
	  left = max( 0, (c-w_size*layer) ); right = min( (mc_src_cols-1), (c+w_size*layer) );
	  up = max( 0, (r-w_size*layer) );   bottom = min( (mc_src_rows-1), (r+w_size*layer) );
	
	  if( mc_src_chans > 3 ){
	    printf("FEATURE_AVE can be calculated with 1 or 3 channels only\n");
	    getchar();
	  }
	  double average[3]={0.0}, stddev[3]={0.0}, asum[3]={0.0};
	  int count = 0;
	  for(rr=up; rr<=bottom; rr++){
	    for(cc=left; cc<=right; cc++){
	      for (int b=0; b < (mc_src_chans); b++){
#if defined (USE_HSV) || defined (USE_CIELab)
		average[b] += (double)mc_src2.data[rr*mc_src_cols*mc_src_chans + cc*mc_src_chans + b];
#else
		average[b] += (double)mc_src.data[rr*mc_src_cols*mc_src_chans + cc*mc_src_chans + b];
#endif
	      }
	      count ++;
	    }
	  }
	  if(count == 0){
	    cout << "weird" << endl;
	    getchar();
	    break;
	  }

	  for (int b=0; b < (mc_src_chans); b++)
	    average[b] /= (double)count;

	  double tmpd[3];
	  for(rr=up; rr<=bottom; rr++){
	    for(cc=left; cc<=right; cc++){
	      for (int b=0; b < (mc_src_chans); b++){
#if defined (USE_HSV) || defined (USE_CIELab)
		tmpd[b] = ((double)mc_src2.data[rr*mc_src_cols*mc_src_chans + cc*mc_src_chans + b] - average[b]);
#else
		tmpd[b] = ((double)mc_src.data[rr*mc_src_cols*mc_src_chans + cc*mc_src_chans + b] - average[b]);
#endif
		asum[b] += tmpd[b];
		stddev[b] += tmpd[b] * tmpd[b];
	      }
	    }
	  }
#ifdef USE_CIELab
	  for (int b=0; b < (mc_src_chans-1); b++){
#else
	  for (int b=0; b < (mc_src_chans); b++){
#endif
	      stddev[b] = sqrt( stddev[b] / (double)count );
	      asum[b] /= (double)count;
	  
	      tmp_feature.feature.push_back( average[b] );
	      feature_counter ++;
	  }
	  tmp_feature_dim_sd.end = feature_counter;
	  tmp_feature_dim_sd.num = tmp_feature_dim_sd.end - tmp_feature_dim_sd.start;
	  tmp_feature_dim.push_back( tmp_feature_dim_sd );
	}
	//#endif
	//#ifdef FEATURE_DCT
	if( feature_options.use_dct ){
	  struct_feature_dimension tmp_feature_dim_dct;
	  tmp_feature_dim_dct.start = feature_counter;
	  //for(int layer=1; layer<=pyramid_layer; layer++){
	  // Rect(x,y,width,height)
	  left = max( 0, (c-w_size*layer) ); right = min( (mc_src_cols-1), (c+w_size*layer) );
	  up = max( 0, (r-w_size*layer) );   bottom = min( (mc_src_rows-1), (r+w_size*layer) );
	  roi_width = right - left;
	  roi_height = bottom - up;
	  if( (roi_width%2) == 1 )  roi_width -= 1;
	  if( (roi_height%2) == 1 ) roi_height -= 1;
	  if( roi_width<=0 || roi_height<=0 ){
	    cout << "window size is too small" << endl;
	    getchar();
	    return;
	  }
	  Rect roi(left, up, roi_width, roi_height);
	  Mat src_mat_roi = src_mat(roi); // this does not take memory

	  // printf("src_mat_roi rows %d cols %d channels %d\n", src_mat_roi.rows, src_mat_roi.cols, src_mat_roi.channels());
	  // imshow( "src_mat", src_mat );
	  // imshow( "src_mat_roi", src_mat_roi );
	  // waitKey(0);
	
	  Mat dct_mat = Mat::zeros( src_mat_roi.rows, src_mat_roi.cols, CV_64FC1 );
	  //src_mat_roi.convertTo( dct_mat, CV_64FC1 );
	  cvtColor( src_mat_roi, dct_mat, CV_BGR2GRAY );
	  dct_mat.convertTo( dct_mat, CV_64FC1 );
	  dct( dct_mat, dct_mat );
	
	  // normalization 0-255
	  dct_mat.convertTo( dct_mat, CV_8UC1 );
	  /// Normalize the result to [ 0, 1 ]
	  //normalize(dct_mat, dct_mat, 0.0, 1.0, NORM_MINMAX, -1, Mat() );

	  int dct_num = w_size;// this is the minimum size
	  for(int i=0; i<dct_num; i++){
	    for(int j=0; j<dct_num; j++){
	      //tmp_feature.feature.push_back( dct_mat.at<double>(i,j) );
	      tmp_feature.feature.push_back( (double)dct_mat.at<uchar>(i,j) );
	      feature_counter ++;
	    }
	  }
	  //}
	  //tmp_feature_dim_dct.type = TYPE_DCT;
	  tmp_feature_dim_dct.end = feature_counter;
	  tmp_feature_dim_dct.num = tmp_feature_dim_dct.end - tmp_feature_dim_dct.start;
	  tmp_feature_dim.push_back( tmp_feature_dim_dct );
	}
	//#endif
	
      // //debug
      // printf("tmp_feature.feature.size() %d\n", tmp_feature.feature.size() );
      // for(int yy=0; yy<tmp_feature.feature.size(); yy++)
      // 	printf("%lf ", tmp_feature.feature[yy]);
      // getchar();

	feature.push_back( tmp_feature );
	feature_dim.clear();
	copy(tmp_feature_dim.begin(), tmp_feature_dim.end(), back_inserter(feature_dim));
      }
    }
    //#if defined (FEATURE_LBP)
    if( feature_options.use_lbp )
      if( lbp_feature ) delete [] lbp_feature;
    //#endif
  }
  

}
