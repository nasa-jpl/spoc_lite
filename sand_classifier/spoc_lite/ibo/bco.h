/*****************************************************************************/

/*  
    @file bco.
    @brief Basic Camera Operations
  author: Yumi Iwashita
*/

/*****************************************************************************/

#ifndef __IBO_BASIC_CAMERA_OPERATION__
#define __IBO_BASIC_CAMERA_OPERATION__

#include <opencv2/opencv.hpp>

namespace ibo
{

struct struct_bco_label
{
  int n_label;
  int rows;
  int cols;
  int chans;
  std::vector<int> data;    /* row-major storage of pixels */
  std::vector<int> label_id; // label_id is 0,1,2,3,... cannot be changed (if you do like 0, 2, 5, ... other process will corrupsed)
  std::vector<int> num_each_label;
  std::vector<int> flag_ignore; // if it is ignore, set as 1. if not, set as -1.
};

struct struct_camera_parameter{
  // from camera parameter files
  double A[3][3]; // internal parameter
  double A_cam0[3][3]; // internal parameter
  double A_cam1[3][3]; // internal parameter
  double ins_R_veh[3][3]; // external parameter (rotation)
  double ins_T_veh[3]; // external parameter (translation)
  double cam_R_ins[3][3]; // external parameter (rotation)
  double cam_T_ins[3]; // external parameter (translation)
  double cam0_R_ins[3][3]; // external parameter (rotation)
  double cam0_T_ins[3]; // external parameter (translation)
  double cam1_R_ins[3][3]; // external parameter (rotation)
  double cam1_T_ins[3]; // external parameter (translation)

  // 1 stereo pair
  // Calculated with IMU
  double ins_R_veh_gt[3][3]; // external parameter (rotation)
  double ins_T_veh_gt[3]; // external parameter (translation)

  // for both 1 and 2 stereo
  // updated every frame in 2 stereo
  std::vector<double> veh_R_wld;//[3][3];
  std::vector<double> veh_T_wld;//[3];
  // PCA related
  double veh_R_pca_c0[3][3];
  double veh_R_pca_c1[3][3];
  double veh_T_pca_c0[3];
  double veh_T_pca_c1[3];
  double cam0_R_pca_c0[3][3];
  double cam0_T_pca_c0[3];
  double cam1_R_pca_c1[3][3];
  double cam1_T_pca_c1[3];
  //double pca_cor_R_pca_c0[3][3];
  //double pca_cor_T_pca_c0[3];
  //double pca_cor_R_pca_c1[3][3];
  //double pca_cor_T_pca_c1[3];

  double veh_R_pca_all[3][3];
  double veh_T_pca_all[3];
  double cam0_R_pca_all[3][3];
  double cam0_T_pca_all[3];
  double cam1_R_pca_all[3][3];
  double cam1_T_pca_all[3];
  double pca_all_R_wld[3][3];
  double pca_all_T_wld[3];

  // 2 stereo pairs (better R matrix between VEH and WLD)
  double cam0_R_veh_gt[3][3];
  double cam0_T_veh_gt[3];
  double cam1_R_veh_gt[3][3];
  double cam1_T_veh_gt[3];

  // this will not be updated
  std::vector<double> vehP_R_wld;//[3][3];
  std::vector<double> vehP_T_wld;//[3];

  // this is more precise. updated at each frame
  std::vector<double> veh_R_vehP;//[3][3];// old version (now this is not updated with PCA) -> this would be updated with PCA results. Initial value is from INS
  std::vector<double> veh_T_vehP;//[3];// old version (now this is not updated with PCA) -> this would be updated with PCA results. Initial value is from INS
  // updated every frame
  //double veh_R_wld[3][3];
  //double veh_T_wld[3];
  double cam0_R_wld[3][3];
  double cam0_T_wld[3];
  double cam1_R_wld[3][3];
  double cam1_T_wld[3];

  // map related ------------------------------------------
  // updated every frame. not implemented yet
  double mmap_R_wld[3][3];
  double mmap_T_wld[3];
  double pxl_R_wld[3][3];
  double pxl_T_wld[3];
  double mmap_R_veh[3][3];
  double mmap_T_veh[3];

  double map_R_wld[3][3];
  double map_T_wld[3];
  double map_R_cam0[3][3];
  double map_T_cam0[3];
  double map_R_cam1[3][3];
  double map_T_cam1[3];

  double map_width; // [m]
  double map_height; // [m]
  double map_scale; // [m]
  int map_cols; //  = map_width / map_scale
  int map_rows; //  = map_height / map_scale

  int rows;
  int cols;
};

struct struct_bco
{
  struct_camera_parameter cp;
  cv::Mat image;
  cv::Mat disp_image;
  cv::Mat image_train;
  cv::Mat image_test;
  cv::Mat projected_image;
  cv::Mat mask_image;
  cv::Mat overlay_image;
  cv::Mat extract_image;
  cv::Mat output_image;

  cv::Mat feature; // CV_32FC2 by SURF
  cv::Mat feature_mask; // CV_8UC1 homography mask
  
};

 void bco_LoadCameraParameter( std::string int_cp_file, std::string ext_cp_file_ins2veh, std::string ext_cp_file_cam2ins, struct_camera_parameter &cp );
 void bco_LoadCameraParameter_int( std::string int_cp_cam0_file, std::string int_cp_cam1_file, struct_camera_parameter &cp );
 void bco_LoadCameraParameter_ext( std::string ext_cp_file, struct_camera_parameter &cp );

 void bco_RemoveNoise( cv::Mat &mat, int threshold_area );
 void bco_RemoveFlyingNoise( cv::Mat &mat, int threshold_area );
 void bco_ExtractNoise( cv::Mat &mat, int threshold_area, std::vector< std::vector<cv::Point> > &contours );
 void bco_FillHole( cv::Mat &mat );
 
 void bco_Morphology_Erosion( cv::Mat &mat, int num ); // smaller
 void bco_Morphology_Dilation( cv::Mat &mat, int num ); // larger
 void bco_Overlay( cv::Mat image, cv::Mat mask_image, cv::Mat &overlay_image, cv::Scalar color );
 void bco_Overlay( cv::Mat image, cv::Mat mask_color_image, cv::Mat &overlay_image );
 void bco_Overlay( cv::Mat image, cv::Mat mask_image, cv::Mat color_image, cv::Mat &overlay_image );
 void bco_Extract( cv::Mat image, cv::Mat mask_image, cv::Mat &extract_image );
 void bco_Extract( cv::Mat image, std::vector< std::vector<cv::Point> > &contours );
 void bco_AlphaBlending( cv::Mat src_mat, cv::Mat &prob_mat_4c );
 
 void bco_calc_and_matching_features( cv::Mat image, cv::Mat image_1, cv::Mat &feature, cv::Mat &feature_1, double ship_area_xy_start[], double ship_area_xy_end[] );
 void bco_calc_and_matching_features( cv::Mat image, cv::Mat image_1, cv::Mat &feature, cv::Mat &feature_1 );
 
 void bco_resize_mat( cv::Mat &mat, double resize_ratio );
 cv::Mat bco_combine_2mats( cv::Mat src_mat1, cv::Mat src_mat2 );
 cv::Mat bco_combine_2mats_vertical( cv::Mat src_mat1, cv::Mat src_mat2 );

 // note: mat_label is signed char (8S) Pixel without label is assigned with -1
 void bco_assign_label( struct_bco_label bco_label_list, cv::Mat &mat_label );

 void bco_convert_to_opposite_color( cv::Mat &mat_in_out, bool flg_fix_intensity=true );// last one is option
 

} // namespace ibo

#endif // __BASIC_CAMERA_OPERATION__
