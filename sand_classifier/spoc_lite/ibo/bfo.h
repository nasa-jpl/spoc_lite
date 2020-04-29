/*
  author: Yumi Iwashita
*/

#ifndef __IBO_BASIC_FEATURE_OPERATION_H__
#define __IBO_BASIC_FEATURE_OPERATION_H__

namespace ibo
{

  enum { TYPE_IMAGE, TYPE_LBP, TYPE_AVE_SD, TYPE_DCT, TYPE_CONT_DIST, TYPE_CONT_DIST_RANGE, TYPE_RANGE, TYPE_LBP_RANGE, TYPE_DCT_RANGE };

struct struct_bfo{
  int label;
  std::vector<double> lbp;
  std::vector<double> feature;
  // feature position
  int x;
  int y;

  // window aroung the feature position
  int left;
  int right;
  int up;
  int bottom;

  // change SetFeature
  int chans;
  int cols;
  int rows;

  uchar status;

  // useful when mutiple objects in one file
  std::string file_str;
  int object_id;

  struct_bfo(){
    file_str = "";
    object_id = 0;
  };
};

struct struct_1dpca{
  cv::Mat eigenVectors;// (rank, dim) row-majored matrix (column is dimension)
  cv::Mat eigenValues;// (rank, 1)
  cv::Mat averageVector; // (1, dim)
};

struct struct_feature_dimension{
  uchar type;
  int start;
  int end;
  int num;
  int flg_normalize=1;//1:do normalization, 0:do not normalize
  int flg_change_feature_as_0=0;//0:do not change, 1:change
};

// svm-style feature load
 // good for load ballancing for gallery data
 int bfo_load_features_ballance( std::string filename, std::vector<struct_bfo> &feature, int next_label=-1 );
 void bfo_load_features( std::string filename, std::vector<struct_bfo> &feature );
 // svm+customized style feature load. Features of multiple objecdts are in one file. 
 // object_id + svm_style_features
 void bfo_load_features_multi_objects( std::string filename, std::vector< std::vector<ibo::struct_bfo>> &feature, std::string file_str="" );
 int bfo_load_features_multi_objects_ballance( std::string filename, std::vector< std::vector<ibo::struct_bfo>> &feature, int next_label=-1, std::string file_str="" );

 void bfo_save_feature( FILE *fp_out, int label, std::vector<double> feature );
 void bfo_save_feature( FILE *fp_out, int label, cv::Mat mat );
 void bfo_save_descriptor_feature( FILE *fp_out, int label, cv::Mat mat );
 void bfo_save_descriptor_feature_roi( FILE *fp_out, int label, int roi_id, cv::Mat mat );
 void bfo_fprintf( FILE *fp_out, int label, std::vector<double> feature );
 void bfo_change_feature_as_0(std::vector<double> &feature, std::vector<struct_feature_dimension> feature_dim);//flg_change_feature_as_0 0:do not change, 1:change
 void bfo_normalize(std::vector<double> &feature, double threshold);
 void bfo_normalize_l1(std::vector<double> &feature, double lower, double upper);
 void bfo_normalize_l1(std::vector<double> &feature, std::vector<struct_feature_dimension> feature_dim, double lower, double upper);
 void bfo_normalize_l2(std::vector<double> &feature);
 void bfo_normalize_l2(std::vector<double> &feature, std::vector<struct_feature_dimension> feature_dim);
 void bfo_apply_1DPCA( std::vector<struct_bfo> data, struct_1dpca &pca );
 void bfo_do_whitening( std::vector<double> &feature, struct_1dpca &pca );

 void bfo_copy_struct_bfo_2_Mat( std::vector<struct_bfo> data, cv::Mat &mat_dst );
 void bfo_copy_struct_bfo_2_Mat( std::vector< std::vector<struct_bfo>> data, cv::Mat &mat_dst );


 //cv::Mat bfo_feature_descriptor( cv::Mat src_mat, std::string feature_type );

} // namespace ibo

#endif // __IBO_BASIC_FEATURE_OPERATION_H__
