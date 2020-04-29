/*
  author: Yumi Iwashita
*/

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

#include <opencv2/opencv.hpp>
//#include <opencv2/nonfree/nonfree.hpp>

#include "bfo.h"
#include "bop.h"

using namespace cv;
using namespace std;

namespace ibo
{

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wunused-variable"


/*
void bfo_calc_feature_LBP( struct_lbp_parameter lbp_parameter, Mat image_mat, int label, vector<double> &lbp )
{
  float *feature = new float[8*lbp_parameter.SDIM]; // RGB -> SDIM*LL dim / DEPTH -> SDIM*LL+LL dim (since NAN value uses one bin）
  float* f = feature;

  Mat lbp_mat;  
  LBP( image_mat, lbp_mat );
  
  //int dim = calc_hist(lbp_mat, (float*)f, lbp_parameter.Flg_LBP);// rotation invariant LBP
  int dim = calc_hist(lbp_mat, (float*)f, NULL);// normal LBP
  
  if(dim!=MAX_NUM_LBP){
    printf("???\n");
    getchar();
  }
  
  for(int i=0;i<dim;i++)
    lbp.push_back( (double)feature[i] );

  if(feature) delete [] feature;
  
}
*/

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

static char *line_r = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
  int len;
  
  if(fgets(line_r,max_line_len,input) == NULL)
    return NULL;
  
  while(strrchr(line_r,'\n') == NULL)
    {
      max_line_len *= 2;
      line_r = (char *) realloc(line_r,max_line_len);
      len = (int) strlen(line_r);
      if(fgets(line_r+len,max_line_len-len,input) == NULL)
	break;
    }
  return line_r;
}

void bfo_load_features( string filename, vector<struct_bfo> &feature )
{
  FILE *fp = fopen( filename.c_str(), "r" );

  if(fp == NULL){
    fprintf(stderr,"can't open input file %s\n",filename.c_str());
    exit(1);
  }
  
  int num_prob = 0;//prob.l
  int elements, max_index, inst_max_index, i, j;
  char *endptr;
  char *idx, *val, *label;
  
  max_line_len = 1024;
  elements = 0;
  //line_r = malloc(char,max_line_len);
  line_r = (char*)malloc(max_line_len);
  while(readline(fp)!=NULL){
    char *p = strtok(line_r," \t"); // label
    
    // features
    while(1){
      p = strtok(NULL," \t");
      if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
	break;
      ++elements;
    }
    //++elements;
    ++num_prob;//prob.l;
  }
  rewind(fp);

  //cout << "elements " << elements << endl;
  int index;

  max_index= 0;
  for(i=0;i<num_prob;i++){
    struct_bfo tmp_feature;
    //tmp_feature.mat = Mat::zeros(1, elements, CV_64FC1);
    
    inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
    readline(fp);
    //prob.x[i] = &x_space[j];// byu
    label = strtok(line_r," \t\n");
    if(label == NULL) // empty line
      exit_input_error(i+1);
    
    //prob.y[i] = strtod(label,&endptr); // byu
    tmp_feature.label = strtod(label,&endptr);// check
    if(endptr == label || *endptr != '\0')
      exit_input_error(i+1);
    
    j = 0;
    while(1){
      idx = strtok(NULL,":");
      val = strtok(NULL," \t");
      
      if(val == NULL)
	break;
      
      //int errno = 0;
      //x_space[j].index = (int) strtol(idx,&endptr,10);
      index = (int) strtol(idx,&endptr,10);
      if(endptr == idx || *endptr != '\0' || index <= inst_max_index)
	exit_input_error(i+1);
      else
	inst_max_index = index;
      
      //x_space[j].value = strtod(val,&endptr);
      //tmp_feature.mat.at<double>(0,j) = strtod(val,&endptr);// check
      tmp_feature.feature.push_back( strtod(val,&endptr) );
      //printf("%lf ", tmp_feature.mat.at<double>(0,j));
      
      if(endptr == val || (*endptr != '\0' && !isspace(*endptr)))
	exit_input_error(i+1);
      
      ++j;
    }
    
    if(inst_max_index > max_index)
      max_index = inst_max_index;

    feature.push_back( tmp_feature );
  }
  
  fclose( fp );
}

int bfo_load_features_ballance( string filename, vector<struct_bfo> &feature, int next_label )
{
  FILE *fp = fopen( filename.c_str(), "r" );

  if(fp == NULL){
    fprintf(stderr,"can't open input file %s\n",filename.c_str());
    exit(1);
  }
  
  int num_prob = 0;//prob.l
  int elements, max_index, inst_max_index, i, j;
  char *endptr;
  char *idx, *val, *label;
  
  max_line_len = 1024;
  elements = 0;
  line_r = (char*)malloc(max_line_len);
  while(readline(fp)!=NULL){
    char *p = strtok(line_r," \t"); // label
    
    // features
    while(1){
      p = strtok(NULL," \t");
      if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
	break;
      ++elements;
    }
    ++num_prob;//prob.l;
  }
  rewind(fp);

  int index;

  max_index = 0;
  for(i=0;i<num_prob;i++){
    struct_bfo tmp_feature;
    
    inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
    readline(fp);
    label = strtok(line_r," \t\n");
    if(label == NULL) // empty line
      exit_input_error(i+1);
    
    tmp_feature.label = strtod(label,&endptr);// check
    if(endptr == label || *endptr != '\0')
      exit_input_error(i+1);
    
    j = 0;
    while(1){
      idx = strtok(NULL,":");
      val = strtok(NULL," \t");
      
      if(val == NULL)
	break;
      
      index = (int) strtol(idx,&endptr,10);
      if(endptr == idx || *endptr != '\0' || index <= inst_max_index)
	exit_input_error(i+1);
      else
	inst_max_index = index;
      
      tmp_feature.feature.push_back( strtod(val,&endptr) );// original
      
      if(endptr == val || (*endptr != '\0' && !isspace(*endptr)))
	exit_input_error(i+1);
      
      ++j;
    }
    
    if(inst_max_index > max_index)
      max_index = inst_max_index;

    if( next_label != tmp_feature.label ){
      feature.push_back( tmp_feature );
      next_label = tmp_feature.label;
    }

  }
  
  fclose( fp );
  return next_label;
}

 // svm+customized style feature load. Features of multiple objecdts are in one file. 
 // object_id + svm_style_features
  void bfo_load_features_multi_objects( string filename, vector< vector<struct_bfo>> &feature, std::string file_str )
{
  FILE *fp = fopen( filename.c_str(), "r" );
  //cout << filename << endl;

  if(fp == NULL){
    fprintf(stderr,"can't open input file %s\n",filename.c_str());
    exit(1);
  }
  
  int num_prob = 0;//prob.l
  int elements, max_index, inst_max_index, i, j;
  char *endptr;
  char *idx, *val, *label, *object_id;
  
  max_line_len = 1024;
  elements = 0;
  line_r = (char*)malloc(max_line_len);
  while(readline(fp)!=NULL){
    char *p = strtok(line_r," \t"); // object_id
    //char *p = strtok(line_r," \t"); // label
    p = strtok(NULL," \t"); // label
    
    // features
    while(1){
      p = strtok(NULL," \t");
      if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
	break;
      ++elements;
    }
    //++elements;
    ++num_prob;//prob.l;
  }
  rewind(fp);

  //cout << "elements " << elements << endl;
  int index;

  // roi_id label 1:xx 2:xx ...
  int pre_object_id = -1;
  max_index = 0;
  vector<struct_bfo> vec_tmp_feature;
  for(i=0;i<num_prob;i++){
    struct_bfo tmp_feature;
    
    inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
    readline(fp);

    object_id = strtok(line_r," \t\n");
    if(object_id == NULL) // empty line
      exit_input_error(i+1);

    //label = strtok(line_r," \t\n");
    label = strtok(NULL," \t\n");
    if(label == NULL) // empty line
      exit_input_error(i+1);
    
    tmp_feature.object_id = strtod(object_id,&endptr);// check
    if(endptr == object_id || *endptr != '\0')
      exit_input_error(i+1);

    tmp_feature.label = strtod(label,&endptr);// check
    if(endptr == label || *endptr != '\0')
      exit_input_error(i+1);

    tmp_feature.file_str = file_str;
    
    j = 0;
    while(1){
      idx = strtok(NULL,":");
      val = strtok(NULL," \t");
      
      if(val == NULL)
	break;
      
      index = (int) strtol(idx,&endptr,10);
      if(endptr == idx || *endptr != '\0' || index <= inst_max_index)
	exit_input_error(i+1);
      else
	inst_max_index = index;
      
      tmp_feature.feature.push_back( strtod(val,&endptr) );// original
      if(endptr == val || (*endptr != '\0' && !isspace(*endptr)))
	exit_input_error(i+1);
      
      ++j;
    }
    
    if(inst_max_index > max_index)
      max_index = inst_max_index;

    if( pre_object_id != tmp_feature.object_id ){
      pre_object_id = tmp_feature.object_id;
      if( vec_tmp_feature.size() > 0 )
	feature.push_back( vec_tmp_feature );
      vec_tmp_feature.clear();
    }
    vec_tmp_feature.push_back( tmp_feature );

  }
  if( vec_tmp_feature.size() > 0 )
    feature.push_back( vec_tmp_feature );
  
  fclose( fp );
}

  int bfo_load_features_multi_objects_ballance( string filename, vector< vector<struct_bfo>> &feature, int next_label, string file_str )
{
  FILE *fp = fopen( filename.c_str(), "r" );

  if(fp == NULL){
    fprintf(stderr,"can't open input file %s\n",filename.c_str());
    exit(1);
  }
  
  int num_prob = 0;//prob.l
  int elements, max_index, inst_max_index, i, j;
  char *endptr;
  char *idx, *val, *label, *object_id;
  
  max_line_len = 1024;
  elements = 0;
  line_r = (char*)malloc(max_line_len);
  while(readline(fp)!=NULL){
    char *p = strtok(line_r," \t"); // object_id
    //char *p = strtok(line_r," \t"); // label
    p = strtok(NULL," \t"); // label
    
    // features
    while(1){
      p = strtok(NULL," \t");
      if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
	break;
      ++elements;
    }
    ++num_prob;//prob.l;
  }
  rewind(fp);

  int index;

  // roi_id label 1:xx 2:xx ...
  int pre_object_id = -1;
  max_index = 0;
  vector<struct_bfo> vec_tmp_feature;
  for(i=0;i<num_prob;i++){
    struct_bfo tmp_feature;
    
    inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
    readline(fp);
    object_id = strtok(line_r," \t\n");
    if(object_id == NULL) // empty line
      exit_input_error(i+1);

    //label = strtok(line_r," \t\n");
    label = strtok(NULL," \t\n");
    if(label == NULL) // empty line
      exit_input_error(i+1);
    
    tmp_feature.object_id = strtod(object_id,&endptr);// check
    if(endptr == object_id || *endptr != '\0')
      exit_input_error(i+1);

    tmp_feature.label = strtod(label,&endptr);// check
    if(endptr == label || *endptr != '\0')
      exit_input_error(i+1);

    tmp_feature.file_str = file_str;
    
    j = 0;
    while(1){
      idx = strtok(NULL,":");
      val = strtok(NULL," \t");
      
      if(val == NULL)
	break;
      
      index = (int) strtol(idx,&endptr,10);
      if(endptr == idx || *endptr != '\0' || index <= inst_max_index)
	exit_input_error(i+1);
      else
	inst_max_index = index;
      
      tmp_feature.feature.push_back( strtod(val,&endptr) );// original
      
      if(endptr == val || (*endptr != '\0' && !isspace(*endptr)))
	exit_input_error(i+1);
      
      ++j;
    }
    
    if(inst_max_index > max_index)
      max_index = inst_max_index;

    if( pre_object_id != tmp_feature.object_id ){
      pre_object_id = tmp_feature.object_id;
      if( vec_tmp_feature.size() > 0 )
	feature.push_back( vec_tmp_feature );
      vec_tmp_feature.clear();
    }

    if( next_label != tmp_feature.label ){
      //feature.push_back( tmp_feature );
      vec_tmp_feature.push_back( tmp_feature );
      next_label = tmp_feature.label;
    }

  }
  if( vec_tmp_feature.size() > 0 )
    feature.push_back( vec_tmp_feature );
  
  fclose( fp );
  return next_label;
}

// output with libsvm format
void bfo_save_feature( FILE *fp_out, int label, vector<double> feature )
{
  fprintf( fp_out, "%d ", label );
  for(int i=0; i<feature.size(); i++)
    fprintf( fp_out, "%d:%lf ", (i+1), feature[i] );
  fprintf( fp_out, "\n" );
  fflush( fp_out );
}

void bfo_save_feature( FILE *fp_out, int label, cv::Mat mat )
{
  int rows = mat.rows;
  int cols = mat.cols;
  int chans = mat.channels();

  fprintf( fp_out, "%d ", label );

  int count = 1;
  if( chans == 3 ){
    for(int r=0; r<rows; r++){
      Vec3b* _mat = mat.ptr<Vec3b>(r);
      for(int c=0; c<cols; c++){
	for(int b=0; b<chans; b++){
	  fprintf( fp_out, "%d:%lf ", count, (double)_mat[c].val[b] );
	  count ++;
	}
      }
    }
  }
  else if( chans == 1 ){
    for(int r=0; r<rows; r++){
      uchar* _mat = mat.ptr<uchar>(r);
      for(int c=0; c<cols; c++){
	fprintf( fp_out, "%d:%lf ", count, (double)_mat[c] );
	count ++;
      }
    }
  }
  fprintf( fp_out, "\n" );
  fflush( fp_out );
}

void bfo_save_descriptor_feature( FILE *fp_out, int label, cv::Mat mat )
{
  int num_sample = mat.rows;
  int dim = mat.cols;

  cout << "bfo_save_descriptor_feature" << endl;

  int r, c;
  for(r=0; r<num_sample; r++){
    float* _mat = mat.ptr<float>(r);
    fprintf( fp_out, "%d ", label );
    for(c=0; c<dim; c++)
      fprintf( fp_out, "%d:%lf ", c+1, _mat[c] );
    fprintf( fp_out, "\n" );
  }    
  fflush( fp_out );
  
}

void bfo_save_descriptor_feature_roi( FILE *fp_out, int label, int roi_id, cv::Mat mat )
{
  int num_sample = mat.rows;
  int dim = mat.cols;

  cout << "bfo_save_descriptor_feature" << endl;

  int r, c;
  for(r=0; r<num_sample; r++){
    float* _mat = mat.ptr<float>(r);
    fprintf( fp_out, "%d %d ", roi_id, label );
    for(c=0; c<dim; c++)
      fprintf( fp_out, "%d:%lf ", c+1, _mat[c] );
    fprintf( fp_out, "\n" );
  }    
  fflush( fp_out );
  
}

void bfo_fprintf( FILE *fp_out, int label, vector<double> feature )
{
  fprintf( fp_out, "%d ", label );
  for(int i=0; i<feature.size(); i++)
    fprintf( fp_out, "%d:%lf ", (i+1), feature[i] );
  fprintf( fp_out, "\n" );
  fflush( fp_out );
}

  //flg_change_feature_as_0 0:do not change, 1:change
  void bfo_change_feature_as_0(vector<double> &feature, vector<struct_feature_dimension> feature_dim)
  {    
    // check feature size
    int dim = feature.size();
    if( dim != feature_dim[feature_dim.size()-1].end ){
      printf("in bfo_change_feature_as_0: input feature_dim is not set properry\n");
      getchar();
    }
    
    for(int ff=0; ff<feature_dim.size(); ff++){
      if( feature_dim[ff].flg_change_feature_as_0 == 1 ){
	for(int i=feature_dim[ff].start; i<feature_dim[ff].end; i++)
	  feature[i] = 0.0;
      }
    }
  }
  
void bfo_normalize_l1(vector<double> &feature, double lower, double upper)
{
  double value_max, value_min, value;
  value_max = -100000000000.0;
  value_min = 100000000000.0;

  for(int i=0; i<feature.size(); i++){
    value = feature[i];
    value_max = max(value_max, value);
    value_min = min(value_min, value);
  }

  for(int i=0; i<feature.size(); i++){
    value = feature[i];
    
    if (value == value_min)
      value = lower;
    else if (value == value_max)
      value = upper;
    else
      value = lower + (upper - lower) *
	(value - value_min) / (value_max - value_min);
    
    feature[i] = value;
  }

}

void bfo_normalize_l1(vector<double> &feature, vector<struct_feature_dimension> feature_dim, double lower, double upper)
{
  double value_max, value_min, value;

  // check feature size
  int dim = feature.size();
  if( dim != feature_dim[feature_dim.size()-1].end ){
    printf("in bfo_normalize: input feature_dim is not set properry\n");
    printf("input feature is %d, but you set as %d\n", dim, feature_dim[feature_dim.size()-1].end );
    getchar();
  }

  for(int ff=0; ff<feature_dim.size(); ff++){
    //printf("ff %d\n", ff);

    value_max = -100000000000.0;
    value_min = 100000000000.0;
    
    if( feature_dim[ff].flg_normalize == 0 ) continue;
    // if( feature_dim[ff].flg_normalize == -1 ){
    //   for(int i=feature_dim[ff].start; i<feature_dim[ff].end; i++)
    // 	value = 0.0;
    //   continue;
    // }

    //for(int i=0; i<feature.size(); i++){
    for(int i=feature_dim[ff].start; i<feature_dim[ff].end; i++){
      value = feature[i];
      value_max = max(value_max, value);
      value_min = min(value_min, value);
    }
    
    //for(int i=0; i<feature.size(); i++){
    for(int i=feature_dim[ff].start; i<feature_dim[ff].end; i++){
      value = feature[i];
      
      if (value == value_min)
	value = lower;
      else if (value == value_max)
	value = upper;
      else
	value = lower + (upper - lower) *
	  (value - value_min) / (value_max - value_min);
      
      feature[i] = value;
    }
  }

}

void bfo_normalize(vector<double> &feature, double threshold)
{
  double value;
  for(int i=0; i<feature.size(); i++){
    value = feature[i] / threshold;    
    feature[i] = value;
  }
}

void bfo_normalize_l2(vector<double> &feature)
{
  double value_total, value;

  value_total = 0;
  for(int i=0; i<feature.size(); i++){
    value = feature[i];
    value_total += (value * value);
  }
  value_total = sqrt( value_total );

  for(int i=0; i<feature.size(); i++){
    value = feature[i];
    feature[i] = (value / value_total);
  }

}

void bfo_normalize_l2(vector<double> &feature, vector<struct_feature_dimension> feature_dim)
{
  double value_total, value;

  // check feature size
  int dim = feature.size();
  if( dim != feature_dim[feature_dim.size()-1].end ){
    printf("in bfo_normalize: input feature_dim is not set properry\n");
    getchar();
  }
  
  for(int ff=0; ff<feature_dim.size(); ff++){

    if( feature_dim[ff].flg_normalize == 0 ) continue;

    value_total = 0;
    //for(int i=0; i<feature.size(); i++){
    for(int i=feature_dim[ff].start; i<feature_dim[ff].end; i++){
      value = feature[i];
      value_total += (value * value);
    }
    value_total = sqrt( value_total );
    
    //for(int i=0; i<feature.size(); i++){
    for(int i=feature_dim[ff].start; i<feature_dim[ff].end; i++){
      value = feature[i];
      feature[i] = (value / value_total);
    }
  }

}

void bfo_apply_1DPCA( vector<struct_bfo> data, struct_1dpca &pca )
{
  int num_data = data.size();
  int dim = data[0].feature.size();

  Mat mat = Mat::zeros( num_data, dim, CV_64FC1 );
  Mat ave_mat = Mat::zeros( 1, dim, CV_64FC1 );

  double* _ave_mat = ave_mat.ptr<double>(0);
  for(int i=0; i<num_data; i++){
    double* _mat = mat.ptr<double>(i);
    for(int j=0; j<dim; j++){
      //mat.at<double>(i,j) = data[i].feature[j];
      //ave_mat.at<double>(0,j) += data[i].feature[j];
      _mat[j] = data[i].feature[j];
      _ave_mat[j] += data[i].feature[j];
    }
  }
  ave_mat /= (double)num_data;

  for(int i=0; i<num_data; i++){
    double* _mat = mat.ptr<double>(i);
    for(int j=0; j<dim; j++)
      //mat.at<double>(i,j) -= ave_mat.at<double>(0,j);
      _mat[j] -= _ave_mat[j];
  }
  
  Mat mat_mul;
  Mat eigenValues;
  Mat eigenVectors;
  
  mat_mul =mat.t()* mat;
  eigen(mat_mul, eigenValues, eigenVectors);
  
  //rankの計算
  int rank = 0;
  for (int i = 0; i < eigenValues.rows; i++){
    if (eigenValues.at<double>(i, 0) != 0)
      rank += 1;
  }
  eigenVectors = eigenVectors.rowRange(Range(0, rank));
  eigenValues = eigenValues.rowRange(Range(0, rank));

  pca.eigenVectors = eigenVectors.clone(); // row-majored matrix (column is dimension)
  pca.eigenValues = eigenValues.clone();
  pca.averageVector = ave_mat.clone();
}

void bfo_do_whitening( vector<double> &feature, struct_1dpca &pca )
{
  int dim = feature.size();
  int rank = pca.eigenVectors.rows;

  // pca.eigenVectors: rank x dim
  // pca.eigenValues: rank x 1
  // pca.averageVector: 1 x dim

  // mat: 1 x dim
  Mat mat = Mat::zeros( 1, dim, CV_64FC1 );
  double* _mat = mat.ptr<double>(0);
  double* _pca_averageVector = pca.averageVector.ptr<double>(0);
  for(int j=0; j<dim; j++)
    //mat.at<double>(0,j) = feature[j] - pca.averageVector.at<double>(0,j);
    _mat[j] = feature[j] - _pca_averageVector[j];

  // new_eigen = eigenVectors(i,:) / sqrt( eigenValues(i,0) )
  Mat new_eigenVectors = pca.eigenVectors.clone();
  for (int i = 0; i < new_eigenVectors.rows; i++){
    double value = sqrt( pca.eigenValues.at<double>(i,0) );
    double* _new_eigenVectors = new_eigenVectors.ptr<double>(i);
    for (int j = 0; j < new_eigenVectors.cols; j++)
      //new_eigenVectors.at<double>(i,j) /= value;
      _new_eigenVectors[j] /= value;
  }

  // project_mat (1 x rank) = mat x pca.eigenVectors^T
  Mat project_mat = Mat::zeros( 1, rank, CV_64FC1 );
  project_mat = mat * new_eigenVectors.t();

  double* _project_mat = project_mat.ptr<double>(0);
  if( feature.size() > 0 ) feature.clear();
  for(int i=0; i<rank; i++)
    //feature.push_back( project_mat.at<double>(0,i) );
    feature.push_back( _project_mat[i] );
}

  // Mat bfo_feature_descriptor( Mat src_mat, std::string feature_type )
  // {
  //   cv::Ptr<cv::FeatureDetector> detector;
  //   cv::Ptr<cv::DescriptorExtractor> extractor;
  //   std::vector<cv::KeyPoint> keypoints;
  //   cv::Mat descriptors;

  //   cv::DenseFeatureDetector detector_dense(
  //       1.f,     //initFeatureScale:   初期の特徴のサイズ（直径）[px]
  //       1,        //featureScaleLevels: 何段階サイズ変更してしてサンプリングするか(>=1)
  //       0.1f,   //featureScaleMul:    ScaleLevelごとにどれくらい拡大縮小するか(!=0)
  //       10,        //initXyStep:         特徴点をどれくらいの量ずらしながらサンプリングするか
  //       0,        //initImgBound:       画像の端からどれくらい離すか(>=0)
  //       true,    //varyXyStepWithScale:    XyStepにもScaleMul を掛けるか
  //       false     //varyImgBoundWithScale:  BoundにもScaleMul を掛けるか
  // 					    );
    
  //   if( feature_type == "SIFT" ){
  //     cv::initModule_nonfree();
  //     detector = FeatureDetector::create("SIFT");
  //     detector->detect(src_mat, keypoints);
  //     extractor = cv::DescriptorExtractor::create("SIFT");
  //     extractor->compute(src_mat, keypoints, descriptors);
  //   }
  //   else if( feature_type == "SURF" ){
  //     cv::initModule_nonfree();
  //     detector = FeatureDetector::create("SURF");
  //     //detector->set("hessianThreshold", 1500.0);
  //     detector->detect(src_mat, keypoints);
  //     extractor = cv::DescriptorExtractor::create("SURF");
  //     extractor->compute(src_mat, keypoints, descriptors);
  //   }
  //   else if( feature_type == "ORB" ){
  //     detector = FeatureDetector::create("ORB");
  //     detector->detect(src_mat, keypoints);
  //     extractor = cv::DescriptorExtractor::create("ORB");
  //     extractor->compute(src_mat, keypoints, descriptors);
  //   }
  //   else if( feature_type == "BRISK" ){
  //     detector = FeatureDetector::create("BRISK");
  //     detector->detect(src_mat, keypoints);
  //     extractor = cv::DescriptorExtractor::create("BRISK");
  //     extractor->compute(src_mat, keypoints, descriptors);
  //   }
  //   if( feature_type == "DENSE_SIFT" ){
  //     cv::initModule_nonfree();
  //     //detector = FeatureDetector::create("Dense");
  //     detector_dense.detect(src_mat, keypoints);
  //     extractor = cv::DescriptorExtractor::create("SIFT");
  //     extractor->compute(src_mat, keypoints, descriptors);
  //   }
  //   else if( feature_type == "DENSE_SURF" ){
  //     cv::initModule_nonfree();
  //     //detector = FeatureDetector::create("Dense");
  //     //detector->set("hessianThreshold", 1500.0);
  //     detector_dense.detect(src_mat, keypoints);
  //     extractor = cv::DescriptorExtractor::create("SURF");
  //     extractor->compute(src_mat, keypoints, descriptors);
  //   }
  //   else if( feature_type == "DENSE_ORB" ){
  //     //detector = FeatureDetector::create("Dense");
  //     detector_dense.detect(src_mat, keypoints);
  //     extractor = cv::DescriptorExtractor::create("ORB");
  //     extractor->compute(src_mat, keypoints, descriptors);
  //   }
  //   else if( feature_type == "DENSE_BRISK" ){
  //     //detector = FeatureDetector::create("Dense");
  //     detector_dense.detect(src_mat, keypoints);
  //     extractor = cv::DescriptorExtractor::create("BRISK");
  //     extractor->compute(src_mat, keypoints, descriptors);
  //   }

  //   return descriptors.clone();
  // }

  void bfo_copy_struct_bfo_2_Mat( std::vector<struct_bfo> data, cv::Mat &mat_dst )
  {
    int num_data = data.size();
    int dim = data[0].feature.size();

    mat_dst = Mat::zeros( num_data, dim, CV_64FC1 );
    int r, c;
    for(r=0; r<num_data; r++){
      double* _mat = mat_dst.ptr<double>(r);
      for(c=0; c<dim; c++)
	_mat[c] = data[r].feature[c];
    }
  }

  void bfo_copy_struct_bfo_2_Mat( std::vector< std::vector<struct_bfo>> data, cv::Mat &mat_dst )
  {
    int rr, r, c;
    int dim = data[0][0].feature.size();
    int num_data = 0;
    for(rr=0; rr<data.size(); rr++)
      num_data += data[rr].size();

    mat_dst = Mat::zeros( num_data, dim, CV_64FC1 );

    num_data = 0;
    for(rr=0; rr<data.size(); rr++){
      for(r=0; r<data[rr].size(); r++){
	double* _mat = mat_dst.ptr<double>(num_data);
	for(c=0; c<dim; c++)
	  _mat[c] = data[rr][r].feature[c];
	num_data ++;
      }
    }
  }
  
#pragma GCC diagnostic pop

} // namespace ibo
