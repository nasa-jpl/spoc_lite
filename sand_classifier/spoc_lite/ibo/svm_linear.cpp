/*
  author: Yumi Iwashita
*/

#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <dirent.h>

#include <time.h>
#include <sys/timeb.h>
#include <sys/time.h>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/nonfree/nonfree.hpp>

#include "bfo.h"
#include "svm_linear.h"

using namespace cv;
using namespace std;
using namespace ibo;

LIBSVM_LINEAR::LIBSVM_LINEAR(vector<struct_bfo> train)
{
  //mat_cols = train[0].mat.cols;
  //mat_rows = train[0].mat.rows;
  //nchannels = train[0].mat.channels();
  
  // set dimention
  //num_dimension = mat_cols * mat_rows * nchannels;
  num_dimension = train[0].feature.size();
  
  // prepare memory for test
  AllStoredData_Test = new struct feature_node[num_dimension + 1];
  AllStoredData_Test[num_dimension].index = -1;

  for(int i=0; i<num_dimension; i++)
    AllStoredData_Test[i].index = i+1;
  flg_AllStoredData_Test = true;
}

LIBSVM_LINEAR::LIBSVM_LINEAR(int num_dimension)
{
  if( num_dimension > 0 ){
    //num_dimension = train[0].feature.size();
    this->num_dimension = num_dimension;
    
    // prepare memory for test
    AllStoredData_Test = new struct feature_node[num_dimension + 1];
    AllStoredData_Test[num_dimension].index = -1;

    for(int i=0; i<num_dimension; i++)
      AllStoredData_Test[i].index = i+1;
    flg_AllStoredData_Test = true;
  }
}

LIBSVM_LINEAR::~LIBSVM_LINEAR()
{
  if( !flg_AllStoredData_Test ) return;

  // CHECK
  svm_linear_free_and_destroy_model(&SvmTrain);//’†g‚ð”j‰ó
  svm_linear_destroy_param(&param);
  delete[] SvmProblem.y;
  delete[] SvmProblem.x;
}

void LIBSVM_LINEAR::DummyGetMemory()
{
  int size = 1;
  SvmProblem.l = size;
  SvmProblem.y = new double[SvmProblem.l];
  SvmProblem.x = new feature_node*[SvmProblem.l];
}


void LIBSVM_LINEAR::SetParameters(int solver_type, int flag_find_C)
//void LIBSVM_LINEAR::SetParameters(int kernel_type, double c, double g)
{
  // CHECK
  // default values
  param.solver_type = solver_type;//L2R_L2LOSS_SVC_DUAL;
  param.C = 1;
  param.eps = INF; // see setting below
  param.p = 0.1;
  param.nr_weight = 0;
  param.weight_label = NULL;
  param.weight = NULL;
  param.init_sol = NULL;

  this->flag_cross_validation = 0;
  this->flag_C_specified = 0;
  this->flag_solver_specified = 0;
  this->flag_find_C = flag_find_C;
  this->nr_fold = 5;
  this->bias = -1;
  //this->flag_predict_probability = 0;  

  if(param.eps == INF){
    switch(param.solver_type){
    case L2R_LR:
    case L2R_L2LOSS_SVC:
      param.eps = 0.01;
      break;
    case L2R_L2LOSS_SVR:
      param.eps = 0.001;
      break;
	case L2R_L2LOSS_SVC_DUAL:
    case L2R_L1LOSS_SVC_DUAL:
    case MCSVM_CS:
    case L2R_LR_DUAL:
      param.eps = 0.1;
      break;
    case L1R_L2LOSS_SVC:
    case L1R_LR:
      param.eps = 0.01;
      break;
    case L2R_L1LOSS_SVR_DUAL:
    case L2R_L2LOSS_SVR_DUAL:
      param.eps = 0.1;
      break;
    }
  }
  
}

void LIBSVM_LINEAR::svmTrain(vector<struct_bfo> train)
{
  num_dimension = train[0].feature.size();
  int num_train_data = train.size();
  
  // store training data to temporal memory
  AllStoredData_Train = new struct feature_node*[num_train_data];
  for (int k = 0; k < num_train_data; k++){
    AllStoredData_Train[k] = new struct feature_node[num_dimension + 1];
    AllStoredData_Train[k][num_dimension].index = -1;
    
    double value = (double)train[k].label;//GetClassID();
    
    int _index_counter = 1;
    //Mat img = train[k].mat.clone();
    //Mat img_tmp;// <- new
    //img.convertTo(img_tmp, CV_32F);// <- new
    
    //for (int i = 0; i < mat_rows; i++){
    //for (int j = 0; j < mat_cols; j++){
    //	for (int c = 0; c < nchannels; c++){
    //	  AllStoredData_Train[k][i*mat_cols*nchannels + j*nchannels + c].value = img_tmp.at<float>(i, j);// <- new
    //	  AllStoredData_Train[k][i*mat_cols*nchannels + j*nchannels + c].index = _index_counter;
    //	  _index_counter++;
    //	}
    //}
    //}
    for(int i=0; i<num_dimension; i++){
      AllStoredData_Train[k][i].value = train[k].feature[i];
      AllStoredData_Train[k][i].index = _index_counter;
      _index_counter++;
    }
    
  }
  
  // prepare training
  SvmProblem.l = num_train_data;
  SvmProblem.n = num_dimension + 1;
  SvmProblem.y = new double[SvmProblem.l];
  SvmProblem.x = new feature_node*[SvmProblem.l];
  SvmProblem.bias = bias;
  
  int num_class = 0;
  int *check_class = new int[num_train_data];
  memset(check_class, 0, sizeof(int)*num_train_data);
  for (int k = 0; k < num_train_data; k++){
    SvmProblem.y[k] = train[k].label;//GetClassID();
    check_class[train[k].label] ++;
    for (int j = 0; j<num_dimension; j++)
      SvmProblem.x[k] = AllStoredData_Train[k];
  }
  for (int k = 0; k < num_train_data; k++){
    if (check_class[k] > 0)
      num_class++;
  }
  
  //int num_concat = 1;
  //int *each_part_dim = new int[num_concat];
  //for (int i = 0; i<num_concat; i++) each_part_dim[i] = num_dimension;
  //svm_linear_train_prepare(&SvmProblem, &param, num_concat, each_part_dim, num_class);
  

  if (flag_find_C)
    do_find_parameter_C();

  
  // training data
  SvmTrain = svm_linear_train(&SvmProblem, &param);
  
  
  if (check_class) delete[] check_class;
  //if (each_part_dim) delete[] each_part_dim;
}

int LIBSVM_LINEAR::svmGetNumClass()
{
  return svm_linear_get_nr_class( SvmTrain );
}

void LIBSVM_LINEAR::do_find_parameter_C()
{
  double start_C, best_C, best_rate;
  double max_C = 1024;
  if (flag_C_specified)
    start_C = param.C;
  else
    start_C = -1.0;
  printf("Doing parameter search with %d-fold cross validation.\n", nr_fold);
  svm_linear_find_parameter_C(&SvmProblem, &param, nr_fold, start_C, max_C, &best_C, &best_rate);
  printf("Best C = %g  CV accuracy = %g%%\n", best_C, 100.0*best_rate);
}

void LIBSVM_LINEAR::svmSaveModel( const char *file_name )
{
  svm_linear_save_model( file_name, this->SvmTrain );
}

void LIBSVM_LINEAR::svmLoadModel( const char *file_name )
{
  this->SvmTrain = svm_linear_load_model( file_name );
}

void LIBSVM_LINEAR::svmPredict(struct_bfo test)
{
  //Mat img = test.mat.clone();
  //Mat img_tmp;// <- new
  //img.convertTo(img_tmp, CV_32F);// <- new

  if( !flg_AllStoredData_Test ){
    num_dimension = test.feature.size();
    // prepare memory for test
    AllStoredData_Test = new struct feature_node[num_dimension + 1];
    AllStoredData_Test[num_dimension].index = -1;

    for(int i=0; i<num_dimension; i++)
      AllStoredData_Test[i].index = i+1;
    flg_AllStoredData_Test = true;
  }
  
  double value = (double)test.label;//GetClassID();
  
  int _index_counter = 1;
  //for (int i = 0; i < mat_rows; i++){
  //  for (int j = 0; j < mat_cols; j++){
  //    for (int c = 0; c < nchannels; c++){
  //	AllStoredData_Test[i*mat_cols*nchannels + j*nchannels + c].value = img_tmp.at<float>(i, j);// <- new
  //	AllStoredData_Test[i*mat_cols*nchannels + j*nchannels + c].index = _index_counter;
  //	_index_counter++;
  //  }
  //  }
  //}
  
#if 0
  // original  
  for(int i=0; i<num_dimension; i++){
    AllStoredData_Test[i].value = test.feature[i];
    AllStoredData_Test[i].index = _index_counter;
    _index_counter++;
  }
#else
  for(int i=0; i<num_dimension; i++){
    feature_node *node = AllStoredData_Test + i;
    node->value = test.feature[i];
  }
#endif
  
  TestData = AllStoredData_Test;
  
  this->svm_predicted_class = (int)svm_linear_predict(SvmTrain, TestData);
}

void LIBSVM_LINEAR::svmPredictProbability(struct_bfo test, double *probability)
{
  if( !flg_AllStoredData_Test ){
    num_dimension = test.feature.size();
    // prepare memory for test
    AllStoredData_Test = new struct feature_node[num_dimension + 1];
    AllStoredData_Test[num_dimension].index = -1;

    for(int i=0; i<num_dimension; i++)
      AllStoredData_Test[i].index = i+1;
    flg_AllStoredData_Test = true;
  }

  double value = (double)test.label;//GetClassID();  
  int _index_counter = 1;
  
#if 0
  // original  
  for(int i=0; i<num_dimension; i++){
    AllStoredData_Test[i].value = test.feature[i];
    AllStoredData_Test[i].index = _index_counter;
    _index_counter++;
  }
#else
  for(int i=0; i<num_dimension; i++){
    feature_node *node = AllStoredData_Test + i;
    node->value = test.feature[i];
  }
#endif
  
  TestData = AllStoredData_Test;
  
  //cout << svm_get_nr_class(SvmTrain) << endl;
  //getchar();
  
  if(!svm_linear_check_probability_model(SvmTrain)){
    fprintf(stderr, "probability output is only supported for logistic regression (like L2R_LR_DUAL)\n");
    exit(1);
  }
  
  this->svm_predicted_class = (int)svm_linear_predict_probability(SvmTrain, TestData, probability);

  //for(int j=0; j<2; j++)
  //printf("predicted_class %d -- %d: %lf\n", this->svm_linear_predicted_class, j, probability[j]);
}
