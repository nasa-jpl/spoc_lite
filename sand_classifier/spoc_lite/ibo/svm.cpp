/*
  author: Yumi Iwashita
*/

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/nonfree/nonfree.hpp>

#include "bfo.h"
#include "svm.h"
#include "libsvm.h"

using namespace cv;
using namespace std;

namespace ibo
{

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wunused-variable"

LIBSVM::LIBSVM(vector<struct_bfo> train)
{
  this->num_dimension = train[0].feature.size();
  //this->num_dimension = num_dimension;
  
  // prepare memory for test
  AllStoredData_Test = new struct svm_node[num_dimension + 1];
  AllStoredData_Test[num_dimension].index = -1;

  for(int i=0; i<num_dimension; i++)
    AllStoredData_Test[i].index = i+1;
  flg_AllStoredData_Test = true;
}

LIBSVM::LIBSVM(int num_dimension)
{
  if( num_dimension > 0 ){
    //num_dimension = train[0].feature.size();
    this->num_dimension = num_dimension;
    
    // prepare memory for test
    AllStoredData_Test = new struct svm_node[num_dimension + 1];
    AllStoredData_Test[num_dimension].index = -1;
    
    for(int i=0; i<num_dimension; i++)
      AllStoredData_Test[i].index = i+1;
    flg_AllStoredData_Test = true;
  }
}
  
LIBSVM::~LIBSVM()
{
  printf("flg_AllStoredData_Test %d\n", flg_AllStoredData_Test );

  if( !flg_AllStoredData_Test ) return;

  svm_free_and_destroy_model(&SvmTrain);//’†g‚ð”j‰ó
  svm_destroy_param(&param);
  delete[] SvmProblem.y;// this is the problem
  delete[] SvmProblem.x;
}

void LIBSVM::DummyGetMemory()
{
  int size = 1;
  SvmProblem.l = size;
  SvmProblem.y = new double[SvmProblem.l];
  SvmProblem.x = new svm_node*[SvmProblem.l];
}

void LIBSVM::SetParameters(int kernel_type, double c, double g)
{
  param.svm_type = C_SVC;
  //param.svm_type = NU_SVC;
  //param.svm_type = ONE_CLASS;
  param.kernel_type = kernel_type;//ORIGINAL;//HI;//CHI2;//RBF;//LINEAR;//POLY//SIGMOID;
  
  param.gamma = pow(2.0, g);
  param.C = pow(2.0, c);
  
  // the following 4 parameters could be changed. 
#if 0 // param1
  param.degree = 0.05;
  param.coef0 = 0.8;
  param.nu = 0.000001;
  param.cache_size = 1;
#else // param2
  param.degree = 3;
  param.coef0 = 0;
  param.nu = 0.5;//0.001067189;//0.5;
  param.cache_size = 100;
  //param.C = 1;
#endif
  
  param.eps = 1e-3;//1e-3;
  param.p = 0.1;
  param.shrinking = 1;
  param.probability = 1;
  //param.probability = 0;
  param.nr_weight = 0;
  param.weight_label = NULL;
  param.weight = NULL;
}

void LIBSVM::SetParameters(int svm_type, int kernel_type, int probability, double c, double g)
{
  param.svm_type = svm_type;// C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR
  param.kernel_type = kernel_type;//ORIGINAL;//HI;//CHI2;//RBF;//LINEAR;//POLY//SIGMOID;
  
  param.gamma = pow(2.0, g);
  param.C = pow(2.0, c);
  
  // the following 4 parameters could be changed. 
#if 0 // param1
  param.degree = 0.05;
  param.coef0 = 0.8;
  param.nu = 0.000001;
  param.cache_size = 1;
#else // param2
  param.degree = 3;
  param.coef0 = 0;
  param.nu = 0.5;//0.001067189;//0.5;
  param.cache_size = 100;
  //param.C = 1;
#endif
  
  param.eps = 1e-3;//1e-3;
  param.p = 0.1;
  param.shrinking = 1;
  param.probability = probability; // 0: off, 1: on
  param.nr_weight = 0;
  param.weight_label = NULL;
  param.weight = NULL;
}

void LIBSVM::SetParameters( svm_parameter input_param, double c, double g )
//int svm_type, int kernel_type, int probability, double c, double g)
{
  param.svm_type = input_param.svm_type;// C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR
  param.kernel_type = input_param.kernel_type;//ORIGINAL;//HI;//CHI2;//RBF;//LINEAR;//POLY//SIGMOID;

  param.degree = input_param.degree;
  param.coef0 = input_param.coef0;
  param.nu = input_param.nu;
  param.cache_size = input_param.cache_size;
  param.probability = input_param.probability; // 0: off, 1: on
  
  param.gamma = pow(2.0, g);
  param.C = pow(2.0, c);

  
  /*  
  // the following 4 parameters could be changed. 
#if 0 // param1
  param.degree = 0.05;
  param.coef0 = 0.8;
  param.nu = 0.000001;
  param.cache_size = 1;
#else // param2
  param.degree = 3;
  param.coef0 = 0;
  param.nu = 0.5;//0.001067189;//0.5;
  param.cache_size = 100;
  //param.C = 1;
#endif
  */

  param.eps = 1e-3;//1e-3;
  param.p = 0.1;
  param.shrinking = 1;
  param.nr_weight = 0;
  param.weight_label = NULL;
  param.weight = NULL;
}

void LIBSVM::svmTrain(vector<struct_bfo> train)
{
  this->num_dimension = train[0].feature.size();
  int num_train_data = train.size();
  
  // store training data to temporal memory
  AllStoredData_Train = new struct svm_node*[num_train_data];
  for (int k = 0; k < num_train_data; k++){
    AllStoredData_Train[k] = new struct svm_node[num_dimension + 1];
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
  SvmProblem.y = new double[SvmProblem.l];
  SvmProblem.x = new svm_node*[SvmProblem.l];
  
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
  
  int num_concat = 1;
  int *each_part_dim = new int[num_concat];
  for (int i = 0; i<num_concat; i++) each_part_dim[i] = num_dimension;
  svm_train_prepare(&SvmProblem, &param, num_concat, each_part_dim, num_class);
  
  // training data
  SvmTrain = svm_train(&SvmProblem, &param);

  this->svm_nr_class = this->SvmTrain->nr_class;
  this->svm_labels = this->SvmTrain->label;
  
  if (check_class) delete[] check_class;
  if (each_part_dim) delete[] each_part_dim;
}

void LIBSVM::svmSaveModel( const char *file_name )
{
  svm_save_model( file_name, this->SvmTrain );
}

void LIBSVM::svmLoadModel( const char *file_name )
{
  this->SvmTrain = svm_load_model( file_name );
  this->svm_nr_class = this->SvmTrain->nr_class;
  this->svm_labels = this->SvmTrain->label;
}

void LIBSVM::svmPredict(struct_bfo test)
{
  if( !flg_AllStoredData_Test ){
    num_dimension = test.feature.size();
    // prepare memory for test
    AllStoredData_Test = new struct svm_node[num_dimension + 1];
    AllStoredData_Test[num_dimension].index = -1;

    for(int i=0; i<num_dimension; i++)
      AllStoredData_Test[i].index = i+1;
    flg_AllStoredData_Test = true;
  }

  //Mat img = test.mat.clone();
  //Mat img_tmp;// <- new
  //img.convertTo(img_tmp, CV_32F);// <- new
  
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
    //AllStoredData_Test[i].index = _index_counter;
    _index_counter++;
  }
#else
  for(int i=0; i<num_dimension; i++){
    svm_node *node = AllStoredData_Test + i;
    node->value = test.feature[i];
    //AllStoredData_Test[i].value = test.feature[i];
    //AllStoredData_Test[i].index = _index_counter;
    //_index_counter++;
  }
#endif
  
  TestData = AllStoredData_Test;
  
  this->svm_predicted_class = svm_predict(SvmTrain, TestData);
}

void LIBSVM::svmPredict(vector<double> feature)
{
    //double value = (double)test.label;//GetClassID();
  
  int _index_counter = 1;

  if( !flg_AllStoredData_Test ){
    num_dimension = feature.size();
    // prepare memory for test
    AllStoredData_Test = new struct svm_node[num_dimension + 1];
    AllStoredData_Test[num_dimension].index = -1;

    for(int i=0; i<num_dimension; i++)
      AllStoredData_Test[i].index = i+1;
    flg_AllStoredData_Test = true;
  }

#if 0
  // original  
  for(int i=0; i<num_dimension; i++){
    AllStoredData_Test[i].value = feature[i];
    //AllStoredData_Test[i].index = _index_counter;
    _index_counter++;
  }
#else
  for(int i=0; i<num_dimension; i++){
    svm_node *node = AllStoredData_Test + i;
    node->value = feature[i];
    //AllStoredData_Test[i].value = test.feature[i];
    //AllStoredData_Test[i].index = _index_counter;
    //_index_counter++;
  }
#endif
  
  TestData = AllStoredData_Test;
  
  this->svm_predicted_class = svm_predict(SvmTrain, TestData);
}

void LIBSVM::svmPredictProbability(struct_bfo test, double *probability)
{
  double value = (double)test.label;//GetClassID();
  
  int _index_counter = 1;
  
  if( !flg_AllStoredData_Test ){
    num_dimension = test.feature.size();
    // prepare memory for test
    AllStoredData_Test = new struct svm_node[num_dimension + 1];
    AllStoredData_Test[num_dimension].index = -1;

    for(int i=0; i<num_dimension; i++)
      AllStoredData_Test[i].index = i+1;
    flg_AllStoredData_Test = true;
  }

#if 0
  // original
  for(int i=0; i<num_dimension; i++){
    AllStoredData_Test[i].value = test.feature[i];
    //AllStoredData_Test[i].index = _index_counter;
    _index_counter++;
  }
#else
  // 3 msec
  for(int i=0; i<num_dimension; i++){
    svm_node *node = AllStoredData_Test + i;
    node->value = test.feature[i];
  }
#endif
  
  TestData = AllStoredData_Test;
  
  //cout << svm_get_nr_class(SvmTrain) << endl;
  //getchar();

  this->svm_predicted_class = svm_predict_probability(SvmTrain, TestData, probability);

  //for(int j=0; j<2; j++)
  //printf("predicted_class %d -- %d: %lf\n", this->svm_predicted_class, j, probability[j]);
}

void LIBSVM::svmPredictProbability(struct_bfo *test, double *probability)
{
  double value = (double)test->label;//GetClassID();
  
  int _index_counter = 1;
  
  if( !flg_AllStoredData_Test ){
    num_dimension = test->feature.size();
    // prepare memory for test
    AllStoredData_Test = new struct svm_node[num_dimension + 1];
    AllStoredData_Test[num_dimension].index = -1;

    for(int i=0; i<num_dimension; i++)
      AllStoredData_Test[i].index = i+1;
    flg_AllStoredData_Test = true;
  }

#if 0
  // original
  for(int i=0; i<num_dimension; i++){
    AllStoredData_Test[i].value = test->feature[i];
    //AllStoredData_Test[i].index = _index_counter;
    _index_counter++;
  }
#else
  for(int i=0; i<num_dimension; i++){
    svm_node *node = AllStoredData_Test + i;
    node->value = test->feature[i];
  }
#endif
  
  TestData = AllStoredData_Test;
  
  this->svm_predicted_class = svm_predict_probability(SvmTrain, TestData, probability);

}

void LIBSVM::svmPredictProbability(vector<double> feature, double *probability)
{
  //double value = (double)test.label;//GetClassID();
  
  int _index_counter = 1;
  
  if( !flg_AllStoredData_Test ){
    num_dimension = feature.size();
    // prepare memory for test
    AllStoredData_Test = new struct svm_node[num_dimension + 1];
    AllStoredData_Test[num_dimension].index = -1;

    for(int i=0; i<num_dimension; i++)
      AllStoredData_Test[i].index = i+1;
    flg_AllStoredData_Test = true;
  }

#if 0
  // original
  for(int i=0; i<num_dimension; i++){
    AllStoredData_Test[i].value = feature[i];
    //AllStoredData_Test[i].index = _index_counter;
    _index_counter++;
  }
#else
  for(int i=0; i<num_dimension; i++){
    svm_node *node = AllStoredData_Test + i;
    node->value = feature[i];
    //AllStoredData_Test[i].value = test.feature[i];
    //AllStoredData_Test[i].index = _index_counter;
    //_index_counter++;
  }
#endif
  
  TestData = AllStoredData_Test;
  
  this->svm_predicted_class = svm_predict_probability(SvmTrain, TestData, probability);

}


#pragma GCC diagnostic pop

} // namespace ibo
