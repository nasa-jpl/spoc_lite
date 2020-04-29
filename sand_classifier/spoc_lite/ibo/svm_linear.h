/*
  author: Yumi Iwashita
*/

#ifndef __SVM_H__
#define __SVM_H__

#include "bfo.h"

#include "linearsvm/linear.h"
#include "linearsvm/tron.h"

#define INF HUGE_VAL

class LIBSVM_LINEAR
{
 private:
  struct problem SvmProblem;
  struct parameter param;
  struct model *SvmTrain;
  struct feature_node *TestData;
  
  struct feature_node **AllStoredData_Train;
  struct feature_node *AllStoredData_Test;
  bool flg_AllStoredData_Test = false;
  
  int mat_cols;
  int mat_rows;
  int nchannels;
  int num_dimension;
  
  int svm_predicted_class;
  
  int flag_cross_validation;
  int flag_C_specified;
  int flag_solver_specified;
  int flag_find_C;
  int nr_fold;
  double bias;
  
  double *svm_predicted_probability;
  //int flag_predict_probability;
  
 public:
  LIBSVM_LINEAR(std::vector<ibo::struct_bfo> train);
  LIBSVM_LINEAR( int num_dimension );
  ~LIBSVM_LINEAR();
  
  //void SetParameters(int kernel_type, double c, double g);
  void SetParameters(int solver_type, int flag_find_C);
  void svmTrain(std::vector<ibo::struct_bfo> train);
  void svmPredict(ibo::struct_bfo test);
  void svmPredictProbability(ibo::struct_bfo test, double *probability);
  int svmGetClass() { return svm_predicted_class; }
  int svmGetNumClass();
  
  void DummyGetMemory();
  void svmSaveModel( const char *file_name );
  void svmLoadModel( const char *file_name );
  
  void do_find_parameter_C();
	
};

#endif

