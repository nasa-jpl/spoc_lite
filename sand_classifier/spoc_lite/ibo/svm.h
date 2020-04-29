/*
  author: Yumi Iwashita
*/

#ifndef __IBO_SUPPORT_VECTOR_MACHINE_OPERATION__
#define __IBO_SUPPORT_VECTOR_MACHINE_OPERATION__

#include <cstdlib>
#include <cstdio>
#include <vector>

#include "bfo.h"
#include "libsvm.h"

namespace ibo
{

class LIBSVM
{
private:
  struct svm_problem SvmProblem;
  struct svm_parameter param;
  struct svm_model *SvmTrain;
  struct svm_node *TestData;
  
  struct svm_node **AllStoredData_Train;
  struct svm_node *AllStoredData_Test;
  bool flg_AllStoredData_Test = false;
  
  int mat_cols;
  int mat_rows;
  int nchannels;
  int num_dimension;
  double *svm_predicted_probability;
  double svm_predicted_class;
  int svm_nr_class;
  int *svm_labels;// "svm_nr_class" labels = label[0]~label[svm_nr_class-1]
  
 public:
  LIBSVM(std::vector<struct_bfo> train);
  LIBSVM( int num_dimension );
  ~LIBSVM();

  // for LINEAR kernel, change c only  
  void SetParameters(int kernel_type, double c, double g);
  void SetParameters(int svm_type, int kernel_type, int probability, double c, double g);
  void SetParameters( svm_parameter input_param, double c, double g );
  void svmTrain(std::vector<struct_bfo> train);
  void svmPredict(struct_bfo test);
  void svmPredict(std::vector<double> feature);
  void svmPredictProbability(struct_bfo test, double *probability);
  void svmPredictProbability(struct_bfo *test, double *probability);
  void svmPredictProbability(std::vector<double> feature, double *probability);
  double svmGetClass() { return svm_predicted_class; }
  double svmGetNumClass() { return svm_nr_class; }
  int svmGetMaxLabel() { return svm_labels[svm_nr_class-1]; }
  int* svmGetLabels() { return svm_labels; }

  void DummyGetMemory();
  void svmSaveModel( const char *file_name );
  void svmLoadModel( const char *file_name );
};

} // namespace ibo

#endif // __IBO_SUPPORT_VECTOR_MACHINE_OPERATION__

