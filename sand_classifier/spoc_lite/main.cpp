/*
 author: Yumi Iwashita

 ./main -s setting.txt

 parameters are defined in "setting.txt"

*/

#include <stdio.h>
#include <string>
#include <vector>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>

#include "classification_ibo.h"
#include "main.h"
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

using namespace std;
using namespace cv;
using namespace ibo;


// do training for each feature vector, not concatenated one. 
// this works only for USE_RF and not implemented for normalization yet
#define TRAIN_4_EACH_FEATURE


int main(int argc, char *argv[])
{
  string fullpath;
  char rname[512], rname_rf[1024];
  char cmd[2048];
  int ret;
  double time_wall;
  double time_wall_all = 0.0;	      
  double time_wall_feature_all = 0.0;	      

  //string setting_file = "setting_svm_ave_2017_0622.txt";
  //string setting_file = "setting_svm_ave_lbp_2017_0622.txt";
  string setting_file = "setting_svm_ave_lbp.txt";
  int n = 0;
  while(++n<argc){
    if(argv[n][0]=='-'){
      switch(argv[n][1]){
      case 'S':
      case 's':
	n++;
	setting_file = argv[n];
	break;
      default:
	break;
      }
    }
  }

  //double *test_new = new double [10];

  struct_file file_info;
  struct_info other_info;
  if( load_settings( setting_file, file_info, other_info ) == 0 ) return 0;



  // setup for libSVM ::::::::::::::::::::::::::::::::::::::::::::::::::
  // parameter tuning for c and g
  int c_start = other_info.c_start, c_end = other_info.c_end, c_gap = other_info.c_gap;
  int g_start = other_info.g_start, g_end = other_info.g_end, g_gap = other_info.g_gap;

  // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // setup for classifier 
  // for SVM ::::::::::::::::::::::::::::::::::
  // choose kernel
  // int kernel_type = other_info.kernel_type;
  // int svm_type = other_info.svm_type;
  // int svm_probability = other_info.svm_probability; // 0:off, 1:on

  // LINEAR: c, RBF: c,g
  if( other_info.classifier_type == "SVM" && other_info.kernel_type == LINEAR ){
    //c_start = c_end;
    g_start = g_end;
  }

  // for LINEAR SVM ::::::::::::::::::::::::::::::::::
  //int solver_type = other_info.solver_type;
  if( other_info.classifier_type == "LINEARSVM" ){
    c_start = c_end;
    g_start = g_end;
  }

  // do not change
  int num_c = (c_end-c_start) / c_gap + 1;
  int num_g = (g_end-g_start) / g_gap + 1;

  // parameter setting for RF ::::::::::::::::::::::::::::::::::::::::::::::::::
  int maxdepth_start = other_info.maxdepth_start, maxdepth_end = other_info.maxdepth_end, maxdepth_gap = other_info.maxdepth_gap;
  int maxtrees_start = other_info.maxtrees_start, maxtrees_end = other_info.maxtrees_end, maxtrees_gap = other_info.maxtrees_gap;

  int num_maxdepth = (maxdepth_end - maxdepth_start ) / maxdepth_gap + 1;
  int num_maxtrees = (maxtrees_end - maxtrees_start ) / maxtrees_gap + 1;
  // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


  char rname_result[2014];
  file_info.rf_eval_file = "/eval_rf_d_%d_t_%d_overlap_%03d_w_%d.txt";
  if( other_info.classifier_type == "SVM" ){
    file_info.svm_model_name = "/svm_type_%d_c_%d_g_%d_w_%d.model";
    file_info.svm_eval_file = "/eval_svm_c_%d_g_%d_overlap_%03d_w_%d.txt";
    file_info.result_file = "/result_svm_type_%d_c_%d-%d_g_%d-%d_w_%d.txt";
    fullpath = file_info.result_fd + file_info.result_file;
    sprintf( rname_result, fullpath.c_str(), other_info.kernel_type, c_start, c_end, g_start, g_end, other_info.window_size );
    file_info.predicted_image_file = "/predicted_svm_%s_type_%d_c_%d_g_%d_w_%d.ppm";
  }
  else if( other_info.classifier_type == "LINEARSVM" ){
    file_info.svm_model_name = "/svmlinear_type_%d_c_%d_g_%d_w_%d.model";
    file_info.svm_eval_file = "/eval_svmlinear_c_%d_g_%d_overlap_%03d_w_%d.txt";
    file_info.result_file = "/result_svmlinear_type_%d_c_%d-%d_g_%d-%d_w_%d.txt";
    fullpath = file_info.result_fd + file_info.result_file;
    sprintf( rname_result, fullpath.c_str(), other_info.solver_type, c_start, c_end, g_start, g_end, other_info.window_size );
    file_info.predicted_image_file = "/predicted_linearsvm_%s_type_%d_c_%d_g_%d_w_%d.ppm";
  }
  else if( other_info.classifier_type == "RF" ){
    file_info.result_file = "/result_rf_d_%d-%d_t_%d-%d_w_%d.txt";
    fullpath = file_info.result_fd + file_info.result_file;
    sprintf( rname_result, fullpath.c_str(), maxdepth_start, maxdepth_end, maxtrees_start, maxtrees_end, other_info.window_size );
    file_info.predicted_image_file = "/predicted_rf_%s_d_%d_t_%d_w_%d.ppm";
  }
  file_info.rf_model_name = "/rf_d_%d_t_%d_w_%d_p_%d_f_%d.model";

  FILE *fp_result = fopen( rname_result, "w" );

  LoadFileName( file_info );

  cout << "file_info.train_file_lists.size() " << file_info.train_file_lists.size() << endl;
  
  struct_lbp_parameter lbp_parameter;
  lbp_parameter.SDIM = makeELBP(8, lbp_parameter.Flg_LBP);
  lbp_parameter.db = other_info.lbp_threshold;

  int nclass = 0;

  double ratio_train = 1.0;
  if( file_info.train_src_fd == file_info.test_src_fd )
    ratio_train = 1.0 - 1.0 / (double)other_info.n_of_n_folds;

  // divide lists_all into training and test randomly
  for(int loop=0; loop<1; loop++){
    // randomly selecting test files
    file_info.reset_flg();

    vector<int> rand_id;
#if 1
    MakeRandomNumber( file_info.train_file_lists.size(), rand_id );
#else
    //debug
    char rname_rand[1024] = "./Ryan_Random.txt";//yumi
    LoadRandomNumber( loop, file_info.train_file_lists.size(), rname_rand, rand_id );
#endif

    int num = 0, id;
    int total_files = file_info.train_file_lists.size();

    // choose train data
    for(int i=0; i<rand_id.size(); i++){
      id = rand_id[i];
      file_info.train_file_lists[id].flg = 1;// used 
      num ++;
      if( num >= (total_files*ratio_train) )
	break;
    }
    cout << "train " << num << endl;
    // choose test data
    num = 0;
    total_files = file_info.test_file_lists.size();
    if( file_info.train_src_fd != file_info.test_src_fd )
      for(int i=0; i<total_files; i+=1){
	file_info.test_file_lists[i].flg = 1;
	num ++;
      }
    else{
      for(int i=0; i<total_files; i+=1){
	if( file_info.train_file_lists[i].flg == 1 )
	  file_info.test_file_lists[i].flg = 0;
	else{
	  file_info.test_file_lists[i].flg = 1;
	  num ++;
	}
      }
    }
    cout << "test " << num << endl;


    printf("bco_label_list\n");
    for(int i=0; i<other_info.bco_label_list.n_label; i++){
      printf("label %d : ", other_info.bco_label_list.label_id[i]);
      for (int b=0; b < other_info.bco_label_list.chans; b++)
	printf("%d ", other_info.bco_label_list.data[i*other_info.bco_label_list.chans + b]);
      printf("\n");
    }
      

    int actual_num_labels = (other_info.bco_label_list.n_label);// - num_ignore_label_rgb);

	  vector<struct_feature_dimension> feature_dim;

	  // Loading train dataset -------------------------------------
	  vector<struct_bfo> train;
	  if( !other_info.load_train_model ){
	    time_wall = buo_get_wall_time();
	    cout << "Loading train dataset -------------------------------------" << endl;
	    // BALANCED LOADING
	    // So load X sets of feature vector of class A, then load X sets of feature vector of class B. Repeat this process. 	  
	    int train_count = 0;
	    int maxnum_feature_one_load = 50;
	    //int maxnum_feature_one_load = 500;
	    int num_label = other_info.bco_label_list.n_label;
	    for(int load_loop=0; load_loop<10; load_loop++){
	      //for(int load_loop=0; load_loop<100; load_loop++){

	      for(int label_loop = 0; label_loop < num_label; label_loop ++){
		int label_next = other_info.bco_label_list.label_id[label_loop];
		if( label_next < 0 ) continue;

		int count = 0;
		int num_loaded = 0;
		int total_loaded = train.size();
		do{
		  count ++;
		  if( count >= file_info.train_file_lists.size() ) break;
		
		  int train_id = rand() % file_info.train_file_lists.size();
		  if( file_info.train_file_lists[train_id].flg == 0 ) continue;

		  //mc_image mc_src;
		  fullpath = file_info.train_src_fd + file_info.train_file_lists[train_id].file_name + file_info.src_extension;
		  Mat mat_src = imread( fullpath.c_str(), CV_LOAD_IMAGE_COLOR );
  	      
		  //mc_label_image mc_label_src;
		  fullpath = file_info.train_label_fd + file_info.train_file_lists[train_id].file_name + file_info.label_extension;
		  Mat mat_label = imread( fullpath.c_str(), CV_LOAD_IMAGE_COLOR );

		  bco_assign_label( other_info.bco_label_list, mat_label );// mc_label_src is converted to label_id (1channels) from rgb

		  SetFeature_BalancedLoad( train, feature_dim, lbp_parameter, mat_src, mat_label, maxnum_feature_one_load, label_next, other_info );
	      
		  num_loaded += (train.size() - total_loaded);
		  if( (train.size()-total_loaded) > maxnum_feature_one_load ) break;
		  total_loaded = train.size();
	      
		}while(1);

		if( num_loaded == 0 ) label_loop -= 1;
		
	      }
	    }

	    time_wall = buo_get_wall_time() - time_wall;
	    printf("Load image, assign labels, and calc feature train %lf [ms]\n", time_wall*1000.0);
	  }// !other_info.load_train_model

	  
	  // load just one test to know feature dimension
	  vector<struct_feature_dimension> feature_dim_4_train;
	  {
	    vector<struct_bfo> test;
	    for(int i=0; i<file_info.test_file_lists.size(); i++){
	      if( file_info.test_file_lists[i].flg == 0 ) continue;
	      //mc_image mc_src;
	      fullpath = file_info.test_src_fd + file_info.test_file_lists[i].file_name + file_info.src_extension;
	      cout << fullpath << endl;
	      Mat mat_src = imread( fullpath.c_str(), CV_LOAD_IMAGE_COLOR );
	      //mc_read_image( mc_src, fullpath.c_str() );

	      //mc_label_image mc_label_src;
	      fullpath = file_info.test_label_fd + file_info.test_file_lists[i].file_name + file_info.label_extension;
	      Mat mat_label = imread( fullpath.c_str(), CV_LOAD_IMAGE_COLOR );
	      if( mat_label.empty() )
		mat_label = Mat::zeros( mat_src.rows, mat_src.cols, CV_8UC3 );

	      bco_assign_label( other_info.bco_label_list, mat_label );// mc_label_src is converted to label_id (1channels) from rgb
	      
	      SetFeature( test, feature_dim, lbp_parameter, mat_src, mat_label, other_info );
	      break;
	    }

	    if( other_info.classifier_type == "RF" ){
#ifdef TRAIN_4_EACH_FEATURE
	      copy( feature_dim.begin(), feature_dim.end(), back_inserter(feature_dim_4_train));
#else
	      int num_feature_dim = feature_dim.size();
	      struct_feature_dimension tmp_feature_dim_4_train;
	      tmp_feature_dim_4_train.start = 0;
	      tmp_feature_dim_4_train.end = feature_dim[num_feature_dim-1].end;
	      tmp_feature_dim_4_train.num = feature_dim[num_feature_dim-1].end;// not num
	      feature_dim_4_train.push_back( tmp_feature_dim_4_train );
#endif
	    }
	  }
	  int num_feature_dim_4_train = feature_dim_4_train.size();

	  
	  // normalization -------------------------------------------------
	  if( other_info.feature_norm_type == "L1" ){
	    cout << "normalization L1" << endl;
	    for(int i=0; i<train.size(); i++)
	      bfo_normalize_l1( train[i].feature, feature_dim, other_info.lower, other_info.upper );
	  }
	  if( other_info.feature_norm_type == "L2" ){
	    cout << "normalization L2" << endl;
	    for(int i=0; i<train.size(); i++)
	      bfo_normalize_l2( train[i].feature, feature_dim );
	  }


	  nclass = 0;
	  for(int i=0; i<other_info.bco_label_list.label_id.size(); i++)
	    nclass = max( nclass, other_info.bco_label_list.label_id[i] );
	  nclass ++;

	  int *conf = new int [nclass * nclass];
	  int *gt_num = new int [nclass];

	  // Classify each test data
	  int gt_class;
	  int estimated_class;
	  //int num_correct = 0;
	  //int num_test_data_all = 0;
	  float result, prob;
	  
	  int sample_num = 0;
	  int i_test = 0;
	  int test_frame_count = 0;
	  time_wall_all = 0.0;
	  time_wall_feature_all = 0.0;
	  
	  //#ifdef USE_SVM //***********************************************************************************:
	  if( other_info.classifier_type == "SVM" || other_info.classifier_type == "LINEARSVM" ){
	    /* ----------------------------------------------------------------------------------
	       Classify each test data with a Machine Learning Technique (Support Vector Machine)
	       ---------------------------------------------------------------------------------- */
	  
	    cout << "------- train/load svm model ------------------" << endl;
	    cout << "c_start " << c_start << " c_end " << c_end << endl;
	    cout << "g_start " << g_start << " g_end " << g_end << endl;

	    int c_count = 0, g_count = 0;
	    for (int c = c_start; c <= c_end; c+=c_gap){
	      g_count = 0;
	      for (int g = g_start; g <= g_end; g+=g_gap){
	      
		cout << "c " << c << " g " << g << endl;
	      
		memset( conf, 0, sizeof(int)*nclass*nclass );
		memset( gt_num, 0, sizeof(int)*nclass );

		struct_eval *eval = new struct_eval [other_info.roc_threshold.num];
		for(int i=0; i<other_info.roc_threshold.num; i++)
		  initialize_eval( eval[i] );

		FILE *fp_eval;
		fullpath = file_info.result_fd + file_info.svm_eval_file;
		sprintf( rname, fullpath.c_str(), c, g, (int)(other_info.window_overlap*100), other_info.window_size );
		fp_eval = fopen( rname, "w" );
	      
		fullpath = file_info.result_fd + file_info.svm_model_name;
		if( other_info.classifier_type == "SVM" )
		  sprintf( rname, fullpath.c_str(), other_info.kernel_type, c, g, other_info.window_size );
		else if( other_info.classifier_type == "LINEARSVM" )
		  sprintf( rname, fullpath.c_str(), other_info.solver_type, c, g, other_info.window_size );

		ClassifierSVMIbo classifier( rname, other_info, c, g );
		if( !other_info.load_train_model ){
		  // train and save model
		  time_wall = buo_get_wall_time();
		  if( other_info.classifier_type == "SVM" ){
		    classifier.svm.SetParameters(other_info.svm_type, other_info.kernel_type, other_info.svm_probability, (double)c, (double)g);
		    classifier.svm.svmTrain(train);
		    classifier.svm.svmSaveModel( rname );
		  }
		  else if( other_info.classifier_type == "LINEARSVM" ){
		    classifier.linear_svm.SetParameters(other_info.solver_type, 1);// 1: automatic parameter search
		    classifier.linear_svm.svmTrain(train);
		    classifier.linear_svm.svmSaveModel( rname );
		  }
		  time_wall = buo_get_wall_time() - time_wall;
		  printf("svmTrain %lf\n", time_wall);
		}
		// else{
		//   // to load pre-trained SVM model
		//   if( other_info.classifier_type == "SVM" ){
		//     classifier.svm.SetParameters(other_info.kernel_type, (double)c, (double)g);
		//     classifier.svm.DummyGetMemory();
		//     cout << rname << endl;
		//     classifier.svm.svmLoadModel( rname );
		//   }
		//   else if( other_info.classifier_type == "LINEARSVM" ){
		//     classifier.linear_svm.SetParameters(other_info.solver_type, 1);// 1: automatic parameter search
		//     classifier.linear_svm.DummyGetMemory();
		//     classifier.linear_svm.svmLoadModel( rname );
		//   }
		//   cout << "load svm model" << endl;
		// }
	      
		cout << "------- predict ------------------" << endl;
		double *probability = new double [actual_num_labels];
		string str_image = "";
		for(int i=0; i<file_info.test_file_lists.size(); i++){//i
		  if( file_info.test_file_lists[i].flg == 0 ) continue;

		  cout << "Loading test dataset ------------------------------------- " << test_frame_count << endl;
		  fullpath = file_info.test_src_fd + file_info.test_file_lists[i].file_name + file_info.src_extension;
		  Mat mat_src = imread( fullpath.c_str(), CV_LOAD_IMAGE_COLOR );
		  cout << fullpath << endl;

		  string str2 ("/");
		  size_t last_position = buo_find_the_last_string( fullpath, str2 );
		  if( last_position > 0 ){
		    str_image = fullpath.substr( 0, (fullpath.size()-4) );
		    str_image = str_image.substr( (last_position+1), str_image.size() );
		  }
		  else
		    str_image = fullpath.substr( last_position, (fullpath.size()-4) );
		
		  fullpath = file_info.test_label_fd + file_info.test_file_lists[i].file_name + file_info.label_extension;
		  Mat mat_label = imread( fullpath.c_str(), CV_LOAD_IMAGE_COLOR );
		  if( mat_label.empty() )
		    mat_label = Mat::zeros( mat_src.rows, mat_src.cols, CV_8UC3 );

		  bco_assign_label( other_info.bco_label_list, mat_label );// mc_label_src is converted to label_id (1channels) from rgb

		  Mat mat_label_4_evaluate_class;// 255 for sand, CV_8UC1
		  Mat mat_prob_4_evaluate_class;//(0.0 to 1.0), CV_32FC1
#if 1
		  //imshow("mat_src", mat_src);

		  // the same process with "original" below, but without time_wall
		  //classifier.classify( mat_src, mat_label_4_evaluate_class, mat_prob_4_evaluate_class, other_info, lbp_parameter );
		  classifier.classify( mat_src, mat_label_4_evaluate_class, mat_prob_4_evaluate_class );
		  
		  // imshow("mat_label_4_evaluate_class", mat_label_4_evaluate_class);
		  // imshow("mat_prob_4_evaluate_class", mat_prob_4_evaluate_class);
		  // waitKey(0);
#else
		  // +++++++++++++++++++++++++++++++++++++++++++++++++
		  // classify() from here --------------------------
		  // +++++++++++++++++++++++++++++++++++++++++++++++++
		  // original
		  vector<struct_bfo> test;
		  time_wall = buo_get_wall_time();
		  //SetFeature( test, feature_dim, lbp_parameter, mat_src, mat_label, other_info );
		  SetFeature( test, feature_dim, lbp_parameter, mat_src, Mat(), other_info );
		  time_wall = buo_get_wall_time() - time_wall;
		  printf("Feature calc %lf [ms]\n", time_wall*1000.0);
		  time_wall_all += time_wall*1000.0;
		  time_wall_feature_all += time_wall*1000.0;
		
		  sample_num = test.size();

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

		  double *probability_frame = new double [sample_num * actual_num_labels];
		  memset( probability_frame, 0, sizeof(double)*sample_num*actual_num_labels );
		
		  // prediction each frame
		  time_wall = buo_get_wall_time();
		  for(int j=0; j<sample_num; j++){

		    if( other_info.classifier_type == "SVM" )
		      classifier.svm.svmPredictProbability(test[j], probability);
		    else if( other_info.classifier_type == "LINEARSVM" )
		      classifier.linear_svm.svmPredictProbability(test[j], probability);

		    memcpy( probability_frame+j*actual_num_labels, probability, sizeof(double)*actual_num_labels );
		  
		    // if( other_info.classifier_type == "SVM" )
		    //   estimated_class = svm.svmGetClass();
		    // else if( other_info.classifier_type == "LINEARSVM" )
		    //   estimated_class = linear_svm.svmGetClass();
		  }
		  time_wall = buo_get_wall_time() - time_wall;
		  time_wall_all += time_wall*1000.0;
		
		  // calc probability at each pixel and display
		  vector<Mat> prob_mat_vec;
		  Mat estimated_class_mat;
		  Size mat_size( mat_src.cols, mat_src.rows );
		  calc_prob_and_get_id_each_pixel( prob_mat_vec, estimated_class_mat, mat_size, probability_frame, test, other_info );

                  make_label_prob_target_id( prob_mat_vec, estimated_class_mat, mat_label_4_evaluate_class, mat_prob_4_evaluate_class, other_info );

		  if( probability_frame ) delete [] probability_frame;
		  // +++++++++++++++++++++++++++++++++++++++++++++++++
		  // classify() till here --------------------------
		  // +++++++++++++++++++++++++++++++++++++++++++++++++


		  // calc ROC
		  evaluate_target_id( prob_mat_vec, estimated_class_mat, mat_label, eval, other_info );
		  
		  Mat overlay_mat;
		  fullpath = file_info.test_src_fd + file_info.test_file_lists[i].file_name + file_info.src_extension;
		  visualize_predicted_target_id( fullpath, estimated_class_mat, prob_mat_vec, overlay_mat, other_info );
		
		  // save overlay_mat
		  if( other_info.save_predicted_image ){
		    fullpath = file_info.result_fd + file_info.predicted_image_file;
		    if( other_info.classifier_type == "SVM" )
		      sprintf( rname, fullpath.c_str(), str_image.c_str(), other_info.svm_type, c, g, other_info.window_size );
		    else if( other_info.classifier_type == "LINEARSVM" )
		      sprintf( rname, fullpath.c_str(), str_image.c_str(), other_info.solver_type, c, g, other_info.window_size );
		    imwrite( rname, overlay_mat );
		  }
#endif

		  i_test ++;

		}// i
	      	      
                if( probability ) delete [] probability;// only for svm
		
		// output --------------------------------------------------	      
		double _CCR = (double)eval[0].tp_ident / (double)(eval[0].tp_ident+eval[0].fn_ident);
		printf("loop %d window_size %d: Correct Classification Rate is %lf (c %d g %d)\ttime\t%lf\t(feature_calc\t%lf\t)\n", 
		       loop, other_info.window_size, _CCR, c, g, time_wall_all/(double)i_test, time_wall_feature_all/(double)i_test);
		fprintf( fp_result, "c\t%d\tg\t%d\tloop\t%d\twindow_size\t%d\tCCR\t%lf\ttime\t%lf\t(feature_calc\t%lf\t)\n", 
			 c, g, loop, other_info.window_size, _CCR, time_wall_all/(double)i_test, time_wall_feature_all/(double)i_test);
		fflush( fp_result );
		for(int i=0; i<other_info.bco_label_list.n_label; i++){
		  printf("label\t%d\t", other_info.bco_label_list.label_id[i]);
		  fprintf(fp_result, "label\t%d\t", other_info.bco_label_list.label_id[i]);
		  for (int b=0; b < other_info.bco_label_list.chans; b++){
		    printf("%d\t", other_info.bco_label_list.data[i*other_info.bco_label_list.chans + b]);
		    fprintf(fp_result, "%d\t", other_info.bco_label_list.data[i*other_info.bco_label_list.chans + b]);
		  }
		  printf("\n");
		  fflush( fp_result );
		  fprintf(fp_result, "\n");
		}
		printf( "time_wall\t%lf (feature calc %lf)\n", 
			time_wall_all/(double)i_test, time_wall_feature_all/(double)i_test);
	      
		for(int i=0; i<nclass; i++){
		  for(int j=0; j<nclass; j++){
		    cout << (double)conf[i*nclass + j]/(double)gt_num[j]*100.0 << " ";
		    fprintf( fp_result, "%lf\t", (double)conf[i*nclass + j]/(double)gt_num[j]*100.0 );
		  }
		  cout << endl;
		  fprintf( fp_result, "\n" );
		  fflush( fp_result );
		}	      
		for(int k=0; k<other_info.roc_threshold.num; k++){
		  printf("tp %d fn %d fp %d tn %d\n", eval[k].tp, eval[k].fn, eval[k].fp, eval[k].tn);
		  eval[k].recall = (double)eval[k].tp / (double)(eval[k].tp+eval[k].fn);
		  eval[k].precision = (double)eval[k].tp / (double)(eval[k].tp+eval[k].fp);
		  eval[k].F = 2.0 * eval[k].precision * eval[k].recall / (eval[k].precision + eval[k].recall);
		  eval[k].FOR = (double)eval[k].fn / (double)(eval[k].tn+eval[k].fn);
		  // eval[k].CCR = (double)num_correct / (double)num_test_data_all;//(double)eval[k].tp_ident / (double)(eval[k].tp_ident+eval[k].fn_ident);
		  // cout << "CCR " << eval[k].CCR << " num_correct " << num_correct << " " << num_test_data_all << endl;
		  eval[k].CCR = (double)eval[k].tp_ident / (double)(eval[k].tp_ident+eval[k].fn_ident);
		  //cout << "CCR " << eval[k].CCR << " eval[k].tp_ident " << eval[k].tp_ident << " (eval[k].tp_ident+eval[k].fn_ident) " << (eval[k].tp_ident+eval[k].fn_ident) << endl;
		  
		  double threshold = other_info.roc_threshold.start + other_info.roc_threshold.gap * (double)k;
		  fprintf( fp_eval, "threshold\t%lf\ttp\t%d\ttn\t%d\tfn\t%d\tfp\t%d\tRecall\t%lf\tPrecision\t%lf\tF\t%lf\tFOR\t%lf\tCCR_ident\t%lf\tsvm_time\t%lf\t(feature_calc\t%lf\t)\n", 
			   threshold, eval[k].tp, eval[k].tn, eval[k].fn, eval[k].fp, eval[k].recall, eval[k].precision, eval[k].F, eval[k].FOR, eval[k].CCR, time_wall_all/(double)i_test, time_wall_feature_all/(double)i_test);
		  fflush( fp_eval );
		}
		fclose( fp_eval );
		// output --------------------------------------------------	      
	      
		if( eval ) delete [] eval;
		g_count ++;
	      }
	      c_count ++;
	    }
	  
	  }
	  //#endif // USE_SVM ***********************************************************************************:
	  //#ifdef USE_RF //***********************************************************************************:
	  if( other_info.classifier_type == "RF" ){

	    /* ----------------------------------------------------------------------------------
	       Classify each test data with a Machine Learning Technique (Random Forest)
	       ---------------------------------------------------------------------------------- */
	    cout << "------- train rf ------------------" << endl;	  
	    int maxdepth_count = 0, maxtrees_count = 0;
	    for (int maxdepth = maxdepth_start; maxdepth <= maxdepth_end; maxdepth += maxdepth_gap){
	      maxtrees_count = 0;
	      for (int maxtrees = maxtrees_start; maxtrees <= maxtrees_end; maxtrees += maxtrees_gap){
		
		cout << "window_size " << other_info.window_size << " maxdepth " << maxdepth << " maxtrees " << maxtrees << endl;
		memset( conf, 0, sizeof(int)*nclass*nclass );
		memset( gt_num, 0, sizeof(int)*nclass );
	      
		struct_eval *eval = new struct_eval [other_info.roc_threshold.num];
		for(int i=0; i<other_info.roc_threshold.num; i++)
		  initialize_eval( eval[i] );
	      
		FILE *fp_eval;
		fullpath = file_info.result_fd + file_info.rf_eval_file;
		sprintf( rname, fullpath.c_str(), maxdepth, maxtrees, (int)(other_info.window_overlap*100), other_info.window_size );
		fp_eval = fopen( rname, "w" );

		// 2 class classification (1 vs the rest)
		int num_category = 2;// this is fixed
	      
		//int id_in_prob = 0;
		int id_category = 0;

		// Prepare RF ------------------------------------------------------------------------	      
		// prepare tree for each category
		CvRTrees** rtree = new CvRTrees* [actual_num_labels*num_feature_dim_4_train];
		for(int jj=0; jj<other_info.bco_label_list.n_label; jj++){
		  id_category = other_info.bco_label_list.label_id[jj];
		  //id_in_prob = id_map_from_label_to_prob[jj];
		  for(int kk=0; kk<num_feature_dim_4_train; kk++)
		    //rtree[id_in_prob*num_feature_dim_4_train + kk] = new CvRTrees;
		    rtree[id_category*num_feature_dim_4_train + kk] = new CvRTrees;
		}
	      
		// train tree first (or load tree first)	 
		// set features into 2 classes for 1 vs the rest classification
		// 1 vs the rest. "1": is class 1, and "the rest" is class 0
		// ----------------------
		// "1 vs the rest" is repeated for "n-1" times (n classes in total)
		// since the probability of the class n can be obtaied from the others like p[n] = 1.0 - p[1] - p[2] - ... - p[n-1]
		// ----------------------
		for(int jj=0; jj<other_info.bco_label_list.n_label; jj++){
		  //id_in_prob = id_map_from_label_to_prob[jj];
		  id_category = other_info.bco_label_list.label_id[jj];
		
		  //if( id_in_prob == (actual_num_labels-1) ) continue; // skip the class n, since this is not necessary for training. (1.0 - \sigma_i P(i))
		  if( id_category == (actual_num_labels-1) ) continue; // skip the class n, since this is not necessary for training. (1.0 - \sigma_i P(i))
		  //if( id_in_prob == rf_skip_training_id_in_prob ) continue; // skip the class n, since this is not necessary for training. (1.0 - \sigma_i P(i))

		  float *priors = new float [num_category];// = {1,1,1,1,1,1,1,1,1,1};  // weights of each classification for classes
		  for(int i=0; i<num_category; i++) priors[i] = 1.0;
		  // (all equal as equal samples of each digit)
		  CvRTParams params = CvRTParams(maxdepth, // max depth
						 5, // min sample count
						 0, // regression accuracy: N/A here
						 false, // compute surrogate split, no missing data
						 15, // max number of categories (use sub-optimal algorithm for larger numbers)
						 priors, // the array of priors
						 false,  // calculate variable importance
						 4,       // number of variables randomly selected at node and used to find the best split(s).
						 maxtrees,	 // 100 max number of trees in the forest
						 0.01f,				// forrest accuracy
						 CV_TERMCRIT_ITER |	CV_TERMCRIT_EPS // termination cirteria
						 );
		
		
		  // train RF----------------------------------------------------------
		  for(int kk=0; kk<num_feature_dim_4_train; kk++){
		    cout << "train RF " << id_category  << " " << kk << " out of " << num_feature_dim_4_train << endl;
		    fullpath = file_info.result_fd + file_info.rf_model_name;
		    sprintf( rname_rf, fullpath.c_str(), maxdepth, maxtrees, other_info.window_size, id_category, kk );
		    cout << rname_rf << endl;
		    //getchar();

		    cout << "feature_dim_4_train[kk] " << feature_dim_4_train[kk].num << endl;

		    if( !other_info.load_train_model ){
		      cout << "set train MatFeature" << endl;
		      Mat mat_train, mat_train_class;
		      // 1 vs the rest. "1": is class 1, and "the rest" is class 0
		      SetMatFeature( train, feature_dim_4_train[kk], id_category, mat_train, mat_train_class );
		      cout << " train was set " << endl;
		  
		      Mat var_type = Mat(mat_train.cols + 1, 1, CV_8U );
		      var_type.setTo(Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical
		      var_type.at<uchar>(mat_train.cols, 0) = CV_VAR_CATEGORICAL;
		  
		      // rtree[id_in_prob*num_feature_dim_4_train + kk]->train(mat_train, CV_ROW_SAMPLE, mat_train_class, 
		      // 							    Mat(), Mat(), var_type, Mat(), params);
		      rtree[id_category*num_feature_dim_4_train + kk]->train(mat_train, CV_ROW_SAMPLE, mat_train_class, 
									    Mat(), Mat(), var_type, Mat(), params);
		      cout << "save RF model" << endl;
		      //rtree[id_in_prob*num_feature_dim_4_train + kk]->save( rname_rf );
		      rtree[id_category*num_feature_dim_4_train + kk]->save( rname_rf );
		    }
		    else{
		      cout << "load RF model" << endl;
		      //rtree[id_in_prob*num_feature_dim_4_train + kk]->load( rname_rf );
		      rtree[id_category*num_feature_dim_4_train + kk]->load( rname_rf );
		    }
		  }

		}
	      
		cout << "------- predict ------------------" << endl;
		string str_image = "";
		for(int i=0; i<file_info.test_file_lists.size(); i++){
		  if( file_info.test_file_lists[i].flg == 0 ) continue;
		
		  cout << "Loading test dataset ------------------------------------- " << test_frame_count << endl;
		  fullpath = file_info.test_src_fd + file_info.test_file_lists[i].file_name + file_info.src_extension;
		  cout << fullpath << endl;
		  Mat mat_src = imread( fullpath.c_str(), CV_LOAD_IMAGE_COLOR );

		  string str2 ("/");
		  size_t last_position = buo_find_the_last_string( fullpath, str2 );
		  if( last_position > 0 ){
		    str_image = fullpath.substr( 0, (fullpath.size()-4) );
		    str_image = str_image.substr( (last_position+1), str_image.size() );
		  }
		  else
		    str_image = fullpath.substr( last_position, (fullpath.size()-4) );
		  // cout << fullpath << endl;
		  // getchar();

		  //mc_label_image mc_label_src;
		  fullpath = file_info.test_label_fd + file_info.test_file_lists[i].file_name + file_info.label_extension;
		  Mat mat_label = imread( fullpath.c_str(), CV_LOAD_IMAGE_COLOR );
		  if( mat_label.empty() )
		    mat_label = Mat::zeros( mat_src.rows, mat_src.cols, CV_8UC3 );
		  
		  //mc_assign_label( other_info.bco_label_list, mc_label_src );// mc_label_src is converted to label_id from rgb
		  bco_assign_label( other_info.bco_label_list, mat_label );// mc_label_src is converted to label_id (1channels) from rgb

		  vector<struct_bfo> test;
		  time_wall = buo_get_wall_time();
		  SetFeature( test, feature_dim, lbp_parameter, mat_src, mat_label, other_info );
		  time_wall = buo_get_wall_time() - time_wall;
		  printf("Feature calc %lf [ms]\n", time_wall*1000.0);
		  time_wall_all += time_wall*1000.0;
		  time_wall_feature_all += time_wall*1000.0;

		  sample_num = test.size();
		
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
		
		  double *probability_frame = new double [sample_num * actual_num_labels];
		  memset( probability_frame, 0, sizeof(double)*sample_num*actual_num_labels );
		
		  //int *estimated_class_all = new int [sample_num];
		  //memset( estimated_class_all, 0, sizeof(int)*sample_num );

		  int num_sample_4_predict = 1;
		  vector<Mat> mat_test, mat_test_class;
		  for(int kk=0; kk<num_feature_dim_4_train; kk++){
		    Mat tmp_mat_test = Mat::zeros( num_sample_4_predict, feature_dim_4_train[kk].num, CV_32FC1 );
		    Mat tmp_mat_test_class = Mat::zeros( num_sample_4_predict, 1, CV_32FC1 );
		    mat_test.push_back( tmp_mat_test );
		    mat_test_class.push_back( tmp_mat_test_class );
		  }
		
		  time_wall = buo_get_wall_time();
		  for(int j=0; j<sample_num; j++){
		    double *tmp_prob = new double [actual_num_labels * num_feature_dim_4_train];

		    for(int kk=0; kk<num_feature_dim_4_train; kk++){
		      for(int r = 0; r < num_sample_4_predict; r++) {
			float* _mat_test_class = mat_test_class[kk].ptr<float>(r);
			float* _mat_test = mat_test[kk].ptr<float>(r);
			_mat_test_class[0] = (float)test[j].label;
			for(int c = 0; c < feature_dim_4_train[kk].num; c++)
			  _mat_test[c] = (float)test[j].feature[c+feature_dim_4_train[kk].start];
		      }
		    
		      // predict with trained models
		      for(int jj=0; jj<other_info.bco_label_list.n_label; jj++){
			//id_in_prob = id_map_from_label_to_prob[jj];
			id_category = other_info.bco_label_list.label_id[jj];

			//if( id_in_prob < (actual_num_labels-1) )
			if( id_category < (actual_num_labels-1) )
			  tmp_prob[id_category*num_feature_dim_4_train + kk] = rtree[id_category*num_feature_dim_4_train + kk]->predict_prob(mat_test[kk], Mat());
			else{
			  prob = 1.0;
			  tmp_prob[id_category*num_feature_dim_4_train + kk] = 1.0;
			  for(int jjj=0; jjj<id_category; jjj++){
			    tmp_prob[id_category*num_feature_dim_4_train + kk] -= tmp_prob[jjj*num_feature_dim_4_train + kk];
			  }
			}
		      }// jj

		    }// kk
		      
		    float max_prob = 0.0;
		    for(int jj=0; jj<other_info.bco_label_list.n_label; jj++){
		      //id_in_prob = id_map_from_label_to_prob[jj];
		      id_category = other_info.bco_label_list.label_id[jj];

		      double prob = 0.0;
		      // A. simple average
		      for(int kk=0; kk<num_feature_dim_4_train; kk++)
			prob += tmp_prob[id_category*num_feature_dim_4_train + kk] / (double)num_feature_dim_4_train;

		      probability_frame[j*actual_num_labels + id_category] = prob;
		      if( max_prob < prob ){
			max_prob = prob;
			estimated_class = (int)id_category;
		      }
		      gt_class = test[j].label;
		    }
		    if( tmp_prob ) delete [] tmp_prob;
		  
		    // id_in_prob = id_map_from_label_to_prob[gt_class];
		    // if( gt_class < 0 ) continue;

		    // conf[estimated_class*nclass + gt_class] ++;
		    // gt_num[gt_class] ++;

		    // //estimated_class_all[j] = estimated_class; 
		  
		    // if (gt_class == estimated_class) num_correct++;
		    // num_test_data_all ++;
		  } // j
		  time_wall = buo_get_wall_time() - time_wall;
		  //printf("predict (each frame)%lf\n", time_wall*1000.0);
		  time_wall_all += time_wall*1000.0;
		

		  // calc probability at each pixel and display
		  vector<Mat> prob_mat_vec;
		  Mat estimated_class_mat;
		  Size mat_size( mat_src.cols, mat_src.rows );
		  calc_prob_and_get_id_each_pixel( prob_mat_vec, estimated_class_mat, mat_size, probability_frame, test, other_info );

		  Mat overlay_mat;
		  fullpath = file_info.test_src_fd + file_info.test_file_lists[i].file_name + file_info.src_extension;
		  visualize_predicted_target_id( fullpath, estimated_class_mat, prob_mat_vec, overlay_mat, other_info );

		  // calc ROC
		  evaluate_target_id( prob_mat_vec, estimated_class_mat, mat_label, eval, other_info );


		  // save overlay_mat
		  if( other_info.save_predicted_image ){
		    fullpath = file_info.result_fd + file_info.predicted_image_file;
		    sprintf( rname, fullpath.c_str(), str_image.c_str(), maxdepth, maxtrees, other_info.window_size );
		    imwrite( rname, overlay_mat );
		  }

		  test_frame_count ++;

		  //if( estimated_class_all ) delete [] estimated_class_all;
		  if( probability_frame ) delete [] probability_frame;	   

		  i_test ++;
		}// i

		// output --------------------------------------------------	      
		double _CCR = (double)eval[0].tp_ident / (double)(eval[0].tp_ident+eval[0].fn_ident);
		printf("window_size %d Correct Classification Rate is %lf\ttime\t%lf\t(feature_calc\t%lf\t)\n", 
		       other_info.window_size, _CCR, time_wall_all/(double)i_test, time_wall_feature_all/(double)i_test);
		fprintf( fp_result, "maxdepth\t%d\tmaxtrees\t%d\twindow_size\t%d\tloop\t%d\tCCR\t%lf\ttime\t%lf\t(feature_calc\t%lf\t)\n", 
			 maxdepth, maxtrees, other_info.window_size, loop, _CCR, time_wall_all/(double)i_test, time_wall_feature_all/(double)i_test);
		fflush( fp_result );
	      
		for(int i=0; i<other_info.bco_label_list.n_label; i++){
		  printf("label\t%d\t", other_info.bco_label_list.label_id[i]);
		  fprintf(fp_result, "label\t%d\t", other_info.bco_label_list.label_id[i]);
		  for (int b=0; b < other_info.bco_label_list.chans; b++){
		    printf("%d\t", other_info.bco_label_list.data[i*other_info.bco_label_list.chans + b]);
		    fprintf(fp_result, "%d\t", other_info.bco_label_list.data[i*other_info.bco_label_list.chans + b]);
		  }
		  printf("\n");
		  fprintf(fp_result, "\n");
		}
	      
		printf( "time_wall\t%lf (feature calc %lf)\n", 
			time_wall_all/(double)i_test, time_wall_feature_all/(double)i_test);

		for(int i=0; i<nclass; i++){
		  for(int j=0; j<nclass; j++){
		    cout << (double)conf[i*nclass + j]/(double)gt_num[j]*100.0 << " ";
		    //cout << (double)conf[i*nclass + j] << "(" << (double)gt_num[j] << ") ";
		    fprintf( fp_result, "%lf\t", (double)conf[i*nclass + j]/(double)gt_num[j]*100.0 );
		  }
		  cout << endl;
		  fprintf( fp_result, "\n" );
		}     
	      
		for(int k=0; k<other_info.roc_threshold.num; k++){
		  printf("tp %d fn %d fp %d tn %d\n", eval[k].tp, eval[k].fn, eval[k].fp, eval[k].tn);
		  eval[k].recall = (double)eval[k].tp / (double)(eval[k].tp+eval[k].fn);
		  eval[k].precision = (double)eval[k].tp / (double)(eval[k].tp+eval[k].fp);
		  eval[k].F = 2.0 * eval[k].precision * eval[k].recall / (eval[k].precision + eval[k].recall);
		  eval[k].FOR = (double)eval[k].fn / (double)(eval[k].tn+eval[k].fn);
		  //eval[k].CCR = (double)num_correct / (double)num_test_data_all;//(double)eval[k].tp_ident / (double)(eval[k].tp_ident+eval[k].fn_ident);
		  eval[k].CCR = (double)eval[k].tp_ident / (double)(eval[k].tp_ident+eval[k].fn_ident);
		
		  //double average_time = time_wall_all / (double)test_frame_count;
		  double threshold = other_info.roc_threshold.start + other_info.roc_threshold.gap * (double)k;
		  fprintf( fp_eval, "threshold\t%lf\ttp\t%d\ttn\t%d\tfn\t%d\tfp\t%d\tRecall\t%lf\tPrecision\t%lf\tF\t%lf\tFOR\t%lf\tCCR_ident\t%lf\tRF_time\t%lf(feature_calc\t%lf\t)\n", 
			   threshold, eval[k].tp, eval[k].tn, eval[k].fn, eval[k].fp, eval[k].recall, eval[k].precision, eval[k].F, eval[k].FOR, eval[k].CCR, time_wall_all/(double)i_test, time_wall_feature_all/(double)i_test);
		  fflush( fp_eval );
		}
		fclose( fp_eval );
		// output --------------------------------------------------	      

		if( eval ) delete [] eval;
		maxtrees_count ++;
	      } // maxtrees
	    
	      maxdepth_count ++;
	    }// maxdepth
	  }
	  //#endif // USE_RF ***********************************************************************************:

	  if(conf) delete [] conf;
	  if(gt_num) delete [] gt_num;


	  //}// window_size

	  //if( id_map_from_label_to_prob  ) delete [] id_map_from_label_to_prob ;
  }// loop
  
   fclose( fp_result );

   return 0;
 }

 void LoadFileName( struct_file &file_info )
 {
   FILE *fp;
   char sdummy[512];

   // load train lists
   fp = fopen(file_info.train_file.c_str(), "r");
   if( fp != NULL ){
     do{
       struct_list tmp_list;
       if( fscanf( fp, "%s", sdummy)==EOF ) break;
       tmp_list.file_name = sdummy;
       tmp_list.flg = 0;
       file_info.train_file_lists.push_back( tmp_list );
     }while(1);
     fclose( fp );
   }

   // load test lists
   fp = fopen(file_info.test_file.c_str(), "r");
   if( fp != NULL ){
     do{
       struct_list tmp_list;
       if( fscanf( fp, "%s", sdummy)==EOF ) break;
       tmp_list.file_name = sdummy;
       tmp_list.flg = 0;
       file_info.test_file_lists.push_back( tmp_list );
     }while(1);
     fclose( fp );
   }
}



 void SetMatFeature( vector<struct_bfo> data, struct_feature_dimension feature_dim_4_train, int class_id, Mat &mat_data, Mat &mat_data_class )
{
  int num_data = data.size();
  //int dimension = data[0].feature.size();
  int dimension = feature_dim_4_train.num;

  mat_data = Mat::zeros( num_data, dimension, CV_32FC1 );
  mat_data_class = Mat::zeros( num_data, 1, CV_32FC1 );
  
  int label = 0;
  for(int i=0; i<num_data; i++){
    if( class_id == data[i].label ) label = 1;
    else label = 0;

    float* _mat_data_class = mat_data_class.ptr<float>(i);
    float* _mat_data = mat_data.ptr<float>(i);
    _mat_data_class[0] = (float)label;
    for(int j=0; j<dimension; j++){
      _mat_data[j] = (float)data[i].feature[j + feature_dim_4_train.start];
    }
  }
}


void MakeRandomNumber( int num_list, vector<int> &rand_id )
{
  if( num_list == 0 ) return;

  unsigned char *flg = new unsigned char [num_list];
  int id, num;

  memset( flg, 0, num_list );
  num = 0;
  do{
    id = rand() % num_list;
    if( flg[id] == 0 ){
      rand_id.push_back( id );
      flg[id] = 1;
      num ++;
    }
    if(num >= num_list)
      break;
  }while(1);

  if( flg ) delete [] flg;
}


void initialize_eval( struct_eval &eval )
{
  eval.tp = 0;
  eval.tn = 0;
  eval.fn = 0;
  eval.fp = 0;
  eval.tp_ident = 0;
  eval.fn_ident = 0;
  eval.recall = .0;
  eval.precision = .0;
  eval.F = .0;
  eval.CCR = .0;
}



void evaluate_target_id( vector<Mat> prob_mat_vec, Mat &estimated_class_mat, Mat mat_label, struct_eval *eval, struct_info other_info )
{
  int src_rows = mat_label.rows;
  int src_cols = mat_label.cols;
  int window_skip = other_info.window_skip;
  int w_size = other_info.w_size;

  // ROC and CCR for target_id
  if( other_info.evaluate_class_id >= 0 ){
    int id_category = other_info.bco_label_list.label_id[other_info.evaluate_class_id];
    //int target_id_in_prob = id_map_from_label_to_prob[target_id_in_mc_label_list];
    for(int r=w_size; r<(src_rows-w_size); r+=window_skip){
      double* _prob_mat_vec = prob_mat_vec[id_category].ptr<double>(r);
      uchar* _estimated_class_mat = estimated_class_mat.ptr<uchar>(r);
      float* _mat_label = mat_label.ptr<float>(r);

      for(int c=w_size; c<(src_cols-w_size); c+=window_skip){
	int gt_class, estimated_class;
	double prob_target;
	gt_class = (int)_mat_label[c];//mc_label_src.data[r*mc_label_src.cols*mc_label_src.chans + c*mc_label_src.chans + 0];
	prob_target = _prob_mat_vec[c];
      
	if( gt_class < 0 ) continue;
      
      
	for(int k=0; k<other_info.roc_threshold.num; k++){
	  // Recognition problem
	  double threshold = other_info.roc_threshold.start + other_info.roc_threshold.gap * (double)k;
	  if( id_category == gt_class ){
	    if( prob_target >= threshold )
	      eval[k].tp ++;
	    else
	      eval[k].fn ++;
	  }
	  else{
	    if( prob_target >= threshold )
	      eval[k].fp ++;
	    else
	      eval[k].tn ++;
	    //}
	  }
	
	  // identification problem
	  if( gt_class == _estimated_class_mat[c] )
	    eval[k].tp_ident ++;
	  else
	    eval[k].fn_ident ++;

	}
      }
    }
  }
  else{
    for(int ii=0; ii<other_info.bco_label_list.n_label; ii++){
      // int id_in_mc_label_list = other_info.bco_label_list.label_id[ii];
      // int id_in_prob = id_map_from_label_to_prob[id_in_mc_label_list];
      int id_category = other_info.bco_label_list.label_id[ii];

      for(int r=w_size; r<(src_rows-w_size); r+=window_skip){
	double* _prob_mat_vec = prob_mat_vec[id_category].ptr<double>(r);
	uchar* _estimated_class_mat = estimated_class_mat.ptr<uchar>(r);
	float* _mat_label = mat_label.ptr<float>(r);

	for(int c=w_size; c<(src_cols-w_size); c+=window_skip){
	  int gt_class, estimated_class;
	  double prob_target;
	  gt_class = (int)_mat_label[c];//mc_label_src.data[r*mc_label_src.cols*mc_label_src.chans + c*mc_label_src.chans + 0];
	  prob_target = _prob_mat_vec[c];
      
	  if( gt_class < 0 ) continue;
      
      
	  for(int k=0; k<other_info.roc_threshold.num; k++){
	    // Recognition problem
	    double threshold = other_info.roc_threshold.start + other_info.roc_threshold.gap * (double)k;
	    if( id_category == gt_class ){
	      if( prob_target >= threshold )
		eval[k].tp ++;
	      else
		eval[k].fn ++;
	    }
	    else{
	      if( prob_target >= threshold )
		eval[k].fp ++;
	      else
		eval[k].tn ++;
	      //}
	    }
	
	    // identification problem
	    if( gt_class == _estimated_class_mat[c] )
	      eval[k].tp_ident ++;
	    else
	      eval[k].fn_ident ++;
	  }
	}
      }
    }
  }
}

void visualize_predicted_target_id( string fullpath, Mat estimated_class_mat, vector<Mat> prob_mat_vec, Mat &overlay_mat, struct_info other_info )
{
  char rname[512];

  Mat rgb_mat = imread( fullpath.c_str(), CV_LOAD_IMAGE_COLOR );
  Mat mask_mat = Mat::zeros( rgb_mat.rows, rgb_mat.cols, CV_8UC3 );
  Mat color_mat = Mat::zeros( rgb_mat.rows, rgb_mat.cols, CV_8UC3 );

  int src_rows = rgb_mat.rows;
  int src_cols = rgb_mat.cols;
  int src_chans = rgb_mat.channels();

  for(int j=0; j<other_info.bco_label_list.n_label; j++){
    int counter = 0, intensity;
    //int id_in_prob = id_map_from_label_to_prob[j];
    int id_category = other_info.bco_label_list.label_id[j];
    //if( id_in_prob < 0 ) continue;

    // shows only other_info.evaluate_class_id
    if( other_info.evaluate_class_id >= 0)
      if( other_info.evaluate_class_id != j ) continue;
		  
    for(double threshold=0.0; threshold<=0.0; threshold+=0.05){
      //for(double threshold=0.4; threshold<=0.4; threshold+=0.05){
		    
      for(int r=0; r<src_rows; r++){
	//for(int r=mc_label_src.rows/4; r<mc_label_src.rows; r++){

	uchar* _estimated_class_mat = estimated_class_mat.ptr<uchar>(r);
	Vec3b* _mask_mat = mask_mat.ptr<Vec3b>(r);
	for(int c=0; c<src_cols; c++){
	  int estimated_class = _estimated_class_mat[c];
	  // this is used for identification problem
	  if( estimated_class != id_category ) continue;
			
	  double prob = prob_mat_vec[id_category].at<double>(r,c);
	  if( prob < threshold ) continue;
			
	  //intensity = (int)min( (prob * 255.0), 255.0 );
	  intensity = (int)min( (prob * 200.0), 255.0 );

	  //debug
	  //intensity = 0;

	  _mask_mat[c].val[0] = intensity;
	  _mask_mat[c].val[1] = intensity;
	  _mask_mat[c].val[2] = intensity;

	  Scalar color;
	  color = Scalar(other_info.bco_label_list.data[j*other_info.bco_label_list.chans+2], 
			 other_info.bco_label_list.data[j*other_info.bco_label_list.chans+1], 
			 other_info.bco_label_list.data[j*other_info.bco_label_list.chans+0] );
	  color_mat.at<Vec3b>(r,c).val[0] = color.val[0];
	  color_mat.at<Vec3b>(r,c).val[1] = color.val[1];
	  color_mat.at<Vec3b>(r,c).val[2] = color.val[2];
			
	}
      }

#if 1
      // jet color
      color_mat = mask_mat.clone();
      applyColorMap(color_mat, color_mat, COLORMAP_JET);
#endif

      bco_Overlay( rgb_mat, mask_mat, color_mat, overlay_mat );
      rgb_mat = overlay_mat.clone();


		    
    }
  }

  sprintf(rname, "overlay");
  imshow( rname, overlay_mat );
  cvWaitKey( 50 );
  //cvWaitKey( 0 );
}

void LoadRandomNumber( int loop, int num_list, char rname_rand[], vector<int> &rand_id )
{
  int loop_id, num, ddummy;
  FILE *fp = fopen( rname_rand, "r" );

  do{
    if( fscanf( fp, "%d %d", &loop_id, &num)==EOF ) break;
    if( num != num_list ){
      cout << "the number of files is different" << endl;
      getchar();
    }

    for(int i=0; i<num; i++){
      fscanf( fp, "%d", &ddummy );
      if(loop == loop_id)
	rand_id.push_back( ddummy );
    }
  }while(1);

  fclose( fp );
}
