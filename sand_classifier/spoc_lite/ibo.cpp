/*
 author: Yumi Iwashita
*/
#include <stdio.h>
#include <string>
#include <vector>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>

#include "ibo.h"
#include "classification_ibo.h"
#include "main.h"

using namespace std;
using namespace cv;
using namespace ibo;

void read_chars_with_space( FILE *fp, string &str )
{
  int c;
  int n = 0;
  int space_flg = 1;
  char rname[2014] = "";

#if 1
  do{
    c = getc( fp );
    if( space_flg ){
      // skip first spaces / tabs
      if( c == ' ' || c == '\t' )
	space_flg = 1;
      else{
	space_flg = 0;
	fseek(fp, -1, SEEK_CUR);
      }
    }
    else{
      if(c != '\n' && c != 0xd){
	//printf("%c", c);
	rname[n] = c;
	n++;
      }
      else
	break;
    }
  }while (1);
  //fseek(fp, 1, SEEK_CUR);
#else
  do{
    c = getc( fp );
    if( space_flg ){
      // skip first spaces / tabs
      if( c == '0x20' || c == '0x09' )
	space_flg = 1;
      else{
	space_flg = 0;
	//fseek(fp, -1, SEEK_CUR);
      }
    }
    else{
      //printf("%c", c);
      _rname[n] = c;
      n++;
    }
  }while (c != '\n' && c != 0xd);
#endif

  str = rname;
  //_rname = rname;
  //cout << "in " << rname << endl;
  //return rname;
}
  

int load_settings( string setting_file_name, struct_file &file_info, struct_info &other_info )
 {
   FILE *fp = fopen( setting_file_name.c_str(), "r" );
   char rname[1024], cmd[1024];
   string str, str_read;
   int color[3];
   int idummy;
   double ddummy;

   cout << "load_seettings file name is " << setting_file_name << endl;

   other_info.bco_label_list.data.clear();
   other_info.bco_label_list.chans = 3;
   other_info.bco_label_list.n_label = 0; 
   
   if( fp == NULL ){
     cout << "could not open file " << setting_file_name << endl;
     return 0;
   }
   
   bool flg = true;
   do{
     if( !flg ) break;
     int c = getc(fp);

     // skip #, %, space and tab and \n
     if( c == '#' || c == '%' || c == ' ' || c == '\t' || c == '\n' ){
       do{
	 c = getc( fp );
	 if( c == EOF ){
	   flg = false;
	   break;
	 }
       }while (c != '\n' && c != 0xd);
     }
     else{
       fseek(fp, -1, SEEK_CUR);
       
       if( fscanf( fp, "%s", rname )==EOF ) break;
       str = rname;
       
       char *_rname;//[2048];
       if( str == "DIR_PATH_TO_TRAIN_IMAGES" ){
	 cout << rname << endl;
	 read_chars_with_space( fp, str_read );
	 cout << str_read << endl;
	 file_info.train_src_fd = str_read;
       }
       else if( str == "DIR_PATH_TO_TEST_IMAGES" ){
	 cout << rname << endl;
	 read_chars_with_space( fp, str_read );
	 cout << str_read << endl;
	 file_info.test_src_fd = str_read;
       }
       else if( str == "DIR_PATH_TO_TRAIN_ANNOTAED_IMAGES" ){
	 cout << rname << endl;
	 read_chars_with_space( fp, str_read );
	 cout << str_read << endl;
	 file_info.train_label_fd = str_read;
       }
       else if( str == "DIR_PATH_TO_TEST_ANNOTAED_IMAGES" ){
	 cout << rname << endl;
	 read_chars_with_space( fp, str_read );
	 cout << str_read << endl;
	 file_info.test_label_fd = str_read;
       }
       else if( str == "LIST_OF_TRAIN_DATA" ){
	 cout << rname << endl;
	 read_chars_with_space( fp, str_read );
	 cout << str_read << endl;
	 file_info.train_file = str_read;
       }
       else if( str == "LIST_OF_TEST_DATA" ){
	 cout << rname << endl;
	 read_chars_with_space( fp, str_read );
	 cout << str_read << endl;
	 file_info.test_file = str_read;
       }
       else if( str == "DIR_PATH_OUTPUT" ){
	 cout << rname << endl;
	 read_chars_with_space( fp, str_read );
	 cout << str_read << endl;
	 file_info.result_fd = str_read;

	 string str2 ("/");
	 size_t last_position = buo_find_the_last_string( file_info.result_fd, str2 );
	 if( last_position == (file_info.result_fd.size()-1) )
	   file_info.result_fd = file_info.result_fd.substr( 0, (last_position) );
	 if( file_info.result_fd == "" )
	   file_info.result_fd = ".";

	 sprintf( cmd, "mkdir %s", file_info.result_fd.c_str() );
	 system(cmd);
       }
       else if( str == "COLOR_IMAGE_EXTENSION" ){
	 cout << rname << endl;
	 read_chars_with_space( fp, str_read );
	 cout << str_read << endl;
	 file_info.src_extension = str_read;
       }
       else if( str == "LABEL_IMAGE_EXTENSION" ){
	 cout << rname << endl;
	 read_chars_with_space( fp, str_read );
	 cout << str_read << endl;
	 file_info.label_extension = str_read;
       }
       else if( str == "N_OF_N_FOLDS" ){
	 cout << rname << endl;
	 fscanf(fp, "%d\n", &idummy);
	 cout << idummy << endl;
	 other_info.n_of_n_folds = idummy;
       }
       else if( str == "SAVE_PREDICTED_IMAGES" ){
	 cout << rname << endl;
	 fscanf(fp, "%d\n", &idummy);
	 cout << idummy << endl;
	 other_info.save_predicted_image = idummy;
       }
       else if( str == "CLASS" ){
	 cout << rname << endl;
	 fscanf(fp, " %d %d %d %d\n", &idummy, &color[0], &color[1], &color[2]);
	 cout << idummy << " " << color[0] << " " << color[1] << " " << color[2] << endl;
	 other_info.bco_label_list.label_id.push_back(other_info.bco_label_list.n_label);
	 other_info.bco_label_list.data.push_back(color[0]); 
	 other_info.bco_label_list.data.push_back(color[1]); 
	 other_info.bco_label_list.data.push_back(color[2]); 
	 other_info.bco_label_list.n_label ++;  
       }
       else if( str == "EVALUATE_CLASS_ID" ){
	 cout << rname << endl;
	 fscanf(fp, "%d\n", &idummy );
	 other_info.evaluate_class_id = idummy;
	 cout << idummy << endl;
       }
       else if( str == "COLOR_SPACE" ){
       	 cout << rname << endl;
       	 read_chars_with_space( fp, str_read );
	 other_info.color_space_type = str_read;
       	 cout << str_read << endl;
       }
       else if( str == "FEATURE_TYPE" ){
       	 cout << rname << endl;
       	 read_chars_with_space( fp, str_read );
	 other_info.feature_type.push_back( str_read );
       	 cout << str_read << endl;
       }
       else if( str == "LBP_THRESHOLD" ){
       	 cout << rname << endl;
	 fscanf(fp, "%d\n", &idummy );
	 other_info.lbp_threshold = idummy;
	 cout << idummy << endl;
       }
       else if( str == "FEATURE_NORMALIZATION" ){
       	 cout << rname << endl;
       	 read_chars_with_space( fp, str_read );
	 other_info.feature_norm_type = str_read;
       	 cout << str_read << endl;
       }
       else if( str == "WINDOW_SIZE" ){
       	 cout << rname << endl;
	 fscanf(fp, "%d\n", &idummy );
	 other_info.window_size = idummy;
	 cout << idummy << endl;
       }
       else if( str == "WINDOW_OVERLAP" ){
       	 cout << rname << endl;
	 fscanf(fp, "%lf\n", &ddummy );
	 other_info.window_overlap = ddummy;
	 cout << ddummy << endl;
       }
       else if( str == "CLASSIFIER" ){
       	 cout << rname << endl;
       	 read_chars_with_space( fp, str_read );
	 other_info.classifier_type = str_read;
       	 cout << str_read << endl;
       }
       else if( str == "LOAD_TRAIN_MODEL" ){
       	 cout << rname << endl;
       	 read_chars_with_space( fp, str_read );
	 if( str_read == "TRUE" )
	   other_info.load_train_model = true;
       	 cout << str_read << endl;
       }
       else if( str == "ROC_THRESHOLD_START" ){
       	 cout << rname << endl;
	 fscanf(fp, "%lf\n", &ddummy );
	 other_info.roc_threshold.start = ddummy;
	 cout << ddummy << endl;
       }
       else if( str == "ROC_THRESHOLD_END" ){
       	 cout << rname << endl;
	 fscanf(fp, "%lf\n", &ddummy );
	 other_info.roc_threshold.end = ddummy;
	 cout << ddummy << endl;
       }
       else if( str == "ROC_THRESHOLD_GAP" ){
       	 cout << rname << endl;
	 fscanf(fp, "%lf\n", &ddummy );
	 other_info.roc_threshold.gap = ddummy;
	 cout << ddummy << endl;
	 other_info.roc_threshold.num = (other_info.roc_threshold.end - other_info.roc_threshold.start) / other_info.roc_threshold.gap + 1;
       }
       else if( str == "KERNEL_TYPE" ){
       	 cout << rname << endl;
       	 read_chars_with_space( fp, str_read );
	 if( str_read == "LINEAR" )
	   other_info.kernel_type = LINEAR;
	 else if( str_read == "ORIGINAL" )
	   other_info.kernel_type = ORIGINAL;
	 else if( str_read == "HI" )
	   other_info.kernel_type = HI;
	 else if( str_read == "CHI2" )
	   other_info.kernel_type = CHI2;
	 else if( str_read == "RBF" )
	   other_info.kernel_type = RBF;
	 else if( str_read == "POLY" )
	   other_info.kernel_type = POLY;
	 else if( str_read == "SIGMOID" )
	   other_info.kernel_type = SIGMOID;
       	 cout << str_read << endl;
	 //cout << other_info.kernel_type << " " << LINEAR << endl;
       }
       else if( str == "SVM_TYPE" ){
       	 cout << rname << endl;
       	 read_chars_with_space( fp, str_read );
	 if( str_read == "C_SVC" )
	   other_info.svm_type = C_SVC;
	 else if( str_read == "NU_SVC" )
	   other_info.svm_type = NU_SVC;
       	 cout << str_read << endl;
       }
       else if( str == "SVM_PROBABILITY" ){
       	 cout << rname << endl;
	 fscanf(fp, "%d\n", &idummy );
	 other_info.svm_probability = idummy;
	 cout << idummy << endl;
       }
       else if( str == "C_START" ){
       	 cout << rname << endl;
	 fscanf(fp, "%d\n", &idummy );
	 other_info.c_start = idummy;
	 cout << idummy << endl;
       }
       else if( str == "C_END" ){
       	 cout << rname << endl;
	 fscanf(fp, "%d\n", &idummy );
	 other_info.c_end = idummy;
	 cout << idummy << endl;
       }
       else if( str == "C_GAP" ){
       	 cout << rname << endl;
	 fscanf(fp, "%d\n", &idummy );
	 other_info.c_gap = idummy;
	 cout << idummy << endl;
       }
       else if( str == "G_START" ){
       	 cout << rname << endl;
	 fscanf(fp, "%d\n", &idummy );
	 other_info.g_start = idummy;
	 cout << idummy << endl;
       }
       else if( str == "G_END" ){
       	 cout << rname << endl;
	 fscanf(fp, "%d\n", &idummy );
	 other_info.g_end = idummy;
	 cout << idummy << endl;
       }
       else if( str == "G_GAP" ){
       	 cout << rname << endl;
	 fscanf(fp, "%d\n", &idummy );
	 other_info.g_gap = idummy;
	 cout << idummy << endl;
       }
       else if( str == "SOLVER_TYPE" ){
       	 cout << rname << endl;
       	 read_chars_with_space( fp, str_read );
	 if( str_read == "L2R_LR" )
	   other_info.solver_type = L2R_LR;
	 else if( str_read == "L2R_L2LOSS_SVC_DUAL" )
	   other_info.solver_type = L2R_L2LOSS_SVC_DUAL;
	 else if( str_read == "L2R_L2LOSS_SVC" )
	   other_info.solver_type = L2R_L2LOSS_SVC;
	 else if( str_read == "L2R_L1LOSS_SVC_DUAL" )
	   other_info.solver_type = L2R_L1LOSS_SVC_DUAL;
	 else if( str_read == "MCSVM_CS" )
	   other_info.solver_type = MCSVM_CS;
	 else if( str_read == "L1R_L2LOSS_SVC" )
	   other_info.solver_type = L1R_L2LOSS_SVC;
	 else if( str_read == "L1R_LR" )
	   other_info.solver_type = L1R_LR;
	 else if( str_read == "L2R_LR_DUAL" )
	   other_info.solver_type = L2R_LR_DUAL;
	 else if( str_read == "L2R_L2LOSS_SVR" )
	   other_info.solver_type = L2R_L2LOSS_SVR;
	 else if( str_read == "L2R_L2LOSS_SVR_DUAL" )
	   other_info.solver_type = L2R_L2LOSS_SVR_DUAL;
       	 cout << str_read << endl;
       }
       else if( str == "MAX_DEPTH_START" ){
       	 cout << rname << endl;
	 fscanf(fp, "%d\n", &idummy );
	 other_info.maxdepth_start = idummy;
	 cout << idummy << endl;
       }
       else if( str == "MAX_DEPTH_END" ){
       	 cout << rname << endl;
	 fscanf(fp, "%d\n", &idummy );
	 other_info.maxdepth_end = idummy;
	 cout << idummy << endl;
       }
       else if( str == "MAX_DEPTH_GAP" ){
       	 cout << rname << endl;
	 fscanf(fp, "%d\n", &idummy );
	 other_info.maxdepth_gap = idummy;
	 cout << idummy << endl;
       }
       else if( str == "MAX_TREES_START" ){
       	 cout << rname << endl;
	 fscanf(fp, "%d\n", &idummy );
	 other_info.maxtrees_start = idummy;
	 cout << idummy << endl;
       }
       else if( str == "MAX_TREES_END" ){
       	 cout << rname << endl;
	 fscanf(fp, "%d\n", &idummy );
	 other_info.maxtrees_end = idummy;
	 cout << idummy << endl;
       }
       else if( str == "MAX_TREES_GAP" ){
       	 cout << rname << endl;
	 fscanf(fp, "%d\n", &idummy );
	 other_info.maxtrees_gap = idummy;
	 cout << idummy << endl;
       }
       // else if( str == "" ){
       // 	 cout << rname << endl;
       // 	 read_chars_with_space( fp, str_read );
       // 	 cout << str_read << endl;
       // }

     }
   }while( 1 );

   if( other_info.bco_label_list.data.empty() ){
     cout << "CLASS is not defined in setting.txt" << endl;
     return 0;
   }

   if( other_info.feature_type.size() == 0 )
     other_info.feature_type.push_back( "AVERAGE" );   
   
   other_info.window_skip = other_info.window_size * (1.0-other_info.window_overlap);
   if( other_info.window_skip == 0 ) other_info.window_skip = 1;// at least move the window 1 pixel

   other_info.w_size = floor(other_info.window_size / 2);
   if(other_info.w_size < 0) other_info.w_size = 0;
   
   fclose( fp );
   return 1;
 }
 
