/*
  author: Yumi Iwashita
*/

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

#include <opencv2/opencv.hpp>
//#include <opencv2/nonfree/nonfree.hpp>

#include "bco.h"
#include "bop.h"

using namespace cv;
using namespace std;

// Standard macros
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif


namespace ibo
{

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wunused-variable"

void bco_LoadCameraParameter( string int_cp_file, string ext_cp_file_ins2veh, string ext_cp_file_cam2ins, struct_camera_parameter &cp )
{
  FILE *fp_int, *fp_ext_ins2veh, *fp_ext_cam2ins;
  char dummy[255];
  double q[4];

  fp_int = fopen( int_cp_file.c_str(), "r" );
  fp_ext_ins2veh = fopen( ext_cp_file_ins2veh.c_str(), "r" );
  fp_ext_cam2ins = fopen( ext_cp_file_cam2ins.c_str(), "r" );

  // load internal camera parameters :::::::::::::::::::::::::::::::::::::::::::::::::::
  cout << int_cp_file << endl;
  int c = getc(fp_int);
  if( c == '#' || c == '%' ){
    //cout << "c ";
    // read the first line
    do{
      c = getc( fp_int );
    }while (c != '\n' && c != 0xd);
   
    // read the rest lines
    uchar flg_Amat_read = 0;
    do{
      int c = getc( fp_int );
      //# or %
      if( c == '#' || c == '%' ){
	do{
	  c = getc( fp_int );
	}while (c != '\n' && c != 0xd);
      }
      // R
      else if( c == 'R' ){
	do{
	  c = getc( fp_int );
	}while (c != '\n' && c != 0xd);
	fscanf( fp_int, "%lf %lf %lf", &q[0], &q[1], &q[2] );
	fscanf( fp_int, "%lf %lf %lf", &q[0], &q[1], &q[2] );
	fscanf( fp_int, "%lf %lf %lf", &q[0], &q[1], &q[2] );
      }
      // T
      else if( c == 'T' ){
	do{
	  c = getc( fp_int );
	}while (c != '\n' && c != 0xd);
	fscanf( fp_int, "%lf", &q[0] );
	fscanf( fp_int, "%lf", &q[0] );
	fscanf( fp_int, "%lf", &q[0] );
      }
      // k1, k2, k3, p1, p2
      else if( c == 'k' || c == 'p' ){
	do{
	  c = getc( fp_int );
	}while (c != '\n' && c != 0xd);
	fscanf( fp_int, "%lf", &q[0] );
      }
      // A
      else if( c == 'A' ){
	do{
	  c = getc( fp_int );
	}while (c != '\n' && c != 0xd);
	fscanf(fp_int, "%lf %lf %lf", &cp.A[0][0], &cp.A[0][1], &cp.A[0][2]);
	fscanf(fp_int, "%lf %lf %lf", &cp.A[1][0], &cp.A[1][1], &cp.A[1][2]);
	fscanf(fp_int, "%lf %lf %lf", &cp.A[2][0], &cp.A[2][1], &cp.A[2][2]);
	flg_Amat_read = 1;
      }
      if( flg_Amat_read ) break;
    }while(1);
  }
  else{
    // A only in int_cp_file
    fseek(fp_int, -1, SEEK_CUR);
    fscanf(fp_int, "%lf %lf %lf", &cp.A[0][0], &cp.A[0][1], &cp.A[0][2]);
    fscanf(fp_int, "%lf %lf %lf", &cp.A[1][0], &cp.A[1][1], &cp.A[1][2]);
    fscanf(fp_int, "%lf %lf %lf", &cp.A[2][0], &cp.A[2][1], &cp.A[2][2]);
  }

  double ratio = 1.0;
  cp.A[0][0] *= ratio;
  cp.A[1][1] *= ratio;
  cp.A[0][2] *= ratio;
  cp.A[1][2] *= ratio;

  printf("%lf %lf %lf\n", cp.A[0][0], cp.A[0][1], cp.A[0][2]);
  printf("%lf %lf %lf\n", cp.A[1][0], cp.A[1][1], cp.A[1][2]);
  printf("%lf %lf %lf\n", cp.A[2][0], cp.A[2][1], cp.A[2][2]);
  //getchar();
  // load internal camera parameters :::::::::::::::::::::::::::::::::::::::::::::::::::


  double R[3][3], T[3];
  fscanf(fp_ext_ins2veh, "%s = (%lf %lf %lf) (%lf %lf %lf %lf)", 
	 //dummy, &cp.ins_T_veh[0], &cp.ins_T_veh[1], &cp.ins_T_veh[2],
	 dummy, &T[0], &T[1], &T[2],
	 //&q[0], &q[1], &q[2], &q[3]);// (x,y,z,angle)
	 &q[2], &q[1], &q[0], &q[3]);// (z,y,x,angle) <- JPL one is this
  printf("%s = (%lf %lf %lf) (%lf %lf %lf %lf)\n", 
	 dummy, T[0], T[1], T[2],
	 q[0], q[1], q[2], q[3]);
  //bop_convQ4ToR33( cp.ins_R_veh, q );
#if 0
  // initial attempt. looks like this is not right
  bop_convQ4ToR33( R, q );
  bop_calcInvR33( R, R );
  bop_calcR33xV3( T, R, T );
  for(int j=0; j<3; j++) T[j] *= -1.0;
#else
  // looks like this is the right one
  bop_convQ4ToR33( R, q );
#endif
  bop_copyR33ToR33( cp.ins_R_veh, R );
  bop_copyV3ToV3( cp.ins_T_veh, T);  

  fscanf(fp_ext_cam2ins, "%s = (%lf %lf %lf) (%lf %lf %lf %lf)", 
	 //dummy, &cp.cam_T_ins[0], &cp.cam_T_ins[1], &cp.cam_T_ins[2],
	 dummy, &T[0], &T[1], &T[2],
	 //&q[0], &q[1], &q[2], &q[3]);// (x,y,z,angle)
	 &q[2], &q[1], &q[0], &q[3]);// (z,y,x,angle) <- JPL one is this
  printf("%s = (%lf %lf %lf) (%lf %lf %lf %lf)\n", 
	 dummy, T[0], T[1], T[2],
	 q[0], q[1], q[2], q[3]);
  //bop_convQ4ToR33( cp.cam_R_ins, q );
#if 0
  // initial attempt. looks like this is not right
  bop_convQ4ToR33( R, q );
  bop_calcInvR33( R, R );
  bop_calcR33xV3( T, R, T );
  for(int j=0; j<3; j++) T[j] *= -1.0;
#else
  // looks like this is the right one
  bop_convQ4ToR33( R, q );
#endif
  bop_copyR33ToR33( cp.cam_R_ins, R );
  bop_copyV3ToV3( cp.cam_T_ins, T);  

  //fprintf(fp_T_cam2ins, "%lf %lf %lf\n", cp.cam_R_ins[0][0], cp.cam_R_ins[0][1], cp.cam_R_ins[0][2]);
  //fprintf(fp_T_cam2ins, "%lf %lf %lf\n", cp.cam_R_ins[1][0], cp.cam_R_ins[1][1], cp.cam_R_ins[1][2]);
  //fprintf(fp_T_cam2ins, "%lf %lf %lf\n", cp.cam_R_ins[2][0], cp.cam_R_ins[2][1], cp.cam_R_ins[2][2]);

  fclose( fp_int );
  fclose( fp_ext_ins2veh );
  fclose( fp_ext_cam2ins );

}

void bco_LoadCameraParameter_int( string int_cp_cam0_file, string int_cp_cam1_file, struct_camera_parameter &cp )
{
  FILE *fp_int_cam0, *fp_int_cam1;
  char dummy[255];
  double q[4], A[3][3];

  fp_int_cam0 = fopen( int_cp_cam0_file.c_str(), "r" );
  fp_int_cam1 = fopen( int_cp_cam1_file.c_str(), "r" );

  // load internal camera parameters :::::::::::::::::::::::::::::::::::::::::::::::::::
  // cam0 ------------------------------------------------------------------
  cout << int_cp_cam0_file << endl;
  int c = getc(fp_int_cam0);
  if( c == '#' || c == '%' ){
    //cout << "c ";
    // read the first line
    do{
      c = getc( fp_int_cam0 );
    }while (c != '\n' && c != 0xd);
   
    // read the rest lines
    uchar flg_Amat_read = 0;
    do{
      int c = getc( fp_int_cam0 );
      //# or %
      if( c == '#' || c == '%' ){
	do{
	  c = getc( fp_int_cam0 );
	}while (c != '\n' && c != 0xd);
      }
      // R
      else if( c == 'R' ){
	do{
	  c = getc( fp_int_cam0 );
	}while (c != '\n' && c != 0xd);
	fscanf( fp_int_cam0, "%lf %lf %lf", &q[0], &q[1], &q[2] );
	fscanf( fp_int_cam0, "%lf %lf %lf", &q[0], &q[1], &q[2] );
	fscanf( fp_int_cam0, "%lf %lf %lf", &q[0], &q[1], &q[2] );
      }
      // T
      else if( c == 'T' ){
	do{
	  c = getc( fp_int_cam0 );
	}while (c != '\n' && c != 0xd);
	fscanf( fp_int_cam0, "%lf", &q[0] );
	fscanf( fp_int_cam0, "%lf", &q[0] );
	fscanf( fp_int_cam0, "%lf", &q[0] );
      }
      // k1, k2, k3, p1, p2
      else if( c == 'k' || c == 'p' ){
	do{
	  c = getc( fp_int_cam0 );
	}while (c != '\n' && c != 0xd);
	fscanf( fp_int_cam0, "%lf", &q[0] );
      }
      // A
      else if( c == 'A' ){
	do{
	  c = getc( fp_int_cam0 );
	}while (c != '\n' && c != 0xd);
	fscanf(fp_int_cam0, "%lf %lf %lf", &A[0][0], &A[0][1], &A[0][2]);
	fscanf(fp_int_cam0, "%lf %lf %lf", &A[1][0], &A[1][1], &A[1][2]);
	fscanf(fp_int_cam0, "%lf %lf %lf", &A[2][0], &A[2][1], &A[2][2]);
	flg_Amat_read = 1;
      }
      if( flg_Amat_read ) break;
    }while(1);
  }
  else{
    // A only in int_cp_file
    fseek(fp_int_cam0, -1, SEEK_CUR);
    fscanf(fp_int_cam0, "%lf %lf %lf", &A[0][0], &A[0][1], &A[0][2]);
    fscanf(fp_int_cam0, "%lf %lf %lf", &A[1][0], &A[1][1], &A[1][2]);
    fscanf(fp_int_cam0, "%lf %lf %lf", &A[2][0], &A[2][1], &A[2][2]);
  }

  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      cp.A_cam0[i][j] = A[i][j];
      //cout << cp.A_cam0[i][j] << " ";
    }
    //cout << endl;
  }

  // cam1 ------------------------------------------------------------------
  cout << int_cp_cam1_file << endl;
  c = getc(fp_int_cam1);
  if( c == '#' || c == '%' ){
    //cout << "c ";
    // read the first line
    do{
      c = getc( fp_int_cam1 );
    }while (c != '\n' && c != 0xd);
   
    // read the rest lines
    uchar flg_Amat_read = 0;
    do{
      int c = getc( fp_int_cam1 );
      //# or %
      if( c == '#' || c == '%' ){
	do{
	  c = getc( fp_int_cam1 );
	}while (c != '\n' && c != 0xd);
      }
      // R
      else if( c == 'R' ){
	do{
	  c = getc( fp_int_cam1 );
	}while (c != '\n' && c != 0xd);
	fscanf( fp_int_cam1, "%lf %lf %lf", &q[0], &q[1], &q[2] );
	fscanf( fp_int_cam1, "%lf %lf %lf", &q[0], &q[1], &q[2] );
	fscanf( fp_int_cam1, "%lf %lf %lf", &q[0], &q[1], &q[2] );
      }
      // T
      else if( c == 'T' ){
	do{
	  c = getc( fp_int_cam1 );
	}while (c != '\n' && c != 0xd);
	fscanf( fp_int_cam1, "%lf", &q[0] );
	fscanf( fp_int_cam1, "%lf", &q[0] );
	fscanf( fp_int_cam1, "%lf", &q[0] );
      }
      // k1, k2, k3, p1, p2
      else if( c == 'k' || c == 'p' ){
	do{
	  c = getc( fp_int_cam1 );
	}while (c != '\n' && c != 0xd);
	fscanf( fp_int_cam1, "%lf", &q[0] );
      }
      // A
      else if( c == 'A' ){
	do{
	  c = getc( fp_int_cam1 );
	}while (c != '\n' && c != 0xd);
	fscanf(fp_int_cam1, "%lf %lf %lf", &A[0][0], &A[0][1], &A[0][2]);
	fscanf(fp_int_cam1, "%lf %lf %lf", &A[1][0], &A[1][1], &A[1][2]);
	fscanf(fp_int_cam1, "%lf %lf %lf", &A[2][0], &A[2][1], &A[2][2]);
	flg_Amat_read = 1;
      }
      if( flg_Amat_read ) break;
    }while(1);
  }
  else{
    // A only in int_cp_file
    fseek(fp_int_cam1, -1, SEEK_CUR);
    fscanf(fp_int_cam1, "%lf %lf %lf", &A[0][0], &A[0][1], &A[0][2]);
    fscanf(fp_int_cam1, "%lf %lf %lf", &A[1][0], &A[1][1], &A[1][2]);
    fscanf(fp_int_cam1, "%lf %lf %lf", &A[2][0], &A[2][1], &A[2][2]);
  }

  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      cp.A_cam1[i][j] = A[i][j];
      //cout << cp.A_cam1[i][j] << " ";
    }
    //cout << endl;
  }
  //getchar();
  // load internal camera parameters :::::::::::::::::::::::::::::::::::::::::::::::::::

  fclose( fp_int_cam0 );
  fclose( fp_int_cam1 );
}

void bco_LoadCameraParameter_ext( string ext_cp_file, struct_camera_parameter &cp )
{
  FILE *fp_ext;
  char dummy[255];
  double q[4], t[3], r[3][3];

  fp_ext = fopen( ext_cp_file.c_str(), "r" );

  int count = 0;
  do{
    int c = getc(fp_ext);
    if( c == '#' || c == '%' ){
      //cout << "c ";
      do{
	c = getc( fp_ext );
      }while (c != '\n' && c != 0xd);
    }
    else{
      fseek(fp_ext, -1, SEEK_CUR);

      if( fscanf(fp_ext, "%s = (%lf %lf %lf) (%lf %lf %lf %lf)", 
		 dummy, &t[0], &t[1], &t[2],
		 //&q[0], &q[1], &q[2], &q[3])==EOF) break;// (x,y,z,angle)
		 &q[2], &q[1], &q[0], &q[3])==EOF) break;// (z,y,x,angle) <- JPL one is this

      printf("%s = (%lf %lf %lf) (%lf %lf %lf %lf)\n", 
	     dummy, t[0], t[1], t[2],
	     q[0], q[1], q[2], q[3]);

#if 0
      // initial attempt. looks like this is not right
      bop_convQ4ToR33( r, q );
      bop_calcInvR33( r, r );
      bop_calcR33xV3( t, r, t );
      for(int j=0; j<3; j++) t[j] *= -1.0;
#else
      // looks like this is the right one
      bop_convQ4ToR33( r, q );
#endif
      
      if( count == 0 ){
	bop_copyR33ToR33( cp.ins_R_veh, r );
	bop_copyV3ToV3( cp.ins_T_veh, t );
      }
      else if( count == 1 ){
	bop_copyR33ToR33( cp.cam0_R_ins, r );
	bop_copyV3ToV3( cp.cam0_T_ins, t );
      }
      else if( count == 2 ){
	bop_copyR33ToR33( cp.cam1_R_ins, r );
	bop_copyV3ToV3( cp.cam1_T_ins, t );
	break;
      }
      count ++;
    }
  }while(1);

  fclose( fp_ext );
}


//void class_bco::Morphology_Erosion( Mat &mat, int num )
void bco_Morphology_Erosion( Mat &mat, int num )
{
  Mat clone_mat = mat.clone();

  int erosion_type;
  int erosion_elem = 2;
  int erosion_size = 1;
  
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
  
  Mat element = getStructuringElement( erosion_type,
				       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
				       Point( erosion_size, erosion_size ) );
  
  /// Apply the erosion operation
  for(int i=0; i<num; i++){
    erode( clone_mat, mat, element );
    clone_mat = mat.clone();
  }
}

//void class_bco::Morphology_Dilation( Mat &mat, int num )
void bco_Morphology_Dilation( Mat &mat, int num )
{
  Mat clone_mat = mat.clone();

  int dilation_type;
  int dilation_elem = 2;
  int dilation_size = 1;

  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
  /// Apply the dilation operation
  for(int i=0; i<num; i++){
    dilate( clone_mat, mat, element );
    clone_mat = mat.clone();
  }

}

//void class_bco::Extract( Mat image, Mat mask_image, Mat &extract_image )
void bco_Extract( Mat image, Mat mask_image, Mat &extract_image )
{
  extract_image = image.clone();
  if( mask_image.channels() == 1 ){
    //mask_image.convertTo( mask_image, CV_8UC3 );
    cvtColor( mask_image, mask_image, CV_GRAY2BGR );
  }

  int color;
  for(int i=0; i<image.rows; i++){
    for(int j=0; j<image.cols; j++){
      color = mask_image.at<Vec3b>(i,j).val[0];
      if( color == 0 ){
	extract_image.at<Vec3b>(i,j).val[0] = 0;
	extract_image.at<Vec3b>(i,j).val[1] = 0;
	extract_image.at<Vec3b>(i,j).val[2] = 0;
      }
    }
  }

}

void bco_Extract( Mat image, vector< vector<Point> > &contours )
{
  Mat tmp_img = image.clone();
  if( image.channels() == 3 )
    cvtColor(tmp_img, tmp_img, CV_RGB2GRAY);

  //vector< vector<Point> > contours;
  vector<Vec4i> hierarchy;
  findContours(tmp_img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));  
}

// mask_image should be 3 channel. if it is 1 channel, type will change.
//void class_bco::Overlay( Mat image, Mat mask_image, Mat &overlay_image, Scalar color )
void bco_Overlay( Mat image, Mat mask_image, Mat &overlay_image, Scalar color )
{
#if 0
  // debug
  // resize
  double size_n = 0.5;
  Mat clone_img(image.rows*size_n, image.cols*size_n, image.type());
  resize(image, clone_img, clone_img.size(), 0, 0, INTER_NEAREST);
  image = clone_img.clone();
#endif

  Mat prob_mat = Mat::zeros( mask_image.rows, mask_image.cols, CV_8UC3 );
  Mat prob_mat_4c = Mat::zeros( mask_image.size(), CV_MAKE_TYPE(mask_image.type(),4) );

  //printf("type %d %d\n", prob_mat.type(), prob_mat_4c.type());

  if( mask_image.channels() == 3 )
    prob_mat = mask_image.clone();
  else{
    // I do not know why but this does not worl... type is not right
    //mask_image.convertTo( prob_mat, CV_8UC3 );
    // should be this one
    cvtColor( mask_image, prob_mat, CV_GRAY2BGR );
    //cout << "in bco_Overlay mask_image should be 3 channels" << endl;
    //getchar();
  }

  //printf("type %d %d\n", prob_mat.type(), prob_mat_4c.type());
  //getchar();

  int from_to[] = { 0,0, 1,1, 2,2, 2,3 };
  mixChannels( &prob_mat, 1, &prob_mat_4c, 1, from_to, 4 );

  // add alpha channel
  double color_ratio;
  for(int i=0; i<prob_mat.rows; i++){
    for(int j=0; j<prob_mat.cols; j++){
      color_ratio = (double)prob_mat.at<Vec3b>(i,j).val[0];// / 255.0;
      if( color_ratio > 200 ){
	prob_mat_4c.at<Vec4b>(i,j).val[0] = color.val[0];
	prob_mat_4c.at<Vec4b>(i,j).val[1] = color.val[1];
	prob_mat_4c.at<Vec4b>(i,j).val[2] = color.val[2];
      }
      prob_mat_4c.at<Vec4b>(i,j).val[3] = color_ratio * 0.4;//0.5;
    }
  }

  bco_AlphaBlending( image, prob_mat_4c );

  overlay_image = prob_mat_4c.clone();

}

void bco_Overlay( Mat image, Mat mask_color_image, Mat &overlay_image )
{
  Mat prob_mat = Mat::zeros( mask_color_image.rows, mask_color_image.cols, CV_8UC3 );
  Mat prob_mat_4c = Mat::zeros( mask_color_image.size(), CV_MAKE_TYPE(mask_color_image.type(),4) );

  if( mask_color_image.channels() == 3 )
    prob_mat = mask_color_image.clone();
  else{
    // I do not know why but this does not worl... type is not right
    //mask_image.convertTo( prob_mat, CV_8UC3 );
    // should be this one
    cvtColor( mask_color_image, prob_mat, CV_GRAY2BGR );
    // cout << "in bco_Overlay mask_image should be 3 channels" << endl;
    // getchar();
  }

  //printf("type %d %d\n", prob_mat.type(), prob_mat_4c.type());
  //getchar();

  int from_to[] = { 0,0, 1,1, 2,2, 2,3 };
  mixChannels( &prob_mat, 1, &prob_mat_4c, 1, from_to, 4 );

  // add alpha channel
  double color_ratio;
  for(int i=0; i<prob_mat.rows; i++){
    for(int j=0; j<prob_mat.cols; j++){
      color_ratio = 0.0;//(double)prob_mat.at<Vec3b>(i,j).val[0];// / 255.0;
      for(int k=0; k<3; k++){
	prob_mat_4c.at<Vec4b>(i,j).val[k] = mask_color_image.at<Vec3b>(i,j).val[k];
	color_ratio = MAX( color_ratio, mask_color_image.at<Vec3b>(i,j).val[k] );
      }
      prob_mat_4c.at<Vec4b>(i,j).val[3] = color_ratio * 0.4;//0.5;
    }
  }

  bco_AlphaBlending( image, prob_mat_4c );

  overlay_image = prob_mat_4c.clone();

}

void bco_Overlay( Mat image, Mat mask_image, Mat color_image, Mat &overlay_image )
{
  Mat prob_mat = Mat::zeros( color_image.rows, color_image.cols, CV_8UC3 );
  Mat prob_mat_4c = Mat::zeros( color_image.size(), CV_MAKE_TYPE(color_image.type(),4) );

  if( mask_image.channels() == 3 )
    prob_mat = mask_image.clone();
  else{
    // I do not know why but this does not worl... type is not right
    //mask_image.convertTo( prob_mat, CV_8UC3 );
    // should be this one
    cvtColor( mask_image, prob_mat, CV_GRAY2BGR );
    // cout << "in bco_Overlay mask_image should be 3 channels" << endl;
    // getchar();
  }

  //printf("type %d %d\n", prob_mat.type(), prob_mat_4c.type());
  //getchar();

  int from_to[] = { 0,0, 1,1, 2,2, 2,3 };
  mixChannels( &prob_mat, 1, &prob_mat_4c, 1, from_to, 4 );

  // add alpha channel
  double color_ratio;
  for(int i=0; i<prob_mat.rows; i++){
    for(int j=0; j<prob_mat.cols; j++){
      color_ratio = 0.0;//(double)prob_mat.at<Vec3b>(i,j).val[0];// / 255.0;
      for(int k=0; k<3; k++){
	prob_mat_4c.at<Vec4b>(i,j).val[k] = color_image.at<Vec3b>(i,j).val[k];
	color_ratio = MAX( color_ratio, prob_mat.at<Vec3b>(i,j).val[k] );
      }
      prob_mat_4c.at<Vec4b>(i,j).val[3] = color_ratio * 0.5;//0.5;
    }
  }

  bco_AlphaBlending( image, prob_mat_4c );

  overlay_image = prob_mat_4c.clone();

}

//void class_bco::AlphaBlending( Mat src_mat, Mat &prob_mat_4c )
void bco_AlphaBlending( Mat src_mat, Mat &prob_mat_4c )
{
  Mat img1, img2;
  img1 = prob_mat_4c.clone();
  img2 = src_mat.clone();

  // BGRAチャンネルに分離
  std::vector<cv::Mat> mv;
  cv::split(img1, mv);
  
  /// 合成処理
  std::vector<cv::Mat> tmp_a;
  cv::Mat alpha, alpha32f;
  // 4番目のチャンネル=アルファ
  alpha = mv[3].clone();
  mv[3].convertTo(alpha32f, CV_32FC1);
  //cv::normalize(alpha32f, alpha32f, 0., 1., cv::NORM_MINMAX);
  for(int i=0; i<alpha32f.rows; i++)
    for(int j=0; j<alpha32f.cols; j++)
      alpha32f.at<float>(i,j) /= 255.0;
  for(int i=0; i<3; i++) tmp_a.push_back(alpha32f);
  cv::Mat alpha32fc3, beta32fc3;
  cv::merge(tmp_a, alpha32fc3);
  cv::Mat tmp_ones = cv::Mat::ones(cv::Size(img2.rows, img2.cols*3), CV_32FC1);
  beta32fc3 = tmp_ones.reshape(3,img2.rows) - alpha32fc3;
  cv::Mat img1_rgb, img1_32f, img2_32f;
  mv.resize(3);
  cv::merge(mv, img1_rgb);
  img1_rgb.convertTo(img1_32f, CV_32FC3);
  img2.convertTo(img2_32f, CV_32FC3);
  // 二つの画像の重み付き和
  cv::Mat blend32f = img1_32f.mul(alpha32fc3) + img2_32f.mul(beta32fc3);
  cv::Mat blend;
  blend32f.convertTo(blend, CV_8UC3);

  // 表示
  //cv::namedWindow("Alpha channel", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
  //cv::namedWindow("Blend", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
  //cv::imshow("Alpha channel", alpha);
  //cv::imshow("Blend", blend);
  //cv::waitKey( 5 );

  prob_mat_4c = blend.clone();
  
  return;  
}

// void bco_calc_and_matching_features( Mat image, Mat image_1, Mat &feature, Mat &feature_1, double ship_area_xy_start[], double ship_area_xy_end[] )
// {
//   //CvMat *ptref, *ptcur;
//   int icnt, n, m, nearest_neighbor, i, j, k, mcnt = 0;
  
//   // C++ & 双方向マッチング
//   cv::Ptr<cv::FeatureDetector> detector_ref;
//   cv::Ptr<cv::FeatureDetector> detector_cur;
//   cv::Ptr<cv::DescriptorExtractor> extractor_ref;
//   cv::Ptr<cv::DescriptorExtractor> extractor_cur;
//   std::vector<cv::KeyPoint> keypoints_ref;
//   std::vector<cv::KeyPoint> keypoints_cur;
//   cv::Mat descriptors_ref;
//   cv::Mat descriptors_cur;
  
//   cv::Mat img_ref_mat = image.clone();//(img_ref.img_gray);
//   cv::Mat img_cur_mat = image_1.clone();//(img_cur.img_gray);

//   //if( img_ref_mat.channels() == 3 )
//   //img_ref_mat.convertTo( img_ref_mat, CV_8UC1 );
//   //if( img_cur_mat.channels() == 3 )
//   //img_cur_mat.convertTo( img_cur_mat, CV_8UC1 );

//   int rows = img_ref_mat.rows;
//   int cols = img_ref_mat.cols;

//   // detector
//   cv::initModule_nonfree();
//   detector_ref = FeatureDetector::create("SURF");
//   detector_cur = FeatureDetector::create("SURF");
//   detector_ref->detect(img_ref_mat, keypoints_ref);
//   detector_cur->detect(img_cur_mat, keypoints_cur);
  
//   // descriptor
//   extractor_ref = cv::DescriptorExtractor::create("SURF");
//   extractor_cur = cv::DescriptorExtractor::create("SURF");
//   extractor_ref->compute(img_ref_mat, keypoints_ref, descriptors_ref);
//   extractor_cur->compute(img_cur_mat, keypoints_cur, descriptors_cur);
  
//   // matcher
//   Ptr<DescriptorMatcher> matcher;
//   vector<DMatch> dmatch, match12, match21;
//   matcher = DescriptorMatcher::create("BruteForce");
//   matcher->match(descriptors_ref, descriptors_cur, match12);
//   matcher->match(descriptors_cur, descriptors_ref, match21);
//   //float maxdistance = 100.0;
//   //matcher->radiusMatch(descriptors_ref, descriptors_cur, match12, maxdistance);
//   //matcher->radiusMatch(descriptors_cur, descriptors_ref, match21, maxdistance);
//   int _idx;
//   int area_xy_s[2] = { ship_area_xy_start[0]*image.cols, ship_area_xy_start[1]*image.rows };
//   int area_xy_e[2] = { ship_area_xy_end[0]*image.cols, ship_area_xy_end[1]*image.rows };
//   cout << area_xy_s[0] << " " << area_xy_e[0] << endl;
//   cout << area_xy_s[1] << " " << area_xy_e[1] << endl;
//   for (size_t i = 0; i < match12.size(); i++){
//     cv::DMatch forward = match12[i];
//     cv::DMatch backward = match21[forward.trainIdx];
//     if (backward.trainIdx == forward.queryIdx){
//       // added by yumi
//       _idx = backward.trainIdx;
//       if( area_xy_s[0] <= keypoints_ref[_idx].pt.x && area_xy_e[0] > keypoints_ref[_idx].pt.x &&
// 	  area_xy_s[1] <= keypoints_ref[_idx].pt.y && area_xy_e[1] > keypoints_ref[_idx].pt.y )
// 	continue;
//       dmatch.push_back(forward);
//     }
//   }

//   cout << "calc_and_matching_features 1" << endl;
  
//   n = dmatch.size();
//   //ptref = cvCreateMat (n, 2, CV_32FC1);
//   //ptcur = cvCreateMat (n, 2, CV_32FC1);
//   Mat ptref(n,1,CV_32FC2);
//   Mat ptcur(n,1,CV_32FC2);


//   //int _idx;
//   for(icnt = 0; icnt < n; icnt++ ) {
//     _idx = dmatch[icnt].queryIdx;
//     //cvmSet(ptref, icnt, 0, (double)keypoints_ref[_idx].pt.x);
//     //cvmSet(ptref, icnt, 1, (double)keypoints_ref[_idx].pt.y);
//     ptref.at<Vec2f>(icnt,0)[0] = (float)keypoints_ref[_idx].pt.x;
//     ptref.at<Vec2f>(icnt,0)[1] = (float)keypoints_ref[_idx].pt.y;
    
//     _idx = dmatch[icnt].trainIdx;
//     //cvmSet(ptcur, icnt, 0, (double)keypoints_cur[_idx].pt.x);
//     //cvmSet(ptcur, icnt, 1, (double)keypoints_cur[_idx].pt.y);
//     ptcur.at<Vec2f>(icnt,0)[0] = (float)keypoints_cur[_idx].pt.x;
//     ptcur.at<Vec2f>(icnt,0)[1] = (float)keypoints_cur[_idx].pt.y;
//   }
  
//   cout << "calc_and_matching_features 2" << endl;

// #if 1
//   // disp matched features
//   int intensity;
//   //Mat tmp = Mat::zeros( rows, cols*2, CV_8UC3 );
//   Mat tmp = Mat::zeros( rows, cols, CV_8UC3 );
//   for(int i=0; i<rows; i++){
//     for(int j=0; j<cols; j++){
//       if(img_ref_mat.channels()==1)
// 	intensity = img_ref_mat.at<uchar>(i,j);
//       else
// 	intensity = img_ref_mat.at<Vec3b>(i,j).val[0];
//       tmp.at<Vec3b>(i,j).val[0] = intensity;
//       tmp.at<Vec3b>(i,j).val[1] = intensity;
//       tmp.at<Vec3b>(i,j).val[2] = intensity;
//       /*
//       if(img_ref_mat.channels()==1)
// 	intensity = img_cur_mat.at<uchar>(i,j);
//       else
// 	intensity = img_cur_mat.at<Vec3b>(i,j).val[0];
//       tmp.at<Vec3b>(i,j+cols).val[0] = intensity;
//       tmp.at<Vec3b>(i,j+cols).val[1] = intensity;
//       tmp.at<Vec3b>(i,j+cols).val[2] = intensity;
//       */
//     }
//   }
//   int x,y,xx,yy;
//   for(int i=0; i<n; i++ ) {
//     x = ptref.at<Vec2f>(i,0)[0];//cvmGet( ptref, i, 0 );
//     y = ptref.at<Vec2f>(i,0)[1];//cvmGet( ptref, i, 1 );
//     xx = ptcur.at<Vec2f>(i,0)[0];//cvmGet( ptcur, i, 0 );
//     yy = ptcur.at<Vec2f>(i,0)[1];//cvmGet( ptcur, i, 1 );
//     //line( tmp, Point(x,y), Point(xx+cols,yy), Scalar(0,0,255), 2, 8 );
//     line( tmp, Point(x,y), Point(xx,yy), Scalar(0,0,255), 2, 8 );
//   }
//   imshow("box", tmp);
//   imwrite( "h_matching.bmp", tmp );
 
//   waitKey( 0 );
// #endif

//   feature = ptref.clone();
//   feature_1 = ptcur.clone();

// }

void bco_RemoveNoise( Mat &mat, int threshold_area )
{

  // 1. noise remove ------------------------------------------------------------------------------
  Mat tmp_img = mat.clone();
  if( mat.channels() == 3 )
    cvtColor(tmp_img, tmp_img, CV_RGB2GRAY);
  Mat tmp_img2 = tmp_img.clone();

  vector< vector<Point> > contours;
  vector<Vec4i> hierarchy;
  findContours(tmp_img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
  
  int max_area = 0, area;
#if 0
  for (int j = 0; j < contours.size(); j++){
    area = contourArea(contours[j]);
    if (area > max_area){
      tmp_img = Mat::zeros(tmp_img.rows, tmp_img.cols, tmp_img.type());
      drawContours(tmp_img, contours, j, Scalar(255, 255, 255), -1, 8);
      max_area = area;
    }
  }
#else
  tmp_img = Mat::zeros(tmp_img.rows, tmp_img.cols, tmp_img.type());
  for (int j = 0; j < contours.size(); j++){
    area = contourArea(contours[j]);
    if (area > threshold_area)
      drawContours(tmp_img, contours, j, Scalar(255, 255, 255), -1, 8);
  }
#endif

  // take AND between tmp_img and mat
  for(int r=0; r<tmp_img.rows; r++){
    for(int c=0; c<tmp_img.cols; c++){
      if( tmp_img2.at<uchar>(r,c) == 0 && tmp_img.at<uchar>(r,c) == 255 )
	tmp_img.at<uchar>(r,c) = 0;
    }
  }
  
  if( mat.channels() == 3 )
    cvtColor(tmp_img, tmp_img, CV_GRAY2RGB);
  mat = tmp_img.clone();
}

// remove noise in the air too. This is for ship stereo
void bco_RemoveFlyingNoise( Mat &mat, int threshold_area )
{

  // 1. noise remove ------------------------------------------------------------------------------
  Mat tmp_img = mat.clone();
  if( mat.channels() == 3 )
    cvtColor(tmp_img, tmp_img, CV_RGB2GRAY);
  Mat tmp_img2 = tmp_img.clone();

  vector< vector<Point> > contours;
  vector<Vec4i> hierarchy;
  findContours(tmp_img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
  
  int max_area = 0, area;
#if 0
  for (int j = 0; j < contours.size(); j++){
    area = contourArea(contours[j]);
    if (area > max_area){
      tmp_img = Mat::zeros(tmp_img.rows, tmp_img.cols, tmp_img.type());
      drawContours(tmp_img, contours, j, Scalar(255, 255, 255), -1, 8);
      max_area = area;
    }
  }
#else
  // find major rect
  Rect major_rect = boundingRect( contours[0] );
  for (int j = 0; j < contours.size(); j++){
    area = contourArea(contours[j]);
    if( area > max_area ){
      major_rect = boundingRect( contours[j] );
      max_area = area;
    }
  }

  tmp_img = Mat::zeros(tmp_img.rows, tmp_img.cols, tmp_img.type());
  for (int j = 0; j < contours.size(); j++){
    area = contourArea( contours[j] );
    Rect tmp_rect = boundingRect( contours[j] );

    // if it is above the major one, considered as flying noise
    if( (tmp_rect.y+tmp_rect.height/2) < (major_rect.y+major_rect.height/2) ) continue;

    if (area > threshold_area)
      drawContours(tmp_img, contours, j, Scalar(255, 255, 255), -1, 8);
  }
#endif

  // take AND between tmp_img and mat
  for(int r=0; r<tmp_img.rows; r++){
    uchar* _tmp_img2 = tmp_img2.ptr<uchar>(r);
    uchar* _tmp_img  = tmp_img.ptr<uchar>(r);
    for(int c=0; c<tmp_img.cols; c++){
#if 0
      if( tmp_img2.at<uchar>(r,c) == 0 && tmp_img.at<uchar>(r,c) == 255 )
	tmp_img.at<uchar>(r,c) = 0;
#else
      if( _tmp_img[c] == 0 && _tmp_img[c] == 255 )
	_tmp_img[c] = 0;
#endif
    }
  }
  
  if( mat.channels() == 3 )
    cvtColor(tmp_img, tmp_img, CV_GRAY2RGB);
  mat = tmp_img.clone();
}

void bco_ExtractNoise( Mat &mat, int threshold_area, vector< vector<Point> > &contours )
{
  Mat tmp_img = mat.clone();
  if( mat.channels() == 3 )
    cvtColor(tmp_img, tmp_img, CV_RGB2GRAY);
  Mat tmp_img2 = tmp_img.clone();

  //vector< vector<Point> > contours;
  vector<Vec4i> hierarchy;
  findContours(tmp_img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
  
  int max_area = 0, area;
  tmp_img = Mat::zeros(tmp_img.rows, tmp_img.cols, tmp_img.type());
  for (int j = 0; j < contours.size(); j++){
    area = contourArea(contours[j]);
    if (area < threshold_area)
      drawContours(tmp_img, contours, j, Scalar(255, 255, 255), -1, 8);
  }

  if( mat.channels() == 3 )
    cvtColor(tmp_img, tmp_img, CV_GRAY2RGB);
  mat = tmp_img.clone();
}

void bco_FillHole( Mat &mat )
{
  Mat tmp_img = mat.clone();
  if( mat.channels() == 3 )
    cvtColor(tmp_img, tmp_img, CV_RGB2GRAY);
  Mat tmp_img2 = tmp_img.clone();

  vector< vector<Point> > contours;
  vector<Vec4i> hierarchy;
  findContours(tmp_img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
  
  int max_area = 0, area;
  tmp_img = Mat::zeros(tmp_img.rows, tmp_img.cols, tmp_img.type());
  for (int j = 0; j < contours.size(); j++){
    //area = contourArea(contours[j]);
    //if (area < threshold_area)
    drawContours(tmp_img, contours, j, Scalar(255, 255, 255), -1, 8);
  }

  if( mat.channels() == 3 )
    cvtColor(tmp_img, tmp_img, CV_GRAY2RGB);
  mat = tmp_img.clone();
}

// void bco_calc_and_matching_features( Mat image, Mat image_1, Mat &feature, Mat &feature_1 )
// {
//   //CvMat *ptref, *ptcur;
//   int icnt, n, m, nearest_neighbor, i, j, k, mcnt = 0;
  
//   // C++ & 双方向マッチング
//   cv::Ptr<cv::FeatureDetector> detector_ref;
//   cv::Ptr<cv::FeatureDetector> detector_cur;
//   cv::Ptr<cv::DescriptorExtractor> extractor_ref;
//   cv::Ptr<cv::DescriptorExtractor> extractor_cur;
//   std::vector<cv::KeyPoint> keypoints_ref;
//   std::vector<cv::KeyPoint> keypoints_cur;
//   cv::Mat descriptors_ref;
//   cv::Mat descriptors_cur;
  
//   cv::Mat img_ref_mat = image.clone();//(img_ref.img_gray);
//   cv::Mat img_cur_mat = image_1.clone();//(img_cur.img_gray);

//   //if( img_ref_mat.channels() == 3 )
//   //img_ref_mat.convertTo( img_ref_mat, CV_8UC1 );
//   //if( img_cur_mat.channels() == 3 )
//   //img_cur_mat.convertTo( img_cur_mat, CV_8UC1 );

//   int rows = img_ref_mat.rows;
//   int cols = img_ref_mat.cols;

//   // detector
//   cv::initModule_nonfree();
//   detector_ref = FeatureDetector::create("SURF");
//   detector_cur = FeatureDetector::create("SURF");
//   detector_ref->detect(img_ref_mat, keypoints_ref);
//   detector_cur->detect(img_cur_mat, keypoints_cur);
  
//   // descriptor
//   extractor_ref = cv::DescriptorExtractor::create("SURF");
//   extractor_cur = cv::DescriptorExtractor::create("SURF");
//   extractor_ref->compute(img_ref_mat, keypoints_ref, descriptors_ref);
//   extractor_cur->compute(img_cur_mat, keypoints_cur, descriptors_cur);
  
//   // matcher
//   Ptr<DescriptorMatcher> matcher;
//   vector<DMatch> dmatch, match12, match21;
//   matcher = DescriptorMatcher::create("BruteForce");
//   matcher->match(descriptors_ref, descriptors_cur, match12);
//   matcher->match(descriptors_cur, descriptors_ref, match21);
//   //float maxdistance = 100.0;
//   //matcher->radiusMatch(descriptors_ref, descriptors_cur, match12, maxdistance);
//   //matcher->radiusMatch(descriptors_cur, descriptors_ref, match21, maxdistance);
//   int _idx;
//   for (size_t i = 0; i < match12.size(); i++){
//     cv::DMatch forward = match12[i];
//     cv::DMatch backward = match21[forward.trainIdx];
//     if (backward.trainIdx == forward.queryIdx)
//       dmatch.push_back(forward);
//   }

//   cout << "calc_and_matching_features 1" << endl;
  
//   n = dmatch.size();
//   //ptref = cvCreateMat (n, 2, CV_32FC1);
//   //ptcur = cvCreateMat (n, 2, CV_32FC1);
//   Mat ptref(n,1,CV_32FC2);
//   Mat ptcur(n,1,CV_32FC2);


//   //int _idx;
//   for(icnt = 0; icnt < n; icnt++ ) {
//     _idx = dmatch[icnt].queryIdx;
//     //cvmSet(ptref, icnt, 0, (double)keypoints_ref[_idx].pt.x);
//     //cvmSet(ptref, icnt, 1, (double)keypoints_ref[_idx].pt.y);
//     ptref.at<Vec2f>(icnt,0)[0] = (float)keypoints_ref[_idx].pt.x;
//     ptref.at<Vec2f>(icnt,0)[1] = (float)keypoints_ref[_idx].pt.y;
    
//     _idx = dmatch[icnt].trainIdx;
//     //cvmSet(ptcur, icnt, 0, (double)keypoints_cur[_idx].pt.x);
//     //cvmSet(ptcur, icnt, 1, (double)keypoints_cur[_idx].pt.y);
//     ptcur.at<Vec2f>(icnt,0)[0] = (float)keypoints_cur[_idx].pt.x;
//     ptcur.at<Vec2f>(icnt,0)[1] = (float)keypoints_cur[_idx].pt.y;
//   }
  
//   cout << "calc_and_matching_features 2" << endl;

// #if 1
//   // disp matched features
//   int intensity;
//   //Mat tmp = Mat::zeros( rows, cols*2, CV_8UC3 );
//   Mat tmp = Mat::zeros( rows, cols, CV_8UC3 );
//   for(int i=0; i<rows; i++){
//     for(int j=0; j<cols; j++){
//       if(img_ref_mat.channels()==1)
// 	intensity = img_ref_mat.at<uchar>(i,j);
//       else
// 	intensity = img_ref_mat.at<Vec3b>(i,j).val[0];
//       tmp.at<Vec3b>(i,j).val[0] = intensity;
//       tmp.at<Vec3b>(i,j).val[1] = intensity;
//       tmp.at<Vec3b>(i,j).val[2] = intensity;
//       /*
//       if(img_ref_mat.channels()==1)
// 	intensity = img_cur_mat.at<uchar>(i,j);
//       else
// 	intensity = img_cur_mat.at<Vec3b>(i,j).val[0];
//       tmp.at<Vec3b>(i,j+cols).val[0] = intensity;
//       tmp.at<Vec3b>(i,j+cols).val[1] = intensity;
//       tmp.at<Vec3b>(i,j+cols).val[2] = intensity;
//       */
//     }
//   }
//   int x,y,xx,yy;
//   for(int i=0; i<n; i++ ) {
//     x = ptref.at<Vec2f>(i,0)[0];//cvmGet( ptref, i, 0 );
//     y = ptref.at<Vec2f>(i,0)[1];//cvmGet( ptref, i, 1 );
//     xx = ptcur.at<Vec2f>(i,0)[0];//cvmGet( ptcur, i, 0 );
//     yy = ptcur.at<Vec2f>(i,0)[1];//cvmGet( ptcur, i, 1 );
//     //line( tmp, Point(x,y), Point(xx+cols,yy), Scalar(0,0,255), 2, 8 );
//     line( tmp, Point(x,y), Point(xx,yy), Scalar(0,0,255), 2, 8 );
//   }
//   imshow("box", tmp);
//   imwrite( "h_matching.bmp", tmp );
 
//   waitKey( 0 );
// #endif

//   feature = ptref.clone();
//   feature_1 = ptcur.clone();

// }

void bco_resize_mat( Mat &mat, double resize_ratio )
{
  //Mat clone_mat = mat.clone();

  Mat resize_img(mat.rows*resize_ratio, mat.cols*resize_ratio, mat.type());
  resize(mat, resize_img, resize_img.size(), 0, 0, INTER_NEAREST);
  mat = resize_img.clone();

  return;
}

  cv::Mat bco_combine_2mats( cv::Mat src_mat1, cv::Mat src_mat2 )
{
  Mat overlay_image;
  
  overlay_image = Mat::zeros( max(src_mat1.rows, src_mat2.rows), (src_mat1.cols + src_mat2.cols), src_mat1.type() );
  
  Mat left_roi_ovly(overlay_image, Rect(0, 0, src_mat1.cols, src_mat2.rows));
  src_mat1.copyTo(left_roi_ovly);
  Mat right_roi_ovly(overlay_image, Rect(src_mat1.cols, 0, src_mat2.cols, src_mat2.rows));
  src_mat2.copyTo(right_roi_ovly);
  
  return overlay_image.clone();
}

  cv::Mat bco_combine_2mats_vertical( cv::Mat src_mat1, cv::Mat src_mat2 )
{
  Mat overlay_image;
  
  overlay_image = Mat::zeros( (src_mat1.rows + src_mat2.rows), max(src_mat1.cols, src_mat2.cols), src_mat1.type() );
  
  Mat left_roi_ovly(overlay_image, Rect(0, 0, src_mat1.cols, src_mat1.rows));
  src_mat1.copyTo(left_roi_ovly);
  Mat right_roi_ovly(overlay_image, Rect(0, src_mat1.rows, src_mat2.cols, src_mat2.rows));
  src_mat2.copyTo(right_roi_ovly);
  
  return overlay_image.clone();
}

  // note: mat_label is float (32F)
  void bco_assign_label( struct_bco_label bco_label_list, cv::Mat &mat_label )
  {
    Mat tmp_mat_label = mat_label.clone();

    int chans = tmp_mat_label.channels();
    int rows = tmp_mat_label.rows;
    int cols = tmp_mat_label.cols;

    // reset mat_label
    mat_label = Mat::zeros( rows, cols, CV_32FC1 );

    uchar color_hit;
    int color_match = -1;
    int n_label = bco_label_list.n_label;
    int intensity;
    int r, c, i, b;
    if( chans == 3 ){
      for (r=0; r < rows; r++){
	Vec3b* _tmp_mat_label = tmp_mat_label.ptr<Vec3b>(r);
	float* _mat_label = mat_label.ptr<float>(r);

	for (c=0; c < cols; c++){
	  color_match = -1;

	  for(i=0; i<n_label; i++){
	    color_hit = 1;
	
	    for (b=0; b < chans; b++){
	      intensity = _tmp_mat_label[c].val[chans-b-1];
	      if( intensity != bco_label_list.data[i*chans + b] ) color_hit = 0;
	    }
	
	    if( color_hit == 1 ){
	      color_match = i;
	      break;
	    }
	  }

	  // this can happen
	  if( color_match == -1 ){
	    //mc_label_src.data[r*cols + c] = 0;
	    _mat_label[c] = -1.0;
	    //getchar();
	    continue;
	  }
      
	  //color_match ++; // avoid class 0 (=no class)
	  //mc_label_src.data[r*cols + c] = mc_label_list.label_id[color_match];// original
	  _mat_label[c] = (float)bco_label_list.label_id[color_match];// original
	}
      }
    }
    else{
      cout << "label image should be 3 channels\n hit any key" << endl;
      getchar();
    }

  }

  // correct name is "opponent color", not "opposite color"
  void bco_convert_to_opposite_color( Mat &mat_in_out, bool flg_fix_intensity )
  {
    int r, b, c;
    int rows = mat_in_out.rows;
    int cols = mat_in_out.cols;
    int chans = mat_in_out.channels();

    if( chans != 3 ){
      cout << "image channel should be 3 for this process" << endl;
      return;
    }

    /*
      O1 = (G-R) / sqrt(2); (min: -255/sqrt(2), max: 255/sqrt(2))
      O2 = (G+R-2B) / sqrt(6); (min: -510/sqrt(6), max: 510/sqrt(6))
      O3 = (G+R+B) / sqrt(3); (min: 0, max: 765/sqrt(3))
    */
    double RR, GG, BB;
    double o[3];
    // changed to memory access
    double v_sqrt2 = sqrt(2);
    double v_sqrt3 = sqrt(3);
    double v_sqrt6 = sqrt(6);
    double v_255_v_sqrt2 = 255.0/v_sqrt2;
    double v_510_v_sqrt2_255 = 510.0/v_sqrt2 / 255.0;
    double v_510_v_sqrt6 = 510.0/v_sqrt6;
    double v_1020_v_sqrt2 = 1020.0/v_sqrt2;
    double v_765_v_sqrt3_255 = 765.0/v_sqrt3 / 255.0;

    // note that opencv is BGR, not RGB
    for (r=0; r < rows; r++){
      Vec3b* _mat_in_out = mat_in_out.ptr<Vec3b>(r);
      for (c=0; c < cols; c++){
	RR = (double)_mat_in_out[c].val[2];
	GG = (double)_mat_in_out[c].val[1];
	BB = (double)_mat_in_out[c].val[0];
	*o = (RR-GG) / v_sqrt2;
	*(o+1) = (RR+GG-2*BB) / v_sqrt6;
	*(o+2) = (RR+GG+BB) / v_sqrt3;

	// normalize
	*o = (*o + v_255_v_sqrt2) / v_510_v_sqrt2_255;
	*(o+1) = (*(o+1) + v_510_v_sqrt6) / v_1020_v_sqrt2 * 255.0;
	*(o+2) = *(o+2) / v_765_v_sqrt3_255;

	// set intensity stable
	if( flg_fix_intensity )
	  *(o+2) = 200;

	_mat_in_out[c].val[2] = (unsigned char)*o;
	_mat_in_out[c].val[1] = (unsigned char)*(o+1);
	_mat_in_out[c].val[0] = (unsigned char)*(o+2);
      }
    }
  }

#pragma GCC diagnostic pop

} // namespace ibo
