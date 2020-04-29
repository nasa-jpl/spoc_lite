/*
  author: Yumi Iwashita

 example can be found in RTD/svm_RF_.../main.cpp
*/

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

#include <opencv2/opencv.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "bco.h"
#include "bop.h"
#include "lbp.h"

using namespace cv;
using namespace std;

namespace ibo
{

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wunused-variable"


// u : parameter　使っていない
// flg : flag to use or ignore values
int makeELBP(int u, int* flg)
{
  int n;
  int s=0;
  for(int k=0;k<256;k++) flg[k] = -1;
  
  int base;
  for(int k=0;k<256;k++){
    if(flg[k]<0) {
      base = k;
      for(int j=0; j<8; j++){
	n = (k<<j & 0xff) | ( (k>>(8-j)) & 0xff);
	flg[n] = base;
      }
      s++;
    }
  }
  return 256;
}

#if 1
void calc_lbp(Mat src_mat, Mat &dst_mat, int db )// db: threshold for intensity difference
{
  Mat gray_mat = Mat::zeros(src_mat.rows, src_mat.cols, CV_8UC1);
  dst_mat = Mat::zeros(src_mat.rows, src_mat.cols, CV_8UC1);
  //dst_mat = src_mat.clone();

  int width = src_mat.cols;
  int height = src_mat.rows;
  
  if(src_mat.channels() != 1)
    cvtColor( src_mat, gray_mat, CV_RGB2GRAY);
  else
    gray_mat = src_mat.clone();
  
  // from here tomorrow  
  for(int y=0; y<height; y++){

    uchar *_gray_mat_p1, *_gray_mat_m1, *_gray_mat;
    if( y<(height-1) )
      _gray_mat_p1 = gray_mat.ptr<uchar>(y+1);
    if( y>=1 )
      _gray_mat_m1 = gray_mat.ptr<uchar>(y-1);
    _gray_mat = gray_mat.ptr<uchar>(y);

    uchar* _dst_mat = dst_mat.ptr<uchar>(y);
    uchar c;
    for(int x=0; x<width; x++){
      //unsigned char c = gray_mat.at<uchar>(y,x);//((unsigned char*)(gray->imageData + gray->widthStep*(y  )))[x];
      c = _gray_mat[x];
      
      if(y<1 || y > height-2 || x<1 || x>width-2) {
	//dst_mat.at<uchar>(y,x) = 0;
	_dst_mat[x] = 0;
      } else {
	// 8 neighbors
	//unsigned char b = 
	//( gray_mat.at<uchar>(y  ,x+1) < c ? 0x00 : 0x01) |
	//( gray_mat.at<uchar>(y+1,x+1) < c ? 0x00 : 0x02) |
	//( gray_mat.at<uchar>(y+1,x  ) < c ? 0x00 : 0x04) |
	//( gray_mat.at<uchar>(y+1,x-1) < c ? 0x00 : 0x08) |
	//( gray_mat.at<uchar>(y  ,x-1) < c ? 0x00 : 0x10) |
	//( gray_mat.at<uchar>(y-1,x-1) < c ? 0x00 : 0x20) |
	//( gray_mat.at<uchar>(y-1,x  ) < c ? 0x00 : 0x40) |
	//( gray_mat.at<uchar>(y-1,x+1) < c ? 0x00 : 0x80);
	//dst_mat.at<uchar>(y,x) = b;
	unsigned char b = 
	  ( (_gray_mat[x+1]   -c) < db ? 0x00 : 0x01) |
	  ( (_gray_mat_p1[x+1]-c) < db ? 0x00 : 0x02) |
	  ( (_gray_mat_p1[x]  -c) < db ? 0x00 : 0x04) |
	  ( (_gray_mat_p1[x-1]-c) < db ? 0x00 : 0x08) |
	  ( (_gray_mat[x-1]   -c) < db ? 0x00 : 0x10) |
	  ( (_gray_mat_m1[x-1]-c) < db ? 0x00 : 0x20) |
	  ( (_gray_mat_m1[x]  -c) < db ? 0x00 : 0x40) |
	  ( (_gray_mat_m1[x+1]-c) < db ? 0x00 : 0x80);
	_dst_mat[x] = b;
      }
      
    }
  }

  return;
}
#else
//void LBP(IplImage *src, IplImage* &dst, IplImage* mask = NULL)
void LBP(Mat src_mat, Mat &dst_mat )
{
  IplImage *src = new IplImage(src_mat);
  IplImage *dst;// = new IplImage(dst_mat);

  int width = src->width;
  int height = src->height;
  
  IplImage *gray  = cvCreateImage( cvSize(width, height), IPL_DEPTH_8U, 1);
  
  if(src->nChannels!=1){
    cvCvtColor(src, gray, CV_RGB2GRAY);
  } else {
    cvCopy(src, gray);
  }
  
  //	cvSmooth(gray,gray,CV_GAUSSIAN);
  
  dst  = cvCreateImage( cvSize(width, height), IPL_DEPTH_8U, 1);
  cvZero(dst);
  
  for(int y=0; y<height; y++){
    for(int x=0; x<width; x++){
      unsigned char c = ((unsigned char*)(gray->imageData + gray->widthStep*(y  )))[x];
      
      if(y<1 || y > height-2 || x<1 || x>width-2) {
	((unsigned char*)(dst->imageData + dst->widthStep*y))[x] = 0;
      } else {
	unsigned char b = 
	  (((unsigned char*)(gray->imageData + gray->widthStep*(y  )))[x+1] < c ? 0x00 : 0x01) |
	  (((unsigned char*)(gray->imageData + gray->widthStep*(y+1)))[x+1] < c ? 0x00 : 0x02) |
	  (((unsigned char*)(gray->imageData + gray->widthStep*(y+1)))[x ]  < c ? 0x00 : 0x04) |
	  (((unsigned char*)(gray->imageData + gray->widthStep*(y+1)))[x-1] < c ? 0x00 : 0x08) |
	  (((unsigned char*)(gray->imageData + gray->widthStep*(y  )))[x-1] < c ? 0x00 : 0x10) |
	  (((unsigned char*)(gray->imageData + gray->widthStep*(y-1)))[x-1] < c ? 0x00 : 0x20) |
	  (((unsigned char*)(gray->imageData + gray->widthStep*(y-1)))[x ]  < c ? 0x00 : 0x40) |
	  (((unsigned char*)(gray->imageData + gray->widthStep*(y-1)))[x+1] < c ? 0x00 : 0x80);
	
	((unsigned char*)(dst->imageData + dst->widthStep*y))[x] = b;
      }
      
      //if(mask){
      //if(((unsigned char*)(mask->imageData + mask->widthStep*y))[x] == 0){
      //  ((unsigned char*)(dst->imageData + dst->widthStep*y))[x] = 0;
      //} 
      //} 
    }
  }
  //if(src) cvReleaseImage(&src);
  
  Mat tmp_mat( dst );
  dst_mat = tmp_mat.clone();

  if(gray) cvReleaseImage(&gray);
  if(dst) cvReleaseImage(&dst);

  return;
}
#endif

#if 0
  //original (IplImage version)
//int calc_hist(IplImage *src, IplImage *mask_img, float* f, IplImage *mask =  NULL, int* flg = NULL)
int calc_lbp_hist( Mat lbp_img_mat, float* f, int* flg = NULL)
{
  IplImage *lbp_img = new IplImage(lbp_img_mat);

  int width = lbp_img->width;
  int height = lbp_img->height;
  int depth = lbp_img->depth;
  
  //cout << width << " " << height << " " << depth << endl;
  if(depth != IPL_DEPTH_8U) return -1;
  
  int n = 0;
  //if(mask){
  //  for(int i=0; i<mask->imageSize; i++) 
  //    if(*((unsigned char*)(mask->imageData) + i) == 0) n++;
  //}
  
  int hist_size = 256;
  float max_value = 0;
  float range_0[] = { 0, 256 };
  float *ranges[] = { range_0 };
  CvHistogram *hist;
  int imageSize = lbp_img->imageSize;
  int dim;
  
  hist = cvCreateHist (1, &hist_size, CV_HIST_ARRAY, ranges, 1);
  
  // (6)ヒストグラムを計算して，スケーリング
  //cvCalcHist (&lbp_img, hist, 0, mask);
  //cvCalcHist (&lbp_img, hist, 0, mask_img);
  cvCalcHist (&lbp_img, hist, 0);
  //	cvGetMinMaxHistValue (hist, 0, &max_value, 0, 0);
  //	cvScale (hist->bins, hist->bins, 1.0 / imageSize, 0);
  
  float w = 1;
  //	float w = pow(2.0,k);
  //	float w = imageSize;
  //	float w = imageSize - n;
  
  if(flg){
    //cout << "calc hist" << endl;
    float *full = new float[hist_size];
    for (int j = 0; j < hist_size; j++){
      full[j] = cvGetReal1D (hist->bins, j) / w;
    }
    
    int gp[37][8];
    int ng[37];
    int kn=0;
    memset(gp,0x00,sizeof(gp));
    memset(ng,0x00,sizeof(ng));
    
    
    for(int k=0;k<256;k++){
      if(k==flg[k]) {
	gp[kn][ng[kn]]= k;
	ng[kn]++;
	kn++;
      } else {
	for(int j=0; j<kn; j++){
	  if(gp[j][0]==flg[k]){
	    gp[j][ng[j]]= k;
	    ng[j]++;
	    break;
	  }
	}
      }
    }
    
    for(int j=0; j<kn; j++){
      if(ng[j]>1){
	IplImage *src_img =  cvCreateImage ( cvSize(ng[j],1), IPL_DEPTH_64F, 1);
	IplImage *realInput;
	IplImage *imaginaryInput;
	IplImage *complexInput;
	IplImage *image_Re;
	IplImage *image_Im;
	int dft_M, dft_N;
	CvMat *dft_A, tmp;
	
	realInput = cvCreateImage (cvGetSize (src_img), IPL_DEPTH_64F, 1);
	imaginaryInput = cvCreateImage (cvGetSize (src_img), IPL_DEPTH_64F, 1);
	complexInput = cvCreateImage (cvGetSize (src_img), IPL_DEPTH_64F, 2);
	
	for(int k=0; k<ng[j]; k++) {
	  ((double*)(src_img->imageData))[k] = full[gp[j][k]];
	}
	
	// (1)入力画像を実数配列にコピーし，虚数配列とマージして複素数平面を構成
	cvScale (src_img, realInput, 1.0, 0.0);
	cvZero (imaginaryInput);
	cvMerge (realInput, imaginaryInput, NULL, NULL, complexInput);
	
	// (2)DFT用の最適サイズを計算し，そのサイズで行列を確保する
	dft_M = 1;//cvGetOptimalDFTSize (src_img->height - 1);
	dft_N = ng[j];//cvGetOptimalDFTSize (src_img->width - 1);
	dft_A = cvCreateMat (dft_M, dft_N, CV_64FC2);
	image_Re = cvCreateImage (cvSize (dft_N, dft_M), IPL_DEPTH_64F, 1);
	image_Im = cvCreateImage (cvSize (dft_N, dft_M), IPL_DEPTH_64F, 1);
	
	// (3)複素数平面をdft_Aにコピーし，残りの行列右側部分を0で埋める
	cvGetSubRect (dft_A, &tmp, cvRect (0, 0, src_img->width, src_img->height));
	cvCopy (complexInput, &tmp, NULL);
	if (dft_A->cols > src_img->width) {
	  cvGetSubRect (dft_A, &tmp, cvRect (src_img->width, 0, dft_A->cols - src_img->width, src_img->height));
	  cvZero (&tmp);
	}
	
	// (4)離散フーリエ変換を行い，その結果を実数部分と虚数部分に分解
	cvDFT (dft_A, dft_A, CV_DXT_FORWARD, complexInput->height);
	cvSplit (dft_A, image_Re, image_Im, 0, 0);
	
	// (5)スペクトルの振幅を計算 Mag = sqrt(Re^2 + Im^2)
	cvPow (image_Re, image_Re, 2.0);
	cvPow (image_Im, image_Im, 2.0);
	cvAdd (image_Re, image_Im, image_Re, NULL);
	cvPow (image_Re, image_Re, 0.5);
	
	for(int k=0; k<ng[j]; k++){
	  full[gp[j][k]] = ((double*)(image_Re->imageData))[k];
	}

	cvReleaseImage(&realInput);
	cvReleaseImage(&imaginaryInput);
	cvReleaseImage(&complexInput);
	cvReleaseImage(&src_img);
	cvReleaseImage(&image_Re);
	cvReleaseImage(&image_Im);
	cvReleaseMat(&dft_A);
      }
    }
    
    dim = 0;
    float* p = f;
    for (int j = 0; j < hist_size; j++){
      if(flg[j]<0) continue;
      *p = full[j];
      p++;
      dim++;
    }
    delete full;
    
    //if(mask){
    //f[dim++] = n / w;
    //}
  } else {
    dim = 0;
    for (int j = 0; j < hist_size; j++){
      f[j] = cvGetReal1D (hist->bins, j) / w;
      dim++;
    }
    //if(mask) f[dim++] = n / w;
  }
  
  //cvReleaseImage(&lbp_img);
  return dim;
}
#else
// new (Mat version)
int calc_lbp_hist( Mat lbp_img_mat, float* f, int* flg = NULL)
{
  int hist_size = 256;
  float max_value = 0;
  float range_0[] = { 0, 256 };
  //const float *ranges[] = { range_0 };
  const float *ranges = range_0;
  //int channels[] = { 0 };
  int channels = 0;

  Mat hist;
  bool uniform = true; 
  bool accumulate = false;
  float w = 1.0;
  int dim = 0;

  if(flg){
    cout << "rotation invariant lbo is not implemeted yet @ calc_lbp_hist" << endl;
    return 0;
  }
  else{
    //calcHist ( &lbp_img_mat, 1, channels, Mat(), hist, 1, &hist_size, &ranges, uniform, accumulate );
    calcHist ( &lbp_img_mat, 1, &channels, Mat(), hist, 1, &hist_size, &ranges );

    dim = 0;
    for (int j = 0; j < hist_size; j++){
#if 0
      f[j] = hist.at<float>(j) / w;
      dim ++;
#else
      // speed up
      float* _hist = hist.ptr<float>(j);
      *(f+j) = _hist[0] / w;
      dim ++;
#endif
    }
  }

  return dim;
}
#endif

#pragma GCC diagnostic pop

} // namespace ibo
