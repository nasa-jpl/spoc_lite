/*
  author: Yumi Iwashita
*/

#ifndef __IBO_LOCAL_BINARY_PATTERNS_H__
#define __IBO_LOCAL_BINARY_PATTERNS_H__

#define MAX_NUM_LBP 256

namespace ibo
{

struct struct_lbp_parameter{
  int Flg_LBP[256];
  int SDIM;
  int db;// thresholds for intensity difference between the center and neighbor pixels. more than or eaqual to 1
};

int makeELBP( int u, int* flg );
void calc_lbp( cv::Mat src_mat, cv::Mat &dst_lbp_mat, int db );// db: threshold for intensity difference
int calc_lbp_hist( cv::Mat lbp_mat, float* f, int* flg );

} // namespace ibo

#endif // __IBO_LOCAL_BINARY_PATTERNS_H__
