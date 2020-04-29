#ifndef __CLASSIFICATION_H__
#define __CLASSIFICATION_H__

#include "ibo/bco.h"
#include "ibo/bco_mc.h"
#include "ibo/buo.h"
#include "ibo/buo.h"
#include "ibo/lbp.h"
#include "ibo/bfo.h"

struct struct_feature_options
{
  bool use_ave = true;
  bool use_lbp = false;
  bool use_dct = false;
  bool use_intensity = false;
};

// choose color space
#define USE_OPPOSITE_COLOR // this uses input image directly
//#define USE_RGB // this uses input image directly
//#define USE_CIELab

#define TRAIN_4_EACH_FEATURE

#include <stdio.h>
#include <string>
#include <vector>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/nonfree/nonfree.hpp>

namespace tc
{
  void setFeature( struct_feature_options feature_options, cv::Mat src_mat, ibo::mc_image mc_src, std::vector<ibo::struct_bfo> &feature, std::vector<ibo::struct_feature_dimension> &feature_dim, int feature_window_size, double feature_window_overlap_ratio, ibo::struct_lbp_parameter lbp_parameter );
}

#endif //__CLASSIFICATION_H__
