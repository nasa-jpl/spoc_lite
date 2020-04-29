#ifndef __IBO_BASIC_CAMERA_OPERATION_MC__
#define __IBO_BASIC_CAMERA_OPERATION_MC__

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>

#include <opencv2/opencv.hpp>

namespace ibo
{

typedef uint8_t pixel_t;

typedef struct mc_image_type
{
    int rows;
    int cols;
    int chans;
    std::vector<pixel_t> data;    /* row-major storage of pixels */
} mc_image;

typedef struct mc_label_image_type
{
    int rows;
    int cols;
    int chans;
    std::vector<int> data;    /* row-major storage of pixels */
} mc_label_image;

typedef struct mc_label_type
{
  int n_label;
  int rows;
  int cols;
  int chans;
  std::vector<pixel_t> data;    /* row-major storage of pixels */
  std::vector<int> label_id; // label_id is 0,1,2,3,... cannot be changed (if you do like 0, 2, 5, ... other process will corrupsed)
  std::vector<int> num_each_label;
  std::vector<int> flag_ignore; // if it is ignore, set as 1. if not, set as -1.
} mc_label;

typedef struct mc_haar_type
{
  std::vector<int> start_x;
  std::vector<int> start_y;
  std::vector<int> end_x;
  std::vector<int> end_y;
  std::vector<int> value; // 1 or -1
} mc_haar;


void mc_convert_to_opposite_color( mc_image &in_out, bool flg_fix_intensity=true );// last one is option
int mc_read_image( mc_image &in, const char *filename );
int mc_read_image( mc_label_image &in, const char *filename );
void mc_image_normalization( mc_image &in_out );
void mc_image_normalization( mc_image &in_out, int window );  // good for images with low dynamic range
void mc_image_normalization( mc_image &in_out, int window, int target_chan );
void mc_image_extract_one_channel( mc_image in, int target_chan, mc_image &out );
void mc_combine_image( std::vector<mc_image> in, mc_image &in_combine );
void mc_save_image( mc_image in, const char *filename );
void mc_apply_dct( mc_image in, int window_size, mc_image &in_dct );
void mc_apply_haar( mc_image in, mc_image &in_haar, std::vector<mc_haar> haar_filter, int window_size );
void mc_make_haar_like_filter( std::vector<mc_haar> &haar_filter, int window_size );
void mc_apply_cnn_filter( mc_image in, mc_image &in_cnn, int window_size );
void mc_export_channels( std::vector<mc_image> in, mc_image &in_ex, int export_channel[], int num_channel );
// color (0,0,0) is ignored as empty label
void mc_load_label( mc_label &mc_label_map, mc_image mc_label_src );
void mc_load_label( mc_label &mc_label_map, mc_label_image mc_label_src );
void mc_assign_label( mc_label mc_label_map, mc_image &mc_label_src ); // based on mc_label_map, assign label-id to each pixel of mc_label_src
void mc_assign_label( mc_label mc_label_map, mc_label_image &mc_label_src ); // based on mc_label_map, assign label-id to each pixel of mc_label_src
int mc_find_label_id( int target_label_rgb[], mc_label mc_label_map );

// works with chans 1 or 3
void mc_copy_image_2_Mat( mc_image in, cv::Mat &mat );
void mc_copy_label_image_2_Mat( mc_label_image in, cv::Mat &mat );
void mc_copy_Mat_2_image( cv::Mat mat, mc_image &out );
void mc_copy_Mat_2_label_image( cv::Mat mat, mc_label_image &out );
void mc_copy_image_2_image( mc_image src, mc_image &out );
 void mc_make_dummy_image( int rows, int cols, int chans, int color[3], mc_image &dst );
 void mc_make_dummy_image( int rows, int cols, int chans, int color[3], mc_label_image &dst );

} // namespace ibo

#endif //#define __IBO_BASIC_CAMERA_OPERATION_MC__
