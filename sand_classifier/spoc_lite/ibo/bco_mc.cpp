#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/nonfree/nonfree.hpp>

#include "bco_mc.h"

using namespace cv;
using namespace std;

namespace ibo
{

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wunused-variable"

  // correct name is "opponent color", not "opposite color"
void mc_convert_to_opposite_color( mc_image &in_out, bool flg_fix_intensity )
{
  int r, b, c;
  int rows = in_out.rows;
  int cols = in_out.cols;
  int chans = in_out.chans;

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
#if 0
  // original
  for (r=0; r < rows; r++){
    for (c=0; c < cols; c++){
      RR = (double)in_out.data[r*cols*chans + c*chans + 0];
      GG = (double)in_out.data[r*cols*chans + c*chans + 1];
      BB = (double)in_out.data[r*cols*chans + c*chans + 2];
      o[0] = (RR-GG) / sqrt(2);
      o[1] = (RR+GG-2*BB) / sqrt(6);
      o[2] = (RR+GG+BB) / sqrt(3);

      // normalize
      o[0] = (o[0] + 255.0/sqrt(2)) / (510.0/sqrt(2)) * 255.0;
      o[1] = (o[1] + 510.0/sqrt(6)) / (1020.0/sqrt(2)) * 255.0;
      o[2] = o[2] / (765.0/sqrt(3)) * 255.0;

      // set intensity stable
      if( flg_fix_intensity )
	o[2] = 200;

      in_out.data[r*cols*chans + c*chans + 0] = (unsigned char)o[0];
      in_out.data[r*cols*chans + c*chans + 1] = (unsigned char)o[1];
      in_out.data[r*cols*chans + c*chans + 2] = (unsigned char)o[2];
    }
  }
#else
  // changed to memory access
  double v_sqrt2 = sqrt(2);
  double v_sqrt3 = sqrt(3);
  double v_sqrt6 = sqrt(6);
  double v_255_v_sqrt2 = 255.0/v_sqrt2;
  double v_510_v_sqrt2_255 = 510.0/v_sqrt2 / 255.0;
  double v_510_v_sqrt6 = 510.0/v_sqrt6;
  double v_1020_v_sqrt2 = 1020.0/v_sqrt2;
  double v_765_v_sqrt3_255 = 765.0/v_sqrt3 / 255.0;

  pixel_t *pixel_t_data;
  for (r=0; r < rows; r++){
    for (c=0; c < cols; c++){
      //pixel_t_data = &in_out.data[r*cols*chans + c*chans];
      RR = (double)in_out.data[r*cols*chans + c*chans + 0];
      GG = (double)in_out.data[r*cols*chans + c*chans + 1];
      BB = (double)in_out.data[r*cols*chans + c*chans + 2];
      // RR = (double)(*pixel_t_data);//in_out.data[r*cols*chans + c*chans + 0];
      // GG = (double)(*pixel_t_data+1);//in_out.data[r*cols*chans + c*chans + 1];
      // BB = (double)(*pixel_t_data+2);//in_out.data[r*cols*chans + c*chans + 2];
      *o = (RR-GG) / v_sqrt2;
      *(o+1) = (RR+GG-2*BB) / v_sqrt6;
      *(o+2) = (RR+GG+BB) / v_sqrt3;

      // normalize
      // *o = (*o + 255.0/v_sqrt2) / (510.0/v_sqrt2) * 255.0;
      // *(o+1) = (*(o+1) + 510.0/v_sqrt6) / (1020.0/v_sqrt2) * 255.0;
      // *(o+2) = *(o+2) / (765.0/v_sqrt3) * 255.0;
      *o = (*o + v_255_v_sqrt2) / v_510_v_sqrt2_255;
      *(o+1) = (*(o+1) + v_510_v_sqrt6) / v_1020_v_sqrt2 * 255.0;
      *(o+2) = *(o+2) / v_765_v_sqrt3_255;

      // set intensity stable
      if( flg_fix_intensity )
	*(o+2) = 200;

      // in_out.data[r*cols*chans + c*chans + 0] = (unsigned char)o[0];
      // in_out.data[r*cols*chans + c*chans + 1] = (unsigned char)o[1];
      // in_out.data[r*cols*chans + c*chans + 2] = (unsigned char)o[2];
      in_out.data[r*cols*chans + c*chans + 0] = (unsigned char)*o;
      in_out.data[r*cols*chans + c*chans + 1] = (unsigned char)*(o+1);
      in_out.data[r*cols*chans + c*chans + 2] = (unsigned char)*(o+2);
      // *pixel_t_data = (unsigned char)*o;
      // *(pixel_t_data+1) = (unsigned char)*(o+1);
      // *(pixel_t_data+2) = (unsigned char)*(o+1);
    }
  }
#endif
}


#if 0
// original
void mc_read_image( mc_image &in, const char *filename )
{
  FILE *file;
  int r, b, c;
  int rows, cols, chans, max;
  int ch_int;
  char type='0';

  if (!strstr(filename, ".pgm") &&
      !strstr(filename, ".PGM") &&
      !strstr(filename, ".ppm") &&
      !strstr(filename, ".PPM"))
    {
      cout << "mc_read_image: Don't recognize the file suffix.\r\n";
      return;
    }
  if ((file = fopen(filename, "rb")) == NULL)
    {
      cout << "cannot open file" << filename << endl;
      return;
    }
  
  /* read header */
  char imgtype = (char) getc(file);
  char* bandcode = new char [2];
  bandcode[0] = (char) getc(file);
  bandcode[1] = '\0';
  type = atoi(bandcode);
  delete [] bandcode;
  char eol = (char) getc(file);

  if (eol != '\n' && eol != ' ')
    {
      cout << "read_image: bad header format.\r\n";
      return;
    }
  
  switch (imgtype)
    {
    case 'P':  /* pgm image */
      switch (type)
        {
        case 1:
        case 4:
	  cout << "mc_read_image: Don't support bitmaps\r\n";
	  return;
	  break;
        case 2:
        case 5:
	  chans = 1;
	  break;
        case 3:
        case 6:
	  chans = 3;
	  break;
        default:
	  cout << "read_image: bad header format.\r\n";
	  return;
	  break;
        }
      break;
    case 'H': /* custom format */
      printf("imgtype %c\n", imgtype);
      chans = type;
      type = 0;
      printf("read_image: custom format, %d channels.\r\n", chans);
      break;
    default:
      cout << "read_image: bad header format.\r\n";
      return;
    }

    /* read comment block */
    while (getc(file) == '#') while (getc(file) != '\n');

    /* read columns and rows, and max value */
    fseek(file, -1, SEEK_CUR);
    fscanf(file,"%d %d %d", &cols, &rows, &max);
    /* Consume the final newline */
    fgetc(file);

    in.rows = rows;
    in.cols = cols;
    in.chans = chans;

    for (r=0; r < rows; r++)
      {
        for (c=0; c < cols; c++)
	  {
            for (b=0; b < chans; b++)
	      {
                switch (type)
		  {
		  case 2:
		  case 3:
                    /* ASCII */
                    if (fscanf(file,"%d", &ch_int) != 1)
		      {
                        cout << "mc_read_image: Syntax error.\r\n";
                        return;
		      }
		    in.data.push_back( ch_int );
                    break;
		  default:
                    /* binary */
		    in.data.push_back( (pixel_t)getc(file) );
                    break;
		  }
	      }
	  }
      }
    fclose(file);
    return;
}
#else
int mc_read_image( mc_image &in, const char *filename )
{
  FILE *file;
  int r, b, c;
  int rows, cols, chans, max;
  int ch_int;
  int type=0;

  if (!strstr(filename, ".pgm") &&
      !strstr(filename, ".PGM") &&
      !strstr(filename, ".ppm") &&
      !strstr(filename, ".PPM"))
    {
      cout << "mc_read_image: Don't recognize the file suffix. Use pgm, ppm.\r\n";
      return 0;
    }
  if ((file = fopen(filename, "rb")) == NULL)
    {
      cout << "cannot open file" << filename << endl;
      return 0;
    }
  
  /* read header */
  char imgtype = (char) getc(file);
  char* bandcode = new char [2];
  bandcode[0] = (char) getc(file);
  bandcode[1] = '\0';
  type = atoi(bandcode);
  delete [] bandcode;

  //printf("bandcode[0] '%c'\n", bandcode[0]);
  printf("imgtype %c type %d\n", imgtype, type);
  if( type == 0 ){// modified version by yumi
    rewind( file );
    char buf[4];
    fscanf( file, "%s %d", buf, &type );
    //printf("0:%c 1:%c 2:%c 3:%c\n", buf[0], buf[1], buf[2], buf[3]);
    imgtype = buf[0];//(char) getc(file);
    //printf("yumi %s %d\n", buf, type);
  }
  //printf("imgtype %c type %d\n", imgtype, type);
  //else{ // original version
  char eol = (char) getc(file);
  if (eol != '\n' && eol != ' ')
    {
      printf("original %d\n", type);
      cout << "read_image: bad header format.\r\n";
      return 0;
    }
  //printf("original %d\n", type);
  //}
  //printf("imgtype %c type %d\n", imgtype, type);
  //getchar();

  switch (imgtype)
    {
    case 'P':  /* pgm image */
      switch (type)
        {
        case 1:
        case 4:
	  cout << "mc_read_image: Don't support bitmaps P\r\n";
	  return 0;
	  break;
        case 2:
        case 5:
	  chans = 1;
	  break;
        case 3:
        case 6:
	  chans = 3;
	  break;
        default:
	  cout << "read_image: bad header format. P\r\n";
	  return 0;
	  break;
        }
      break;
    case 'H': /* custom format */
      printf("imgtype %c\n", imgtype);
      chans = type;
      type = 0;
      printf("read_image: custom format, %d channels. H\r\n", chans);
      break;
    default:
      cout << "read_image: bad header format. D\r\n";
      return 0;
    }

    /* read comment block */
    while (getc(file) == '#') while (getc(file) != '\n');

    /* read columns and rows, and max value */
    fseek(file, -1, SEEK_CUR);
    fscanf(file,"%d %d %d", &cols, &rows, &max);
    /* Consume the final newline */
    fgetc(file);

    in.rows = rows;
    in.cols = cols;
    in.chans = chans;

    for (r=0; r < rows; r++)
      {
        for (c=0; c < cols; c++)
	  {
            for (b=0; b < chans; b++)
	      {
                switch (type)
		  {
		  case 2:
		  case 3:
                    /* ASCII */
                    if (fscanf(file,"%d", &ch_int) != 1)
		      {
                        cout << "mc_read_image: Syntax error.\r\n";
                        return 0;
		      }
		    in.data.push_back( ch_int );
                    break;
		  default:
                    /* binary */
		    in.data.push_back( (pixel_t)getc(file) );
                    break;
		  }
	      }
	  }
      }
    fclose(file);
    return 1;
}

int mc_read_image( mc_label_image &in, const char *filename )
{
  FILE *file;
  int r, b, c;
  int rows, cols, chans, max;
  int ch_int;
  int type=0;

  if (!strstr(filename, ".pgm") &&
      !strstr(filename, ".PGM") &&
      !strstr(filename, ".ppm") &&
      !strstr(filename, ".PPM"))
    {
      cout << "mc_read_image: Don't recognize the file suffix.\r\n";
      return 0;
    }
  if ((file = fopen(filename, "rb")) == NULL)
    {
      cout << "cannot open file" << filename << endl;
      return 0;
    }
  
  /* read header */
  char imgtype = (char) getc(file);
  char* bandcode = new char [2];
  bandcode[0] = (char) getc(file);
  bandcode[1] = '\0';
  type = atoi(bandcode);
  delete [] bandcode;

  //printf("bandcode[0] '%c'\n", bandcode[0]);
  printf("imgtype %c type %d\n", imgtype, type);
  if( type == 0 ){// modified version by yumi
    rewind( file );
    char buf[4];
    fscanf( file, "%s %d", buf, &type );
    //printf("0:%c 1:%c 2:%c 3:%c\n", buf[0], buf[1], buf[2], buf[3]);
    imgtype = buf[0];//(char) getc(file);
    //printf("yumi %s %d\n", buf, type);
  }
  //printf("imgtype %c type %d\n", imgtype, type);
  //else{ // original version
  char eol = (char) getc(file);
  if (eol != '\n' && eol != ' ')
    {
      printf("original %d\n", type);
      cout << "read_image: bad header format.\r\n";
      return 0;
    }
  //printf("original %d\n", type);
  //}
  //printf("imgtype %c type %d\n", imgtype, type);
  //getchar();

  switch (imgtype)
    {
    case 'P':  /* pgm image */
      switch (type)
        {
        case 1:
        case 4:
	  cout << "mc_read_image: Don't support bitmaps P\r\n";
	  return 0;
	  break;
        case 2:
        case 5:
	  chans = 1;
	  break;
        case 3:
        case 6:
	  chans = 3;
	  break;
        default:
	  cout << "read_image: bad header format. P\r\n";
	  return 0;
	  break;
        }
      break;
    case 'H': /* custom format */
      printf("imgtype %c\n", imgtype);
      chans = type;
      type = 0;
      printf("read_image: custom format, %d channels. H\r\n", chans);
      break;
    default:
      cout << "read_image: bad header format. D\r\n";
      return 0;
    }

    /* read comment block */
    while (getc(file) == '#') while (getc(file) != '\n');

    /* read columns and rows, and max value */
    fseek(file, -1, SEEK_CUR);
    fscanf(file,"%d %d %d", &cols, &rows, &max);
    /* Consume the final newline */
    fgetc(file);

    in.rows = rows;
    in.cols = cols;
    in.chans = chans;

    for (r=0; r < rows; r++)
      {
        for (c=0; c < cols; c++)
	  {
            for (b=0; b < chans; b++)
	      {
                switch (type)
		  {
		  case 2:
		  case 3:
                    /* ASCII */
                    if (fscanf(file,"%d", &ch_int) != 1)
		      {
                        cout << "mc_read_image: Syntax error.\r\n";
                        return 0;
		      }
		    in.data.push_back( ch_int );
                    break;
		  default:
                    /* binary */
		    in.data.push_back( (pixel_t)getc(file) );
                    break;
		  }
	      }
	  }
      }
    fclose(file);
    return 1;
}
#endif

void mc_image_normalization( mc_image &in_out )
{
  int r, b, c;
  int rows = in_out.rows;
  int cols = in_out.cols;
  int chans = in_out.chans;

  double average[3] = { 0 };
  double sd[3] = { 0 };
  double intensity[3];
  int rr, cc;

  for (r=0; r < rows; r++){
    for (c=0; c < cols; c++){
      for (b=0; b < chans; b++)
	average[b] += (unsigned char)in_out.data[r*cols*chans + c*chans + b];
    }
  }
  for (b=0; b < chans; b++)
    average[b] /= (double)(rows*cols);

#if 0
  //debug
  double total_average = 0;
  for (b=0; b < chans; b++)
    total_average += average[b];
  for (b=0; b < chans; b++)
    average[b] = total_average / (double)chans;
#endif

  for (r=0; r < rows; r++){
    for (c=0; c < cols; c++){
      for (b=0; b < chans; b++)
	sd[b] += (average[b]-(double)in_out.data[r*cols*chans + c*chans + b])*(average[b]-(double)in_out.data[r*cols*chans + c*chans + b]);
    }
  }
  for (b=0; b < chans; b++)
    sd[b] = sqrt( sd[b] /  (double)(rows*cols) );
  

  for (r=0; r < rows; r++){
    for (c=0; c < cols; c++){
      for (b=0; b < chans; b++){
	intensity[b] = (unsigned char)in_out.data[r*cols*chans + c*chans + b];

#if 1
	intensity[b] = (intensity[b] - (int)average[b]) + 127;
#else
	intensity[b] = (intensity[b] - (int)average[b]) / sd[b] + 127;
#endif
	intensity[b] = min( (double)intensity[b], 255.0 );
	intensity[b] = max( (double)intensity[b], 0.0 );

	in_out.data[r*cols*chans + c*chans + b] = (int)intensity[b];
      }
    }
  }
  
}

  // good for images with low dynamic range
void mc_image_normalization( mc_image &in_out, int window )
{
  int r, b, c, rr, cc;
  int wr, wc;
  int rows = in_out.rows;
  int cols = in_out.cols;
  int chans = in_out.chans;

  double average[3] = { 0 };
  double intensity[3];

  mc_image ave_mc;
  ave_mc.rows  = rows;
  ave_mc.cols  = cols;
  ave_mc.chans = chans;
  for (r=0; r < rows; r++){
    for (c=0; c < cols; c++){
      // window
      int count = 0;
      memset( average, 0, sizeof(double)*3 );
      for(wr=-window; wr<=window; wr++){
	rr = r + wr;
	if( rr < 0 || rr >= rows ) continue;

	for(wc=-window; wc<=window; wc++){
	  cc = c + wc;
	  if( cc < 0 || cc >= cols ) continue;
	  for (b=0; b < chans; b++)
	    average[b] += (unsigned char)in_out.data[rr*cols*chans + cc*chans + b];
	  count ++;
	}
      }
      for (b=0; b < chans; b++){
	average[b] /= (double)count;
	ave_mc.data.push_back( (pixel_t)average[b] );
      }
    }
  }
  
  for (r=0; r < rows; r++){
    for (c=0; c < cols; c++){
      for (b=0; b < chans; b++){
	intensity[b] = (unsigned char)in_out.data[r*cols*chans + c*chans + b];
	average[b] = (unsigned char)ave_mc.data[r*cols*chans + c*chans + b];

	intensity[b] = intensity[b] - (int)average[b] + 127;
	intensity[b] = min( (double)intensity[b], 255.0 );
	intensity[b] = max( (double)intensity[b], 0.0 );

	in_out.data[r*cols*chans + c*chans + b] = (int)intensity[b];
      }
    }
  }
  
}

  // good for images with low dynamic range
  // normalize values at a specific channel
  void mc_image_normalization( mc_image &in_out, int window, int target_chan )
{
  int r, b, c, rr, cc;
  int wr, wc;
  int rows = in_out.rows;
  int cols = in_out.cols;
  int chans = in_out.chans;

  if( chans <= target_chan ){
    printf("mc_image_normalization: target_chan is too large\n");
    return;
  }

  double average[3] = { 0 };
  double intensity[3];

  mc_image ave_mc;
  ave_mc.rows  = rows;
  ave_mc.cols  = cols;
  ave_mc.chans = chans;
  for (r=0; r < rows; r++){
    for (c=0; c < cols; c++){
      // window
      int count = 0;
      memset( average, 0, sizeof(double)*3 );
      for(wr=-window; wr<=window; wr++){
	rr = r + wr;
	if( rr < 0 || rr >= rows ) continue;

	for(wc=-window; wc<=window; wc++){
	  cc = c + wc;
	  if( cc < 0 || cc >= cols ) continue;
	  for (b=0; b < chans; b++)
	    average[b] += (unsigned char)in_out.data[rr*cols*chans + cc*chans + b];
	  count ++;
	}
      }
      for (b=0; b < chans; b++){
	average[b] /= (double)count;
	ave_mc.data.push_back( (pixel_t)average[b] );
      }
    }
  }
  
  for (r=0; r < rows; r++){
    for (c=0; c < cols; c++){
      for (b=0; b < chans; b++){
	intensity[b] = (unsigned char)in_out.data[r*cols*chans + c*chans + b];
	average[b] = (unsigned char)ave_mc.data[r*cols*chans + c*chans + b];

	if( b == target_chan ){
	  intensity[b] = intensity[b] - (int)average[b] + 127;
	  intensity[b] = min( (double)intensity[b], 255.0 );
	  intensity[b] = max( (double)intensity[b], 0.0 );

	  //debug
	  //intensity[b] = 200;
	}

	in_out.data[r*cols*chans + c*chans + b] = (int)intensity[b];
      }
    }
  }
  
}

  void mc_image_extract_one_channel( mc_image in, int target_chan, mc_image &out )
{
  int r, b, c;
  int rows = in.rows;
  int cols = in.cols;
  int chans = in.chans;
  int intensity;

  out.rows  = rows;
  out.cols  = cols;
  out.chans = 1;

  b = target_chan;
  for (r=0; r < rows; r++){
    for (c=0; c < cols; c++){
      intensity = (unsigned char)in.data[r*cols*chans + c*chans + b];
      out.data.push_back( (pixel_t)intensity );
    }
  }
  
}

void mc_combine_image( vector<mc_image> in, mc_image &in_combine )
{
  int num_in = in.size();
  
  // check image size
  int rows = in[0].rows;
  int cols = in[0].cols;
  for(int i=0; i<num_in; i++){
    if( (rows != in[i].rows) || (cols != in[i].cols) ){
      cout << "image size is different\n" << endl;
      return;
    }
  }

  int chans = 0;

  // initialize
  in_combine.chans = 0;
  for(int i=0; i<num_in; i++)
      in_combine.chans += in[i].chans;
  in_combine.rows = in[0].rows;
  in_combine.cols = in[0].cols;
  in_combine.data.resize( (in_combine.rows * in_combine.cols * in_combine.chans) );  

  int tmp_chans = 0;
  for(int i=0; i<num_in; i++){
    chans = in[i].chans;
    
    for (int r=0; r < rows; r++){
      for (int c=0; c < cols; c++){
	for (int b=0; b < chans; b++){
	  in_combine.data[r*in_combine.cols*in_combine.chans + c*in_combine.chans + (tmp_chans+b)] = in[i].data[r*cols*chans + c*chans + b];
	}
      }
    }
    tmp_chans += chans;
  }
}

void mc_export_channels( vector<mc_image> in, mc_image &in_ex, int export_channel[], int num_channel )
{
  int num_in = in.size();
  //int num_channel = sizeof(export_channel) / sizeof(int);

  int i = 0;
  int rows = in[i].rows;
  int cols = in[i].cols;
  int chans = in[i].chans;

  in_ex.rows = rows;
  in_ex.cols = cols;
  in_ex.chans = num_channel;
  in_ex.data.resize( (in_ex.rows * in_ex.cols * in_ex.chans) );  
  int chan_count = 0;
  for(int b=0; b<chans; b++){
    if( export_channel[chan_count] >= chans ){
      cout << "export_channel is too big" << endl;
      return;
    }
    
    if( b == export_channel[chan_count] ){
      for (int r=0; r < in_ex.rows; r++)
	for (int c=0; c < in_ex.cols; c++)
	  in_ex.data[r*in_ex.cols*in_ex.chans + c*in_ex.chans + chan_count] = in[i].data[r*cols*chans + c*chans + b];
      chan_count ++;
    }
    if( chan_count >= num_channel ) break;
  }

}

#if 0
// original
void mc_save_image( mc_image in, const char *filename )
{
  int r, b, c;
  FILE *file;
  if ((file = fopen(filename, "wb")) == NULL)
    {
      cout << "mc_write_image: Can't open file for writing.\r\n";
      return;
    }
  if (!strstr(filename, ".pgm") &&
      !strstr(filename, ".PGM") &&
      !strstr(filename, ".PPM") &&
      !strstr(filename, ".ppm"))
    {
      cout << "mc_write_image: Don't recognize image type.";
      fclose(file);
      return;
    }
  
  switch(in.chans)
    {
    case 1:
      fprintf(file, "P5\n");
      fprintf(file, "%d %d\n", in.cols, in.rows);
      fprintf(file, "255\n");
      break;
    case 3:
      fprintf(file, "P6\n");
      fprintf(file, "%d %d\n", in.cols, in.rows);
      fprintf(file, "255\n");
      break;
    default:
      cout << "mc_write_image: writing special pgm format";
      /* our convention for channels > 3 */
      fprintf(file, "H%i\n",in.chans);// 16 shinsu
      fprintf(file, "%d %d\n", in.cols, in.rows);
      fprintf(file, "255\n");
      break;
    }
  
  for (r=0; r < in.rows; r++)
    for (c=0; c < in.cols; c++)
      for (b=0; b < in.chans; b++)
	//putc(pixel_to_uchar(mc_get(img, r, c, b)), file);
	putc((unsigned char)in.data[r*in.cols*in.chans + c*in.chans + b], file);

  fclose(file);
  return;
}
#else
void mc_save_image( mc_image in, const char *filename )
{
  int r, b, c;
  FILE *file;
  if ((file = fopen(filename, "wb")) == NULL)
    {
      cout << "mc_write_image: Can't open file for writing.\r\n";
      return;
    }
  if (!strstr(filename, ".pgm") &&
      !strstr(filename, ".PGM") &&
      !strstr(filename, ".PPM") &&
      !strstr(filename, ".ppm"))
    {
      cout << "mc_write_image: Don't recognize image type.";
      fclose(file);
      return;
    }
  
  switch(in.chans)
    {
    case 1:
#if 0 // original
      fprintf(file, "P5\n");
      fprintf(file, "%d %d\n", in.cols, in.rows);
      fprintf(file, "255\n");
#else
      fprintf(file, "P5 ");
      fprintf(file, "%d %d ", in.cols, in.rows);
      fprintf(file, "255\n");
#endif
      break;
    case 3:
      fprintf(file, "P6\n");
      fprintf(file, "%d %d\n", in.cols, in.rows);
      fprintf(file, "255\n");
      break;
    default:
      cout << "mc_write_image: writing special pgm format";
      /* our convention for channels > 3 */
      fprintf(file, "H %d\n",in.chans);
      fprintf(file, "%d %d\n", in.cols, in.rows);
      fprintf(file, "255\n");
      break;
    }
  
  for (r=0; r < in.rows; r++)
    for (c=0; c < in.cols; c++)
      for (b=0; b < in.chans; b++)
	//putc(pixel_to_uchar(mc_get(img, r, c, b)), file);
	putc((unsigned char)in.data[r*in.cols*in.chans + c*in.chans + b], file);

  fclose(file);
  return;
}
#endif


void mc_apply_dct( mc_image in, int window_size, mc_image &in_dct )
{
  int r, b, c;
  int rows = in.rows;
  int cols = in.cols;
  int chans = in.chans;

  int b1_num = 16;
  int b2_num = 1;
  in_dct.rows = rows;
  in_dct.cols = cols;
  in_dct.chans = b1_num * b2_num;//9;// maximum due to the current implementation of mc_read_image

  in_dct.data.resize( (in_dct.rows * in_dct.cols * in_dct.chans) );

  double intensity;
  int rr, cc;

#if 1
  // just for display
  Mat in_mat = Mat::zeros( rows, cols, CV_8UC1 );
  for (r=0; r < rows; r++){
    for (c=0; c < cols; c++){
      intensity = 0;
      for (b=0; b < chans; b++)
	intensity += (unsigned char)in.data[r*cols*chans + c*chans + b];
      intensity /= (double)chans;
      in_mat.at<uchar>(r,c) = (uchar)intensity;
    }
  }
  imshow( "in", in_mat );
  waitKey( 5 );
#endif

  vector<Mat> vec_dct_mat;
  for (r=0; r < rows; r++){
    if( r<window_size/2 || r>(rows-window_size/2) ) continue;

    for (c=0; c < cols; c++){
      if( c<window_size/2 || c>(cols-window_size/2) ) continue;

      // for opencv implementation reason, the size of dct_mat is converted into even number
      Mat dct_mat = Mat::zeros( window_size+1, window_size+1, CV_64FC1 );
      //Mat dct_mat_d = Mat::zeros( window_size+1, window_size+1, CV_8UC1 );
      for(int r_w=-window_size/2; r_w<=(window_size/2+1); r_w++){
	for(int c_w=-window_size/2; c_w<=(window_size/2+1); c_w++){
	  rr = r + r_w;
	  cc = c + c_w;
	  intensity = 0;
	  for (b=0; b < chans; b++)
	    intensity += (unsigned char)in.data[rr*cols*chans + cc*chans + b];
	  intensity /= (double)chans;
	  dct_mat.at<double>((r_w+window_size/2),(c_w+window_size/2)) = intensity;
	  //dct_mat_d.at<uchar>((r_w+window_size/2),(c_w+window_size/2)) = (uchar)intensity;
	}
      }

      //dct_mat.convertTo( dct_mat, CV_8UC1 );
      dct( dct_mat, dct_mat );
      vec_dct_mat.push_back( dct_mat.clone() );

#if 0 // for visualization
      Mat idct_mat;
      dct( dct_mat, idct_mat, DCT_INVERSE );
      dct_mat.convertTo( idct_mat, CV_8UC1 );// normalization is done automatically 
      imshow( "test", idct_mat );
      waitKey( 0 );
#endif
    }
  }

  int counter = 0;
  double *max_value = new double [in_dct.chans];
  memset( max_value, 0, sizeof(double)*in_dct.chans );
  for(int i=0; i<vec_dct_mat.size(); i++){
    //Mat dct_mat = vec_dct_mat[i].clone();
    //imshow( "debug", dct_mat );
    //waitKey( 50 );
      /*
    for(int b1=0; b1<b1_num; b1++)
      for(int b2=0; b2<b2_num; b2++)
	max_value[b1*b2_num+b2] = max( fabs(vec_dct_mat[i].at<double>(b1,b2)), max_value[b1*b2_num+b2] );
      */
    for(int b1=0; b1<b1_num; b1++)
      max_value[b1] = max( fabs(vec_dct_mat[i].at<double>(b1,b1)), max_value[b1] );
  }

  for (r=0; r < rows; r++){
    if( r<window_size/2 || r>(rows-window_size/2) ) continue;
    
    for (c=0; c < cols; c++){
      if( c<window_size/2 || c>(cols-window_size/2) ) continue;
      
      Mat dct_mat = vec_dct_mat[counter].clone();
      counter ++;
#if 0
      // normalize with maximum component for each dct_mat
      dct_mat.convertTo( dct_mat, CV_8UC1 );// normalization is done automatically 
      for(int b1=0; b1<b1_num; b1++)
	for(int b2=0; b2<b2_num; b2++)
	  in_dct.data[r*cols*3 + c*3 + (b1+b2)] = abs(dct_mat.at<uchar>(b1,b2));// assumption that convertTo(CV_8UC1) is done
#else
      // normalize at each component
      /*
      for(int b1=0; b1<b1_num; b1++){
	for(int b2=0; b2<b2_num; b2++){
 	  //in_dct.data[r*in_dct.cols*in_dct.chans + c*in_dct.chans + (b1*b2_num+b2)] = abs(dct_mat.at<double>(b1,b2)/max_idct) * 255;
	  in_dct.data[r*in_dct.cols*in_dct.chans + c*in_dct.chans + (b1*b2_num+b2)] = abs(dct_mat.at<double>(b1,b2)) / max_value[b1*b2_num+b2] * 255;
	}
      }
      */
      for(int b1=0; b1<b1_num; b1++)
        in_dct.data[r*in_dct.cols*in_dct.chans + c*in_dct.chans + (b1)] = abs(dct_mat.at<double>(b1,b1)) / max_value[b1] * 255;
#endif
      
    }
  }
  if( max_value ) delete [] max_value;
  
}

void mc_apply_haar( mc_image in, mc_image &in_haar, vector<mc_haar> haar_filter, int window_size )
{
  int r, b, c;
  int rows = in.rows;
  int cols = in.cols;
  int chans = in.chans;
  int num_haar_like = haar_filter.size();
  
  in_haar.rows = rows;
  in_haar.cols = cols;
  in_haar.chans = min(num_haar_like, 9);// maximum due to the current implementation of mc_read_image
  in_haar.data.resize( (in_haar.rows * in_haar.cols * in_haar.chans) );

  double intensity;
  int rr, cc;

#if 0
  // just for display
  Mat in_mat = Mat::zeros( rows, cols, CV_8UC1 );
  for (r=0; r < rows; r++){
    for (c=0; c < cols; c++){
      intensity = 0;
      for (b=0; b < chans; b++)
	intensity += (unsigned char)in.data[r*cols*chans + c*chans + b];
      intensity /= (double)chans;
      in_mat.at<uchar>(r,c) = (uchar)intensity;

      if( (uchar)intensity > 0 ){
	printf("%d\t", (uchar)intensity );
	getchar();
      }
    }
  }
  imshow( "in", in_mat );
  waitKey( 5 );
#endif

  // create integral image
  Mat tmp_out(rows,cols,CV_8UC1);
  Mat integral_mat = Mat::zeros( rows, cols, CV_64FC1 );
  for (r=0; r < rows; r++){
    for (c=0; c < cols; c++){
      intensity = 0;
      for (b=0; b < chans; b++)
	intensity += (unsigned char)in.data[r*cols*chans + c*chans + b];
      intensity /= (double)chans;

      if( r==0 && c>0 )
	integral_mat.at<double>(r,c) = integral_mat.at<double>(r,c-1) + intensity;
      else if( c==0 && r>0 )
	integral_mat.at<double>(r,c) = integral_mat.at<double>(r-1,c) + intensity;
      else if( c>0 && r>0 )
	integral_mat.at<double>(r,c) = integral_mat.at<double>(r-1,c) + integral_mat.at<double>(r,c-1) + intensity - integral_mat.at<double>(r-1,c-1);
      else
	integral_mat.at<double>(r,c) = intensity;
      
      tmp_out.at<uchar>(r,c) = (uchar)intensity;
      //if( (uchar)intensity > 0 ){
      //printf("%lf %lf\n", intensity, integral_mat.at<double>(r,c));
        //cout << chans << " " << (uchar)intensity << " " << (uchar)integral_mat.at<double>(r,c) << " = " << (uchar)intensity << " + ..." << endl; 
        //getchar();
      //}
    }
  }
  imshow( "tmp", tmp_out );
  waitKey( 5 );

#if 0
  // checking integral image
  for (r=1; r < rows; r++){
    for (c=1; c < cols; c++){
      int sum = 0;
      b = 0;
      in_haar.data[r*in_haar.cols*in_haar.chans + c*in_haar.chans + b]  = integral_mat.at<double>(r,c) - integral_mat.at<double>(r-1,c) - integral_mat.at<double>(r,c-1) + integral_mat.at<double>(r-1,c-1);
    }
  }
#endif
  
#if 1
  int num_haar_box;
  int value, sum, area;
  for(b=0; b<num_haar_like; b++){
    num_haar_box = haar_filter[b].value.size();

    double average = 0.0;
    double max_value = 0.0;
    Mat tmp_haar = Mat::zeros( rows, cols, CV_64FC1 );
    for (r=0; r < rows; r++){
      if( r<window_size/2 || r>=(rows-window_size/2-1) ) continue;
      
      for (c=0; c < cols; c++){
	if( c<window_size/2 || c>=(cols-window_size/2-1) ) continue;
	
	rr = r - window_size/2;
	cc = c - window_size/2;
	sum = 0;
	area = 0;
	for(int bb=0; bb<num_haar_box; bb++){
	  sum += (double)haar_filter[b].value[bb] * (
             integral_mat.at<double>((rr+haar_filter[b].start_x[bb]),(cc+haar_filter[b].start_y[bb])) // +1
	       + integral_mat.at<double>((rr+haar_filter[b].end_x[bb]),(cc+haar_filter[b].end_y[bb])) // +2
	       - integral_mat.at<double>((rr+haar_filter[b].start_x[bb]),(cc+haar_filter[b].end_y[bb])) // -3
	       - integral_mat.at<double>((rr+haar_filter[b].end_x[bb]),(cc+haar_filter[b].start_y[bb]))); // -4
	  area += haar_filter[b].value[bb] * (haar_filter[b].end_x[bb]-haar_filter[b].start_x[bb]) * (haar_filter[b].end_y[bb]-haar_filter[b].start_y[bb]);
        }

	//if( area == 0 )
	//tmp_haar.at<double>(r,c) = (double)abs(sum);
	//else
	//tmp_haar.at<double>(r,c) = (double)abs(sum) / (double)abs(area);
	tmp_haar.at<double>(r,c) = (double)abs(sum);// / (double)(window_size*window_size);
	
	average += abs(sum);
	max_value = max(max_value, fabs(sum));

	//if(abs(sum)==31200988)
	//printf( "%d: %d %d %d\n", abs(sum), num_haar_box, r, c );
	//else if( abs(sum)<31200988 && abs(sum)>205536 )//&& r<450 && c<450 )
	    //printf( "%d: %d %d %d\n", abs(sum), num_haar_box, r, c );

#if 0
	if( area == 0 )
	  in_haar.data[r*in_haar.cols*in_haar.chans + c*in_haar.chans + b] = abs(sum);
	else
	  in_haar.data[r*in_haar.cols*in_haar.chans + c*in_haar.chans + b] = abs(sum) / abs(area);

	in_haar.data[r*in_haar.cols*in_haar.chans + c*in_haar.chans + b] = (uchar)min( 255.0, (double)in_haar.data[r*in_haar.cols*in_haar.chans + c*in_haar.chans + b] );
	in_haar.data[r*in_haar.cols*in_haar.chans + c*in_haar.chans + b] = (uchar)max( 0.0, (double)in_haar.data[r*in_haar.cols*in_haar.chans + c*in_haar.chans + b] );
	//cout << in_haar.data[r*in_haar.cols*in_haar.chans + c*in_haar.chans + b] << endl;
#endif
      }
    }

    //printf("average %lf max %lf\n", average/(double)(rows*cols), max_value);
#if 1
    double mat_min, mat_max;
    cv::minMaxLoc( tmp_haar, &mat_min, &mat_max);
    //printf("%lf %lf\n", mat_min, mat_max);
    tmp_haar = (tmp_haar-mat_min) / (mat_max-mat_min) * 255.0;
    tmp_haar.convertTo( tmp_haar, CV_8UC1 );
    for (r=0; r < rows; r++)
      for (c=0; c < cols; c++)
      in_haar.data[r*in_haar.cols*in_haar.chans + c*in_haar.chans + b] = tmp_haar.at<uchar>(r,c);
#endif
  }
#endif

#if 0
  // just for display
  Mat tmp_mat = Mat::zeros( rows, cols, CV_8UC1 );
  for (b=0; b < in_haar.chans; b++){
    for (r=0; r < rows; r++){
      for (c=0; c < cols; c++){
	intensity = (unsigned char)in_haar.data[r*in_haar.cols*in_haar.chans + c*in_haar.chans + b];
	tmp_mat.at<uchar>(r,c) = (uchar)intensity;
      }
    }
    cout << b << endl;
    imshow( "in", tmp_mat );
    waitKey( 0 );
  }
#endif


}

void mc_make_haar_like_filter( vector<mc_haar> &haar_filter, int window_size )
{
  int divnum_x, divnum_y;
  int value;

#if 1
  // no filter (to get original image)
  for( divnum_x=1; divnum_x<=1; divnum_x++){ 
    mc_haar tmp_filter;
    for(int j=0; j<divnum_x; j++){
      value = (int)pow( -1.0, (double)j );
      tmp_filter.start_x.push_back( j*window_size/divnum_x ); tmp_filter.start_y.push_back( 0 );
      tmp_filter.end_x.push_back( (j+1)*window_size/divnum_x ); tmp_filter.end_y.push_back( window_size );
      tmp_filter.value.push_back( value );
    }
    haar_filter.push_back( tmp_filter );
  }
#endif

#if 0 //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  // window size is fixed and divinding into small pieces inside the windows
#if 1
  // x direction (0 degree)
  divnum_y = 1;
  for( divnum_x=2; divnum_x<=4; divnum_x++){ 
    mc_haar tmp_filter;
    for(int j=0; j<divnum_x; j++){
      value = (int)pow( -1.0, (double)j );
      tmp_filter.start_x.push_back( j*window_size/divnum_x ); tmp_filter.start_y.push_back( 0 );
      tmp_filter.end_x.push_back( (j+1)*window_size/divnum_x ); tmp_filter.end_y.push_back( window_size );
      tmp_filter.value.push_back( value );
    }
    haar_filter.push_back( tmp_filter );
  }
#endif

#if 1
  // x-y direction (45 degree)
  for( divnum_x=2; divnum_x<=4; divnum_x++){ 
  //for( divnum_x=2; divnum_x<=9; divnum_x++){ 
    divnum_y = divnum_x;
    //    for( divnum_y=2; divnum_y<=4; divnum_y++){ 
      mc_haar tmp_filter;
      for(int i=0; i<divnum_y; i++){
        for(int j=0; j<divnum_x; j++){
          value = (int)pow( -1.0, (double)(i+j) );
	  tmp_filter.start_x.push_back( j*window_size/divnum_x ); tmp_filter.start_y.push_back( i*window_size/divnum_y );
	  tmp_filter.end_x.push_back( (j+1)*window_size/divnum_x ); tmp_filter.end_y.push_back( (i+1)*window_size/divnum_y );
	  tmp_filter.value.push_back( value );
        }
      }
      haar_filter.push_back( tmp_filter );
      //}
  }
#endif

#if 1
  // y direction (90 degree)
  divnum_x = 1;
  for( divnum_y=2; divnum_y<=4; divnum_y++){ 
    mc_haar tmp_filter;
    for(int i=0; i<divnum_y; i++){
      value = (int)pow( -1.0, (double)i );
      tmp_filter.start_x.push_back( 0 ); tmp_filter.start_y.push_back( i*window_size/divnum_y );
      tmp_filter.end_x.push_back( window_size ); tmp_filter.end_y.push_back( (i+1)*window_size/divnum_y );
      tmp_filter.value.push_back( value );
    }
    haar_filter.push_back( tmp_filter );
  }
#endif
#else //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  // window size changes
  int x_rf, y_rf;
#if 1
  // x direction (0 degree)
  divnum_y = 1;
  for( divnum_x=2; divnum_x<=4; divnum_x++){ 
    mc_haar tmp_filter;
    x_rf = window_size/2 - window_size/divnum_x;
    y_rf = 0;
    for(int j=0; j<2; j++){
      value = (int)pow( -1.0, (double)j );
      tmp_filter.start_x.push_back( x_rf+j*window_size/divnum_x ); tmp_filter.start_y.push_back( y_rf+0 );
      tmp_filter.end_x.push_back( x_rf+(j+1)*window_size/divnum_x ); tmp_filter.end_y.push_back( y_rf+window_size );
      tmp_filter.value.push_back( value );
    }
    haar_filter.push_back( tmp_filter );
  }
#endif

#if 1
  // x-y direction (45 degree)
  for( divnum_x=2; divnum_x<=4; divnum_x++){ 
  //for( divnum_x=2; divnum_x<=9; divnum_x++){ 
    divnum_y = divnum_x;
    //    for( divnum_y=2; divnum_y<=4; divnum_y++){ 
      x_rf = window_size/2 - window_size/divnum_x;
      y_rf = window_size/2 - window_size/divnum_y;
      mc_haar tmp_filter;
      for(int i=0; i<2; i++){
        for(int j=0; j<2; j++){
          value = (int)pow( -1.0, (double)(i+j) );
	  tmp_filter.start_x.push_back( x_rf+j*window_size/divnum_x ); tmp_filter.start_y.push_back( y_rf+i*window_size/divnum_y );
	  tmp_filter.end_x.push_back( x_rf+(j+1)*window_size/divnum_x ); tmp_filter.end_y.push_back( y_rf+(i+1)*window_size/divnum_y );
	  tmp_filter.value.push_back( value );
        }
      }
      haar_filter.push_back( tmp_filter );
      //}
  }
#endif

#if 1
  // y direction (90 degree)
  divnum_x = 1;
  for( divnum_y=2; divnum_y<=4; divnum_y++){ 
    x_rf = 0;
    y_rf = window_size/2 - window_size/divnum_y;
    mc_haar tmp_filter;
    for(int i=0; i<2; i++){
      value = (int)pow( -1.0, (double)i );
      tmp_filter.start_x.push_back( x_rf+0 ); tmp_filter.start_y.push_back( y_rf+i*window_size/divnum_y );
      tmp_filter.end_x.push_back( x_rf+window_size ); tmp_filter.end_y.push_back( y_rf+(i+1)*window_size/divnum_y );
      tmp_filter.value.push_back( value );
    }
    haar_filter.push_back( tmp_filter );
  }
#endif
#endif

#if 0
  // debug
  int num_haar_like = haar_filter.size();
  for(int b=1; b<num_haar_like; b++){
    Mat tmp = Mat::zeros( window_size, window_size, CV_8UC3 );

    int num_haar_box = haar_filter[b].value.size();    
    for(int bb=0; bb<num_haar_box; bb++){
      cout << haar_filter[b].value[bb] << " (" << haar_filter[b].start_x[bb] << " " << haar_filter[b].start_y[bb] << ") -> (" << haar_filter[b].end_x[bb] << " " << haar_filter[b].end_y[bb] << ")" << endl;
      if(haar_filter[b].value[bb]==1)
	rectangle(tmp, Point(haar_filter[b].start_x[bb],haar_filter[b].start_y[bb]), Point(haar_filter[b].end_x[bb],haar_filter[b].end_y[bb]), Scalar(0,0,0), -1);
      else
	rectangle(tmp, Point(haar_filter[b].start_x[bb],haar_filter[b].start_y[bb]), Point(haar_filter[b].end_x[bb],haar_filter[b].end_y[bb]), Scalar(255,255,255), -1);
    }
    cout << endl;

    imshow("box", tmp);
    waitKey( 0 );
  }
#endif

}

void mc_apply_cnn_filter( mc_image in, mc_image &in_cnn, int window_size )
{
  int r, b, c;
  int rows = in.rows;
  int cols = in.cols;
  int chans = in.chans;
  double intensity;
  
  in_cnn.rows = rows;
  in_cnn.cols = cols;
  in_cnn.chans = 4;// maximum due to the current implementation of mc_read_image
  in_cnn.data.resize( (in_cnn.rows * in_cnn.cols * in_cnn.chans) );  

  std::vector<cv::Mat> W_conv1, W_conv2;
  W_conv1.push_back((cv::Mat_<float>(5,5) << 0.55445600, 0.01994691, -0.01522893, 0.07922035, 0.19061980, 0.41065201, -0.02583857, -0.31515610, -0.24336697, -0.35852617, 0.13309394, -0.25537738, -0.14495069, 0.44799170, -0.01394634, 0.54505253, -0.13158214, -0.03420714, 0.16443619, -0.07657393, 0.57273328, 0.25234050, 0.21659411, 0.32713521, -0.25873408));
  W_conv1.push_back((cv::Mat_<float>(5,5) << 0.00429894, 0.15968244, 0.00932316, 0.18426569, -0.08853491, -0.07288036, 0.04574790, 0.13092177, 0.16242144, -0.04668861, -0.02863846, 0.10431144, 0.21963508, 0.77156800, 0.18773293, -0.45077473, -0.36364922, -0.03406430, 0.24074237, 0.21139050, -0.26109561, -0.38609326, -0.20167707, -0.24799600, -0.02700261));
  W_conv1.push_back((cv::Mat_<float>(5,5) << -0.05923871, 0.14358938, -0.08340947, -0.03284870, 0.00274592, -0.11157122, -0.04093757, 0.08876786, 0.19927344, 0.32248035, -0.02847243, -0.04642048, -0.06551967, 0.19278078, 0.11324178, -0.08204728, -0.11211036, -0.13771932, 0.06982047, -0.05791488, -0.14726172, -0.29424852, 0.04167874, 0.09158552, 0.14486761));
  W_conv1.push_back((cv::Mat_<float>(5,5) << -0.14192091, -0.20860809, -0.06248078, -0.03212799, -0.25188538, -0.21325788, -0.34347138, -0.06574018, 0.41498446, -0.10104620, -0.17410275, -0.45236680, 0.01950049, 0.80293530, 0.44285595, -0.29130852, -0.15294062, 0.13089778, 0.47403297, 0.26176241, -0.22368491, -0.05676161, 0.21057217, 0.22118913, -0.03274488));

  W_conv2.push_back((cv::Mat_<float>(5,5) << -0.59849662, -0.36762032, -0.18186949, -0.20310119, -0.03621410, -0.18522088, -0.03164810, 0.23309754, 0.29061079, 0.14495869, -0.19156840, -0.00027692, 0.56925476, 0.28800854, 0.11092661, -0.75480098, -0.03820177, 0.87098062, 0.43171209, 0.35841164, -1.01080120, -0.32175410, 0.57555795, 0.63252467, 0.26138601));
  W_conv2.push_back((cv::Mat_<float>(5,5) << -0.54301137, -0.16504331, -0.13128099, -0.34817740, -0.52921635, -0.07408112, 0.10902073, 0.29768428, -0.01765074, -0.33459085, 0.16329710, 0.36315545, 0.35110372, 0.04835239, -0.12213374, -0.30893832, 0.39775833, 0.69598264, 0.15334041, -0.41393989, -0.32654810, 0.10708908, 0.32890737, 0.04562226, -0.03167849));
  W_conv2.push_back((cv::Mat_<float>(5,5) << -0.85858291, -0.69391584, -0.59749830, -0.55682111, -0.69355947, -0.49175876, 0.02434204, 0.06963897, 0.02018656, -0.38013807, -0.71551836, 0.24509235, 0.49967524, 0.11069350, -0.69399166, -0.91335440, -0.05561139, 0.54036504, 0.12755442, -0.71201944, -1.58911431, -0.73992079, -0.26907235, -0.49641228, -0.51354927));
  W_conv2.push_back((cv::Mat_<float>(5,5) << -0.19247173, -0.11761874, -0.07696011, -0.51842028, -0.51264316, 0.18300590, 0.27981314, 0.20824613, -0.03458434, -0.50555217, 0.10611386, 0.50673592, 0.40806067, -0.02661739, -0.17873298, 0.22291602, 0.54374295, 0.48040888, 0.10056540, -0.25230309, 0.00423653, 0.21503825, 0.31625813, 0.16698126, -0.12613815));
  W_conv2.push_back((cv::Mat_<float>(5,5) << 0.34617379, 0.05624257, -0.07425122, -0.03529512, 0.59230876, 0.13005126, -0.64486557, -0.76499176, -0.44333765, -0.22843120, 0.47443947, 0.17696416, 0.04469018, 0.17809692, -0.15338738, -1.74107206, -0.00986454, 0.38045362, 0.64574611, -0.04663575, -3.91075587, -1.06400597, 0.21841812, 0.10146497, -0.20711291));
  W_conv2.push_back((cv::Mat_<float>(5,5) << 0.57028413, 0.70846909, 0.56928325, -0.20957096, -0.06029505, 0.38060677, 0.44486332, 0.23770417, -0.46978575, -0.71465498, 0.15477127, -0.10161539, -0.06314643, -0.88709974, -1.44865692, -0.04772629, -0.13213110, -0.05305346, -0.58722901, -1.23442233, 0.58794570, 0.11426157, -0.26146844, -0.49418879, -0.44119388));
  W_conv2.push_back((cv::Mat_<float>(5,5) << -0.43828291, -0.77731657, -1.14518714, -0.87951183, -0.93357652, -0.08753683, -0.31377217, -0.25029859, -0.57706136, -0.91210485, 0.09272303, -0.04308754, 0.13342081, 0.08282879, -0.82124549, -0.61629897, -0.52857482, -0.13128780, -0.06168590, -0.90789866, 0.46622121, 0.28165919, -0.03031399, -0.02887219, -0.56055903));
  W_conv2.push_back((cv::Mat_<float>(5,5) << -0.12151616, -0.18505563, -0.16960159, -2.68591809, -3.67791128, 0.08683524, 0.11787523, 0.76599985, -0.44256988, -3.17236900, -0.16515732, -0.41908213, 0.26912960, 0.60453898, 0.40424445, 0.37168801, -0.06562776, -0.08927178, 0.04663891, 0.57183552, 0.88509196, 0.21415944, 0.21953627, 0.11590288, 0.45024380));
  W_conv2.push_back((cv::Mat_<float>(5,5) << -0.03878759, 0.10770328, -0.04413297, -0.07424930, 0.10643736, -0.15876877, -0.10313674, 0.03190772, 0.11733593, -0.19092067, -0.46375242, 0.02481622, 0.34748319, 0.50999695, 0.03166209, -0.07286190, 0.13256881, 0.36956900, 0.24954532, 0.10104424, -0.20633012, 0.04763134, 0.19961850, -0.03183565, -0.09511987));
  W_conv2.push_back((cv::Mat_<float>(5,5) << 0.33117142, 0.28524828, 0.22524284, 0.29873168, 0.02230724, 0.43906337, 0.37044892, 0.30979112, 0.00338414, -0.03967099, 0.30562323, 0.21806188, 0.22982866, -0.10431956, -0.12077820, -0.09532276, 0.14073086, 0.03861385, -0.16436796, -0.12675643, -0.31388688, -0.20395523, -0.15270674, -0.13378836, -0.30023742));
  W_conv2.push_back((cv::Mat_<float>(5,5) << 0.27932671, 0.21415290, 0.28827575, 0.11469740, 0.16604917, 0.24970590, 0.46967664, 0.47904304, 0.16493349, -0.16506611, 0.45420438, 0.37692848, 0.60656887, 0.49866414, -0.03366613, 0.44286877, 0.54183209, 0.52792364, 0.18181978, -0.02441703, 0.28613916, 0.46247506, 0.27816358, 0.27001113, -0.05159573));
  W_conv2.push_back((cv::Mat_<float>(5,5) << 0.06591789, 0.07545210, 0.29326102, -0.07924090, -0.20269556, 0.46488094, 0.47650808, 0.40772685, 0.02330978, -0.23249437, 0.36348784, 0.51850152, 0.47370365, 0.10118309, -0.16490313, 0.14544441, 0.08324146, 0.41573375, 0.14396638, 0.12724462, -0.04612691, -0.27008671, 0.07311086, -0.19751546, -0.08409805));
  W_conv2.push_back((cv::Mat_<float>(5,5) << 0.81226337, 0.33197209, 0.20745450, -0.02463252, 0.67325145, 0.64400923, 0.21382351, -0.27156824, -0.23182757, 0.30769503, 0.50344825, -0.05044780, -0.10509825, 0.35552090, 0.43773457, -2.76497293, -0.09648598, -0.10205749, 0.21109325, 0.22783390, -5.25449610, 0.06604446, 0.85586405, 0.07478090, -0.29158694));
  W_conv2.push_back((cv::Mat_<float>(5,5) << 0.99014843, 0.63152039, 0.20588402, -0.15993689, -0.18763793, 0.44033498, 0.42147946, 0.34238505, -0.46497670, -1.74711335, 0.40661505, 0.16902825, 0.64583343, -0.33342561, -2.78664732, 0.38891947, -0.01368539, 0.55359191, 0.11107136, -1.76884246, 0.46469960, 0.36069474, -0.06629296, -0.30086732, -0.80445635));
  W_conv2.push_back((cv::Mat_<float>(5,5) << 0.14423873, -0.74169827, -0.83854723, -0.18533881, 0.01898623, -0.06039404, -0.77016944, -0.23686559, 0.07773846, -0.25571564, -0.00051817, -0.42274103, 0.21930450, 0.17645845, -0.90621519, -0.25258428, -0.25355154, 0.05643197, -0.16433661, -1.53286171, -0.54245597, -0.11189485, -0.15531707, -0.85069287, -1.73638415));
  W_conv2.push_back((cv::Mat_<float>(5,5) << -0.58751196, -0.85477555, -0.49054265, -3.34977293, -2.85317564, -0.66518492, -0.39467737, 0.44934869, -1.44462311, -2.87440443, -0.58279115, -0.38404211, 0.66901451, 0.78501707, -0.63239521, -0.07424013, -0.29446393, 0.05967544, 0.73685485, 0.28674820, 0.29347235, 0.08444874, -0.01645537, 0.30866852, 0.51401186));

  float b_conv1[] = {-0.39814457,  0.01438079,  0.26769125,  0.03359091};
  float b_conv2[] = {-0.02889748, -0.04417422,  0.0682012 , -0.0891505};
  float W_fc[] = {-1.7177496 , -0.85658294, -2.05604148, -1.26670563};
  float b_fc = 2.7525692;

  cv::Mat img(rows,cols,CV_32FC1), out(rows,cols,CV_32FC1);
  cv::Mat tmp_out(rows,cols,CV_8UC1);
  for (r=0; r < rows; r++){
    for (c=0; c < cols; c++){
      intensity = 0.0;
      for (b=0; b < chans; b++)
	intensity += (unsigned char)in.data[r*cols*chans + c*chans + b];
      intensity /= (double)chans;
      img.at<float>(r,c) = (float)intensity;
    }
  }

  double mat_min, mat_max;
  double mat_min_all, mat_max_all;
  cv::Mat ones = cv::Mat::ones(15,15,CV_32F);
  cv::Mat tmp(rows, cols, CV_32FC1);
  cv::Mat tmp2(rows, cols, CV_32FC1);
  cv::Mat tmp3(rows, cols, CV_32FC1);
  std::vector<cv::Mat> hidden;
#if 1
  // 1st layer
  for (int i = 0; i < 4; i++) {
    cv::filter2D(img, tmp, -1, W_conv1[i], cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101);
    tmp += b_conv1[i];
    cv::threshold(tmp, tmp, 0.0, 0.0, cv::THRESH_TOZERO); // ReLU
    
    cv::minMaxLoc( tmp, &mat_min, &mat_max);
    tmp3 = (tmp-mat_min) / (mat_max-mat_min) * 255.0;
    tmp3.convertTo( tmp, CV_8UC1 );
    //imshow("out", tmp );
    //waitKey( 0 );

    if(i==0){
      mat_min_all = mat_min;
      mat_max_all = mat_max;
    }
    else{
      mat_min_all = min( mat_min_all, mat_min );
      mat_max_all = max( mat_max_all, mat_max );
    }
    //cout << mat_min << " " << mat_max << " : "<< mat_min_all << " " << mat_max_all << endl;
    
    //for (r=0; r < rows; r++)
    //for (c=0; c < cols; c++)
    //in_cnn.data[r*in_cnn.cols*in_cnn.chans + c*in_cnn.chans + i] = tmp_out.at<uchar>(r,c);//sum / (window_size*window_size);
    
    hidden.push_back(tmp.clone());
  }
  for (int i = 0; i < 4; i++) {
    //tmp3 = (hidden[i]-mat_min_all) / (mat_max_all-mat_min_all) * 255.0;
    //tmp3.convertTo( tmp_out, CV_8UC1 );
    //imshow("out", tmp_out );
    //waitKey( 0 );
    hidden[i].convertTo( tmp_out, CV_8UC1 );
    for (r=0; r < rows; r++)
      for (c=0; c < cols; c++)
	in_cnn.data[r*in_cnn.cols*in_cnn.chans + c*in_cnn.chans + i] = tmp_out.at<uchar>(r,c);//sum / (window_size*window_size);
  }
#else
  // 2nd layer
  std::vector<cv::Mat> hidden_tmp;
  for (int i = 0; i < 4; i++) {
    cv::filter2D(img, tmp, -1, W_conv1[i], cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101);
    tmp += b_conv1[i];
    cv::threshold(tmp, tmp, 0.0, 0.0, cv::THRESH_TOZERO); // ReLU
    hidden.push_back(tmp.clone());
  }
  for (int i = 0; i < 4; i++) {
    tmp2.setTo(0);
    for (int j = 0; j < 4; j++) {
      cv::filter2D(hidden[j], tmp, -1, W_conv2[4*j+i], cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101);
      tmp2 += tmp;
      
      //cv::minMaxLoc( tmp, &mat_min, &mat_max);
      //tmp3 = (tmp-mat_min) / (mat_max-mat_min) * 255.0;
      //tmp3.convertTo( tmp_out, CV_8UC1 );
      //imshow("out", tmp_out );
      //waitKey( 0 );
    }

    hidden_tmp.push_back(tmp2.clone());

    //cout << i << endl;
    cv::minMaxLoc( tmp2, &mat_min, &mat_max);
    //tmp3 = (tmp2-mat_min) / (mat_max-mat_min) * 255.0;
    //tmp3.convertTo( tmp_out, CV_8UC1 );
    //imshow("out", tmp_out );
    //waitKey( 0 );

    if(i==0){
      mat_min_all = mat_min;
      mat_max_all = mat_max;
    }
    else{
      mat_min_all = min( mat_min_all, mat_min );
      mat_max_all = max( mat_max_all, mat_max );
    }

    //for (r=0; r < rows; r++)
    //for (c=0; c < cols; c++)
    //in_cnn.data[r*in_cnn.cols*in_cnn.chans + c*in_cnn.chans + i] = tmp_out.at<uchar>(r,c);//sum / (window_size*window_size);

    tmp2 += b_conv2[i];
    cv::threshold(tmp2, tmp2, 0.0, 0.0, cv::THRESH_TOZERO); // ReLU
    cv::dilate(tmp2, tmp2, ones, cv::Point(-1,-1), 1, cv::BORDER_REFLECT_101); // Max filter
    out += W_fc[i]*tmp2;
  }
  for (int i = 0; i < 4; i++) {
    tmp3 = (hidden_tmp[i]-mat_min_all) / (mat_max_all-mat_min_all) * 255.0;
    tmp3.convertTo( tmp_out, CV_8UC1 );
    //imshow("out", tmp_out );
    //waitKey( 0 );
    for (r=0; r < rows; r++)
      for (c=0; c < cols; c++)
	in_cnn.data[r*in_cnn.cols*in_cnn.chans + c*in_cnn.chans + i] = tmp_out.at<uchar>(r,c);//sum / (window_size*window_size);
  }
#endif

}

void mc_load_label( mc_label &mc_label_list, mc_image mc_label_src )
{
  int chans = mc_label_src.chans;
  int rows = mc_label_src.rows;
  int cols = mc_label_src.cols;

  if( mc_label_list.data.size() == 0 ){
    mc_label_list.n_label = 0;
    mc_label_list.chans = chans;
    mc_label_list.rows = rows;
    mc_label_list.cols = cols;
  }

  if( chans!=mc_label_list.chans || rows!=mc_label_list.rows || cols!=mc_label_list.cols ){
    cout << "properties of mc_label_list are not the same with those of mc_label_src" << endl;
    return;
  }

  int intensity;
  for (int r=0; r < rows; r++){
    for (int c=0; c < cols; c++){
      
#if 0
      // ignore intensity = 0
      intensity = 0;
      for (int b=0; b < chans; b++)
	intensity += mc_label_src.data[r*cols*chans + c*chans + b];
      if( intensity == 0 )  continue;
#endif      

      if( mc_label_list.n_label == 0 ){
	// set new label
	for (int b=0; b < chans; b++){
	  intensity = mc_label_src.data[r*cols*chans + c*chans + b];
	  mc_label_list.data.push_back( intensity );
	}
	mc_label_list.num_each_label.push_back( 1 );

	// label_id is 0,1,2,3,... cannot be changed (if you do like 0, 2, 5, ... other process will corrupsed)
	mc_label_list.label_id.push_back( mc_label_list.n_label );

	mc_label_list.flag_ignore.push_back( -1 );

	mc_label_list.n_label ++;

      }
      else{
	int color_match = -1;
	for(int i=0; i<mc_label_list.n_label; i++){
	  uchar color_hit = 1;

	  for (int b=0; b < chans; b++){
	    intensity = mc_label_src.data[r*cols*chans + c*chans + b];	    
	    if( intensity != mc_label_list.data[i*chans + b] ) color_hit = 0;
	  }

	  if( mc_label_list.data[i*chans+0]==215 && mc_label_list.data[i*chans+1]==228 && mc_label_list.data[i*chans+2]==189 )
	    getchar();

	  if( color_hit == 1 ){
	    color_match = i;
	    break;
	  }
	}

	if( color_match == -1 ){
	  // set new label
	  for (int b=0; b < chans; b++){
	    intensity = mc_label_src.data[r*cols*chans + c*chans + b];
	    mc_label_list.data.push_back( intensity );
	  }
	  mc_label_list.num_each_label.push_back( 1 );
	  //mc_label_list.label_id( mc_label_list.n_label );

	  // label_id is 0,1,2,3,... cannot be changed (if you do like 0, 2, 5, ... other process will corrupsed)
	  mc_label_list.label_id.push_back( mc_label_list.n_label );

	  mc_label_list.flag_ignore.push_back( -1 );
	
	  mc_label_list.n_label ++;
	}
	else
	  mc_label_list.num_each_label[color_match] ++;

      }      
    }
  }
  
}

// color (0,0,0) will be ignored
void mc_load_label( mc_label &mc_label_list, mc_label_image mc_label_src )
{
  int chans = mc_label_src.chans;
  int rows = mc_label_src.rows;
  int cols = mc_label_src.cols;

  int color_ignore[3] = { 0 };

  if( mc_label_list.data.size() == 0 ){
    mc_label_list.n_label = 0;
    mc_label_list.chans = chans;
    mc_label_list.rows = rows;
    mc_label_list.cols = cols;
  }

  if( chans!=mc_label_list.chans || rows!=mc_label_list.rows || cols!=mc_label_list.cols ){
    cout << "properties of mc_label_list are not the same with those of mc_label_src" << endl;
    return;
  }

  int intensity;
  for (int r=0; r < rows; r++){
    for (int c=0; c < cols; c++){
      
      // ignore intensity = 0
      bool flg_ignore = true;
      for (int b=0; b < chans; b++){
	intensity = mc_label_src.data[r*cols*chans + c*chans + b];
	if( color_ignore[b] != intensity ) flg_ignore = false;
      }
      if( flg_ignore ) continue;
      
      if( mc_label_list.n_label == 0 ){
	// set new label
	for (int b=0; b < chans; b++){
	  intensity = mc_label_src.data[r*cols*chans + c*chans + b];
	  mc_label_list.data.push_back( intensity );
	}
	mc_label_list.num_each_label.push_back( 1 );

	// label_id is 0,1,2,3,... cannot be changed (if you do like 0, 2, 5, ... other process will corrupsed)
	mc_label_list.label_id.push_back( mc_label_list.n_label );

	mc_label_list.flag_ignore.push_back( -1 );

	mc_label_list.n_label ++;

      }
      else{
	int color_match = -1;
	for(int i=0; i<mc_label_list.n_label; i++){
	  uchar color_hit = 1;

	  for (int b=0; b < chans; b++){
	    intensity = mc_label_src.data[r*cols*chans + c*chans + b];	    
	    if( intensity != mc_label_list.data[i*chans + b] ) color_hit = 0;
	  }

	  if( mc_label_list.data[i*chans+0]==215 && mc_label_list.data[i*chans+1]==228 && mc_label_list.data[i*chans+2]==189 )
	    getchar();

	  if( color_hit == 1 ){
	    color_match = i;
	    break;
	  }
	}

	if( color_match == -1 ){
	  // set new label
	  for (int b=0; b < chans; b++){
	    intensity = mc_label_src.data[r*cols*chans + c*chans + b];
	    mc_label_list.data.push_back( intensity );
	  }
	  mc_label_list.num_each_label.push_back( 1 );
	  //mc_label_list.label_id( mc_label_list.n_label );

	  // label_id is 0,1,2,3,... cannot be changed (if you do like 0, 2, 5, ... other process will corrupsed)
	  mc_label_list.label_id.push_back( mc_label_list.n_label );

	  mc_label_list.flag_ignore.push_back( -1 );
	
	  mc_label_list.n_label ++;
	}
	else
	  mc_label_list.num_each_label[color_match] ++;

      }      
    }
  }
  
}


// based on mc_label_list, assign label-id to each pixel of mc_label
void mc_assign_label( mc_label mc_label_list, mc_image &mc_label_src )
{
  mc_image tmp_mc_label_src;
  tmp_mc_label_src.rows = mc_label_src.rows;
  tmp_mc_label_src.cols = mc_label_src.cols;
  tmp_mc_label_src.chans = mc_label_src.chans;
  
  tmp_mc_label_src.data.reserve(mc_label_src.data.size());
  copy(mc_label_src.data.begin(), mc_label_src.data.end(), back_inserter(tmp_mc_label_src.data));

  int chans = tmp_mc_label_src.chans;
  int rows = tmp_mc_label_src.rows;
  int cols = tmp_mc_label_src.cols;

  // resize mc_label_src
  mc_label_src.chans = 1;
  mc_label_src.data.clear();
  mc_label_src.data.resize( rows * cols );

  // if( chans!=mc_label_list.chans || rows!=mc_label_list.rows || cols!=mc_label_list.cols ){
  //   cout << "properties of mc_label_list are not the same with those of tmp_mc_label_src" << endl;
  //   return;
  // }

  //int ratio = 255 / mc_label_list.n_label;

  int intensity;
  for (int r=0; r < rows; r++){
    for (int c=0; c < cols; c++){
      
#if 0
      // ignore intensity = 0
      intensity = 0;
      for (int b=0; b < chans; b++)
	intensity += tmp_mc_label_src.data[r*cols*chans + c*chans + b];
      if( intensity == 0 ){
	mc_label_src.data[r*cols + c] = 0;
	continue;
      }
#endif
      
      int color_match = -1;
      for(int i=0; i<mc_label_list.n_label; i++){
	uchar color_hit = 1;
	
	for (int b=0; b < chans; b++){
	  intensity = tmp_mc_label_src.data[r*cols*chans + c*chans + b];	    
	  if( intensity != mc_label_list.data[i*chans + b] ) color_hit = 0;
	}
	
	if( color_hit == 1 ){
	  color_match = i;
	  break;
	}
      }

      // this can happen
      if( color_match == -1 ){
	cout << "assign_label, unknown class" << endl;
	mc_label_src.data[r*cols + c] = -1;
	//getchar();
	continue;
      }
      
      //color_match ++; // avoid class 0 (=no class)
      mc_label_src.data[r*cols + c] = mc_label_list.label_id[color_match];// original
      //mc_label_src.data[r*cols + c] = mc_label_list.label_id[color_match]*50;//debug
      
    }
  }
  
}

void mc_assign_label( mc_label mc_label_list, mc_label_image &mc_label_src )
{
  mc_image tmp_mc_label_src;
  tmp_mc_label_src.rows = mc_label_src.rows;
  tmp_mc_label_src.cols = mc_label_src.cols;
  tmp_mc_label_src.chans = mc_label_src.chans;
  
  tmp_mc_label_src.data.reserve(mc_label_src.data.size());
  copy(mc_label_src.data.begin(), mc_label_src.data.end(), back_inserter(tmp_mc_label_src.data));

  int chans = tmp_mc_label_src.chans;
  int rows = tmp_mc_label_src.rows;
  int cols = tmp_mc_label_src.cols;

  // resize mc_label_src
  mc_label_src.chans = 1;
  mc_label_src.data.clear();
  mc_label_src.data.resize( rows * cols );

  // if( chans!=mc_label_list.chans || rows!=mc_label_list.rows || cols!=mc_label_list.cols ){
  //   cout << "properties of mc_label_list are not the same with those of tmp_mc_label_src" << endl;
  //   return;
  // }

  //int ratio = 255 / mc_label_list.n_label;

  int intensity;
  for (int r=0; r < rows; r++){
    for (int c=0; c < cols; c++){
      
#if 0
      // ignore intensity = 0
      intensity = 0;
      for (int b=0; b < chans; b++)
	intensity += tmp_mc_label_src.data[r*cols*chans + c*chans + b];
      if( intensity == 0 ){
	mc_label_src.data[r*cols + c] = 0;
	continue;
      }
#endif
      
      int color_match = -1;
      for(int i=0; i<mc_label_list.n_label; i++){
	uchar color_hit = 1;
	
	for (int b=0; b < chans; b++){
	  intensity = tmp_mc_label_src.data[r*cols*chans + c*chans + b];	    
	  if( intensity != mc_label_list.data[i*chans + b] ) color_hit = 0;
	}
	
	if( color_hit == 1 ){
	  color_match = i;
	  break;
	}
      }

      // this can happen
      if( color_match == -1 ){
	//cout << "assign_label, unknown class" << endl;
	mc_label_src.data[r*cols + c] = -1;
	//getchar();
	continue;
      }

      //cout << color_match << " ";
      
      mc_label_src.data[r*cols + c] = mc_label_list.label_id[color_match];// original
      //mc_label_src.data[r*cols + c] = mc_label_list.label_id[color_match]*50;//debug
      
    }
  }
  
}

int mc_find_label_id( int target_label_rgb[], mc_label mc_label_list )
{
  int target_id = -1;

  int chans = mc_label_list.chans;
  int rows = mc_label_list.rows;
  int cols = mc_label_list.cols;

  int intensity;
  for(int i=0; i<mc_label_list.n_label; i++){
    uchar color_hit = 1;

    //printf("label %d : ", mc_label_list.label_id[i]);
    //for (int b=0; b < chans; b++)
    //printf("%d ", mc_label_list.data[i*chans + b]);
    //printf("\n");

    for (int b=0; b < chans; b++)
      if( target_label_rgb[b] != mc_label_list.data[i*chans + b] ) 
	color_hit = 0;

    if( color_hit == 1 ){
      target_id = i;
      break;
    }
  }
  //getchar();

  return target_id;
}

// works with chans 1 or 3
void mc_copy_image_2_Mat( mc_image in, Mat &mat )
{
  int r, c, b;
  
  if( in.chans == 1 )
    mat = Mat::zeros( in.rows, in.cols, CV_8UC1 );
  else if( in.chans == 3 )
    mat = Mat::zeros( in.rows, in.cols, CV_8UC3 );
  else{
    cout << "mc_copy_image_2_Mat works only for chans 1 or 3" << endl;
    return;
  }

  if( in.chans == 1 ){
    for (r=0; r < in.rows; r++){
      uchar* _mat = mat.ptr<uchar>(r);
      for (c=0; c < in.cols; c++)
	_mat[c] = (unsigned char)in.data[r*in.cols + c];
    }
  }
  else if( in.chans == 3 ){
    for (r=0; r < in.rows; r++){
      Vec3b* _mat = mat.ptr<Vec3b>(r);
      for (c=0; c < in.cols; c++){
	for (b=0; b < in.chans; b++)
	  _mat[c].val[2-b] = (unsigned char)in.data[r*in.cols*in.chans + c*in.chans + b];
      }
    }
  }
	
}

// works with chans 1 or 3
void mc_copy_label_image_2_Mat( mc_label_image in, cv::Mat &mat )
{
  int r, c, b;
  
  if( in.chans == 1 )
    mat = Mat::zeros( in.rows, in.cols, CV_8UC1 );
  else if( in.chans == 3 )
    mat = Mat::zeros( in.rows, in.cols, CV_8UC3 );
  else{
    cout << "mc_copy_label_image_2_Mat works only for chans 1 or 3" << endl;
    return;
  }

  if( in.chans == 1 ){
    for (r=0; r < in.rows; r++){
      uchar* _mat = mat.ptr<uchar>(r);
      for (c=0; c < in.cols; c++)
	_mat[c] = (unsigned char)in.data[r*in.cols + c];
    }
  }
  else if( in.chans == 3 ){
    for (r=0; r < in.rows; r++){
      Vec3b* _mat = mat.ptr<Vec3b>(r);
      for (c=0; c < in.cols; c++){
	for (b=0; b < in.chans; b++)
	  _mat[c].val[2-b] = (unsigned char)in.data[r*in.cols*in.chans + c*in.chans + b];
      }
    }
  }
	
}

// works with chans 1 or 3
void mc_copy_Mat_2_image( Mat mat, mc_image &out )
{
  int r, c, b;
  int intensity;
  out.rows = mat.rows;
  out.cols = mat.cols;
  out.chans = mat.channels();

  if( out.data.size() > 0 )
    out.data.clear();

  if( out.chans == 1 ){
    for (r=0; r < out.rows; r++){
      uchar* _mat = mat.ptr<uchar>(r);
      for (c=0; c < out.cols; c++)
	out.data.push_back( _mat[c] );
    }
  }
  else if( out.chans == 3 ){
    for (r=0; r < out.rows; r++){
      Vec3b* _mat = mat.ptr<Vec3b>(r);
      for (c=0; c < out.cols; c++){
	//for (b=0; b < out.chans; b++)
	for (b=(out.chans-1); b >= 0; b--)
	  out.data.push_back( _mat[c].val[b] );
      }
    }
  }
  else{
    cout << "mc_copy_Mat_2_image works only for chans 1 or 3" << endl;
    return;
  }
	
}

// works with chans 1 or 3
void mc_copy_Mat_2_label_image( Mat mat, mc_label_image &out )
{
  int r, c, b;
  int intensity;
  out.rows = mat.rows;
  out.cols = mat.cols;
  out.chans = mat.channels();

  if( out.data.size() > 0 )
    out.data.clear();

  if( out.chans == 1 ){
    for (r=0; r < out.rows; r++){
      uchar* _mat = mat.ptr<uchar>(r);
      for (c=0; c < out.cols; c++)
	out.data.push_back( _mat[c] );
    }
  }
  else if( out.chans == 3 ){
    for (r=0; r < out.rows; r++){
      Vec3b* _mat = mat.ptr<Vec3b>(r);
      for (c=0; c < out.cols; c++){
	//for (b=0; b < out.chans; b++)
	for (b=(out.chans-1); b >= 0; b--)
	  out.data.push_back( _mat[c].val[b] );
      }
    }
  }
  else{
    cout << "mc_copy_Mat_2_image works only for chans 1 or 3" << endl;
    return;
  }
	
}

void mc_copy_image_2_image( mc_image src, mc_image &out )
{
  int r, c, b;
  int intensity;
  out.rows = src.rows;
  out.cols = src.cols;
  out.chans = src.chans;

  if( out.data.size() > 0 )
    out.data.clear();

  copy(src.data.begin(), src.data.end(), back_inserter(out.data));
}
  
  void mc_make_dummy_image( int rows, int cols, int chans, int color[3], mc_image &dst )
  {
    dst.rows = rows;
    dst.cols = cols;
    dst.chans = chans;

    for (int r=0; r < rows; r++)
      for (int c=0; c < cols; c++)
	for (int b=0; b < chans; b++)
	  dst.data.push_back( color[b] );
  }

  void mc_make_dummy_image( int rows, int cols, int chans, int color[3], mc_label_image &dst )
  {
    dst.rows = rows;
    dst.cols = cols;
    dst.chans = chans;

    for (int r=0; r < rows; r++)
      for (int c=0; c < cols; c++)
	for (int b=0; b < chans; b++)
	  dst.data.push_back( color[b] );
  }
  

#pragma GCC diagnostic pop

} // namespace ibo
