/*****************************************************************************/

/**
    @file buo.
    @brief Basic Useful Operations
  author: Yumi Iwashita
**/

/*****************************************************************************/

#ifndef __IBO_BASIC_USEFUL_OPERATION__
#define __IBO_BASIC_USEFUL_OPERATION__

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

//#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <complex>
//#include <string.h>

#include <time.h>
#include <sys/timeb.h>
#include <sys/time.h>


enum { X_AXIS, Y_AXIS, Z_AXIS };

namespace ibo
{

inline double buo_rand_pm_1(){
  double value = rand() % 2;
  value = pow( -1.0, value );
  
  value = value * (rand()%RAND_MAX) / RAND_MAX;
  return value;
}

inline double buo_get_wall_time(){
  struct timeval time;
  if (gettimeofday(&time,NULL))
    return 0;
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

// this is to get clock cycle
inline double buo_get_cpu_time(){
  return (double)clock() / CLOCKS_PER_SEC;
}


template <class T>
inline void buo_ObtainRotateMat( int type, double angle, T R)
{
  angle = angle / 180.0 * M_PI;
  
  if( type == X_AXIS )
    {
      R[0][0] = 1.0; R[0][1] = 0.0; R[0][2] = 0.0;
      R[1][0] = 0.0; R[1][1] = cos(angle); R[1][2] = -sin(angle);
      R[2][0] = 0.0; R[2][1] = sin(angle); R[2][2] = cos(angle);
    }
  else if( type == Y_AXIS )
    {
      R[0][0] = cos(angle); R[0][1] = 0.0; R[0][2] = sin(angle);
      R[1][0] = 0.0; R[1][1] = 1.0; R[1][2] = 0.0;
      R[2][0] = -sin(angle); R[2][1] = 0.0; R[2][2] = cos(angle);
    }
  else if( type == Z_AXIS )
    {
      R[0][0] = cos(angle); R[0][1] = -sin(angle); R[0][2] = 0.0;
      R[1][0] = sin(angle); R[1][1] = cos(angle); R[1][2] = 0.0;
      R[2][0] = 0.0; R[2][1] = 0.0; R[2][2] = 1.0;    
    }
}

 std::size_t buo_find_the_last_string( std::string input_string, std::string find_string );

} // namespace ibo

#endif // __IBO_BASIC_USEFUL_OPERATION__
