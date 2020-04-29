/**
  author: Yumi Iwashita
**/

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

#include "buo.h"

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

  size_t buo_find_the_last_string( string input_string, string find_string )
  {
    size_t last_position = 0;
    for(size_t str2_position = 0; str2_position<input_string.size(); str2_position++ ){
      size_t found = input_string.find(find_string, str2_position);
      if (found!=std::string::npos){
	//std::cout << "first '/' found at: " << found << '\n';	
	last_position = found;
      }
    }
    return last_position;
  }

} // closing namespace
