#ifndef IMAGE_PREPROCESS_CUH_
#define IMAGE_PREPROCESS_CUH_

#include <stdio.h>


void NearestNeighborResizeNormCHW(float* d_dst_data,
                                  int dst_w, int dst_h,
                                  unsigned char* d_src_data,
                                  int src_w, int src_h,
                                  float scale,
                                  cudaStream_t Stream);


#endif
