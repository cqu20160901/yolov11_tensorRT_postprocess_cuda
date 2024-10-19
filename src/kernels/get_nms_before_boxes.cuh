#ifndef GET_NMS_BEFORE_BOXES_CUH__
#define GET_NMS_BEFORE_BOXES_CUH__

#include <stdio.h>
#include "../common_struct.hpp"

void GetNmsBeforeBoxes(float *SrcInput, int AnchorCount, int ClassNum, float ObjectThresh, int NMSBeforeMaxNum, DetectRect* OutputRects, int *OutputCount, cudaStream_t Stream);


#endif

