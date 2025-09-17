#include "image_preprocess.cuh"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void NearestNeighborResizeNormCHWKernel(float* dst_output,         // [c, h, w]
                                                   int dst_c, int dst_h, int dst_w,
                                                   unsigned char* src_input,  // [h, w, c] 
                                                   int src_h, int src_w, int src_c,
                                                   float scale)
{

    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;

    if (w >= dst_w || h >= dst_h) return;

    const float fx = static_cast<float>(w) * (static_cast<float>(src_w) / dst_w);
    const float fy = static_cast<float>(h) * (static_cast<float>(src_h) / dst_h);

    int src_x = __float2int_rn(fx);
    int src_y = __float2int_rn(fy);
    
    src_x = max(0, min(src_x, src_w - 1));
    src_y = max(0, min(src_y, src_h - 1));

    const int src_offset = (src_y * src_w + src_x) * src_c;

    const int spatial_offset = h * dst_w + w;

    for (int c = 0; c < min(dst_c, src_c); ++c)
    {
        const uint8_t input_value = src_input[src_offset + c];
        const int channel_offset = c * dst_h * dst_w;
        dst_output[channel_offset + spatial_offset] = static_cast<float>(input_value) * scale;
    }
}

void NearestNeighborResizeNormCHW(float* d_dst,
                                  int dst_w, int dst_h,
                                  unsigned char* d_src,
                                  int src_w, int src_h,
                                  float scale,
                                  cudaStream_t Stream) {
 
    dim3 blockSize(16, 16, 2);
    dim3 gridSize((dst_w + blockSize.x - 1) / blockSize.x, (dst_h + blockSize.y - 1) / blockSize.y, (3 + blockSize.z - 1) / blockSize.z);
    
    NearestNeighborResizeNormCHWKernel<<<gridSize, blockSize, 0, Stream>>>(d_dst, 3, dst_h, dst_w, d_src, src_h, src_w, 3, scale);
}