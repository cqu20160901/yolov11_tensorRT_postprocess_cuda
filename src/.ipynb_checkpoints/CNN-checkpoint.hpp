#ifndef CNN_HPP
#define CNN_HPP

#include "NvInfer.h"
#include "postprocess_cuda.hpp"
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <nppdefs.h>

#include "common_struct.hpp"

class CNN
{
public:
    CNN(const std::string &OnnxFilePath, const std::string &SaveTrtFilePath, int BatchSize, int InputChannel, int InputImageWidth, int InputImageHeight);
    ~CNN();

    void Inference(cv::Mat &SrcImage);

    std::vector<float> DetectiontRects_;

private:
    void ModelInit();
    void PrepareImage(cv::Mat &SrcImage, std::vector<float> &PreprocessResult);
    void PrepareImage(cv::Mat &vec_img, void *InputBuffer);

    std::string OnnxFilePath_;
    std::string SaveTrtFilePath_;

    int BatchSize_ = 0;
    int InputChannel_ = 0;
    int InputImageWidth_ = 0;
    int InputImageHeight_ = 0;
    int ModelOutputSize_ = 0;

    nvinfer1::ICudaEngine *PtrEngine_ = nullptr;
    nvinfer1::IExecutionContext *PtrContext_ = nullptr;
    cudaStream_t Stream_;

    void *Buffers_[10];
    std::vector<int64_t> BuffersDataSize_;

    Npp8u *GpuSrcImgBuf_ = nullptr;     // gpu：装 src 图像
    Npp8u *GpuImgResizeBuf_ = nullptr;  //  gpu 装 resize后的图像
    Npp32f *GpuImgF32Buf_ = nullptr;    // gpu: int8 转 F32
    Npp32f *GpuDataPlanes_ = nullptr;

    Npp32f MeanScale_[3] = {0.00392157, 0.00392157, 0.00392157 };
    int DstOrder_[3] = {2, 1, 0 };
    Npp32f* DstPlanes_[3];

    GetResultRectYolov8 Postprocess_;
    const int NmsBeforeMaxNum_ = 512;
    int* GpuOutputCount_ = nullptr;
    DetectRect *GpuOutputRects_ = nullptr;

    int* CpuOutputCount_ = nullptr;
    DetectRect *CpuOutputRects_ = nullptr;
};

#endif
