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
    void PrepareImage(cv::Mat &SrcImage, void *InputBuffer, cudaStream_t Stream);

    std::string OnnxFilePath_;
    std::string SaveTrtFilePath_;

    int BatchSize_ = 0;
    int InputChannel_ = 0;
    int InputImageWidth_ = 0;
    int InputImageHeight_ = 0;
    unsigned char *GpuSrcImage_ = nullptr;
    
    nvinfer1::ICudaEngine *PtrEngine_ = nullptr;
    nvinfer1::IExecutionContext *PtrContext_ = nullptr;
    cudaStream_t Stream_;

    void *Buffers_[10];
    std::vector<int64_t> BuffersDataSize_;

    GetResultRectYolov11 Postprocess_;
    const int NmsBeforeMaxNum_ = 512;
    int* GpuOutputCount_ = nullptr;
    DetectRect *GpuOutputRects_ = nullptr;

    int* CpuOutputCount_ = nullptr;
    DetectRect *CpuOutputRects_ = nullptr;
};

#endif
