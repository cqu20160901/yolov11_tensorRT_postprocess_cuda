#include "CNN.hpp"
#include "common/common.hpp"
#include <algorithm>
#include <chrono>
#include "kernels/get_nms_before_boxes.cuh"
#include "kernels/image_preprocess.cuh"


CNN::CNN(const std::string &OnnxFilePath, const std::string &SaveTrtFilePath, int BatchSize, int InputChannel, int InputImageWidth, int InputImageHeight)
{
    OnnxFilePath_ = OnnxFilePath;
    SaveTrtFilePath_ = SaveTrtFilePath;

    BatchSize_ = BatchSize;
    InputChannel_ = InputChannel;
    InputImageWidth_ = InputImageWidth;
    InputImageHeight_ = InputImageHeight;

    ModelInit();
}

CNN::~CNN()
{
    // release the stream and the Buffers
    cudaStreamDestroy(Stream_);

    for (int i = 0; i < BuffersDataSize_.size(); i ++)
    {
        cudaFree(Buffers_[i]);
    }

    // destroy the engine
    if (nullptr != PtrContext_)
    {
        PtrContext_->destroy();
    }

    if (nullptr != PtrEngine_)
    {
        PtrEngine_->destroy();
    }

    if (nullptr != GpuOutputCount_)
    {
        cudaFree(GpuOutputCount_);
    }
    
    if (nullptr != GpuSrcImage_)
    {
        cudaFree(GpuSrcImage_);
    }

    if (nullptr != CpuOutputCount_)
    {
        free(CpuOutputCount_);
    }
    if (nullptr != CpuOutputRects_)
    {
        free(CpuOutputRects_);
    }
    
    std::cout << "CNN deconstructed finished." << std::endl;
}

void CNN::ModelInit()
{
    std::fstream existEngine;
    existEngine.open(SaveTrtFilePath_, std::ios::in);
    if (existEngine)
    {
        ReadTrtFile(SaveTrtFilePath_, PtrEngine_);
        assert(PtrEngine_ != nullptr);
    }
    else
    {
        OnnxToTRTModel(OnnxFilePath_, SaveTrtFilePath_, PtrEngine_, BatchSize_);
        assert(PtrEngine_ != nullptr);
    }
    
    assert(PtrEngine_ != nullptr);
    PtrContext_ = PtrEngine_->createExecutionContext();
    PtrContext_->setOptimizationProfile(0);
    auto InputDims = nvinfer1::Dims4 {BatchSize_, InputChannel_, InputImageHeight_, InputImageWidth_};
    PtrContext_->setBindingDimensions(0, InputDims);

    cudaStreamCreate(&Stream_);

    int64_t TotalSize = 0;
    int nbBindings = PtrEngine_->getNbBindings();
    BuffersDataSize_.resize(nbBindings);
    for (int i = 0; i < nbBindings; ++ i)
    {
        nvinfer1::Dims dims = PtrEngine_->getBindingDimensions(i);
        nvinfer1::DataType dtype = PtrEngine_->getBindingDataType(i);
        TotalSize = Volume(dims) * 1 * GetElementSize(dtype);
        BuffersDataSize_[i] = TotalSize;
        cudaMalloc(&Buffers_[i], TotalSize);

        if (0 == i)
        {
            std::cout << "input node name: "<< PtrEngine_->getBindingName(i) << ", dims: " << dims.nbDims << std::endl;
        }
        else
        {
            std::cout << "output node" << i - 1 << " name: "<< PtrEngine_->getBindingName(i) << ", dims: " << dims.nbDims << std::endl;
        }

        for (int j = 0; j < dims.nbDims; j++) 
        {
            std::cout << "demension[" << j << "], size = " << dims.d[j] << std::endl;
        }
        std::cout << "TotalSize = " << TotalSize << std::endl;
    }

    cudaMalloc(&GpuOutputCount_, sizeof(int));
    cudaMalloc(&GpuOutputRects_, sizeof(DetectRect) * NmsBeforeMaxNum_);
    
    cudaMalloc(&GpuSrcImage_, 810 * 1080 * 3 * sizeof(unsigned char));  // 输入图像的尺寸,通常输出图像尺寸是一致的
    
    CpuOutputCount_ = (int *)malloc(sizeof(int));
    CpuOutputRects_ = (DetectRect *)malloc(sizeof(DetectRect) * NmsBeforeMaxNum_);
}


void CNN::Inference(cv::Mat &SrcImage)
{
    DetectiontRects_.clear();
    if(PtrContext_ == nullptr)
    {
        std::cout << "Error, PtrContext_" << std::endl;
    }
    
    PrepareImage(SrcImage, Buffers_[0], Stream_);

    PtrContext_->enqueueV2(Buffers_, Stream_, nullptr);

    cudaMemsetAsync(GpuOutputCount_, 0, 4, Stream_);
    GetNmsBeforeBoxes((float*)Buffers_[1], Postprocess_.CoordIndex, Postprocess_.ClassNum, Postprocess_.ObjectThresh, NmsBeforeMaxNum_, GpuOutputRects_, GpuOutputCount_, Stream_);

    cudaMemcpyAsync(CpuOutputCount_, GpuOutputCount_, sizeof(int), cudaMemcpyDeviceToHost, Stream_);
    cudaMemcpyAsync(CpuOutputRects_, GpuOutputRects_, sizeof(DetectRect) * NmsBeforeMaxNum_, cudaMemcpyDeviceToHost, Stream_);
    
    cudaStreamSynchronize(Stream_);

    // Postprocess
    int ret = Postprocess_.GetConvDetectionResult(CpuOutputRects_, CpuOutputCount_, DetectiontRects_);
}

void CNN::PrepareImage(cv::Mat &SrcImage, void *InputBuffer, cudaStream_t Stream)
{
    cudaError_t err = cudaMemcpy(GpuSrcImage_, (void*)SrcImage.data, SrcImage.cols * SrcImage.rows * 3 * sizeof(char), cudaMemcpyHostToDevice);
    NearestNeighborResizeNormCHW((float*)InputBuffer, InputImageWidth_, InputImageHeight_, GpuSrcImage_, SrcImage.cols, SrcImage.rows, 0.0039215, Stream);     
}