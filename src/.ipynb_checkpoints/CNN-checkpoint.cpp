#include "CNN.hpp"
#include "common/common.hpp"
#include <algorithm>
#include <chrono>


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
        if (i >= 1)
        {
            delete OutputData_[i];
        }
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

    if (nullptr != GpuSrcImgBuf_)
    {
        cudaFree(GpuSrcImgBuf_);
    }

    if (nullptr != GpuImgResizeBuf_)
    {
        cudaFree(GpuImgResizeBuf_);
    }

    if (nullptr != GpuImgF32Buf_)
    {
        cudaFree(GpuImgF32Buf_);
    }

    if (nullptr != GpuDataPlanes_)
    {
        cudaFree(GpuDataPlanes_);
    }

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
    OutputData_.resize(nbBindings - 1);
    for (int i = 0; i < nbBindings; ++ i)
    {
        nvinfer1::Dims dims = PtrEngine_->getBindingDimensions(i);
        nvinfer1::DataType dtype = PtrEngine_->getBindingDataType(i);
        TotalSize = Volume(dims) * 1 * GetElementSize(dtype);

        BuffersDataSize_[i] = TotalSize;
        cudaMalloc(&Buffers_[i], TotalSize);
        if (i >= 1)
        {
            OutputData_[i - 1] = new float[int(TotalSize / sizeof(float))];
        }

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
    }

    PreprocessResult_.resize(BatchSize_ * InputImageWidth_ * InputImageHeight_ * InputChannel_);
}


void CNN::Inference(cv::Mat &SrcImage)
{
    DetectiontRects_.clear();
    if(PtrContext_ == nullptr)
    {
        std::cout << "Error, PtrContext_" << std::endl;
    }

    PrepareImage(SrcImage, PreprocessResult_);
    cudaMemcpyAsync(Buffers_[0], PreprocessResult_.data(), BuffersDataSize_[0], cudaMemcpyHostToDevice, Stream_);
    // PrepareImage(SrcImage, Buffers_[0]);

    PtrContext_->enqueueV2(Buffers_, Stream_, nullptr);

    for (int i = 1; i < BuffersDataSize_.size(); i++)
    {
        cudaMemcpyAsync(OutputData_[i - 1], Buffers_[i], BuffersDataSize_[i], cudaMemcpyDeviceToHost, Stream_);
    }

    cudaStreamSynchronize(Stream_);
    
    // Postprocess
    static GetResultRectYolov8 Postprocess;
    int ret = Postprocess.GetConvDetectionResult(OutputData_, DetectiontRects_);
}

void CNN::PrepareImage(cv::Mat &SrcImage, std::vector<float> &PreprocessResult)
{
    float *Imagedata = PreprocessResult.data();

    cv::Mat rsz_img;
    cv::resize(SrcImage, rsz_img, cv::Size(InputImageWidth_, InputImageHeight_));
    rsz_img.convertTo(rsz_img, CV_32FC3, 1.0 / 255);

    // HWC TO CHW
    int channelLength = InputImageWidth_ * InputImageHeight_;
    std::vector<cv::Mat> split_img = {cv::Mat(InputImageHeight_, InputImageWidth_, CV_32FC1, Imagedata + channelLength * 2),
                                      cv::Mat(InputImageHeight_, InputImageWidth_, CV_32FC1, Imagedata + channelLength * 1),
                                      cv::Mat(InputImageHeight_, InputImageWidth_, CV_32FC1, Imagedata + channelLength * 0)};

    cv::split(rsz_img, split_img);
}


void CNN::PrepareImage(cv::Mat &SrcImage, void *InputBuffer)
{
#if 0
    int src_width = SrcImage.cols;
    int src_height = SrcImage.rows;
    int src_channel = SrcImage.channels();

    NppiSize dstSize = {InputImageWidth_, InputImageHeight_};
    NppiRect dstROI = {0, 0, InputImageWidth_, InputImageHeight_};
    if (GpuImgResizeBuf_ == nullptr)
    {
        cudaMalloc(&GpuImgResizeBuf_, InputImageWidth_ * InputImageHeight_ * src_channel * sizeof(uchar));
        cudaMalloc(&GpuImgF32Buf_, InputImageWidth_ * InputImageHeight_ * src_channel * sizeof(float));
        cudaMalloc(&GpuDataPlanes_, InputImageWidth_ * InputImageHeight_ * src_channel * sizeof(float));
    }

 
    NppiSize srcSize = {src_width, src_height};
    NppiRect srcROI = {0, 0, src_width, src_height};
    if(GpuSrcImgBuf_ == nullptr)
    {
        cudaMalloc(&GpuSrcImgBuf_, src_width * src_height * src_channel * sizeof(uchar));   
    }

    DstPlanes_[0] = GpuDataPlanes_;
    DstPlanes_[1] = GpuDataPlanes_ + InputImageWidth_ * InputImageHeight_;
    DstPlanes_[2] = GpuDataPlanes_ + InputImageWidth_ * InputImageHeight_ * 2;

    // 将cpu图像拷贝到gpu
    cudaMemcpy(GpuSrcImgBuf_, (void *)SrcImage.data, src_width * src_height * src_channel, cudaMemcpyHostToDevice);
    
    // resize 
    nppiResize_8u_C3R(GpuSrcImgBuf_, src_width * src_channel, srcSize, srcROI, GpuImgResizeBuf_, InputImageWidth_ * src_channel, dstSize, dstROI, NPPI_INTER_LINEAR);
    
    // bgr 转 rgb
    nppiSwapChannels_8u_C3IR(GpuImgResizeBuf_, InputImageWidth_ * src_channel, dstSize, DstOrder_);
    
    // int8(uchar) 转 f32
    nppiConvert_8u32f_C3R(GpuImgResizeBuf_, InputImageWidth_ * src_channel, GpuImgF32Buf_, InputImageWidth_ * src_channel * sizeof(float), dstSize);
   
    // 减均值、除方差
    nppiMulC_32f_C3IR(MeanScale_, GpuImgF32Buf_, InputImageWidth_ * src_channel * sizeof(float), dstSize);

    nppiCopy_32f_C3P3R(GpuImgF32Buf_, InputImageWidth_ * src_channel * sizeof(float), DstPlanes_, InputImageWidth_ * sizeof(float), dstSize);
 
    cudaMemcpyAsync(InputBuffer, GpuDataPlanes_, src_channel * InputImageWidth_ * InputImageHeight_ * sizeof(float), cudaMemcpyDeviceToDevice, Stream_);
#endif
}
