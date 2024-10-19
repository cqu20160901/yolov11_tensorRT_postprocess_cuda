#include "get_nms_before_boxes.cuh"


__global__ void GetNmsBeforeBoxesKernel(float *SrcInput, int AnchorCount, int ClassNum, float ObjectThresh, int NmsBeforeMaxNum, DetectRect* OutputRects, int *OutputCount)
{
    int ThreadId = blockIdx.x * blockDim.x + threadIdx.x;
   
    if (ThreadId >= AnchorCount)
    {
        return;
    }

    float* XywhConf = SrcInput + ThreadId;
    float CenterX = 0, CenterY = 0, CenterW = 0, CenterH = 0;

    float MaxScore = 0;
    int MaxIndex = 0;

    DetectRect TempRect;
    for (int j = 4; j < ClassNum + 4; j ++) 
    {
        if (4 == j)
        {
            MaxScore = XywhConf[j * AnchorCount];
            MaxIndex = j;   
        } 
        else 
        {
            if (MaxScore <  XywhConf[j * AnchorCount])
            {
                MaxScore = XywhConf[j * AnchorCount];
                MaxIndex = j;   
            }
        }  
    }

    if (MaxScore > ObjectThresh)
    {
        int index = atomicAdd(OutputCount, 1);
    
        if (index > NmsBeforeMaxNum)
        {
            return;
        }

        CenterX = XywhConf[0 * AnchorCount];
        CenterY = XywhConf[1 * AnchorCount];
        CenterW = XywhConf[2 * AnchorCount];
        CenterH = XywhConf[3 * AnchorCount ];

        TempRect.classId = MaxIndex - 4;
        TempRect.score = MaxScore;
        TempRect.xmin = CenterX - 0.5 * CenterW;
        TempRect.ymin = CenterY - 0.5 * CenterH;
        TempRect.xmax = CenterX + 0.5 * CenterW;
        TempRect.ymax = CenterY + 0.5 * CenterH;

        OutputRects[index] = TempRect;
    }
}


void GetNmsBeforeBoxes(float *SrcInput, int AnchorCount, int ClassNum, float ObjectThresh, int NmsBeforeMaxNum, DetectRect* OutputRects, int *OutputCount, cudaStream_t Stream)
{
    int Block = 512;
    int Grid = (AnchorCount + Block - 1) / Block;

    GetNmsBeforeBoxesKernel<<<Grid, Block, 0, Stream>>>(SrcInput, AnchorCount, ClassNum, ObjectThresh, NmsBeforeMaxNum, OutputRects, OutputCount);
    return;
}


