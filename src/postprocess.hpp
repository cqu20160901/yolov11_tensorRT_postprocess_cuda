#ifndef _POSTPROCESS_H_
#define _POSTPROCESS_H_

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <vector>

typedef struct
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    int classId;
} DetectRect;

// yolov11
class GetResultRectYolov11
{
public:
    GetResultRectYolov11();

    ~GetResultRectYolov11();

    int GetConvDetectionResult(std::vector<float *> &BlobPtr, std::vector<float> &DetectiontRects);

private:
    const int ClassNum = 80;

    int InputW = 640;
    int InputH = 640;

    int MapSize[3][2] = {{80, 80}, {40, 40}, {20, 20}};
    int CoordIndex = 0;

    float NmsThresh = 0.45;
    float ObjectThresh = 0.5;
};

#endif
