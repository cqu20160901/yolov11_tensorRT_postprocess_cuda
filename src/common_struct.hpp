#ifndef COMMON_STRUCT_HPP
#define COMMON_STRUCT_HPP

struct DetectRect
{
    float classId;
    float score;
    float xmin;
    float ymin;
    float xmax;
    float ymax;
};

#endif
