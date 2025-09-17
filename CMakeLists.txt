cmake_minimum_required(VERSION 3.5)

project(yolo_trt)

set(CMAKE_CXX_STANDARD 14)

# CUDA
find_package(CUDA REQUIRED)

message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")

# TensorRT
set(TENSORRT_ROOT /zhangqian/code/autodl-tmp/TensorRT-8.6.1.6)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include/)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNXPARSER})
message(STATUS "Find TensorRT libs: ${TENSORRT_LIBRARY}")

# OpenCV
find_package(OpenCV REQUIRED)

message(STATUS "Find OpenCV include at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Find OpenCV libraries: ${OpenCV_LIBRARIES}")

set(COMMON_INCLUDE ./common)
set(KERNELS_INCLUDE ./kernels)

include_directories(${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${COMMON_INCLUDE} ${KERNELS_INCLUDE})

# set(CMAKE_CUDA_ARCHITECTURES 86)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CUDA_GEN_CODE "-gencode=arch=compute_89,code=sm_89")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --default-stream=per-thread --threads 0  -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")

enable_language(CUDA)

SET(CMAKE_PREFIX_PATH "/usr/local/cuda")

add_executable(yolo_trt main.cpp src/CNN.cpp src/postprocess_cuda.cpp src/kernels/get_nms_before_boxes.cu src/kernels/image_preprocess.cu)

target_link_libraries(yolo_trt ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY})


# export LD_LIBRARY_PATH=/zhangqian/code/autodl-tmp/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH