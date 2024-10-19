# yolov8 tensorRT 的 C++ 部署

本示例中，包含完整的代码、模型、测试图片、测试结果。

TensorRT版本：TensorRT-7.1.3.4

## 导出onnx模型

导出适配本实例的onnx模型参考[【yolov8 导出onnx-2023年11月15日版本】](https://blog.csdn.net/zhangqian_1/article/details/134438275)。


## 编译

修改 CMakeLists.txt 对应的TensorRT位置

![17012439652981](https://github.com/cqu20160901/yolov8_tensorRT_Cplusplus/assets/22290931/68586322-556a-42ef-8444-1064b42c86f9)

```powershell
cd yolov8_tensorRT_Cplusplus
mkdir build
cd build
cmake ..
make
```

## 运行

```powershell
# 运行时如果.trt模型存在则直接加载，若不存会自动先将onnx转换成 trt 模型，并存在给定的位置，然后运行推理。
cd build
./yolo_trt
```

## 测试效果

onnx 测试效果

![image](https://github.com/cqu20160901/yolov8_tensorRT_Cplusplus/assets/22290931/8574c0ce-fc56-4b3c-9c7e-ec31e29b01ed)

tensorRT 测试效果

![result](https://github.com/cqu20160901/yolov8_tensorRT_Cplusplus/assets/22290931/29a8115d-a5ce-4c58-9b1a-c48766cdfcd5)

tensorRT 时耗

![image](https://github.com/user-attachments/assets/6490069e-4d8a-48f3-88e7-58eb70ae3abe)



## 替换模型说明

1）按照本实例给的导出onnx方式导出对应的onnx；导出的onnx模型建议simplify后再转trt模型。

2）注意修改后处理相关 postprocess.hpp 中相关的参数（类别、输入分辨率等）。

修改相关的路径
```cpp
    std::string OnnxFile = "/zhangqian/workspaces1/TensorRT/yolov8_trt_Cplusplus/models/yolov8n_ZQ.onnx";
    std::string SaveTrtFilePath = "/zhangqian/workspaces1/TensorRT/yolov8_trt_Cplusplus/models/yolov8n_ZQ.trt";
    cv::Mat SrcImage = cv::imread("/zhangqian/workspaces1/TensorRT/yolov8_trt_Cplusplus/images/test.jpg");

    int img_width = SrcImage.cols;
    int img_height = SrcImage.rows;

    CNN YOLO(OnnxFile, SaveTrtFilePath, 1, 3, 640, 640);
    YOLO.Inference(SrcImage);

    for (int i = 0; i < YOLO.DetectiontRects_.size(); i += 6)
    {
        int classId = int(YOLO.DetectiontRects_[i + 0]);
        float conf = YOLO.DetectiontRects_[i + 1];
        int xmin = int(YOLO.DetectiontRects_[i + 2] * float(img_width) + 0.5);
        int ymin = int(YOLO.DetectiontRects_[i + 3] * float(img_height) + 0.5);
        int xmax = int(YOLO.DetectiontRects_[i + 4] * float(img_width) + 0.5);
        int ymax = int(YOLO.DetectiontRects_[i + 5] * float(img_height) + 0.5);

        char text1[256];
        sprintf(text1, "%d:%.2f", classId, conf);
        rectangle(SrcImage, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(255, 0, 0), 2);
        putText(SrcImage, text1, cv::Point(xmin, ymin + 15), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    }

    imwrite("/zhangqian/workspaces1/TensorRT/yolov8_trt_Cplusplus/images/result.jpg", SrcImage);

    printf("== obj: %d \n", int(float(YOLO.DetectiontRects_.size()) / 6.0));

```

## 特别说明

本示例只是用来测试流程，模型效果并不保证，且代码整理的布局合理性没有做过多的考虑。

## 相关链接

[yolov8 瑞芯微 RKNN 的 C++部署](https://github.com/cqu20160901/yolov8n_onnx_tensorRT_rknn_horizon)

[yolov8 瑞芯微RKNN和地平线Horizon芯片仿真测试部署-2023年11月15日版本](https://blog.csdn.net/zhangqian_1/article/details/134438275)

[yolov8 瑞芯微RKNN和地平线Horizon芯片仿真测试部署](https://blog.csdn.net/zhangqian_1/article/details/128918268)


## 2024-10-06

### 1）预处理优化

原来为：用 opencv 进行预处理(resize-转rgb-转float-减均值除方差) 

修改为：用 cuda 提供的 nppi 库进行预处理(resize-转rgb-转float-减均值除方差)

    优化效果：10FPS情况下 CPU 占用减少 62%
