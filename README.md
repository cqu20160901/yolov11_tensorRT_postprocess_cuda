# yolov11 tensorRT 的 C++ 部署，后处理用cuda实现的

本示例中，包含完整的代码、模型、测试图片、测试结果。

后处理部分用cuda 核函数实现，并不是全部后处理都用cuda实现；cpu实现后处理部分[【参考链接】](https://github.com/cqu20160901/yolov11_tensorRT_postprocess_cuda/tree/yolov11_postprocess_cpu)

TensorRT版本：TensorRT-8.6.1.6

## 导出onnx模型

按照yolov11官方导出的方式如下：

```python
from ultralytics import YOLO
model = YOLO(model='yolov11n.pt')  # load a pretrained model (recommended for training)
results = model(task='detect', source=r'./bus.jpg', save=True)  # predict on an image

model.export(format="onnx", imgsz=640, simplify=True)

```

## 编译

修改 CMakeLists.txt 对应的TensorRT位置

![image](https://github.com/user-attachments/assets/ac92b3d7-855a-40ac-9b5f-a3fabd262634)


```powershell
cd yolov11_tensorRT_postprocess_cuda
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

![image](https://github.com/user-attachments/assets/da904ce0-4e0c-414e-9339-39dca4747328)

tensorRT 测试效果

![image](https://github.com/cqu20160901/yolov11_tensorRT_postprocess_cuda/blob/main/images/result.jpg)

tensorRT 时耗

示例中用cpu对图像进行预处理、用rtx4090显卡、模型yolov11n（输入分辨率640x640，80个类别）、量化成FP16模型

![image](https://github.com/user-attachments/assets/f3fd18e8-9f7f-4a2e-8a3e-132a47583a09)



## 替换模型说明

修改相关的路径
```cpp

int main()
{
    std::string OnnxFile = "/root/autodl-tmp/yolov11_tensorRT_postprocess_cuda/models/yolov11n.onnx";
    std::string SaveTrtFilePath = "/root/autodl-tmp/yolov11_tensorRT_postprocess_cuda/models/yolov11n.trt";
    cv::Mat SrcImage = cv::imread("/root/autodl-tmp/yolov11_tensorRT_postprocess_cuda/images/test.jpg");

    int img_width = SrcImage.cols;
    int img_height = SrcImage.rows;
    std::cout << "img_width: " << img_width << " img_height: " << img_height << std::endl;

    CNN YOLO(OnnxFile, SaveTrtFilePath, 1, 3, 640, 640);
    
    auto t_start = std::chrono::high_resolution_clock::now();
    int Temp = 2000;
    
    int SleepTimes = 0;
    for (int i = 0; i < Temp; i++)
    {
        YOLO.Inference(SrcImage);
        std::this_thread::sleep_for(std::chrono::milliseconds(SleepTimes));
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "Info: " << Temp << " times infer and postprocess ave cost: " << total_inf / float(Temp) - SleepTimes << " ms." << std::endl;


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

    imwrite("/root/autodl-tmp/yolov11_tensorRT_postprocess_cuda/images/result.jpg", SrcImage);

    printf("== obj: %d \n", int(float(YOLO.DetectiontRects_.size()) / 6.0));

    return 0;
}

```
