# OpenVINO-YOLOv8-Seg

OpenVINO™ 实现YOLOv8-Seg实例分割模型推理

开发环境是 Windows + Visual Studio Community 2022。

本文的码云代码仓：

git clone https://gitee.com/ppov-nuc/yolov8_openvino_cpp.git

02. 导出 YOLOv8-Seg OpenVINO™ IR 模型

YOLOv8 是 Ultralytics 公司基于 YOLO 框架，发布的一款面向物体检测与跟踪、实例分割、图像分类和姿态估计任务的 SOTA 模型工具套件。

首先用命令 ：

pip install -r requirements.txt

安装  ultralytics 和 openvino-dev 。

然后使用命令：

yolo export model=yolov8n-seg.pt format=openvino half=True

导出 FP16 精度的 OpenVINO™ IR 模型，如下图所示。

![image](https://github.com/wangzhenlin123/OpenVINO-YOLOv8-Seg/assets/51401216/d2c9ef54-b8a4-41ba-9f39-ac5b0df45300)


接着使用命令：

benchmark_app -m yolov8n-seg.xml -d GPU.1

03. 使用 OpenVINO™ C++ API 

编写 YOLOv8-Seg 实例分割模型推理程序使用 OpenVINO™ C++ API 编写 YOLOv8-Seg 实例分割模型推理程序主要有5个典型步骤：

1采集图像&图像解码

2图像数据预处理

3AI 推理计算(基于 OpenVINO™ C++ API )

4对推理结果进行后处理

5将处理后的结果可视化

YOLOv8-Seg 实例分割模型推理程序的图像数据预处理和AI推理计算的实现方式跟 YOLOv8 目标检测模型推理程序的实现方式几乎一模一样，可以直接复用。

3.1 图像数据预处理
使用 Netron 打开 yolov8n-seg.onnx 

![image](https://github.com/wangzhenlin123/OpenVINO-YOLOv8-Seg/assets/51401216/32c131ca-001f-4619-b680-a1dc3b274e5a)

输入节点的名字：“ images ”；数据：float32[1,3,640,640]
输出节点1的名字：“ output0 ”；数据：float32[1,116,8400]。其中116的前84个字段跟  YOLOv8 目标检测模型输出定义完全一致，即cx,cy,w,h 和80类的分数；后32个字段为掩膜置信度，用于计算掩膜数据。
输出节点2的名字：“ output1 ”；数据：float32[1,32,160,160]。output0 后32个字段与 output1 的数据做矩阵乘法后得到的结果，即为对应目标的掩膜数据
图像数据预处理的目标就是将任意尺寸的图像数据转变为形状为[1,3,640,640]，精度为 FP32 的张量。YOLOv8-Seg 模型的输入尺寸为正方形，为了解决将任意尺寸数据放缩为正方形带来的图像失真问题，在图像放缩前，采用 letterbox 算法先保持图像的长宽比，如下图所示，然后再使用 cv::dnn::blobFromImage 函数对图像进行放缩。

3.2 AI 同步推理计算
用 OpenVINO™ C++ API 实现同步推理计算，主要有七步：

1实例化 Core 对象：ov::Core core;

2编译并载入模型：core.compile_model();

3创建推理请求：infer_request = compiled_model.create_infer_request()；

4读取图像数据并完成预处理；

5将输入数据传入模型：infer_request.set_input_tensor(input_tensor);

6启动推理计算：infer_request.infer();

7获得推理结果：output0 = infer_request.get_output_tensor(0) ; 

output1 = infer_request.get_output_tensor(1) ;


3.3 推理结果后处理
实例分割推理程序的后处理是从结果中拆解出预测别类（class_id），类别分数（class_score），类别边界框（box）和类别掩膜（mask）。

在英特尔® 独立显卡 A770m 上获得了较好的推理计算性能。

