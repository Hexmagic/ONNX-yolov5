## 介绍

使用ONNX部署yolov5模型，这里挑选ultralytics公司的[Yolov5](https://github.com/ultralytics/yolov5.git)提供的预训练模型。


## 环境
1. GCC需要支持C++11标准
2. Opencv 4.5.1

## 准备工作

0. 我们需要到[release页面](https://github.com/ultralytics/yolov5/releases)下载yolov5的预训练模型，例如选择yolv5s.pt

1. 然后使用`trace.py`转换成scrptmodel:

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
python3.7 export.py --data data/coco128.yaml --weights yolov5s.pt --include onnx
```

我们会得到一个yolov5s.onnx


## 编译代码:
```bash
git clone https://github.com/Hexmagic/ONNX_yolov5.git
mkdir build&&cd build
cmake ..
make -j4
cd ..
./build/main yolov5s.onnx data/images/zidane.jpg
```


默认输出会在assets/output.jpg
<center>
<img src="assets/output.jpg">
</center>

