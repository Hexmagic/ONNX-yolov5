## Introdution

Deploy ultralytics [Yolov5](https://github.com/ultralytics/yolov5.git) pretained model with C++ language ;

<div align="center">
<img src="assets/output.jpg">
</div>




## Env

1. GCC 7.5
2. Opencv 4.5.4

## Get ONNX Model 

1. go to  yolov5 [release page](https://github.com/ultralytics/yolov5/releases) download yolov5 pretrained model（official onnx can't work right)，such as yolv5s.pt

2. use`trace.py` convert yolov5s.pt to yolov5.onnx:

    ```bash
    git clone https://github.com/ultralytics/yolov5.git
    cd yolov5
    python3.7 export.py --data data/coco128.yaml --weights yolov5s.pt --include onnx
    ```


## Build 

```bash
git clone https://github.com/Hexmagic/ONNX_yolov5.git
mkdir build&&cd build
cmake ..
make -j4
cd ..
./build/main yolov5s.onnx data/images/zidane.jpg
```



