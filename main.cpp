#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include  "detector.h"
using namespace cv;
using namespace std;
/* main */


int main(int argc, char *argv[])
{
    // 默认参数
    string model_path = argv[1];
    string img_path = argv[2];
    //string img_path = "data/images/bus.jpg";
    //string model_path = "/Users/mix/yolov5/yolov5s.onnx";
    Config config = {0.25f, 0.45f, model_path, "data/coco.names"};
    cout<<"Load Model"<<endl;
    Detector detector(config);
    cout << "Read Image" << endl;
    Mat img = imread(img_path, IMREAD_COLOR);
    auto detection = detector.detect(img);
    detector.letterBoxImage(img);
    detector.postProcess(img,detection);
    imwrite("assets/output.jpg",img);
    cout << "detect Image And Save to assets/output.jpg" << endl;
    return 0;
}
