#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "loguru.hpp"
#include "detector.h"
using namespace cv;
using namespace std;
/* main */

int main(int argc, char *argv[])
{
    // 默认参数
    string model_path = argv[1];
    string img_path = argv[2];
    //string model_path = "3_best.onnx";
    //string img_path = "data/images/zidane.jpg";
    loguru::init(argc, argv);
    Config config = {0.25f, 0.45f, model_path, "data/coco.names", Size(640, 640),false};
    LOG_F(INFO,"Start main process");
    Detector detector(config);
    LOG_F(INFO,"Load model done ..");
    Mat img = imread(img_path, IMREAD_COLOR);
    LOG_F(INFO,"Read image from %s", img_path.c_str());
    Detection detection = detector.detect(img);
    LOG_F(INFO,"Detect process finished");
    Colors cl = Colors();
    detector.postProcess(img, detection,cl);
    LOG_F(INFO,"Post process done save image to assets/output.jpg");
    imwrite("assets/output.jpg", img);
    cout << "detect Image And Save to assets/output.jpg" << endl;
    return 0;
}
