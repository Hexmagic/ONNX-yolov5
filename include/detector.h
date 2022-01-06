#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>


using namespace cv;
using namespace std;

struct PadInfo{
    float scale;
    int top;
    int left;
};

struct Detection{
    PadInfo info;
    std::vector<Mat> detection;
};

struct Config
{
    float confThreshold;
    float nmsThreshold;
    string weightPath;
    string classNamePath;
};

class Detector{
    public:
        Detector(Config&config);
        Detection detect(Mat&img);
        void postProcess(Mat&img,Detection&detection);
        PadInfo letterBoxImage(Mat &img);
    private:
        float nmsThreshold = 0.45;
        float confThreshold = 0.25;
        int inWidth = 640;
        int inHeight = 640;
        vector<string> classNames;
        dnn::Net model;
        void drawPredection(Mat&img,vector<Rect>&boxes,vector<float>&sc,vector<string>&clsNames, vector<int>&ind);
};
