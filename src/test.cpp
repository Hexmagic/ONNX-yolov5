#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
int main(){
    Mat img = imread("data/images/bus.jpg");
    imshow("bus", img);
    waitKey(0);
}
