#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
using namespace cv;
using namespace std;

Mat letterbox(Mat &img, Size new_shape, Scalar color, bool _auto, bool scaleFill, bool scaleup, int stride)
{
    float width = img.cols;
    float height = img.rows;
    float r = min(new_shape.width / width, new_shape.height / height);
    if (!scaleup)
        r = min(r, 1.0f);
    int new_unpadW = int(round(width * r));
    int new_unpadH = int(round(height * r));
    int dw = new_shape.width - new_unpadW;
    int dh = new_shape.height - new_unpadH;
    if (_auto)
    {
        dw %= stride;
        dh %= stride;
    }
    dw /= 2, dh /= 2;
    Mat dst;
    resize(img, dst, Size(new_unpadW, new_unpadH), 0, 0, INTER_LINEAR);
    int top = int(round(dh - 0.1));
    int bottom = int(round(dh + 0.1));
    int left = int(round(dw - 0.1));
    int right = int(round(dw + 0.1));
    copyMakeBorder(dst, dst, top, bottom, left, right, BORDER_CONSTANT, color);
    return dst;
}

int main(int argc, char *argv[])
{
    string model_path = argv[1];
    dnn::Net model = dnn::readNetFromONNX(model_path);
    Mat img0 = imread("data/images/bus.jpg");
    Mat img = letterbox(img0, Size(640, 640), Scalar(114, 114, 114), true, false, true, 32);
    Mat blob;    
    dnn::blobFromImage(img, blob, 1 / 255.0f, Size(img.cols, img.rows), Scalar(0, 0, 0), true, false);
    model.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    model.setPreferableTarget(dnn::DNN_TARGET_CPU);
    model.setInput(blob);
    vector<string> outLayerNames = model.getUnconnectedOutLayersNames();
    vector<Mat> result;
    model.forward(result, outLayerNames);

    Mat out = Mat(result[0].size[1], result[0].size[2], CV_32F, result[0].ptr<float>());
    vector<Rect> boxes;
    vector<int> indices;
    vector<float> scores;
    for (int r = 0; r < out.size[0]; r++)
    {
        float cx = out.at<float>(r, 0);
        float cy = out.at<float>(r, 1);
        float w = out.at<float>(r, 2);
        float h = out.at<float>(r, 3);
        float sc = out.at<float>(r, 4);
        Mat confs = out.row(r).colRange(5,85);
        confs*=sc;
        double minV=0,maxV=0;
        double *minI=&minV;
        double *maxI=&maxV;
        minMaxIdx(confs,minI,maxI);
        scores.push_back(maxV);
        boxes.push_back(Rect(cx - w / 2, cy - h / 2, w, h));
        indices.push_back(r);
    }
    dnn::NMSBoxes(boxes, scores, 0.25f, 0.45f, indices);
    for (auto &ind : indices)
    {
        cout << indices[ind] << ":" << scores[ind] << endl;
    }
}
