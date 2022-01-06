#include "detector.h"

Detector::Detector(Config &config)
{
    this->nmsThreshold = config.nmsThreshold;
    this->confThreshold = config.confThreshold;
    ifstream ifs(config.classNamePath);
    string line;
    while (getline(ifs, line))
        this->classNames.push_back(line);
    ifs.close();
    this->model = dnn::readNetFromONNX(config.weightPath);
    this->model.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    this->model.setPreferableTarget(dnn::DNN_TARGET_CPU);
}

PadInfo Detector::letterBoxImage(Mat &image)
{
    float row = image.rows * 1.0f;
    float col = image.cols * 1.0f;
    float scale = max(row / this->inHeight, col / this->inWidth);
    int dst_col = col / scale;
    int dst_row = row / scale;
    resize(image, image, Size(dst_col, dst_row));
    int left = (this->inWidth - dst_col) / 2;
    int top = (this->inHeight - dst_row) / 2;
    int right = (this->inWidth - dst_col + 1) / 2;
    int bottom = (this->inHeight - dst_row + 1) / 2;
    cv::copyMakeBorder(image, image, top, bottom, left, right, BORDER_CONSTANT, Scalar(114, 114, 114));
    return {scale, top, left};
}

Detection Detector::detect(Mat &img)
{
    // 预处理 添加border
    Mat im;
    img.copyTo(im);
    cout << im.size() << endl;
    PadInfo padInfo = letterBoxImage(im);
    Mat blob;
    dnn::blobFromImage(im, blob, 1 / 255.0f, Size(this->inWidth, this->inHeight), Scalar(0, 0, 0), true, false);
    std::vector<string> outLayerNames = this->model.getUnconnectedOutLayersNames();
    std::vector<Mat> outs;
    this->model.setInput(blob);
    this->model.forward(outs,outLayerNames);
    return {padInfo, outs};
}
void Detector::postProcess(Mat &img, Detection &detection)
{

    std::vector<Mat> outs = detection.detection;

    cout<<outs[0].size[1]<<":"<<outs[0].size[2]<<endl;
    Mat out(outs[0].size[1], outs[0].size[2],CV_32F,outs[0].ptr<float>());
    std::vector<Rect> boxes;
    std::vector<float> scores;
    std::vector<int> indices;
    std::vector<int> classIndexList;
    for (int r = 0;  r< out.rows;r++){
        float cx = out.at<float>(r, 0);
        float cy = out.at<float>(r, 1);
        float w = out.at<float>(r, 2);
        float h = out.at<float>(r, 3);
        float sc = out.at<float>(r, 4);
        scores.push_back(sc);
        boxes.push_back(Rect(cx-w/2, cy-h/2, w, h));
        indices.push_back(r);
        int maxI = 0;
        float maxV = out.at<float>(r, 5);
        for (int i = 1; i < 80; i++)
        {
            if(out.at<float>(r, i+5)>maxV){
                maxV = out.at<float>(r, i + 5);
                maxI = i;
            }
        }
        classIndexList.push_back(maxI);
    }

    dnn::NMSBoxes(boxes, scores,this->confThreshold,this->nmsThreshold,indices);
    std::vector<string> rstNames;
    for(int i=0;i<indices.size();i++){
        string name = this->classNames[classIndexList[indices[i]]];
        rstNames.push_back(name);
    }
    drawPredection(img,boxes,scores,rstNames,indices);
}

void Detector::drawPredection(Mat &img, std::vector<Rect> &boxes, std::vector<float> &scores, std::vector<string> &clsNames, std::vector<int> &ind)
{
    for (int i = 0; i < ind.size(); i++)
    {
        Rect rect = boxes[ind[i]];
        float score = scores[ind[i]];
        string name = clsNames[i];
        rectangle(img, rect, Scalar(0, 0, 255));
        cout << name << endl;
        putText(img, name, Point(rect.x, rect.y), FONT_HERSHEY_PLAIN, 1.2, Scalar(0, 255, 0));
    }
    imshow("rst", img);
    waitKey(0);
}
