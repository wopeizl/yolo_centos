#pragma once
//#include "network.h"
#include "image.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//extern network;
class yolo_wrapper {
public:
    yolo_wrapper(int gpu_id, char *datacfg, char *cfgfile, char *weightfile, char* name_list, float _thresh, float _hier_thresh);
    ~yolo_wrapper() {}

    static bool prepare(const cv::Mat &img, image* output);
    bool predict(image* input, cv::Mat& output);
    static bool postprocess(const cv::Mat &img, image* output);

private:
    //extern network net;
    float w, h, thresh, hier_thresh;
    image **alphabet;
    char **names;
};
