
#include "yolo_wrapper.hpp"
#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "option_list.h"
#include "blas.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

//extern "C" list *read_data_cfg(char *filename);

network net;

yolo_wrapper::yolo_wrapper(int gpu_id, char *datacfg, char *cfgfile, char *weightfile, char* name_listfile, float _thresh, float _hier_thresh) {
    if (gpu_index >= 0) {
        cuda_set_device(gpu_index);
    }

    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", name_listfile);
    names = get_labels(name_list);

    alphabet = load_alphabet();
    net = parse_network_cfg(cfgfile);

    thresh = _thresh;
    hier_thresh = _hier_thresh;

    load_weights(&net, weightfile);

    set_batch_network(&net, 1);
    srand(2222222);
}

bool yolo_wrapper::prepare(const cv::Mat &img, image* output) {
    unsigned char *data = (unsigned char *)img.data;
    int h = img.rows;
    int w = img.cols;
    int c = img.channels();
    int step = img.step;
    image im  = make_image(w, h, c);
    int i, j, k, count = 0;;

    for (k = 0; k < c; ++k) {
        for (i = 0; i < h; ++i) {
            for (j = 0; j < w; ++j) {
                im.data[count++] = data[i*step + j*c + k] / 255.;
            }
        }
    }

    rgbgr_image(im);

    image out = resize_image(im, 0, 0);
    free_image(im);
    *output = letterbox_image(out, w, h);

    return true;
}


bool yolo_wrapper::predict(image* input, cv::Mat& output) {
    image& im = *input;
    float nms = .4;

    /*float* out = */network_predict(net, input->data);

    layer l = net.layers[net.n - 1];
    box *boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for (int j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes + 1, sizeof(float *));

    get_region_boxes(l, im.w, im.h, net.w, net.h, thresh, probs, boxes, 0, 0, hier_thresh, 1);
    if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);

    draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);

    return true;
}

bool yolo_wrapper::postprocess(const cv::Mat &img, image* output) {

    return true;
}
