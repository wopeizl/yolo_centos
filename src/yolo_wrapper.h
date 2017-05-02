#pragma once

void yolo_init_net(int gpu_id, const char *cfgfile
    , const char *weightfile
    , const char* namelist
    , const char* labeldir
    , float thresh
    , float hier_thresh);

void yolo_predict(void* input, void* output);
