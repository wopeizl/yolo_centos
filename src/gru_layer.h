
#ifndef GRU_LAYER_H
#define GRU_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

layer make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize);

void forward_gru_layer(layer l, network net);
void backward_gru_layer(layer l, network net);
void update_gru_layer(layer l, int batch, float learning_rate, float momentum, float decay);

#ifdef YOLO_GPU
void forward_gru_layer_gpu(layer l, network net);
void backward_gru_layer_gpu(layer l, network net);
void update_gru_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay);
void push_gru_layer(layer l);
void pull_gru_layer(layer l);
#endif

#endif

