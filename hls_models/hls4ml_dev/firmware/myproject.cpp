//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t global_in[N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1],
    result_t layer43_out[N_LAYER_43]
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=global_in complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer43_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=global_in,layer43_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight44_t, 400>(w44, "w44.txt");
        nnet::load_weights_from_txt<bias44_t, 16>(b44, "b44.txt");
        nnet::load_weights_from_txt<weight45_t, 12800>(w45, "w45.txt");
        nnet::load_weights_from_txt<bias45_t, 32>(b45, "b45.txt");
        nnet::load_weights_from_txt<weight43_t, 15680>(w43, "w43.txt");
        nnet::load_weights_from_txt<bias43_t, 10>(b43, "b43.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer32_t layer32_out[OUT_DEPTH_32*OUT_HEIGHT_32*OUT_WIDTH_32];
    #pragma HLS ARRAY_PARTITION variable=layer32_out complete dim=0
    nnet::transpose_3d<input_t, layer32_t, config32>(global_in, layer32_out); // Transpose_0

    layer44_t layer44_out[OUT_HEIGHT_44*OUT_WIDTH_44*N_FILT_44];
    #pragma HLS ARRAY_PARTITION variable=layer44_out complete dim=0
    nnet::conv_2d_cl<layer32_t, layer44_t, config44>(layer32_out, layer44_out, w44, b44); // Conv2D_Conv_0

    layer34_t layer34_out[OUT_HEIGHT_33*OUT_WIDTH_33*N_FILT_33];
    #pragma HLS ARRAY_PARTITION variable=layer34_out complete dim=0
    nnet::relu<layer44_t, layer34_t, ReLU_config34>(layer44_out, layer34_out); // Relu_0

    layer35_t layer35_out[OUT_HEIGHT_35*OUT_WIDTH_35*N_FILT_35];
    #pragma HLS ARRAY_PARTITION variable=layer35_out complete dim=0
    nnet::pooling2d_cl<layer34_t, layer35_t, config35>(layer34_out, layer35_out); // MaxPool_0

    layer45_t layer45_out[OUT_HEIGHT_45*OUT_WIDTH_45*N_FILT_45];
    #pragma HLS ARRAY_PARTITION variable=layer45_out complete dim=0
    nnet::conv_2d_cl<layer35_t, layer45_t, config45>(layer35_out, layer45_out, w45, b45); // Conv2D_Conv_1

    layer37_t layer37_out[OUT_HEIGHT_36*OUT_WIDTH_36*N_FILT_36];
    #pragma HLS ARRAY_PARTITION variable=layer37_out complete dim=0
    nnet::relu<layer45_t, layer37_t, ReLU_config37>(layer45_out, layer37_out); // Relu_1

    layer38_t layer38_out[OUT_HEIGHT_38*OUT_WIDTH_38*N_FILT_38];
    #pragma HLS ARRAY_PARTITION variable=layer38_out complete dim=0
    nnet::pooling2d_cl<layer37_t, layer38_t, config38>(layer37_out, layer38_out); // MaxPool_1

    auto& layer39_out = layer38_out;
    nnet::dense<layer38_t, result_t, config43>(layer39_out, layer43_out, w43, b43); // Dense_MatMul_0

}
