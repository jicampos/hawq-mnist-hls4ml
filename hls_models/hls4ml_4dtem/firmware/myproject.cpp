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
    result_t layer79_out[N_LAYER_64]
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=global_in complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer79_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=global_in,layer79_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<model_default_t, 784>(scale68, "scale68.txt");
        nnet::load_weights_from_txt<model_default_t, 784>(bias68, "bias68.txt");
        nnet::load_weights_from_txt<model_default_t, 784>(s80, "s80.txt");
        nnet::load_weights_from_txt<model_default_t, 784>(b80, "b80.txt");
        nnet::load_weights_from_txt<weight84_t, 400>(w84, "w84.txt");
        nnet::load_weights_from_txt<bias84_t, 16>(b84, "b84.txt");
        nnet::load_weights_from_txt<model_default_t, 12544>(scale76, "scale76.txt");
        nnet::load_weights_from_txt<model_default_t, 12544>(bias76, "bias76.txt");
        nnet::load_weights_from_txt<model_default_t, 12544>(scale71, "scale71.txt");
        nnet::load_weights_from_txt<model_default_t, 12544>(bias71, "bias71.txt");
        nnet::load_weights_from_txt<model_default_t, 12544>(scale72, "scale72.txt");
        nnet::load_weights_from_txt<model_default_t, 12544>(bias72, "bias72.txt");
        nnet::load_weights_from_txt<model_default_t, 3136>(scale81, "scale81.txt");
        nnet::load_weights_from_txt<model_default_t, 3136>(bias81, "bias81.txt");
        nnet::load_weights_from_txt<weight85_t, 12800>(w85, "w85.txt");
        nnet::load_weights_from_txt<bias85_t, 32>(b85, "b85.txt");
        nnet::load_weights_from_txt<model_default_t, 6272>(scale77, "scale77.txt");
        nnet::load_weights_from_txt<model_default_t, 6272>(bias77, "bias77.txt");
        nnet::load_weights_from_txt<model_default_t, 6272>(scale74, "scale74.txt");
        nnet::load_weights_from_txt<model_default_t, 6272>(bias74, "bias74.txt");
        nnet::load_weights_from_txt<model_default_t, 6272>(scale75, "scale75.txt");
        nnet::load_weights_from_txt<model_default_t, 6272>(bias75, "bias75.txt");
        nnet::load_weights_from_txt<model_default_t, 1568>(scale82, "scale82.txt");
        nnet::load_weights_from_txt<model_default_t, 1568>(bias82, "bias82.txt");
        nnet::load_weights_from_txt<weight83_t, 15680>(w83, "w83.txt");
        nnet::load_weights_from_txt<bias83_t, 10>(b83, "b83.txt");
        nnet::load_weights_from_txt<model_default_t, 10>(scale79, "scale79.txt");
        nnet::load_weights_from_txt<model_default_t, 10>(bias79, "bias79.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer47_t layer47_out[OUT_DEPTH_47*OUT_HEIGHT_47*OUT_WIDTH_47];
    #pragma HLS ARRAY_PARTITION variable=layer47_out complete dim=0
    nnet::transpose_3d<input_t, layer47_t, config47>(global_in, layer47_out); // Transpose_0

    layer68_t layer68_out[OUT_DEPTH_47*OUT_HEIGHT_47*OUT_WIDTH_47];
    #pragma HLS ARRAY_PARTITION variable=layer68_out complete dim=0
    layer80_t layer80_out[OUT_DEPTH_47*OUT_HEIGHT_47*OUT_WIDTH_47];
    #pragma HLS ARRAY_PARTITION variable=layer80_out complete dim=0
    nnet::normalize<layer68_t, layer80_t, config80>(layer68_out, layer80_out, s80, b80); // bn_Div_0

    layer84_t layer84_out[OUT_HEIGHT_84*OUT_WIDTH_84*N_FILT_84];
    #pragma HLS ARRAY_PARTITION variable=layer84_out complete dim=0
    nnet::conv_2d_cl<layer80_t, layer84_t, config84>(layer80_out, layer84_out, w84, b84); // Conv2D_Conv_0

    layer76_t layer76_out[OUT_HEIGHT_50*OUT_WIDTH_50*N_FILT_50];
    #pragma HLS ARRAY_PARTITION variable=layer76_out complete dim=0
    nnet::normalize<layer84_t, layer76_t, config76>(layer84_out, layer76_out, scale76, bias76); // bn_Mul_0

    layer52_t layer52_out[OUT_HEIGHT_50*OUT_WIDTH_50*N_FILT_50];
    #pragma HLS ARRAY_PARTITION variable=layer52_out complete dim=0
    nnet::relu<layer76_t, layer52_t, ReLU_config52>(layer76_out, layer52_out); // Relu_0

    layer71_t layer71_out[OUT_HEIGHT_50*OUT_WIDTH_50*N_FILT_50];
    #pragma HLS ARRAY_PARTITION variable=layer71_out complete dim=0
    layer72_t layer72_out[OUT_HEIGHT_50*OUT_WIDTH_50*N_FILT_50];
    #pragma HLS ARRAY_PARTITION variable=layer72_out complete dim=0
    layer54_t layer54_out[OUT_HEIGHT_54*OUT_WIDTH_54*N_FILT_54];
    #pragma HLS ARRAY_PARTITION variable=layer54_out complete dim=0
    nnet::pooling2d_cl<layer72_t, layer54_t, config54>(layer72_out, layer54_out); // MaxPool_0

    layer81_t layer81_out[OUT_HEIGHT_54*OUT_WIDTH_54*N_FILT_54];
    #pragma HLS ARRAY_PARTITION variable=layer81_out complete dim=0
    nnet::normalize<layer54_t, layer81_t, config81>(layer54_out, layer81_out, scale81, bias81); // bn_Div_1

    layer85_t layer85_out[OUT_HEIGHT_85*OUT_WIDTH_85*N_FILT_85];
    #pragma HLS ARRAY_PARTITION variable=layer85_out complete dim=0
    nnet::conv_2d_cl<layer81_t, layer85_t, config85>(layer81_out, layer85_out, w85, b85); // Conv2D_Conv_1

    layer77_t layer77_out[OUT_HEIGHT_56*OUT_WIDTH_56*N_FILT_56];
    #pragma HLS ARRAY_PARTITION variable=layer77_out complete dim=0
    nnet::normalize<layer85_t, layer77_t, config77>(layer85_out, layer77_out, scale77, bias77); // bn_Mul_1

    layer58_t layer58_out[OUT_HEIGHT_56*OUT_WIDTH_56*N_FILT_56];
    #pragma HLS ARRAY_PARTITION variable=layer58_out complete dim=0
    nnet::relu<layer77_t, layer58_t, ReLU_config58>(layer77_out, layer58_out); // Relu_1

    layer74_t layer74_out[OUT_HEIGHT_56*OUT_WIDTH_56*N_FILT_56];
    #pragma HLS ARRAY_PARTITION variable=layer74_out complete dim=0
    layer75_t layer75_out[OUT_HEIGHT_56*OUT_WIDTH_56*N_FILT_56];
    #pragma HLS ARRAY_PARTITION variable=layer75_out complete dim=0
    layer60_t layer60_out[OUT_HEIGHT_60*OUT_WIDTH_60*N_FILT_60];
    #pragma HLS ARRAY_PARTITION variable=layer60_out complete dim=0
    nnet::pooling2d_cl<layer75_t, layer60_t, config60>(layer75_out, layer60_out); // MaxPool_1

    layer61_t layer61_out[OUT_DEPTH_61*OUT_HEIGHT_61*OUT_WIDTH_61];
    #pragma HLS ARRAY_PARTITION variable=layer61_out complete dim=0
    nnet::transpose_3d<layer60_t, layer61_t, config61>(layer60_out, layer61_out); // Transpose_1

    auto& layer62_out = layer61_out;
    layer82_t layer82_out[N_SIZE_0_62];
    #pragma HLS ARRAY_PARTITION variable=layer82_out complete dim=0
    nnet::normalize<layer61_t, layer82_t, config82>(layer62_out, layer82_out, scale82, bias82); // bn_Div_2

    layer83_t layer83_out[N_LAYER_83];
    #pragma HLS ARRAY_PARTITION variable=layer83_out complete dim=0
    nnet::dense<layer82_t, layer83_t, config83>(layer82_out, layer83_out, w83, b83); // Dense_MatMul_0

    nnet::normalize<layer83_t, result_t, config79>(layer83_out, layer79_out, scale79, bias79); // bn_Mul_2

}
