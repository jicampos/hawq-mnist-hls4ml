#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 1
#define N_INPUT_2_1 28
#define N_INPUT_3_1 28
#define OUT_DEPTH_47 28
#define OUT_HEIGHT_47 28
#define OUT_WIDTH_47 1
#define OUT_HEIGHT_84 28
#define OUT_WIDTH_84 28
#define N_FILT_84 16
#define OUT_HEIGHT_50 28
#define OUT_WIDTH_50 28
#define N_FILT_50 16
#define OUT_HEIGHT_54 14
#define OUT_WIDTH_54 14
#define N_FILT_54 16
#define OUT_HEIGHT_85 14
#define OUT_WIDTH_85 14
#define N_FILT_85 32
#define OUT_HEIGHT_56 14
#define OUT_WIDTH_56 14
#define N_FILT_56 32
#define OUT_HEIGHT_60 7
#define OUT_WIDTH_60 7
#define N_FILT_60 32
#define OUT_DEPTH_61 32
#define OUT_HEIGHT_61 7
#define OUT_WIDTH_61 7
#define N_SIZE_0_62 1568
#define N_LAYER_83 10
#define N_LAYER_64 10

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> layer47_t;
typedef ap_fixed<12,12,AP_RND_CONV,AP_SAT> layer68_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> layer80_t;
typedef ap_fixed<16,6> layer84_t;
typedef ap_fixed<8,8,AP_RND_CONV,AP_SAT> weight84_t;
typedef ap_fixed<16,16,AP_RND_CONV,AP_SAT> bias84_t;
typedef ap_fixed<16,6> layer76_t;
typedef ap_fixed<16,6> layer52_t;
typedef ap_fixed<18,8> Relu_0_table_t;
typedef ap_fixed<8,8,AP_RND_CONV,AP_SAT> layer71_t;
typedef ap_fixed<16,6> layer72_t;
typedef ap_fixed<16,6> layer54_t;
typedef ap_fixed<16,6> layer81_t;
typedef ap_fixed<16,6> layer85_t;
typedef ap_fixed<8,8,AP_RND_CONV,AP_SAT> weight85_t;
typedef ap_fixed<16,16,AP_RND_CONV,AP_SAT> bias85_t;
typedef ap_fixed<16,6> layer77_t;
typedef ap_fixed<16,6> layer58_t;
typedef ap_fixed<18,8> Relu_1_table_t;
typedef ap_fixed<8,8,AP_RND_CONV,AP_SAT> layer74_t;
typedef ap_fixed<16,6> layer75_t;
typedef ap_fixed<16,6> layer60_t;
typedef ap_fixed<16,6> layer61_t;
typedef ap_fixed<16,6> layer82_t;
typedef ap_fixed<8,8,AP_RND_CONV,AP_SAT> weight83_t;
typedef ap_fixed<16,16,AP_RND_CONV,AP_SAT> bias83_t;
typedef ap_fixed<16,6> layer83_t;
typedef ap_uint<1> layer83_index;
typedef ap_fixed<16,6> result_t;

#endif
