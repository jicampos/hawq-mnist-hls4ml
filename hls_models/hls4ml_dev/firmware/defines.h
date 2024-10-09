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
#define OUT_DEPTH_32 28
#define OUT_HEIGHT_32 28
#define OUT_WIDTH_32 1
#define OUT_HEIGHT_44 28
#define OUT_WIDTH_44 28
#define N_FILT_44 16
#define OUT_HEIGHT_33 28
#define OUT_WIDTH_33 28
#define N_FILT_33 16
#define OUT_HEIGHT_35 14
#define OUT_WIDTH_35 14
#define N_FILT_35 16
#define OUT_HEIGHT_45 14
#define OUT_WIDTH_45 14
#define N_FILT_45 32
#define OUT_HEIGHT_36 14
#define OUT_WIDTH_36 14
#define N_FILT_36 32
#define OUT_HEIGHT_38 7
#define OUT_WIDTH_38 7
#define N_FILT_38 32
#define N_SIZE_0_39 1568
#define N_LAYER_43 10

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> layer32_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> layer44_t;
typedef ap_fixed<8,8,AP_RND_CONV,AP_SAT> weight44_t;
typedef ap_fixed<16,16,AP_RND_CONV,AP_SAT> bias44_t;
typedef ap_fixed<16,6> layer34_t;
typedef ap_fixed<18,8> Relu_0_table_t;
typedef ap_fixed<16,6> layer35_t;
typedef ap_fixed<16,6> layer45_t;
typedef ap_fixed<8,8,AP_RND_CONV,AP_SAT> weight45_t;
typedef ap_fixed<16,16,AP_RND_CONV,AP_SAT> bias45_t;
typedef ap_fixed<16,6> layer37_t;
typedef ap_fixed<18,8> Relu_1_table_t;
typedef ap_fixed<16,6> layer38_t;
typedef ap_fixed<8,8,AP_RND_CONV,AP_SAT> weight43_t;
typedef ap_fixed<16,16,AP_RND_CONV,AP_SAT> bias43_t;
typedef ap_fixed<16,6> result_t;
typedef ap_uint<1> layer43_index;

#endif
