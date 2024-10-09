#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
#include "nnet_utils/nnet_code_gen.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_array.h"
#include "nnet_utils/nnet_conv2d.h"
#include "nnet_utils/nnet_conv2d_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_pooling_stream.h"
#include "nnet_utils/nnet_stream.h"
 
//hls-fpga-machine-learning insert weights
#include "weights/w44.h"
#include "weights/b44.h"
#include "weights/w45.h"
#include "weights/b45.h"
#include "weights/w43.h"
#include "weights/b43.h"

//hls-fpga-machine-learning insert layer-config
// Transpose_0
struct config32 : nnet::transpose_config {
    static const unsigned depth = 1;
    static const unsigned height = 28;
    static const unsigned width = 28;
    static constexpr unsigned perm[3] = {1,2,0};
};

// Conv2D_Conv_0
struct config44_mult : nnet::dense_config {
    static const unsigned n_in = 25;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 1024;
    static const unsigned strategy = nnet::latency;
    typedef model_default_t accum_t;
    typedef bias44_t bias_t;
    typedef weight44_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config44 : nnet::conv2d_config {
    static const unsigned pad_top = 2;
    static const unsigned pad_bottom = 2;
    static const unsigned pad_left = 2;
    static const unsigned pad_right = 2;
    static const unsigned in_height = 28;
    static const unsigned in_width = 28;
    static const unsigned n_chan = 1;
    static const unsigned filt_height = 5;
    static const unsigned filt_width = 5;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 16;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 28;
    static const unsigned out_width = 28;
    static const unsigned reuse_factor = 1024;
    static const unsigned n_zeros = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 28;
    static const unsigned min_width = 28;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 784;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_44<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef bias44_t bias_t;
    typedef weight44_t weight_t;
    typedef config44_mult mult_config;
};
const ap_uint<config44::filt_height * config44::filt_width> config44::pixels[] = {0};

// Relu_0
struct ReLU_config34 : nnet::activ_config {
    static const unsigned n_in = 12544;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1024;
    typedef Relu_0_table_t table_t;
};

// MaxPool_0
struct config35 : nnet::pooling2d_config {
    static const unsigned in_height = 28;
    static const unsigned in_width = 28;
    static const unsigned n_filt = 16;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = pool_height;
    static const unsigned filt_width = pool_width;
    static const unsigned n_chan = n_filt;

    static const unsigned out_height = 14;
    static const unsigned out_width = 14;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse_factor = 1024;
    typedef model_default_t accum_t;
};

// Conv2D_Conv_1
struct config45_mult : nnet::dense_config {
    static const unsigned n_in = 400;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 800;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias45_t bias_t;
    typedef weight45_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config45 : nnet::conv2d_config {
    static const unsigned pad_top = 2;
    static const unsigned pad_bottom = 2;
    static const unsigned pad_left = 2;
    static const unsigned pad_right = 2;
    static const unsigned in_height = 14;
    static const unsigned in_width = 14;
    static const unsigned n_chan = 16;
    static const unsigned filt_height = 5;
    static const unsigned filt_width = 5;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 32;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 14;
    static const unsigned out_width = 14;
    static const unsigned reuse_factor = 800;
    static const unsigned n_zeros = 86;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 14;
    static const unsigned min_width = 14;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 196;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_45<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef bias45_t bias_t;
    typedef weight45_t weight_t;
    typedef config45_mult mult_config;
};
const ap_uint<config45::filt_height * config45::filt_width> config45::pixels[] = {0};

// Relu_1
struct ReLU_config37 : nnet::activ_config {
    static const unsigned n_in = 6272;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1024;
    typedef Relu_1_table_t table_t;
};

// MaxPool_1
struct config38 : nnet::pooling2d_config {
    static const unsigned in_height = 14;
    static const unsigned in_width = 14;
    static const unsigned n_filt = 32;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = pool_height;
    static const unsigned filt_width = pool_width;
    static const unsigned n_chan = n_filt;

    static const unsigned out_height = 7;
    static const unsigned out_width = 7;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse_factor = 1024;
    typedef model_default_t accum_t;
};

// Dense_MatMul_0
struct config43 : nnet::dense_config {
    static const unsigned n_in = 1568;
    static const unsigned n_out = 10;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 784;
    static const unsigned n_zeros = 230;
    static const unsigned n_nonzeros = 15450;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias43_t bias_t;
    typedef weight43_t weight_t;
    typedef layer43_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};


#endif
