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
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_batchnorm_stream.h"
#include "nnet_utils/nnet_conv2d.h"
#include "nnet_utils/nnet_conv2d_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_pooling_stream.h"
#include "nnet_utils/nnet_stream.h"
 
//hls-fpga-machine-learning insert weights
#include "weights/scale68.h"
#include "weights/bias68.h"
#include "weights/s80.h"
#include "weights/b80.h"
#include "weights/w84.h"
#include "weights/b84.h"
#include "weights/scale76.h"
#include "weights/bias76.h"
#include "weights/scale71.h"
#include "weights/bias71.h"
#include "weights/scale72.h"
#include "weights/bias72.h"
#include "weights/scale81.h"
#include "weights/bias81.h"
#include "weights/w85.h"
#include "weights/b85.h"
#include "weights/scale77.h"
#include "weights/bias77.h"
#include "weights/scale74.h"
#include "weights/bias74.h"
#include "weights/scale75.h"
#include "weights/bias75.h"
#include "weights/scale82.h"
#include "weights/bias82.h"
#include "weights/w83.h"
#include "weights/b83.h"
#include "weights/scale79.h"
#include "weights/bias79.h"

//hls-fpga-machine-learning insert layer-config
// Transpose_0
struct config47 : nnet::transpose_config {
    static const unsigned depth = 1;
    static const unsigned height = 28;
    static const unsigned width = 28;
    static constexpr unsigned perm[3] = {1,2,0};
};

// bn_Div_0
struct config80 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_DEPTH_47*OUT_HEIGHT_47*OUT_WIDTH_47;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 32;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// Conv2D_Conv_0
struct config84_mult : nnet::dense_config {
    static const unsigned n_in = 25;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 32;
    static const unsigned strategy = nnet::latency;
    typedef model_default_t accum_t;
    typedef bias84_t bias_t;
    typedef weight84_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config84 : nnet::conv2d_config {
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
    static const unsigned reuse_factor = 32;
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
    using fill_buffer = nnet::fill_buffer_84<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef bias84_t bias_t;
    typedef weight84_t weight_t;
    typedef config84_mult mult_config;
};
const ap_uint<config84::filt_height * config84::filt_width> config84::pixels[] = {0};

// bn_Mul_0
struct config76 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_84*OUT_WIDTH_84*N_FILT_84;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 32;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// Relu_0
struct ReLU_config52 : nnet::activ_config {
    static const unsigned n_in = 12544;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 32;
    typedef Relu_0_table_t table_t;
};

// MaxPool_0
struct config54 : nnet::pooling2d_config {
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
    static const unsigned reuse_factor = 32;
    typedef model_default_t accum_t;
};

// bn_Div_1
struct config81 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_54*OUT_WIDTH_54*N_FILT_54;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 32;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// Conv2D_Conv_1
struct config85_mult : nnet::dense_config {
    static const unsigned n_in = 400;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 25;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias85_t bias_t;
    typedef weight85_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config85 : nnet::conv2d_config {
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
    static const unsigned reuse_factor = 25;
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
    using fill_buffer = nnet::fill_buffer_85<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef bias85_t bias_t;
    typedef weight85_t weight_t;
    typedef config85_mult mult_config;
};
const ap_uint<config85::filt_height * config85::filt_width> config85::pixels[] = {0};

// bn_Mul_1
struct config77 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_85*OUT_WIDTH_85*N_FILT_85;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 32;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// Relu_1
struct ReLU_config58 : nnet::activ_config {
    static const unsigned n_in = 6272;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 32;
    typedef Relu_1_table_t table_t;
};

// MaxPool_1
struct config60 : nnet::pooling2d_config {
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
    static const unsigned reuse_factor = 32;
    typedef model_default_t accum_t;
};

// Transpose_1
struct config61 : nnet::transpose_config {
    static const unsigned depth = 7;
    static const unsigned height = 7;
    static const unsigned width = 32;
    static constexpr unsigned perm[3] = {2,0,1};
};

// bn_Div_2
struct config82 : nnet::batchnorm_config {
    static const unsigned n_in = N_SIZE_0_62;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 32;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// Dense_MatMul_0
struct config83 : nnet::dense_config {
    static const unsigned n_in = 1568;
    static const unsigned n_out = 10;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 32;
    static const unsigned n_zeros = 230;
    static const unsigned n_nonzeros = 15450;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias83_t bias_t;
    typedef weight83_t weight_t;
    typedef layer83_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// bn_Mul_2
struct config79 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_83;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 32;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};


#endif
