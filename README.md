# hawq-mnist-hls4ml

This project implements a small CNN in HAWQ for quantization aware training, and then exports to QONNX for hls4ml ingestion.

## Setup

Clone repo and HAWQ submodule

```bash
git clone --recursive https://github.com/jicampos/hawq-mnist-hls4ml.git
```

## Model Summary

```bash
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
CNN                                      [1, 10]                   --
├─Conv2d: 1-1                            [1, 16, 28, 28]           416
├─ReLU: 1-2                              [1, 16, 28, 28]           --
├─MaxPool2d: 1-3                         [1, 16, 14, 14]           --
├─Conv2d: 1-4                            [1, 32, 14, 14]           12,832
├─ReLU: 1-5                              [1, 32, 14, 14]           --
├─MaxPool2d: 1-6                         [1, 32, 7, 7]             --
├─Flatten: 1-7                           [1, 1568]                 --
├─Linear: 1-8                            [1, 10]                   15,690
==========================================================================================
Total params: 28,938
Trainable params: 28,938
Non-trainable params: 0
Total mult-adds (M): 2.86
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.15
Params size (MB): 0.12
Estimated Total Size (MB): 0.27
==========================================================================================
```

## Relevant Links

HAWQ [https://github.com/Zhen-Dong/HAWQ]

QONNX [https://github.com/fastmachinelearning/qonnx]

Original PyTorch Implementation [https://github.com/julesmuhizi/pytorch_hls4ml]
