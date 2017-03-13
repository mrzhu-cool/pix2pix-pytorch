# pix2pix-pytorch

PyTorch implementation of [Image-to-Image Translation Using Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004v1.pdf).

Based on [pix2pix](https://phillipi.github.io/pix2pix/) by Isola et al.

The examples from the paper: 

<img src="result/examples.jpg" width = "766" height = "282" alt="examples" align=center />

## Prerequisites

+ Linux
+ Python with numpy
+ NVIDIA GPU + CUDA 8.0 + CuDNNv5.1
+ pytorch
+ torchvision

## Getting Started

+ Clone this repo:

    git clone git@github.com:mrzhu-cool/pix2pix-pytorch.git
    cd pix2pix-pytorch

+ Get dataset

    unzip dataset/facades.zip

+ Train the model:

    python train.py --dataset facades --nEpochs 200 --cuda

+ Test the model:

    python test.py --dataset facades --model checkpoint/facades/netG_model_epoch_200.pth --cuda

## Acknowledgments

This is a rough code implementation in order to practice pytorch and there may be some 
incorrect implementation in the code. 

So the effect of this implementation is not as good as the [original one](https://github.com/phillipi/pix2pix). 

Therefore, I really hope that someone can give valuable suggestions.