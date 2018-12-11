# pix2pix-pytorch

PyTorch implementation of [Image-to-Image Translation Using Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004v1.pdf).

Based on [pix2pix](https://phillipi.github.io/pix2pix/) by [Phillip Isola](https://github.com/phillipi) et al.

The examples from the paper: 

<img src="examples.jpg" width = "766" height = "282" alt="examples" align=center />

## Prerequisites

+ Linux
+ Python, Numpy, PIL
+ pytorch 0.4.0
+ torchvision 0.2.1

## Getting Started

+ Clone this repo:

    git clone git@github.com:mrzhu-cool/pix2pix-pytorch.git
    cd pix2pix-pytorch

+ Get dataset

    unzip dataset/facades.zip

+ Train the model:

    python train.py --dataset facades --cuda

+ Test the model:

    python test.py --dataset facades --cuda

## Acknowledgments

This code is a simple implementation of [pix2pix](https://phillipi.github.io/pix2pix/). Easier to understand. Note that we use a downsampling-resblocks-upsampling structure instead of the unet structure in this code, therefore the results of this code may inconsistent with the results presented in the paper.

Highly recommend the more sophisticated and organized code [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) by [Jun-Yan Zhu](https://github.com/junyanz).
