### Signal Processing Meets SGD: From Momentum to Filter

This repository contains code to reproduce results: "Signal Processing Meets SGD: From Momentum to Filter".

This repo heavily depends on the official implementation of AdaBlief: https://github.com/juntang-zhuang/Adabelief-Optimizer
Code for optimizers in the literature are forked from public implementations, please see the comments in corresponding files in the folder "optimizers".

### Dependencies
python 3.10
pytorch 2.0.1
torchvision 0.15.2
CUDA 11.8
Ubuntu20.04
jupyter notebook
AdaBound  (Please instal by "pip install adabound")
AdaBlief  (Please instal by "pip install adabelief_pytorch")


### Training and evaluation code

CUDA_VISIBLE_DEVICES=0 python main.py --optimizer sgdf --eps 1e-8 --Train --dataset cifar10 

--optim: name of optimizers, choices include ['sgdf', 'sgd', 'adam', 'adamw', 'yogi', 'msvag', 'radam', 'fromage', 'adabound', 'rmsprop']
--lr: learning rate
--eps: epsilon value used for optimizers

The code will automatically generate a folder containing model weights, a csv file containing the FID score, and a separate folder containing 64,000 fake images


### Running time
On a single GTX 2080Ti GPU, training a one round takes 1 hours for a single optimzer. To run all experiments would take 1 hours x 9 optimizers x 5 repeats = 45 hours
