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


### Visualization of pre-trained curves
Please use the jupyter notebook "visualization.ipynb" to visualize the training and test curves of different optimizers. We provide logs for pre-trained models (9 optimizers x 3 models = 27 pre-trained curves) in the folder "curve".



### Training and evaluation code

(1) train network with
CUDA_VISIBLE_DEVICES=0 python main.py --optim sgdf --lr 0.5 --eps 1e-8 --beta1 0.9 --beta2 0.999 

--optim: name of optimizers, choices include ['sgdf', 'sgd', 'adam', 'radam', 'adamw', 'msvag', 'adabound', 'sophia', 'lion', 'adabelief']
--lr: learning rate
--eps: epsilon value used for optimizers. Note that Yogi uses a default of 1e-03, other optimizers typically uses 1e-08
--beta1, --beta2: beta values in adaptive optimizers
--momentum: momentum used for SGD.s

(2) visualize using the notebook "visualization.ipynb"



### Running time
On a single GTX 2080Ti GPU, training a ResNet typically takes 100-120 minutes for a single optimzer. To run all experiments would take 5 hours for 9 optimizers all models = 45 hours
