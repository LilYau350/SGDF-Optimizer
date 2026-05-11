<h1 align="center">Dynamic Momentum Recalibration in Online Gradient Learning</h1>

</h1>
  <p align="center">
    <a>Zhipeng Yao</a><sup>1,2</sup> •
    <a href="https://ruiyu0.github.io/Rui Yu">Rui Yu</a><sup>2</sup> •
    <a>Guisong Chang</a> <sup>3</sup> •
    <a>Ying Li</a><sup>1</sup> •
    <a href="https://scholar.google.com/citations?hl=zh-CN&user=tVpADYIAAAAJ&view_op=list_works&sortby=pubdate">Yu Zhang</a><sup>1</sup> •
    <a href="https://scholar.google.com/citations?user=SdS4SdgAAAAJ&hl=zh-CN&oi=sra">Dazhou Li</a><sup>1</sup>
  </p>
</h1>

<p align="center">
  <sup>1</sup> Shenyang University of Chemical Technology &nbsp;&nbsp;
  <sup>2</sup> University of Louisville &nbsp;&nbsp;
  <sup>3</sup> Northeastern University &nbsp;&nbsp;
</p>

<h4 align="center">Official repository of the paper</h4>

<h4 align="center">CVPR 2026</h4>

<p align="center">
  <img src="https://img.shields.io/badge/arXiv-2603.06120-b31b1b.svg" alt="arXiv"> <img src="https://img.shields.io/badge/PyTorch-%3E%3D1.10-red.svg" alt="PyTorch">
</p>  
:pushpin:Abstract, *Stochastic Gradient Descent (SGD) and its momentum variants form the backbone of deep learning optimization, yet the underlying dynamics of their gradient behavior remain insufficiently understood. In this work, we reinterpret gradient updates through the lens of signal processing and reveal that fixed momentum coefficients inherently distort the balance between bias and variance, leading to skewed or suboptimal parameter updates. To address this, we propose SGDF (SGD with Filter), an optimizer inspired by the principles of Optimal Linear Filtering. SGDF computes an online, time-varying gain to dynamically refine gradient estimation by minimizing the mean-squared error, thereby achieving an optimal trade-off between noise suppression and signal preservation. Furthermore, our approach could extend to other optimizers, showcasing its broad applicability to optimization frameworks. Extensive experiments across diverse architectures and benchmarks demonstrate SGDF surpasses conventional momentum methods and achieves performance on par with or surpassing state-of-the-art optimizers*  

## :mortar_board: Related GitHub Repositories
* Some of the experimental code in our paper was borrowed from the following repositories, thanks to these authors for open source.
<br> https://github.com/tomgoldstein/loss-landscape
<br> https://github.com/amirgholami/PyHessian
<br> https://github.com/juntang-zhuang/Adabelief-Optimizer
<br> https://github.com/jeonsworld/ViT-pytorch

## :bookmark_tabs: Table of Hyper-parameters 
### :wrench:Hyper-parameters in PyTorch
* :warning:Note weight decay varies with tasks, for different tasks the weight decay is untuned from the original repository (only changed the optimizer and other hyper-parameters).

|           Task            | lr   | beta1 | beta2 | epsilon | weight_decay |
| :-----------------------: | ---- | ----- | ----- | ------- | ------------ |
|           Cifar           | 5e-1 | 0.9   | 0.999 | 1e-8    | 5e-4         |
|         ImageNet          | 5e-1 | 0.5   | 0.999 | 1e-8    | 1e-4         |
|            ViT            | 5e-1 | 0.9   | 0.999 | 1e-8    | 0            |
|           WGAN            | 1e-2 | 0.5   | 0.999 | 1e-8    | 0            |
| Object detection (PASCAL) | 1e-2 | 0.9   | 0.999 | 1e-8    | 1e-4         |

## :book: Citation
If you find this repository useful, please consider giving a star :star: and citing our work :page_facing_up::)

```bibtex
@article{yao2026dynamic,
  title={Dynamic Momentum Recalibration in Online Gradient Learning},
  author={Yao, Zhipeng and Yu, Rui and Chang, Guisong and Li, Ying and Zhang, Yu and Li, Dazhou},
  journal={arXiv preprint arXiv:2603.06120},
  year={2026}
}
```
