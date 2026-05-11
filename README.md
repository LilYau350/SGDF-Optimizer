<h1 align="center">Dynamic Momentum Recalibration in Online Gradient Learning</h1>

<p align="center">
  <a>Zhipeng Yao</a><sup>1,2</sup> •
  <a href="https://ruiyu0.github.io/Rui-Yu">Rui Yu</a><sup>2</sup> •
  <a>Guisong Chang</a><sup>3</sup> •
  <a>Ying Li</a><sup>1</sup> •
  <a href="https://scholar.google.com/citations?hl=zh-CN&user=tVpADYIAAAAJ&view_op=list_works&sortby=pubdate">Yu Zhang</a><sup>1</sup> •
  <a href="https://scholar.google.com/citations?user=SdS4SdgAAAAJ&hl=zh-CN&oi=sra">Dazhou Li</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup> Shenyang University of Chemical Technology &nbsp;&nbsp;
  <sup>2</sup> University of Louisville &nbsp;&nbsp;
  <sup>3</sup> Northeastern University
</p>

<h4 align="center">Official repository of the paper</h4>

<h4 align="center">CVPR 2026</h4>

<p align="center">
  <a href="https://arxiv.org/abs/2603.06120"><img src="https://img.shields.io/badge/arXiv-2603.06120-b31b1b.svg" alt="arXiv"></a>
  <a href="https://pytorch.org/get-started/locally/"><img src="https://img.shields.io/badge/PyTorch-2.0.0%2B-red.svg" alt="PyTorch"></a>
</p>


## 📌 Abstract

*Stochastic Gradient Descent (SGD) and its momentum variants form the backbone of deep learning optimization, yet the underlying dynamics of their gradient behavior remain insufficiently understood. In this work, we reinterpret gradient updates through the lens of signal processing and reveal that fixed momentum coefficients inherently distort the balance between bias and variance, leading to skewed or suboptimal parameter updates. To address this, we propose SGDF (SGD with Filter), an optimizer inspired by the principles of Optimal Linear Filtering. SGDF computes an online, time-varying gain to dynamically refine gradient estimation by minimizing the mean-squared error, thereby achieving an optimal trade-off between noise suppression and signal preservation. Furthermore, our approach can be extended to other optimizers, showcasing its broad applicability to optimization frameworks. Extensive experiments across diverse architectures and benchmarks demonstrate that SGDF surpasses conventional momentum methods and achieves performance on par with or beyond state-of-the-art optimizers.*


## 🏆 Main Results

SGDF achieves consistent improvements across image classification, object detection, and ViT post-training benchmarks.

| Benchmark | Model / Task | Baseline | SGDF |
| :-------- | :----------- | :------: | :--: |
| ImageNet Classification | ResNet18 Top-1 Acc. (%) | SGD: 70.23 | **70.51 ± 0.05** |
| ImageNet Classification | ResNet18 Top-5 Acc. (%) | SGD: 89.40 | **89.69 ± 0.16** |
| ImageNet Classification | ResNet50 Top-1 Acc. (%) | SGD: 76.13 | **76.72 ± 0.09** |
| ImageNet Classification | DenseNet161 Top-1 Acc. (%) | SGD: 77.13 | **78.34 ± 0.08** |
| Object Detection | Faster R-CNN on PASCAL VOC mAP (%) | SGD: 80.43 | **83.81** |
| ViT Post-training | ViT-B/32 on CIFAR-100 Top-1 Acc. (%) | SGD: 90.62 ± 0.07 | **91.44 ± 0.13** |
| ViT Post-training | ViT-B/32 on Oxford-IIIT-Pets Top-1 Acc. (%) | SGD: 89.71 ± 0.32 | **92.68 ± 0.04** |
| ViT Post-training | ViT-L/32 on CIFAR-100 Top-1 Acc. (%) | SGD: 91.30 ± 0.17 | **92.20 ± 0.14** |

> 📌 SGDF consistently improves over SGD and conventional momentum-based optimizers across CNNs, object detection, and frozen-backbone ViT post-training.
> 
## 🎓 Related GitHub Repositories

Some of the experimental code in our paper was borrowed from the following repositories. We sincerely thank these authors for their excellent open-source contributions.

- https://github.com/tomgoldstein/loss-landscape
- https://github.com/amirgholami/PyHessian
- https://github.com/juntang-zhuang/Adabelief-Optimizer
- https://github.com/jeonsworld/ViT-pytorch


## 🛠️ Hyper-parameters

### 🔧 Hyper-parameters in PyTorch

> ⚠️ **Note:** Weight decay varies across tasks. For each task, we follow the original repository settings and only replace the optimizer and its related hyper-parameters.

|           Task            | lr   | beta1 | beta2 | epsilon | weight_decay |
| :-----------------------: | :--- | :---- | :---- | :------ | :----------- |
|           CIFAR           | 5e-1 | 0.9   | 0.999 | 1e-8    | 5e-4         |
|         ImageNet          | 5e-1 | 0.5   | 0.999 | 1e-8    | 1e-4         |
|            ViT            | 5e-1 | 0.9   | 0.999 | 1e-8    | 0            |
|           WGAN            | 1e-2 | 0.5   | 0.999 | 1e-8    | 0            |
| Object Detection (PASCAL) | 1e-2 | 0.9   | 0.999 | 1e-8    | 1e-4         |


## 📚 Citation

If you find this repository useful, please consider giving it a star ⭐ and citing our work 📄.

```bibtex
@article{yao2026dynamic,
  title={Dynamic Momentum Recalibration in Online Gradient Learning},
  author={Yao, Zhipeng and Yu, Rui and Chang, Guisong and Li, Ying and Zhang, Yu and Li, Dazhou},
  journal={arXiv preprint arXiv:2603.06120},
  year={2026}
}
