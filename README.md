<h1 align="center">SGDF Optimizer</h1>
<h3 align="center">Signal Processing Meets SGD: From Momentum to Filter</h3>

## Related GitHub Repositories
* Some of the experimental code in our paper was borrowed from the following repositories, thanks to these authors for open source.
<br> https://github.com/tomgoldstein/loss-landscape
<br> https://github.com/amirgholami/PyHessian
<br> https://github.com/juntang-zhuang/Adabelief-Optimizer/tree/update_0.2.0


## Table of Hyper-parameters 
### Hyper-parameters in PyTorch
* Note weight decay varies with tasks, for different tasks the weight decay is untuned from the original repository (only changed the optimizer and other hyper-parameters).

|   Task   |  lr | beta1 | beta2 | epsilon | weight_decay | 
|:--------:|-----|-------|-------|---------|--------------|
| Cifar    | 3e-1 | 0.9   | 0.999 | 1e-8    | 5e-4        | 
| ImageNet | 5e-1 |0.5   | 0.999 | 1e-8    | 1e-4         | 
| Object detection (PASCAL) | 1e-2 | 0.9   | 0.999 | 1e-8 | 1e-4         | 
| WGAN | 1e-2 |0.5| 0.999 | 1e-8   | 0            | 
