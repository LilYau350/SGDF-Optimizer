## Table of Hyper-parameters 
### Hyper-parameters in PyTorch
* Note weight decay varies with tasks, for different tasks the weight decay is untuned from the original repository (only changed the optimizer and other hyper-parameters).

|   Task   |  lr | beta1 | beta2 | epsilon | weight_decay | 
|:--------:|-----|-------|-------|---------|--------------|
| Cifar    | 5e-1 | 0.9   | 0.999 | 1e-8    | 5e-4        | 
| ImageNet | 5e-1 |0.5   | 0.999 | 1e-8    | 1e-4         | 
| Object detection (PASCAL) | 1e-2 | 0.9   | 0.999 | 1e-8 | 1e-4         | 
| WGAN | 1e-2 |0.5| 0.999 | 1e-8   | 0            | 
| ViT | 5e-1 |0.9| 0.999 | 1e-8   | 0            | 

