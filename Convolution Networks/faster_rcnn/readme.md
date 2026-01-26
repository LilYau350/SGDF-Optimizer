# Faster R-CNN

## This project is mostly source code from the official torchvision module of pytorch
* https://github.com/pytorch/vision/tree/master/torchvision/models/detection

## Environment configuration:
* python 3.10
* pytorch 2.0.1
* torchvision 0.15.2
* CUDA 11.8
* Ubuntu 20.04
* See `requirements.txt` for detailed environment configuration.

## File structure:
``
  ├── backbone: Feature extraction network.
  ├── network_files: Faster R-CNN network (including Fast R-CNN and RPN modules)
  ├── train_utils: training and validation related modules (including cocotools)
  ├── my_dataset.py: customised dataset for reading VOC dataset
  ├── train_mobilenet.py: use MobileNetV2 as backbone for training.
  ├── train_resnet50_fpn.py: use resnet50+FPN as backbone for training
  ├── train_multi_GPU.py: for users with multiple GPUs
  ├── predict.py: a simple prediction script that uses the trained weights for prediction testing
  ├── validation.py: validate/test the COCO metrics of the data with the trained weights and generate record_mAP.txt file
  └── pascal_voc_classes.json: pascal_voc labels file
``.

## Pre-training weights download address (download and put in backbone folder):
* MobileNetV2 weights (rename to `mobilenet_v2.pth` after download and put in `bakcbone` folder): https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
* Resnet50 weights (download and rename to `resnet50.pth`, then put in `bakcbone` folder): https://download.pytorch.org/models/resnet50-0676ba61.pth
* ResNet50+FPN weights: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
* Note, remember to rename the downloaded pre-training weights, e.g. the `fasterrcnn_resnet50_fpn_coco.pth` file is read in train_resnet50_fpn.py, not `fasterrcnn_resnet50_fpn_coco.pth`.
  Not `fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`, then just put it into the current project root directory.
 
 
## Dataset, this article uses PASCAL VOC2007 and VOC2012 dataset
* Pascal VOC2007&2012 train/val dataset download address:
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_11-May-2007.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
* Weights obtained on VOC2012 dataset using ResNet50+FPN and migration learning: link:https://pan.baidu.com/s/1ifilndFRtAV5RDZINSHj5w Extract code:dsz8

## Training method
* Ensure that the dataset is prepared in advance
* Make sure to download the corresponding pre-trained model weights in advance.
* To train mobilenetv2+fasterrcnn, use the train_mobilenet.py training script directly.
* To train resnet50+fpn+fasterrcnn, use the train_resnet50_fpn.py training script directly.
* To train with multiple GPUs, use the `python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_GPU.py` command, where the `nproc_per_node` parameter is the number of GPUs to use.
* If you want to specify which GPUs you want to use, prefix the command with `CUDA_VISIBLE_DEVICES=0,3` (e.g., I only want to use GPUs 1 and 4 of the devices).
* `CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_GPU.py`

## Caution.
* When using the training script, be careful to set `--data-path` (VOC_root) to the **root folder where you store your `VOCdevkit` folder **.
* The `-results.txt` saved during training is the COCO metrics for each epoch on the validation set, the first 12 values are the COCO metrics, and the last two values are the average training loss and the learning rate.
* When using the prediction script, set `train_weights` to your own generated weight path.

Translated with www.DeepL.com/Translator (free version)


