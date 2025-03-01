import logging
import os
import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

logger = logging.getLogger(__name__)

def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(args.data_dir, 'train')  # Update with your ImageNet train data directory
    val_dir = os.path.join(args.data_dir, 'val')  # Update with your ImageNet validation data directory
    trainset = datasets.ImageFolder(train_dir, transform=transform_train)
    valset = datasets.ImageFolder(val_dir, transform=transform_val) 

    return trainset, valset

