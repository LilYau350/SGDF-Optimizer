import logging
import os
import torch
import torch.distributed as dist
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import tools.dist_util as dist_util

logger = logging.getLogger(__name__)


def get_loader(args):
    if not dist_util.is_main_process():
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "CIFAR-10":
        if dist_util.is_main_process():
            trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
            testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test) 
        else:
            trainset = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_train)
            testset = datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_test)   
            
    elif args.dataset == "CIFAR-100":
        if dist_util.is_main_process():
            trainset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
            testset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test) 
        else:
            trainset = datasets.CIFAR100(root="./data", train=True, download=False, transform=transform_train)
            testset = datasets.CIFAR100(root="./data", train=False, download=False, transform=transform_test) 
        
    elif args.dataset == "Food101":
        trainset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=transform_train)
        testset = datasets.ImageFolder(os.path.join(args.data_dir, 'test'), transform=transform_test) 

    elif args.dataset == "Oxford-102-Flowers":
        trainset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=transform_train)
        testset = datasets.ImageFolder(os.path.join(args.data_dir, 'test'), transform=transform_test) 
        
    elif args.dataset == "Oxford_IIIT_Pets":
        trainset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=transform_train)
        testset = datasets.ImageFolder(os.path.join(args.data_dir, 'test'), transform=transform_test) 
        
    elif args.dataset == "ImageNet":
        trainset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=transform_train)
        testset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=transform_test) 
        
    if not dist_util.is_main_process():
        torch.distributed.barrier()

    return trainset, testset

def build_dataset(args):
    trainset, valset = get_loader(args)
    if args.parallel:
        world_size = dist.get_world_size()  
        per_gpu_batch_size = args.batch_size // world_size  
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=dist.get_rank())
        val_sampler = torch.utils.data.distributed.DistributedSampler(valset, num_replicas=world_size, rank=dist.get_rank())
    else:
        per_gpu_batch_size = args.batch_size
        train_sampler = None  
        val_sampler = None 

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=per_gpu_batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=per_gpu_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)
    
    # Data loading code
    train_loader, val_loader = DataPrefetcher(train_loader), DataPrefetcher(val_loader)
    
    return train_loader, val_loader, train_sampler, val_sampler

class DataPrefetcher():
    def __init__(self, loader, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None

    def __len__(self):
        return len(self.loader)
    
    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self.preload()
            count += 1
            yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break
