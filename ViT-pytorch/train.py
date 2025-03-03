# coding=utf-8
from __future__ import absolute_import, division, print_function
import math
import logging
import argparse
import os
import random
import numpy as np
import pandas as pd
import time
import warnings
import torch.multiprocessing as mp
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import shutil
import torch
import torch.distributed as dist
from tools.checkpoint import build_optimizer, save_checkpoint, load_checkpoint
from models.modeling import VisionTransformer, CONFIGS
from tools.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from tools.data_utils import build_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import tools.dist_util as dist_util
from tools.accuracy import AverageMeter, ProgressMeter, accuracy

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_dir', metavar='DIR', default='/home/workspace/zhipeng/ILSVRC2012',
                    help='Path to the ImageNet train data directory')
parser.add_argument("--name", required=False, default="default_run", help="Name of this run. Used for monitoring.")
parser.add_argument("--model_type", choices=["ViT-S_16", "ViT-S_32", 
                                             "ViT-B_16", "ViT-B_32", 
                                             "ViT-L_16", "ViT-L_32", 
                                             "ViT-H_14", "R50-ViT-B_16"],
                    default="ViT-B_32", help="Which variant to use.")

parser.add_argument("--pretrained_dir", type=str, default='./pretrained/ViT-B_32.npz', help="Where to search for pretrained ViT models.")
parser.add_argument('--resume', type=str, default=None, help='Path to the checkpoint to resume from')
parser.add_argument("--batch_size", default=128, type=int, help="Total batch size for training.")
parser.add_argument("--img_size", default=384, type=int, help="Resolution size")
parser.add_argument("--dataset", type=str, default='CIFAR-10', choices=['CIFAR-10', 'CIFAR-100', 'Food101', 'Oxford-102-Flowers', 'Oxford_IIIT_Pets', 'ImageNet'], help="Dataset to train on")
parser.add_argument("--num_classes", default=10, type=int, help="Number of classes in the dataset.")
parser.add_argument("--num_workers", default=8, type=int, help="Number of worker processes to use for data loading.")

parser.add_argument("--epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument('--save_epoch', default=300, type=int, metavar='N', help='number of epochs to save checkpoint')

parser.add_argument('--optimizer', default='sgdf', type=str)
parser.add_argument("--lr", default=0.25, type=float, help="The initial learning rate for optimizer.")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--beta1', default=0.9, type=float, help='beta1 in SGDF')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 in SGDF')
parser.add_argument('--eps', default=1e-8, type=float, help='eps in SGDF')
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
parser.add_argument('--weight_decouple', default=True, type=bool, help='Weight decouple in Adabelief')

parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--max_grad_norm", default=-1.0, type=float, help="Max gradient norm.")
parser.add_argument("--warmup_steps", default=0, type=int, help="Number of warm-up steps before learning rate cosine annealing.")
parser.add_argument("--decay_type", choices=["constant", "cosine", "linear"], default="cosine", help="How to decay the learning rate.")

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=2025, type=int, help="random seed for initialization")
parser.add_argument('--amp', default=True, type=bool, help='Use mixed precision training with torch.cuda.amp')
parser.add_argument("--parallel", default=False, type=bool, help="Use multi-GPU training")

args = parser.parse_args()

best_acc1 = 0
# global_step = 0
logger = logging.getLogger(__name__)


def main():
    args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    
    main_worker(args)
    
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def main_worker(args):

    filename = 'model-{}-dataset-{}-optimizer-{}-lr-{}-epochs-{}-eps{}-beta1{}-beta2{}-wd{}-batch-{}'.\
        format(args.model_type, args.dataset, args.optimizer, args.lr, args.epochs, args.eps, args.beta1, args.beta2, args.weight_decay, args.batch_size)
        
    if dist_util.is_main_process():
        print(filename)

    global best_acc1
    
    # create model
    config = CONFIGS[args.model_type]

    if args.parallel:
        dist_util.setup_dist()  
        local_rank = int(os.getenv('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=args.num_classes)
    
    if args.pretrained_dir:
        model.load_from(np.load(args.pretrained_dir))
    
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    if dist_util.is_main_process():
        print(num_params)   
        
    model = model.to(device)
    
    if args.parallel:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    train1, train5, test1, test5 = [], [], [], []
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model=model, optimizer=optimizer)
        start_epoch = checkpoint['step']
    else:
        start_epoch = 0
        
    train_loader, val_loader, train_sampler, val_sampler = build_dataset(args)
    
    steps_per_epoch = len(train_loader) 
    # print(train_loader)
    
    total_steps = (steps_per_epoch * args.epochs) // args.gradient_accumulation_steps
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = build_optimizer(args, model)    
    
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=total_steps)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=total_steps) 
        
    # Initialize GradScaler if AMP is enabled
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)        
    
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(start_epoch, args.epochs):
        if args.parallel:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        #t_total = args.num_steps
        # train for one epoch
        _train1, _train5 = train(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, device, args)
        # evaluate on validation set
        _test1, _test5 = validate(val_loader, model, criterion, device, args)
        
        train1.append(_train1.data.cpu().numpy())
        train5.append(_train5.data.cpu().numpy())
        test1.append(_test1.data.cpu().numpy())
        test5.append(_test5.data.cpu().numpy())
        results = {}
        results['train1'] = train1
        results['train5'] = train5
        results['test1'] = test1
        results['test5'] = test5
        df = pd.DataFrame(data = results)
        df.to_csv(filename+'.csv')

        # remember best acc@1 and save checkpoint
        is_best = _test1 > best_acc1
        best_acc1 = max(_test1, best_acc1)
        
        if is_best or epoch % args.save_epoch == 0:
            save_checkpoint(args, epoch, model, optimizer)
            

def sync(target, output):
    """Synchronize both target and output by gathering from all GPUs and concatenating."""
    # Create lists to hold the gathered target and output from all GPUs
    gathered_target = [torch.zeros_like(target) for _ in range(dist.get_world_size())]
    gathered_output = [torch.zeros_like(output) for _ in range(dist.get_world_size())]

    # Gather all targets and outputs from all GPUs
    dist.all_gather(gathered_target, target)
    dist.all_gather(gathered_output, output)

    # Concatenate the gathered targets and outputs along the first dimension (batch dimension)
    combined_target = torch.cat(gathered_target, dim=0)
    combined_output = torch.cat(gathered_output, dim=0)

    return combined_target, combined_output

def train(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, device, args):
    # global global_step
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        math.ceil(len(train_loader) / args.gradient_accumulation_steps),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # Switch to train mode
    model.train()

    end = time.time()
    
    # Initialize gradient accumulation
    optimizer.zero_grad()  # Ensure gradients are cleared before training
    for step, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        current_step = step + epoch * len(train_loader)
        # Move images and targets to the correct device
        images = images.to(device, non_blocking=True)  # Use device
        target = target.to(device, non_blocking=True)  # Use device

        # Forward pass with AMP
        with torch.cuda.amp.autocast(enabled=args.amp):
            output, _ = model(images)
            loss = criterion(output, target) / args.gradient_accumulation_steps  # Divide loss for gradient accumulation

        if dist.is_available() and dist.is_initialized():
            target, output = sync(target, output)   
            
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item() * args.gradient_accumulation_steps, images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # Backward pass
        if args.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update optimizer and clear gradients if accumulation is complete
        if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
            if args.max_grad_norm > 0:  # Apply gradient clipping if enabled
                if args.amp:
                    scaler.unscale_(optimizer)  # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if args.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()  # Reset gradients after updating

            # Update learning rate scheduler
            scheduler.step(current_step)
            # global_step += 1  # Increment global step
        batch_time.update(time.time() - end)
        end = time.time()

        if (step + 1) % args.gradient_accumulation_steps == 0:  
            if (step + 1) // args.gradient_accumulation_steps % args.print_freq == 0:
                progress.display((step + 1) // args.gradient_accumulation_steps)

    return top1.avg, top5.avg

def validate(val_loader, model, criterion, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # Move images and targets to the correct device
            images = images.to(device, non_blocking=True)  # Use device
            target = target.to(device, non_blocking=True)  # Use device

            # Forward pass with mixed precision (AMP)
            with torch.cuda.amp.autocast(enabled=args.amp):
                output, _ = model(images)  # Forward pass
                loss = criterion(output, target)  # Compute loss
                
            if dist.is_available() and dist.is_initialized():
                target, output = sync(target, output)   
                
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000
    
if __name__ == "__main__":
    main()
