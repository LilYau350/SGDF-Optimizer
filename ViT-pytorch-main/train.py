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
import torch.optim
from optimizers import AdaBound, AdaBelief, RAdam, SGDF, MSVAG
from models.modeling import VisionTransformer, CONFIGS
from tools.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from tools.data_utils import get_loader
from torch.nn.parallel import DistributedDataParallel as DDP
import tools.dist_util as dist_util


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_dir', metavar='DIR', default='/home/workspace/zhipeng/ILSVRC2012',
                    help='Path to the ImageNet train data directory')
parser.add_argument("--name", required=False, default="default_run",
                    help="Name of this run. Used for monitoring.")
parser.add_argument("--model_type", choices=["ViT-S_16", "ViT-S_32", 
                                             "ViT-B_16", "ViT-B_32", 
                                             "ViT-L_16", "ViT-L_32", 
                                             "ViT-H_14", "R50-ViT-B_16"],
                    default="ViT-S_32", help="Which variant to use.")
parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                    help="Where to search for pretrained ViT models.")

parser.add_argument("--output_dir", default="output", type=str,
                    help="The output directory where checkpoints will be written.")
parser.add_argument("--num_workers", default=4, type=int,
                    help="Number of worker processes to use for data loading.")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument("--img_size", default=224, type=int,
                    help="Resolution size")
parser.add_argument("--batch_size", default=512, type=int,
                    help="Total batch size for training.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--max_grad_norm", default=-1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--warmup_steps", default=10000, type=int, help="Number of warm-up steps before learning rate cosine annealing.")
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument("--epochs", default=300, type=int, help="Total number of training epochs to perform.")
parser.add_argument('--save_epoch', default=60, type=int, metavar='N',
                    help='number of epochs to save checkpoint')
parser.add_argument("--decay_type", choices=["constant", "cosine", "linear"], default="cosine",
                    help="How to decay the learning rate.")
parser.add_argument('--optimizer', default='sgdf', type=str)
parser.add_argument("--lr", default=3e-2, type=float,
                    help="The initial learning rate for optimizer.")
parser.add_argument("--weight_decay", default=1e-4, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--eps', default=1e-8, type=float, help='eps in SGDF')
parser.add_argument('--beta1', default=0.5, type=float, help='beta1 in SGDF')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 in SGDF')
parser.add_argument('--weight_decouple', default=True, type=bool, help='Weight decouple in Adabelief')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int, 
                    help="random seed for initialization")
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--amp', action='store_true',
                    help='Use mixed precision training with torch.cuda.amp')
parser.add_argument("--parallel", default=False, type=bool, help="Use multi-GPU training")

args = parser.parse_args()

best_acc1 = 0
# global_step = 0
logger = logging.getLogger(__name__)


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    
    main_worker(args)

def build_optimizer(args, model):
        # define optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'adabound':
        optimizer = AdaBound(model.parameters(), args.lr, eps=args.eps, betas=(args.beta1, args.beta2))
    elif args.optimizer == 'adabelief':
        optimizer = AdaBelief(model.parameters(), args.lr, eps=args.eps, betas=(args.beta1, args.beta2), weight_decouple = args.weight_decouple, 
                              weight_decay = args.weight_decay, fixed_decay = args.fixed_decay, rectify=False)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, eps=args.eps, betas=(args.beta1, args.beta2), weight_decay = args.weight_decay)
    elif args.optimizer == 'radam':
        optimizer = RAdam(model.parameters(), args.lr, eps=args.eps, betas=(args.beta1, args.beta2), weight_decay = args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, eps=args.eps, betas=(args.beta1, args.beta2), weight_decay = args.weight_decay)
    elif args.optimizer == 'sgdf':
        optimizer = SGDF(model.parameters(), args.lr, eps=args.eps, betas=(args.beta1, args.beta2), weight_decay = args.weight_decay)
    elif args.optimizer == 'msvag':
        optimizer = MSVAG(model.parameters(), args.lr, eps=args.eps, betas=(args.beta1, args.beta2), weight_decay = args.weight_decay)
    else:
        print('Optimizer not found')
        
    return optimizer


def main_worker(args, model):

    filename = 'model-{}-optimizer-{}-lr-{}-epochs-{}-eps{}-beta1{}-beta2{}-wd{}-batch-{}'.format(args.model_type, args.optimizer, args.lr, args.epochs, args.eps, args.beta1, args.beta2, args.weight_decay, args.batch_size)
    if dist_util.is_main_process():
        print(filename)

    global best_acc1
    
    # create model
    num_classes = 1000    
    config = CONFIGS[args.model_type]

    if args.parallel:
        dist_util.setup_dist()  
        local_rank = int(os.getenv('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    # model.load_from(np.load(args.pretrained_dir))
    
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    if dist_util.is_main_process():
        print(num_params)   
        
        
    model = model.to(device)
    
    if args.parallel:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    '''
    model.half()  # convert to half precision
    for layer in model.modules():
      if isinstance(layer, nn.BatchNorm2d):
        layer.float()
    '''

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = build_optimizer(args, model)

    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            
            checkpoint = torch.load(args.resume, map_location=device)
            
            if args.start_epoch is None:
                args.start_epoch = checkpoint['epoch'] + 1
                df = pd.read_csv(filename+'.csv')
                train1, train5, test1, test5 = df['train1'].tolist(), df['train5'].tolist(), df['test1'].tolist(), df['test5'].tolist()
            else: # if specify start epoch, and resume from checkpoint, not resume previous accuracy curves
                train1, train5, test1, test5 = [], [], [], []
            best_acc1 = checkpoint['best_acc1']
            
            best_acc1 = best_acc1.to(device) if isinstance(best_acc1, torch.Tensor) else best_acc1
            
            model.load_state_dict(checkpoint['state_dict'])

            if not args.reset_resume_optim:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        if args.start_epoch is None:
            args.start_epoch = 0
        train1, train5, test1, test5 = [], [], [], []

    cudnn.benchmark = True
    trainset, valset, train_loader, val_loader = get_loader(args)
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
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=per_gpu_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    
    # Data loading code
    train_loader, val_loader = DataPrefetcher(train_loader), DataPrefetcher(val_loader)

    steps_per_epoch = len(train_loader)  
    # print(train_loader)
    
    total_steps = (steps_per_epoch * args.epochs) // args.gradient_accumulation_steps
    
    warmup_steps = 4096 / (args.batch_size * args.gradient_accumulation_steps) * args.warmup_steps * (300 / args.epochs)
    
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)

    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps) 
        
    # Initialize GradScaler if AMP is enabled
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)        
    
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            #t_total = args.num_steps
        # train for one epoch
        # _train1, _train5 = train(train_loader, model, criterion, optimizer, scheduler, epoch, args)
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

        if dist_util.is_main_process():
            save_checkpoint({
                'epoch': epoch + 1,
                'model_type': args.model_type,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename = filename, epoch=epoch, save_epoch = args.save_epoch)
            

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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', epoch = 0, save_epoch=30):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'_model_best.pth.tar')
    if (epoch + 1) % save_epoch == 0:
        torch.save(state, '{}-epoch-{}'.format(filename, epoch))
        
def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000
    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
      
if __name__ == "__main__":
    main()
