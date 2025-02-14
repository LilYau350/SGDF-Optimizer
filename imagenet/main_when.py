import argparse
import os
import random
import shutil
import time
import warnings
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from adabound import AdaBound
from shufflenet import *
from senet import *
import pandas as pd
from SGD_GC import SGD_GC
from adabelief_pytorch import AdaBelief
from AdamW import AdamW
from RAdam import RAdam
from SGDF import SGDF
#from MSVAG import MSVAG
import dist_util
from torch.nn import SyncBatchNorm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.transforms import AutoAugmentPolicy
import torchvision.datasets

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',default='/root/data1/ILSVRC2012',#'/data1/ILSVRC2012',#
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg11',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--optimizer', default='sgdf', type=str)
parser.add_argument('--centralize', default=False, dest='centralize', action = 'store_true')
parser.add_argument('--reset', default=False, dest='reset', action = 'store_true')
parser.add_argument('--warmup', default=False, dest='warmup', action = 'store_true')
parser.add_argument('--reset_resume_optim', default=False, dest='reset_resume_optim', action = 'store_true')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--when', nargs='+', type=int, default=[30,60,90])
parser.add_argument('--save_epoch', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup_epoch', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--fixed_decay', action='store_true')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=2025, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--eps', default=1e-8, type=float, help='eps in Adabelief')
parser.add_argument('--beta1', default=0.5, type=float, help='beta1 in Adabelief')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 in Adabelief')
parser.add_argument('--weight_decouple', default=True, type=bool, help='Weight decouple in Adabelief')
parser.add_argument("--parallel", default=False, type=bool, help="Use multi-GPU training")
parser.add_argument('--amp', default=False, type=bool, help='Use AMP for mixed precision training')
parser.add_argument('--lr_decay', default='cosine', type=str, choices=['cosine', 'stage'], help='Choise how to lr decay.')


best_acc1 = 0


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


def main_worker(args):
    filename = 'model-{}-optimizer-{}-lr-{}-epochs-{}-eps{}-beta1{}-beta2{}-wd{}-batch{}-lr_decay-{}'.format(
        args.arch, args.optimizer, args.lr, args.epochs, args.eps, args.beta1, args.beta2, args.weight_decay, args.batch_size, args.lr_decay)
    
    if dist_util.is_main_process():
        print(filename)

    global best_acc1
    
    if args.parallel:
        dist_util.setup_dist()  
        local_rank = int(os.getenv('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # create model
    if args.pretrained:
        if dist_util.is_main_process():
            print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        if dist_util.is_main_process():
            print("=> creating model '{}'".format(args.arch))
        if args.arch == 'resnet50':
            model = models.resnet50()
        elif args.arch == 'vgg11':
            model = models.vgg11_bn()
        elif args.arch == 'densenet161':
            model = models.densenet161()
        elif args.arch == 'shufflenet_v2_x0_5':
            model = shufflenet_v2_x0_5(pretrained=False)
        elif args.arch == 'se_resnet18':
            model = se_resnet18()
        else:
            model = models.__dict__[args.arch]()
            
    model = model.to(device)
    
    if args.parallel:
        model = SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    '''
    model.half()  # convert to half precision
    for layer in model.modules():
      if isinstance(layer, nn.BatchNorm2d):
        layer.float()
    '''

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    if args.optimizer == 'sgd' and (not args.centralize):
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd' and args.centralize:
        optimizer = SGD_GC(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'adabound':
        optimizer = AdaBound(model.parameters(), args.lr, eps=args.eps, betas=(args.beta1, args.beta2))
    elif args.optimizer == 'adabelief':
        optimizer = AdaBelief(model.parameters(), args.lr, eps=args.eps, betas=(args.beta1, args.beta2), weight_decouple = args.weight_decouple, 
                              weight_decay = args.weight_decay, fixed_decay = args.fixed_decay, rectify=False)
    elif args.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), args.lr, eps=args.eps, betas=(args.beta1, args.beta2), weight_decay = args.weight_decay)
    elif args.optimizer == 'sgdf':
        optimizer = SGDF(model.parameters(), args.lr, eps=args.eps, betas=(args.beta1, args.beta2), weight_decay = args.weight_decay)
    #elif args.optimizer == 'msvag':
    #    optimizer = MSVAG(model.parameters(), args.lr, eps=args.eps, betas=(args.beta1, args.beta2), weight_decay = args.weight_decay)
    else:
        if dist_util.is_main_process():
            print('Optimizer not found')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            if dist_util.is_main_process():
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
                
            if dist_util.is_main_process():
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            if dist_util.is_main_process():
                print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        if args.start_epoch is None:
            args.start_epoch = 0
        train1, train5, test1, test5 = [], [], [], []

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    
    if args.parallel:
        world_size = dist.get_world_size()  
        per_gpu_batch_size = args.batch_size // world_size  

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=dist.get_rank())
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=dist.get_rank())        
    else:
        per_gpu_batch_size = args.batch_size  
        train_sampler = None  
        val_sampler = None         

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=per_gpu_batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=per_gpu_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    train_loader, val_loader = DataPrefetcher(train_loader), DataPrefetcher(val_loader)
    
    if args.lr_decay == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, verbose=True)
        
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.when, gamma=0.1, verbose=True)

        
    # If using mixed precision, initialize the GradScaler
    scaler = GradScaler() if args.amp else None   
    
    if args.evaluate:
        validate(val_loader, model, criterion, device, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.parallel:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        # train for one epoch
        _train1, _train5 = train(train_loader, model, criterion, optimizer, epoch, device, scheduler, scaler, args)
        scheduler.step()        

        # evaluate on validation set
        acc1, _test5 = validate(val_loader, model, criterion, device, args)
        
        if dist_util.is_main_process():
            train1.append(_train1.data.cpu().numpy())
            train5.append(_train5.data.cpu().numpy())
            test1.append(acc1.data.cpu().numpy())
            test5.append(_test5.data.cpu().numpy())
            results = {}
            results['train1'] = train1
            results['train5'] = train5
            results['test1'] = test1
            results['test5'] = test5
            df = pd.DataFrame(data = results)
            df.to_csv(filename+'.csv')

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

        if dist_util.is_main_process():
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename = filename, epoch=epoch, save_epoch = args.save_epoch)
            
        if args.parallel: 
            dist.barrier() 


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


def train(train_loader, model, criterion, optimizer, epoch, device, scheduler, scaler, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Move images and targets to the correct device
        images = images.to(device, non_blocking=True)  # Use device
        target = target.to(device, non_blocking=True)  # Use device
        
        # Use autocast for mixed precision if enabled
        with autocast(enabled=args.amp):
            # compute output
            output = model(images)
            loss = criterion(output, target)
                     
        if dist.is_available() and dist.is_initialized():
            target, output = sync(target, output)       

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.detach().item(), images.size(0))
        
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        # Compute gradient and do SGD step, use scaler for mixed precision
        optimizer.zero_grad()

        if args.amp:
            # Scales the loss and performs backward pass with mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()  # Update the scale for next iteration
        else:
            # Standard backward pass without mixed precision
            loss.backward()
            optimizer.step()
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Display progress
        if i % args.print_freq == 0:
            progress.display(i)
    
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
            #target = target.half()
            #images = images.half()

            # compute output
            output = model(images)
            loss = criterion(output, target)
            
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
            
        # TODO: this should also be done with the ProgressMeter
        if dist_util.is_main_process():
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', epoch = 0, save_epoch=30):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'_model_best.pth.tar')
    if (epoch + 1) % save_epoch == 0:
        torch.save(state, '{}-epoch-{}'.format(filename, epoch))

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

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, args, optimizer, use_warmup=False, last_epoch=-1):
        self.lr = args.lr  
        self.warmup_epoch = args.warmup_epoch  
        self.total_epochs = args.epochs 
        self.eta_min = args.eta_min if hasattr(args, 'eta_min') else 0
        self.use_warmup = use_warmup 
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if self.use_warmup and epoch < self.warmup_epoch:
            target_lr = self.lr * epoch / self.warmup_epoch
            return [target_lr]
        else:
            progress = (epoch - self.warmup_epoch) / (self.total_epochs - self.warmup_epoch) if epoch >= self.warmup_epoch else 0
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            target_lr = (self.lr - self.eta_min) * cosine_decay + self.eta_min
            return [target_lr]

    def step(self):
        super().step()

class WarmupStageScheduler(_LRScheduler):
    def __init__(self, args, optimizer, use_warmup=True, last_epoch=-1):
        self.lr = args.lr  
        self.warmup_epoch = args.warmup_epoch  
        self.total_epochs = args.epochs 
        self.when = args.when
        self.decay_factor = args.decay_factor if hasattr(args, 'decay_factor') else 0.1
        self.use_warmup = use_warmup 
        super(WarmupStageScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if self.use_warmup and epoch < self.warmup_epoch:
            target_lr = self.lr * (epoch+1) / self.warmup_epoch
            return [target_lr]
        else:
            if epoch in self.when:
                self.lr *= self.decay_factor
            return [self.lr]

    def step(self):
        super().step()
        
# Seems to speed up training by ~2%
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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if dist_util.is_main_process():
            print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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
    

if __name__ == '__main__':  
    main()
