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
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.models as models
from adabound import AdaBound
from shufflenet import *
from senet import *
import pandas as pd
from optimizers import SGD_GC, SGDF, AdaBound, AdaBelief, AdamW, Lion, SophiaG, RAdam, MSVAG
import dist_util
from torch.nn import SyncBatchNorm
import torch.distributed as dist
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy
from nvidia.dali.types import DALIInterpType


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',default='/data/ILSVRC2012',#'/data1/ILSVRC2012',#
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--optimizer', default='sgdf', type=str)
parser.add_argument('--centralize', default=False, dest='centralize', action = 'store_true')
parser.add_argument('--reset', default=False, dest='reset', action = 'store_true')
parser.add_argument('--reset_resume_optim', default=False, dest='reset_resume_optim', action = 'store_true')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--when', nargs='+', type=int, default=[30,60,90])
parser.add_argument('--warmup_epoch', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup', default=False, dest='warmup', action = 'store_true')
parser.add_argument('--save_epoch', default=120, type=int, metavar='N',
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
parser.add_argument('--amp', default=True, type=bool, help='Use AMP for mixed precision training')
parser.add_argument('--lr_decay', default='cosine', type=str, choices=['cosine', 'stage'], help='Choise how to lr decay.')


best_acc1 = 0

def create_model(args):
        # create model
    if args.pretrained:
        if dist_util.is_main_process():
            print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        if dist_util.is_main_process():
            print("=> creating model '{}'".format(args.arch))
        if args.arch == 'resnet18':
            model = models.resnet18()
        elif args.arch == 'resnet34':
            model = models.resnet34()
        elif args.arch == 'resnet50':
            model = models.resnet50()     
                     
        elif args.arch == 'vgg11':
            model = models.vgg11_bn()
        elif args.arch == 'vgg13':
            model = models.vgg13_bn()
            
        elif args.arch == 'densenet121':
            model = models.densenet121()            
        elif args.arch == 'densenet161':
            model = models.densenet161()
            
        elif args.arch == 'shufflenet_v2_x0_5':
            model = shufflenet_v2_x0_5(pretrained=False)
        elif args.arch == 'se_resnet18':
            model = se_resnet18()
        else:
            model = models.__dict__[args.arch]()
            
    return model


def create_optimzer(model, args):
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

    return optimizer


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

        
def count_files(data_dir):
    return sum(len(files) for _, _, files in os.walk(data_dir))


def get_dali_loader(data_dir, batch_size, num_threads, device_id, seed, crop_size, is_training=True):
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist!")

    pipeline = DALIPipeline(
        data_dir=data_dir,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        seed=seed,
        crop_size=crop_size,
        is_training=is_training
    )
    pipeline.build()

    # Automatically determine the number of samples (epoch size) from the pipeline
    try:
        epoch_size = pipeline.epoch_size("Reader")
    except RuntimeError:
        # Fallback to manual file counting if metadata is unavailable
        epoch_size = count_files(data_dir)

    # Return the iterator for provided pipeline
    return DALIGenericIterator(
        pipelines=pipeline,
        output_map=["images", "labels"],  # Map pipeline outputs to keys
        size=epoch_size,                  # Number of samples in the epoch
        auto_reset=True                   # Automatically reset after one epoch
    )
    
class DALIPipeline(Pipeline):
    def __init__(self, data_dir, batch_size, num_threads, device_id, seed, crop_size,
                 is_training=True, num_shards=1, shard_id=0, prefetch_queue_depth=3):
        super().__init__(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed,
            prefetch_queue_depth=prefetch_queue_depth,
        )

        self.jpegs, self.labels = fn.readers.file(
            file_root=data_dir,
            random_shuffle=is_training,
            name="Reader",
            num_shards=num_shards,
            shard_id=shard_id,
            read_ahead=True,
            stick_to_shard=True,
        )

        self.jpegs, self.labels = fn.readers.file(
            file_root=data_dir,
            random_shuffle=is_training,
            name="Reader",
            num_shards=num_shards, shard_id=shard_id,
            read_ahead=True, stick_to_shard=True,
            dont_use_mmap=False  # NVMe
        )
        if is_training:
            images = fn.decoders.image_random_crop(
                self.jpegs, device="mixed", output_type=types.RGB,
                random_area=[0.08, 1.0],
                random_aspect_ratio=[3/4, 4/3],
            )
            images = fn.resize(images, device="gpu",
                               size=(crop_size, crop_size),
                               interp_type=DALIInterpType.INTERP_LINEAR)
            mirror = fn.random.coin_flip(0.5)
        else:
            images = fn.decoders.image(self.jpegs, device="mixed", output_type=types.RGB)
            images = fn.resize(images, device="gpu", resize_shorter=256,
                               interp_type=DALIInterpType.INTERP_LINEAR)
            images = fn.crop(images, device="gpu", crop=(crop_size, crop_size),
                             crop_pos_x=0.5, crop_pos_y=0.5)
            mirror = 0

        self.images = fn.crop_mirror_normalize(
            images / 255.0,
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
            mirror=mirror, output_layout="CHW", dtype=types.FLOAT
        )
        self.labels = fn.cast(self.labels, dtype=types.INT64)

    def define_graph(self):
        return self.images, self.labels

        
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


def train(train_loader, model, criterion, optimizer, epoch, device, scaler, args):
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
    for i, data in enumerate(train_loader):
        images = data[0]["images"] 
        target = data[0]["labels"]
        
        if target.dim() == 2 and target.size(1) == 1:
            target = target.squeeze(1)
        target = target.to(device, dtype=torch.long, non_blocking=True)
        
        # measure data loading time
        data_time.update(time.time() - end)
        
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
        for i, data in enumerate(val_loader):
            images = data[0]["images"] 
            target = data[0]["labels"]           
            
            if target.dim() == 2 and target.size(1) == 1:
                target = target.squeeze(1)
            target = target.to(device, dtype=torch.long, non_blocking=True)
            
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
    
def _atomic_save(obj, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path) 

def save_checkpoint(filename, epoch, model, optimizer):
    checkpoint_dir = os.path.join('checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)

    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': int(epoch),
    }

    out = os.path.join(checkpoint_dir, f"{filename}_{epoch}.pth")
    _atomic_save(state, out)
    print(f"Checkpoint saved: {out}")


def load_checkpoint(ckpt_path, model=None, optimizer=None, device='cpu'):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    checkpoint = torch.load(ckpt_path, map_location=device)

    if model is not None:
        model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    start_epoch = int(checkpoint.get('epoch', 0))
    return start_epoch, checkpoint


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, ' 
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        
    filename = 'model-{}-optimizer-{}-lr-{}-epochs-{}-eps{}-beta1{}-beta2{}-wd{}-batch{}-lr_decay-{}-seed{}'.format(
        args.arch, args.optimizer, args.lr, args.epochs, args.eps, args.beta1, args.beta2, args.weight_decay, args.batch_size, args.lr_decay, args.seed)
    
    if dist_util.is_main_process():
        print(filename)

    global best_acc1
    
    if args.parallel:
        dist_util.setup_dist()  
        local_rank = int(os.getenv('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
           
    model = create_model(args).to(device)
            
    # model = model
    
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

    optimizer = create_optimzer(model, args)
        
    csv_path = f"{filename}.csv"

    if args.resume and os.path.isfile(args.resume):
        if dist_util.is_main_process():
            print(f"=> loading checkpoint '{args.resume}'")

        opt_obj = None if getattr(args, "reset_resume_optim", False) else optimizer

        saved_epoch, _ = load_checkpoint(args.resume, model=model, optimizer=opt_obj, device=device)

        args.start_epoch = int(saved_epoch) + 1

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df.iloc[:saved_epoch].copy()  
            train1 = df['train1'].tolist() if 'train1' in df else []
            train5 = df['train5'].tolist() if 'train5' in df else []
            test1  = df['test1'].tolist()  if 'test1'  in df else []
            test5  = df['test5'].tolist()  if 'test5'  in df else []
        else:
            train1, train5, test1, test5 = [], [], [], []

        if dist_util.is_main_process():
            print(f"=> resume done, next epoch = {args.start_epoch}")

    else:
        if dist_util.is_main_process() and args.resume:
            print(f"=> no checkpoint found at '{args.resume}'")
        args.start_epoch = 0
        train1, train5, test1, test5 = [], [], [], []

    
    if args.parallel:
        world_size = dist.get_world_size()  
        rank = torch.distributed.get_rank()  
        per_gpu_batch_size = args.batch_size // world_size  
    else:
        per_gpu_batch_size = args.batch_size 
        rank = 0  
        world_size = 1

    # Create DALI training pipeline
    train_pipeline = DALIPipeline(
        data_dir=os.path.join(args.data, 'train'),
        batch_size=per_gpu_batch_size,
        num_threads=args.workers,
        device_id=rank,
        seed=args.seed,
        crop_size=224,
        is_training=True,
        num_shards=world_size,  # Number of data shards
        shard_id=rank,
        prefetch_queue_depth=3
    )
    train_pipeline.build()

    train_loader = DALIGenericIterator(
        pipelines=train_pipeline,
        output_map=["images", "labels"],
        # size=train_pipeline.epoch_size("Reader") // world_size,
        auto_reset=True,
        # fill_last_batch=True,
        reader_name="Reader", 
        last_batch_policy=LastBatchPolicy.PARTIAL 
    )

    # Create DALI validation pipeline
    val_pipeline = DALIPipeline(
        data_dir=os.path.join(args.data, 'val'),
        batch_size=per_gpu_batch_size,
        num_threads=args.workers,
        device_id=rank,
        seed=args.seed,
        crop_size=224,
        is_training=False,
        num_shards=world_size,
        shard_id=rank,
        prefetch_queue_depth=3
    )
    val_pipeline.build()

    val_loader = DALIGenericIterator(
        pipelines=val_pipeline,
        output_map=["images", "labels"],
        # size=val_pipeline.epoch_size("Reader") // world_size,
        auto_reset=True,
        reader_name="Reader", 
        last_batch_policy=LastBatchPolicy.PARTIAL  
    )

    if args.lr_decay == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, verbose=True)
        
    elif args.lr_decay == 'stage':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.when, gamma=0.1, verbose=True)
    
    else:
        raise ValueError(f"{args.lr_decay} does not exist!")
        
        
    # If using mixed precision, initialize the GradScaler
    scaler = GradScaler() if args.amp else None   
    
    if args.evaluate:
        validate(val_loader, model, criterion, device, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        
        # train for one epoch
        _train1, _train5 = train(train_loader, model, criterion, optimizer, epoch, device, scaler, args)
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
        if is_best or (epoch+1) % args.save_epoch ==0:
            if dist_util.is_main_process():
                save_checkpoint(filename, epoch, model, optimizer)
            
        if args.parallel: 
            dist.barrier() 
            
            
if __name__ == '__main__':  
    main()


