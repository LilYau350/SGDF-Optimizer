import os
import torch
import torch.distributed as dist
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import tools.dist_util as dist_util
import torch.optim
from optimizers import AdaBound, AdaBelief, RAdam, SGDF, MSVAG

def build_optimizer(args, model):
        # define optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
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

def save_checkpoint(args, epoch, model, optimizer):
    if dist_util.is_main_process():
        checkpoint_dir = os.path.join('checkpoint')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        
        filename = os.path.join(checkpoint_dir, f"{args.model_type}_{args.optimizer}.pth")
        
        torch.save(state, filename)
        print(f"Checkpoint saved: {filename}")


def load_checkpoint(ckpt_path, model=None, optimizer=None):
    if dist_util.is_main_process():
        print('==> Resuming from checkpoint..')
    assert os.path.exists(ckpt_path), 'Error: checkpoint {} not found'.format(ckpt_path)
    checkpoint = torch.load(ckpt_path)
    if model:
        model.load_state_dict(checkpoint['model'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint