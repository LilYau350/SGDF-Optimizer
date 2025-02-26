"""Train CIFAR100 with PyTorch."""
from __future__ import print_function

import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import os
import argparse
import time
import csv
import random
import numpy as np
from models import ResNet34, DenseNet121,vgg11
from torch.optim import Adam, SGD, RAdam, AdamW
from optimizers import MSVAG, SGDF, AdaBound, Lion, SophiaG, MomentumKOALA, AdaBelief



def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('--total_epoch', default=200, type=int, help='Total number of training epochs')
    parser.add_argument('--decay_epoch', default=150, type=int, help='Number of epochs to decay learning rate')
    parser.add_argument('--model', default='resnet', type=str, help='model',
                        choices=['resnet', 'densenet', 'vgg'])
    parser.add_argument('--optim', default='adam', type=str, help='optimizer',
                        choices=['sgdf', 'sgd', 'adam', 'radam', 'adamw','msvag', 'lion', 'sophia', 'adabelief', 'koala-m'])
    parser.add_argument('--run', default=0, type=int, help='number of runs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='learning rate decay')
    parser.add_argument('--final_lr', default=0.1, type=float,
                        help='final learning rate of AdaBound')
    parser.add_argument('--gamma', default=1e-3, type=float,
                        help='convergence speed term of AdaBound')
    parser.add_argument('--eps', default=1e-8, type=float, help='eps for var adam')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2') 
    parser.add_argument('--rho', default=0.04, type=float, help='rho for sophia') 
    # KOALA specific args
    parser.add_argument('--r', type=float, help='None for adaptive', default=None)
    parser.add_argument('--sw', type=float, default=0.1)
    parser.add_argument('--sv', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--sigma', type=float, default=0.1)
    
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--batchsize', type=int, default=128, help='batch size')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay for optimizers')
    parser.add_argument('--reset', action = 'store_true',
                        help='whether reset optimizer at learning rate decay')     
    return parser

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_dataset(args):
    print('==> Preparing data..')
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=2)

    return train_loader, test_loader


def get_ckpt_name(model='resnet', optimizer='sgd', lr=0.001, final_lr=0.1, momentum=0.9,
                  beta1=0.9, beta2=0.999, gamma=1e-3, eps=1e-8, weight_decay=5e-4, rho=0.04,
                  r=None, sw= 0.1, sv=0.1, alpha=0.9, sigma=0.1,
                  reset = False, run = 0,):
    name = {
        'sgdf': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2, weight_decay, eps, run),
        'sgd': 'lr{}-momentum{}-wdecay{}-run{}'.format(lr, momentum,weight_decay, run),
        'adam': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2, weight_decay, eps, run),
        'radam': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2, weight_decay, eps, run),
        'adamw': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2, weight_decay, eps, run),
        'msvag': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2, weight_decay, eps, run),
        'lion': 'lr{}-betas{}-{}-wdecay{}-run{}'.format(lr, beta1, beta2, weight_decay,  run),
        'sophia': 'lr{}-betas{}-{}-rho{}-wdecay{}-run{}'.format(lr, beta1, beta2, rho, weight_decay, run),
        'adabelief': 'lr{}-betas{}-{}-eps{}-wdecay{}-run{}'.format(lr, beta1, beta2, eps, weight_decay, run),
        'adabound': 'lr{}-betas{}-{}-final_lr{}-gamma{}-wdecay{}-run{}'.format(lr, beta1, beta2, final_lr, gamma, weight_decay, run),
        'koala-m': 'lr{}-r{}-sw{}-sv{}-alpha{}-sigma{}-wdecay{}-run{}'.format(lr, r, sw, sv, alpha, sigma, weight_decay, run),

    }[optimizer]
    return '{}-{}-{}-reset{}'.format(model, optimizer, name, str(reset))



def load_checkpoint(ckpt_name):
    print('==> Resuming from checkpoint..')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, 'checkpoint')
    path = os.path.join(checkpoint_dir, ckpt_name)
    assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
    assert os.path.exists(path), 'Error: checkpoint {} not found'.format(ckpt_name)
    return torch.load(path)


def build_model(args, device, ckpt=None):
    print('==> Building model..')
    net = {
        'resnet': ResNet34,
        'densenet': DenseNet121,
        'vgg':vgg11,
    }[args.model]()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if ckpt:
        net.load_state_dict(ckpt['net'])

    return net


def create_optimizer(args, model_params):
    args.optim = args.optim.lower()
    if args.optim == 'sgdf':
        return SGDF(model_params, args.lr, betas=(args.beta1, args.beta2),
                        weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim =='sgd':
        return SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        return Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'radam':
        return RAdam(model_params, args.lr, betas=(args.beta1, args.beta2),
                        weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'adamw':
        return AdamW(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'msvag':
        return MSVAG(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'lion':
        return Lion(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay)
    elif args.optim == 'sophia':
        return SophiaG(model_params, args.lr, betas=(args.beta1, args.beta2),
                          rho=args.rho, weight_decay=args.weight_decay)
    elif args.optim == 'adabelief':
        return AdaBelief(model_params, args.lr, betas=(args.beta1, args.beta2),
                        weight_decay=args.weight_decay, eps=args.eps, amsgrad=False, 
                        weight_decouple=False, fixed_decay=False, rectify=False,
                        degenerated_to_sgd=False, print_change_log=False)
    elif args.optim == 'adabound':
        return AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma,
                        weight_decay=args.weight_decay)
    elif args.optim == 'koala-m':
        return MomentumKOALA(model_params, lr=args.lr, r=args.r, sw=args.sw, sv=args.sv, 
                             alpha=args.alpha, sigma=args.sigma,
                             weight_decay=args.weight_decay)
    else:
        print('Optimizer not found')


def train(net, epoch, device, data_loader, optimizer, criterion, args):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if args.optim == 'koala-m':
            loss_var = torch.mean(torch.pow(loss, 2))
            optimizer.update(loss, loss_var)
        else:
            optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print('train acc %.3f' % accuracy)
    print('train loss %.3f' % loss.item())
    return {'loss': train_loss / len(data_loader), 'accuracy': accuracy}

def test(net, device, data_loader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(' test acc %.3f' % accuracy)
    print('test loss %.3f'% loss.item())
    return {'loss': test_loss / len(data_loader), 'accuracy': accuracy}

def adjust_learning_rate(optimizer, epoch, step_size=50, gamma=0.1, reset = False):
    for param_group in optimizer.param_groups:
        if epoch % step_size==0 and epoch>0:
            param_group['lr'] *= gamma

    if  epoch % step_size==0 and epoch>0 and reset:
        optimizer.reset()

def main():
    parser = get_parser()
    args = parser.parse_args()
    set_random_seed(args.run)
    train_loader, test_loader = build_dataset(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_name = get_ckpt_name(model=args.model, optimizer=args.optim, lr=args.lr, final_lr=args.final_lr,
                              momentum=args.momentum, eps=args.eps,
                              beta1=args.beta1, beta2=args.beta2, 
                              r=args.r, sw=args.sw, sv=args.sv,
                              alpha=args.alpha, sigma=args.sigma,
                              rho=args.rho, gamma= args.gamma,                              
                              reset=args.reset, run=args.run,
                              weight_decay=args.weight_decay)
    print('ckpt_name')
    
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    curve_dir = os.path.join(base_dir, 'curve') 
    checkpoint_dir = os.path.join(base_dir, 'checkpoint')  
    
    if args.resume:
        ckpt = load_checkpoint(ckpt_name)
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']

        curve = os.path.join('curve', ckpt_name)     
        curve = torch.load(curve)
        train_accuracies = curve['train_acc']
        test_accuracies = curve['test_acc']
    else:
        ckpt = None
        best_acc = 0
        start_epoch = -1
        train_accuracies = []
        test_accuracies = []

    net = build_model(args, device, ckpt=ckpt)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, net.parameters())

    for epoch in range(start_epoch + 1, args.total_epoch):
        start = time.time()
        adjust_learning_rate(optimizer, epoch, step_size=args.decay_epoch, gamma=args.lr_gamma, reset = args.reset)
        train_metrics = train(net, epoch, device, train_loader, optimizer, criterion, args)
        test_metrics = test(net, device, test_loader, criterion)
        train_acc = train_metrics['accuracy']
        test_acc = test_metrics['accuracy']
        end = time.time()
        print('Time: {}'.format(end - start))

        # Save checkpoint.
        if test_acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            torch.save(state, os.path.join(checkpoint_dir, ckpt_name))
            best_acc = test_acc

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # Ensure curve directory exists
        if not os.path.isdir(curve_dir):
            os.mkdir(curve_dir)
            
        # Save accuracies in the curve directory
        torch.save({'train_acc': train_accuracies, 'test_acc': test_accuracies},
                    os.path.join(curve_dir, ckpt_name))
   
if __name__ == '__main__':
    main()
