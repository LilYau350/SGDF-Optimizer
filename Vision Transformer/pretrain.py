import os
import argparse
import json
import time
import datetime
from pathlib import Path

import torch
from tqdm import tqdm
from torchvision.datasets import ImageFolder

import timm
from timm.data import create_transform, create_loader, Mixup
from timm.scheduler import CosineLRScheduler
from timm.utils import accuracy, AverageMeter, ModelEmaV2
from timm.loss import SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2

from accelerate import Accelerator
from accelerate.utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser("ViT Training (ViT Default with Lion Hints)")

    # Basic setup
    parser.add_argument('--data-path', type=str, default='/data/ImageNet/ILSVRC2012')
    parser.add_argument('--output-dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--resume', type=str, default=None)

    # Model parameters
    parser.add_argument('--model', type=str, default='vit_base_patch16_224')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--num-classes', type=int, default=1000)
    parser.add_argument('--drop-path', type=float, default=0.0) # Lion uses 0.1
    parser.add_argument('--drop', type=float, default=0.1)      # Standard ViT 0.1

    # Training parameters
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--grad-accum-steps', type=int, default=1)
    parser.add_argument('--clip-grad', type=float, default=1.0) # Standard ViT 1.0

    # Optimizer parameters
    parser.add_argument('--opt', type=str, default='adamw')
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--weight-decay', type=float, default=0.3) # Lion uses 0.1
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)

    # Augmentation parameters
    parser.add_argument('--mixup', type=float, default=0.0)             # Lion uses 0.5
    parser.add_argument('--cutmix', type=float, default=0.0)
    parser.add_argument('--label-smoothing', type=float, default=0.1)  # Standard ViT 0.1
    parser.add_argument('--re-prob', type=float, default=0.0)
    parser.add_argument('--aa', type=str, default=None)                # Lion uses 'rand-m15-n2'

    # EMA parameters
    parser.add_argument('--model-ema', action='store_true', default=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.9999)

    parser.add_argument('--warmup-steps', type=int, default=10000)

    return parser.parse_args()

def build_dataloaders(args, accelerator, batch_size_per_device):
    train_transform = create_transform(
        input_size=args.img_size, is_training=True, auto_augment=args.aa, 
        interpolation='bicubic', re_prob=args.re_prob,
    )
    val_transform = create_transform(
        input_size=args.img_size, is_training=False, interpolation='bicubic',
    )
    train_set = ImageFolder(os.path.join(args.data_path, 'train'), transform=train_transform)
    val_set = ImageFolder(os.path.join(args.data_path, 'val'), transform=val_transform)

    train_loader = create_loader(
        train_set, input_size=(3, args.img_size, args.img_size),
        batch_size=batch_size_per_device, is_training=True,
        num_workers=args.num_workers, distributed=accelerator.use_distributed,
    )
    val_loader = create_loader(
        val_set, input_size=(3, args.img_size, args.img_size),
        batch_size=batch_size_per_device, is_training=False,
        num_workers=args.num_workers, distributed=accelerator.use_distributed,
    )
    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0:
        mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, 
                         label_smoothing=args.label_smoothing, num_classes=args.num_classes)
    return train_loader, val_loader, mixup_fn

def evaluate(model, loader, accelerator):
    model.eval()
    meter = AverageMeter()
    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images)
            outputs, targets = accelerator.gather_for_metrics((outputs, targets))
            acc1 = accuracy(outputs.detach(), targets, topk=(1,))[0]
            meter.update(acc1.item(), targets.size(0))
    return meter.avg

def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision='fp16', gradient_accumulation_steps=args.grad_accum_steps)
    set_seed(args.seed)

    num_gpus = accelerator.num_processes
    batch_size_per_device = (args.batch_size // args.grad_accum_steps) // num_gpus
    
    model = timm.create_model(args.model, pretrained=False, num_classes=args.num_classes, 
                              drop_path_rate=args.drop_path, drop_rate=args.drop)
    train_loader, val_loader, mixup_fn = build_dataloaders(args, accelerator, batch_size_per_device)
    optimizer = create_optimizer_v2(model, opt=args.opt, lr=args.lr, weight_decay=args.weight_decay, 
                                    betas=(args.beta1, args.beta2), eps=args.eps)
    
    total_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    scheduler = CosineLRScheduler(optimizer, t_initial=total_steps, warmup_t=args.warmup_steps, 
                                  warmup_lr_init=1e-6, t_in_epochs=False)

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    model_ema = ModelEmaV2(accelerator.unwrap_model(model), decay=args.model_ema_decay) if args.model_ema else None
    criterion = SoftTargetCrossEntropy() if mixup_fn is not None else torch.nn.CrossEntropyLoss()
    
    results_dir = Path(args.output_dir)
    log_file = results_dir / "log.csv"
    if accelerator.is_main_process:
        results_dir.mkdir(parents=True, exist_ok=True)
        if not log_file.exists():
            with open(log_file, "w") as f: f.write("Epoch,Train_Acc1,Val_Acc1,EMA_Val_Acc1\n")

    start_epoch, global_step, best_acc = 0, 0, 0.0

    # Enhanced Resume logic
    if args.resume:
        accelerator.print(f"Resuming from checkpoint: {args.resume}")
        accelerator.load_state(args.resume)
        
        # Manually load EMA state from checkpoint folder
        if model_ema is not None:
            ema_path = Path(args.resume) / "model_ema.bin"
            if ema_path.exists():
                model_ema.load_state_dict(torch.load(ema_path, map_location='cpu'))
            else:
                accelerator.print("Warning: EMA state not found in checkpoint. Initializing from current model.")

        state_file = Path(args.resume) / "training_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)
                start_epoch = state.get("epoch", 0) + 1
                global_step = state.get("global_step", 0)
                best_acc = state.get("best_acc", 0.0)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss_meter, train_acc1_meter = AverageMeter(), AverageMeter()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), disable=not accelerator.is_main_process)

        for step, (samples, targets) in pbar:
            with accelerator.accumulate(model):
                if mixup_fn is not None: samples, targets = mixup_fn(samples, targets)
                outputs = model(samples)
                loss = criterion(outputs, targets)

                with torch.no_grad():
                    acc_targets = targets.argmax(dim=1) if mixup_fn else targets
                    acc1 = accuracy(outputs.detach(), acc_targets, topk=(1,))[0]
                
                train_loss_meter.update(loss.detach().item(), samples.size(0))
                train_acc1_meter.update(acc1.item(), samples.size(0))

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.clip_grad: accelerator.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
                    optimizer.zero_grad()
                    if model_ema: model_ema.update(model)
                    scheduler.step_update(global_step)
                    global_step += 1
            
            if accelerator.is_main_process and step % args.print_freq == 0:
                pbar.set_description(f"Epoch {epoch} Loss: {train_loss_meter.avg:.4f} Acc: {train_acc1_meter.avg:.2f}")

        # Evaluation
        val_acc = evaluate(model, val_loader, accelerator)
        ema_acc = evaluate(model_ema.module, val_loader, accelerator) if model_ema else 0.0

        if accelerator.is_main_process:
            print(f"Epoch {epoch} | Val: {val_acc:.2f} | EMA: {ema_acc:.2f}")
            with open(log_file, "a") as f: f.write(f"{epoch},{train_acc1_meter.avg:.4f},{val_acc:.4f},{ema_acc:.4f}\n")

            if max(val_acc, ema_acc) > best_acc:
                best_acc = max(val_acc, ema_acc)
                accelerator.save_model(model, results_dir / "best_model")
                if model_ema: torch.save(model_ema.state_dict(), results_dir / "best_model_ema.pth")

            # Save full state including optimizer and EMA
            ckpt_dir = results_dir / "last_checkpoint"
            accelerator.save_state(ckpt_dir)
            if model_ema: torch.save(model_ema.state_dict(), ckpt_dir / "model_ema.bin")
            with open(ckpt_dir / "training_state.json", "w") as f:
                json.dump({"epoch": epoch, "global_step": global_step, "best_acc": best_acc}, f)

if __name__ == "__main__":
    main()