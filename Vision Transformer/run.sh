#!/bin/bash

# NUM_GPUS=8
# GLOBAL_BATCH_SIZE=4096
# GRAD_ACCUM_STEPS=1
# LR=3e-3
# DATA_PATH="/data/ImageNet/ILSVRC2012"

# accelerate launch \
#     --multi_gpu \
#     --mixed_precision fp16 \
#     --num_processes $NUM_GPUS \
#     main.py \
#     --model vit_base_patch16_224 \
#     --data-path $DATA_PATH \
#     --output-dir ./results_vit_original \
#     --batch-size $GLOBAL_BATCH_SIZE \
#     --grad-accum-steps $GRAD_ACCUM_STEPS \
#     --lr $LR \
#     --weight-decay 0.3 \
#     --drop 0.1 \
#     --label-smoothing 0.1 \
#     --clip-grad 1.0 \
#     --epochs 300 \
#     --warmup-steps 10000 \
#     --print-freq 100 

# Hardware and Distribution
NUM_GPUS=8
GLOBAL_BATCH_SIZE=4096
# If you face OOM with 512 batch per GPU, increase GRAD_ACCUM_STEPS to 4 or 8
GRAD_ACCUM_STEPS=1 
DATA_PATH="/data/ImageNet/ILSVRC2012"

# Lion Hyperparameters (Aligned with Table 12 in Lion paper)
LR=3e-4             # Lion usually uses a smaller LR than AdamW (e.g., 3e-4 or 4e-4)
WD=0.1              # Lion suggests 0.1 for ViT-B/16 (AdamW was 0.3)
MIXUP=0.5           # Lion uses Mixup 0.5 for ViT-B
RA_SPEC="rand-m15-n2" # RandAugment (2, 15)

accelerate launch \
    --multi_gpu \
    --mixed_precision fp16 \
    --num_processes $NUM_GPUS \
    main.py \
    --model vit_base_patch16_224 \
    --data-path $DATA_PATH \
    --output-dir ./results_lion_vit \
    --batch-size $GLOBAL_BATCH_SIZE \
    --grad-accum-steps $GRAD_ACCUM_STEPS \
    --opt lion \
    --lr $LR \
    --weight-decay $WD \
    --drop-path 0.1 \
    --drop 0.1 \
    --mixup $MIXUP \
    --aa $RA_SPEC \
    --label-smoothing 0.1 \
    --clip-grad 1.0 \
    --epochs 300 \
    --warmup-steps 10000 \
    --model-ema \
    --model-ema-decay 0.9999 \
    --print-freq 100