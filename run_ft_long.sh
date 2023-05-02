#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,6,7

sleep 2s

# sgd

python3 -m torch.distributed.launch --nproc_per_node=6 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 18 --batch-size 400 --lr 0.045 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=base --experiment vit_ft_long_gsam --initial-checkpoint models/ViT-S_16_gsam.npz --warmup-epochs 2 --cooldown-epochs 2 

sleep 20s

# sam

python3 -m torch.distributed.launch --nproc_per_node=6 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 18 --batch-size 400 --lr 0.045 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft_long_gsam --initial-checkpoint models/ViT-S_16_gsam.npz --warmup-epochs 2 --cooldown-epochs 2 --rho 0.01

sleep 20s


# sam only bn
python3 -m torch.distributed.launch --nproc_per_node=6 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 18 --batch-size 400 --lr 0.045 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft_long_gsam --initial-checkpoint models/ViT-S_16_gsam.npz --warmup-epochs 2 --cooldown-epochs 2 --rho 0.1 --only_bn

sleep 20s

# sam adaptive
python3 -m torch.distributed.launch --nproc_per_node=6 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 18 --batch-size 400 --lr 0.045 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft_long_gsam --initial-checkpoint models/ViT-S_16_gsam.npz --warmup-epochs 2 --cooldown-epochs 2 --rho 0.1 --isASAM

sleep 20s

# sam only bn adaptive

python3 -m torch.distributed.launch --nproc_per_node=6 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 18 --batch-size 400 --lr 0.045 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft_long_gsam --initial-checkpoint models/ViT-S_16_gsam.npz --warmup-epochs 2 --cooldown-epochs 2 --rho 1.0 --only_bn --isASAM

sleep 20s


# sam adaptive layerwise
python3 -m torch.distributed.launch --nproc_per_node=6 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 18 --batch-size 400 --lr 0.045 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft_long_gsam --initial-checkpoint models/ViT-S_16_gsam.npz --warmup-epochs 2 --cooldown-epochs 2 --rho 0.001 --isASAM --layerwise


sleep 20s


# sam only bn adaptive layerwise

python3 -m torch.distributed.launch --nproc_per_node=6 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 18 --batch-size 400 --lr 0.045 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft_long_gsam --initial-checkpoint models/ViT-S_16_gsam.npz --warmup-epochs 2 --cooldown-epochs 2 --rho 0.001 --only_bn --isASAM --layerwise

sleep 20s

# sam adaptive elementwise linf 
python3 -m torch.distributed.launch --nproc_per_node=6 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 18 --batch-size 400 --lr 0.045 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft_long_gsam --initial-checkpoint models/ViT-S_16_gsam.npz --warmup-epochs 2 --cooldown-epochs 2 --rho 0.0001 --isASAM --elementwise_linf

sleep 20s


# sam only bn adaptive elementwise linf

python3 -m torch.distributed.launch --nproc_per_node=6 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 18 --batch-size 400 --lr 0.045 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft_long_gsam --initial-checkpoint models/ViT-S_16_gsam.npz --warmup-epochs 2 --cooldown-epochs 2 --rho 0.01 --only_bn --isASAM --elementwise_linf


