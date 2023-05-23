#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7

# sleep 2s

# sgd

# python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=base --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 

# sleep 20s

# sam
# python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 0.1

# sleep 20s

python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 0.001

sleep 20s

# python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 0.01

# sleep 20s


# sam only bn
# python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 0.01 --only_bn 

# sleep 20s


# python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 1.0 --only_bn

# sleep 20s


# python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 0.1 --only_bn

# sleep 20s


# sam adaptive
# python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 1. --isASAM

# sleep 20s


# python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 0.1 --isASAM

# sleep 20s

python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 0.01 --isASAM

sleep 20s

# sam only bn adaptive
# python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 0.1 --only_bn --isASAM 

# sleep 20s


# python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 1.0 --only_bn --isASAM

# sleep 20s


# python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 10. --only_bn --isASAM 

# sleep 20s


python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 0.0001 --isASAM --layerwise

sleep 20s

# sam adaptive layerwise
# python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 0.001 --isASAM --layerwise

# sleep 20s


# python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 0.01 --isASAM --layerwise

# sleep 20s


# sam only bn adaptive layerwise
# python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 0.1 --only_bn --isASAM --layerwise

# sleep 20s


# python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 0.01 --only_bn --isASAM --layerwise

# sleep 20s

python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 0.001 --only_bn --isASAM --layerwise

sleep 20s


# sam adaptive elementwise linf 
python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 0.00001 --isASAM --elementwise_linf

 sleep 20s 

# python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 0.001 --isASAM --elementwise_linf

# sleep 20s


# python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 0.0001 --isASAM --elementwise_linf

# sleep 20s


# sam only bn adaptive elementwise linf


python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 0.1 --only_bn --isASAM --elementwise_linf

sleep 20s

# python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 0.001 --only_bn --isASAM --elementwise_linf

# sleep 20s


# python3 -m torch.distributed.launch --nproc_per_node=7 train.py /scratch/nsingh/imagenet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 0.01 --only_bn --isASAM --elementwise_linf


