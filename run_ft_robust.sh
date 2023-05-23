#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --min-lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=base --rho 0.1 --experiment vit_ft_robust_short --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5

python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --min-lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --rho 0.2 --experiment vit_ft_robust_short --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --min-lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --rho 0.3 --experiment vit_ft_robust_short --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --min-lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --rho 0.05 --experiment vit_ft_robust_short --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --min-lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --rho 0.1 --experiment vit_ft_robust_short --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --min-lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --rho 0.5 --experiment vit_ft_robust_short --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5

# sleep 2s

# sgd

#python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=base --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 

# sleep 20s

# sam
#python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 0.1

#sleep 20s
  
# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 0.001

# sleep 20s

# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 0.01

# sleep 20s


# sam only bn
# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 0.01 --only_bn 

# sleep 20s


# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 1.0 --only_bn

# sleep 20s


# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 0.1 --only_bn

# sleep 20s


# sam adaptive
# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 1. --isASAM

# sleep 20s


# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 0.1 --isASAM

# sleep 20s

# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 0.01 --isASAM

# sleep 20s

# sam only bn adaptive
# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 0.1 --only_bn --isASAM 

# sleep 20s


# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 1.0 --only_bn --isASAM

# sleep 20s


# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 10. --only_bn --isASAM 

# sleep 20s


# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 0.0001 --isASAM --layerwise

# sleep 20s

# sam adaptive layerwise
# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 0.001 --isASAM --layerwise

# sleep 20s


# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 0.01 --isASAM --layerwise

# sleep 20s


# sam only bn adaptive layerwise
# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 0.1 --only_bn --isASAM --layerwise

# sleep 20s


# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 0.01 --only_bn --isASAM --layerwise

# sleep 20s

# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 0.001 --only_bn --isASAM --layerwise

# sleep 20s


# sam adaptive elementwise linf 
# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 0.00001 --isASAM --elementwise_linf

# sleep 20s 

# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 0.001 --isASAM --elementwise_linf

# sleep 20s


# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 0.0001 --isASAM --elementwise_linf

# sleep 20s


# sam only bn adaptive elementwise linf


# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 0.1 --only_bn --isASAM --elementwise_linf

# sleep 20s

# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 0.001 --only_bn --isASAM --elementwise_linf

# sleep 20s


# python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --model vit_s --epochs 2 --batch-size 160 --lr 1e-4 --sched cosine --mean 0. --std 1. --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment robust_vit_ft --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 1 --cooldown-epochs 0 --warmup-lr 1e-5 --rho 0.01 --only_bn --isASAM --elementwise_linf


