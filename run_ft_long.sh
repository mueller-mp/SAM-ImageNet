#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


sleep 2s

# # adamw

python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=base --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 

# sleep 20s

# # sam

python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 0.01
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 0.001
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 0.1

# sleep 20s


# # sam only bn
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 0.1 --only_bn
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 0.01 --only_bn
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 1. --only_bn

# sleep 20s

# # sam adaptive
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 0.01 --isASAM
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 0.1 --isASAM
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 1. --isASAM

# sleep 20s

# # sam only bn adaptive

python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 0.1 --only_bn --isASAM
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 1.0 --only_bn --isASAM
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 10.0 --only_bn --isASAM

# sleep 20s


# # sam adaptive layerwise
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 0.0001 --isASAM --layerwise
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 0.001 --isASAM --layerwise
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 0.01 --isASAM --layerwise


# sleep 20s


# # sam only bn adaptive layerwise

python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 0.001 --only_bn --isASAM --layerwise
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 0.01 --only_bn --isASAM --layerwise
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 0.1 --only_bn --isASAM --layerwise

# sleep 20s

# # sam adaptive elementwise linf 
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 0.00001 --isASAM --elementwise_linf
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 0.0001 --isASAM --elementwise_linf
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 0.001 --isASAM --elementwise_linf

# sleep 20s


# # sam only bn adaptive elementwise linf

python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 0.001 --only_bn --isASAM --elementwise_linf
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 0.01 --only_bn --isASAM --elementwise_linf
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch/nsingh/imagenet/ --opt adamw --model vit_s --mean 0. --std 1. --epochs 18 --batch-size 160 --lr 1e-4 --sched cosine --weight-decay 1e-4 --smoothing 0.0  --sam_variant=sam --experiment vit_robust_long_adamw --initial-checkpoint models/vit_s_cvst_robust.pt --warmup-epochs 2 --cooldown-epochs 2 --rho 0.1 --only_bn --isASAM --elementwise_linf
