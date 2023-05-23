### Creating a conda environment with the required dependencies:

`conda create -n imagenet_samon -f environment.yml
 conda activate imagenet_samon`

in case your GPUs are incompatible with the pytorch version, try:
`pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`
### Training the RN50 from scratch
For ResNet50 training from scratch, run e.g.:
##### SGD
`python3 -m torch.distributed.launch --nproc_per_node=8 train.py /path/to/ImageNet --model resnet50 --epochs 90 --batch-size 64 --lr 0.2 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --sam_variant=base --experiment test` 

##### SAM
`python3 -m torch.distributed.launch --nproc_per_node=8 train.py /path/to/ImageNet --model resnet50 --epochs 90 --batch-size 64 --lr 0.2 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --sam_variant=sam --experiment test --rho 0.05` 

##### elem l-2 onlyNorm
`python3 -m torch.distributed.launch --nproc_per_node=8 train.py /path/to/ImageNet --model resnet50 --epochs 90 --batch-size 64 --lr 0.2 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --sam_variant=base --experiment test --rho 0.01 --isASAM --only_bn` 

##### elem l-inf onlyNorm
`python3 -m torch.distributed.launch --nproc_per_node=8 train.py /path/to/ImageNet --model resnet50 --epochs 90 --batch-size 64 --lr 0.2 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --sam_variant=base --experiment test --rho 1. --isASAM --elementwise_linf --only_bn` 

##### layerwise l-2 onlyNorm
`python3 -m torch.distributed.launch --nproc_per_node=8 train.py /path/to/ImageNet --model resnet50 --epochs 90 --batch-size 64 --lr 0.2 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --sam_variant=base --experiment test --rho 0.05 --isASAM  --layerwise --only_bn` 

By removing the flag '--only_bn' and adjusting the perturbation radius '--rho 0.05' you can switch to the SAM-all variants


#### Finetuning the ViT-S
For finetuning the ViT-S with 21k pretraining:
download the weights into the models folder:

`cd models
wget gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz
cd ..`

For training with elementwise l-inf, the command is:
##### elem l-inf onlyNorm
`python3 -m torch.distributed.launch --nproc_per_node=7 train.py /path/to/ImageNet/ --model vit_small_patch16_224 --epochs 9 --batch-size 128 --lr 0.017 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --sam_variant=sam --experiment vit_ft --initial-checkpoint models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --warmup-epochs 1 --cooldown-epochs 0 --rho 0.01 --only_bn --isASAM --elementwise_linf`

The flags '--isASAM', '--elementwise_linf', '--layerwise', '--sam_variant', '--rho' can then be adjusted like above for the other sam-variants and the perturbation radii reported in the paper.
