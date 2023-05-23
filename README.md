I use `sbatch submit_script.job` for in order to submit to slurm. 
You can also run the following directly in order to run e.g. layerwise-l2 ASAM on ImageNet: 

`python3 -m torch.distributed.launch --nproc_per_node=8 train.py /scratch_local/datasets/ImageNet2012 --model vit_small_patch16_224 --epochs 90 --batch-size 64 --lr 0.2 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema --sam_variant=sam --experiment vit_new --rho 0.01 --isASAM  --layerwise` 

Examples for how to run SAM, SGD, only-bn, etc can be found in run_ft_long.sh

The logic for the optimizer can be found in `timm_esam/optim/layer_dp_sam.py`

The most current yml file is esam_vm.yml

